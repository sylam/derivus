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

"""What a copy of the record proves about a day on the desk - fourteen invariants and their
evidence.

Pure over a HOME and, where one is given, the SCRIPT of what was asked. The oracle folds the record
the way every other reader does and re-runs the writer's own decisions through the writer's own
functions, so an answer here is a property of the bytes rather than of the process that made them.
It is read as a REPLICA, which is the honest posture - verification is always local - and a
chain-only home still answers the four questions that cost no key while naming the ten it cannot.

What the record cannot hold is taken as DATA. The SCRIPT says what was ASKED: which attempts the
writer refused, which were turned away short of it, the lane each run was submitted in, the rows a
settlement file instructed and the collateral calls a seat read - so `asked x recorded` is
checkable where `recorded` alone is only half the question. The DIARY - its keys and the amounts it
determines - and the P&L are the engine's, so a replica reports those as not assessed and the caller
holding an engine hands them in.

Each invariant is one function answering `{held, evidence}`, where `held` is null for a question
this home or this script could not put, and `evidence` is the rows that decided it.
"""
import json
import math

from .canon import canonical_bytes, content_hash
from .capability import (
    CAPABILITIES_POLICY, CAPABILITY_EVENTS, PORTFOLIO_TYPES, SCOPED_TYPES, apply_event,
    declarable, deepest, evaluate, holders, initial_state, nodes, scope_of, stray, under,
    verb_for)
from .errors import SpineRefusal
from .log import GENESIS_PREV, SpineLog, as_of_key
from .policy import (
    DESIGNATIONS_SECTION, TIERS_POLICY, TIERS_SECTION, TOLERANCE_POLICY, TOLERANCE_SECTION,
    compare, in_force, tiers_in_force)
from .projections import CSA, PROJECTORS, fold
from .store import BlobStore
from .vocabulary import (
    ADMIN, BLOB_FIELDS, EVENT_TYPES, EVENT_VERB, RECOVERY, WRITER, WRITER_VOICE, cited_blobs,
    is_hash, is_text)
from .verbs import (
    FEE, HELD, PAYMENT, REPLAY_FIELDS, SETTLED, STANDING, fill_key, marks_of)
from .verify import NOT_ASSESSED, verify_home

#: The fourteen questions, in the order a report answers them. The names are the report's keys.
INVARIANTS = ('copies_agree', 'nothing_outside_its_seat', 'every_refusal_is_a_denial',
              'an_amended_plan_is_a_new_approval', 'closes_superseded_never_edited',
              'every_numbers_replay_tuple', 'the_diary_equals_the_filings', 'no_delete',
              'duplicates_coalesce', 'nothing_outside_its_scope', 'the_pnl_is_additive',
              'cash_reconciles', 'every_close_marked', 'the_call_is_the_formula')

#: The one projector that opens no body, which is the whole of what a keyless copy can compare.
KEYLESS_PROJECTORS = ('activity',)
#: The fold an amendment is held against: where the terms it restrikes were held.
POSITIONS = PROJECTORS['positions']
#: The folds the money is held to: what each position cost, the money moved, and the paper.
COSTS, CASH, AGREEMENTS = PROJECTORS['costs'], PROJECTORS['cash'], PROJECTORS['agreements']
#: What carries a ticket - a trade, and a restrike of one.
TICKETED = ('fill', 'amendment')

#: The script's one section, and the keys an act of it may carry - each read by one invariant. An
#: act naming none of them is a step the oracle has no question about.
ACTS = 'acts'
DENIED, REFUSED, LANE, REPLAY = 'denied', 'refused', 'lane', 'replay'
INSTRUCTED, CALLED = 'instructed', 'called'
#: The P&L figures a window is the sum of its days in - the new-deal split is not among them, a
#: trade being new against the end of the window it was done in - and the result class the
#: deployment's tolerance policy declares their epsilon under.
ADDITIVE = ('premiums', 'payments', 'fees', 'pnl', 'realised', 'unrealised')
PNL = 'pnl'
#: What names the marks run a P&L reads at one end: the day it valued, the position it cut the
#: book at, and the values, job and result it attested.
MARKED = ('day', 'lsn', 'values_hash', 'job', 'result')

#: What a denial is identified by - the writer's own body, minus the position it landed at.
DENIAL_FIELDS = ('subject', 'verb', 'book', 'attempted_type')


def unasked(why):
    """The answer to an invariant nothing put the question to: not assessed, and why."""
    return {'held': None, 'evidence': ['{}: {}'.format(NOT_ASSESSED, why)]}


def answer(failures, evidence):
    """`{held, evidence}` from the failures found and the rows read: held where nothing failed, and
    the failures first either way, since what broke is what a reader came for."""
    return {'held': not failures, 'evidence': list(failures) + list(evidence)}


def copies_agree(log, against, entitled):
    """Two homes carry one head hash at a common LSN and fold to byte-equal rows there.

    The chain is re-derived on both - over ciphertext, so a copy holding no key still answers - and
    the comparison is taken at the SHALLOWER head, since a replica behind its hub is a replica that
    has not pulled yet rather than one that disagrees.
    """
    if against is None:
        return unasked('this reading names no second copy to compare against (--against)')
    mirror = SpineLog(against)
    try:
        reports = (verify_home(log.home, entitled=False), verify_home(against, entitled=False))
        common = min(report['head_lsn'] for report in reports)
        heads = [GENESIS_PREV if not common else one.frame_at(common)['event_hash']
                 for one in (log, mirror)]
        failures = [] if heads[0] == heads[1] else [
            'LSN {}: {} carries {} and {} carries {} - the two are copies of different histories'
            .format(common, log.home, heads[0], against, heads[1])]
        names = sorted(PROJECTORS) if entitled else KEYLESS_PROJECTORS
        for name in names:
            rows = [canonical_bytes(PROJECTORS[name].rows(fold(one, PROJECTORS[name], lsn=common)))
                    for one in (log, mirror)]
            if rows[0] != rows[1]:
                failures.append(
                    'the {} rows at LSN {} are {} bytes here and {} bytes at {} - one history has '
                    'two readings'.format(name, common, len(rows[0]), len(rows[1]), against))
        return answer(failures, [
            '{} compared with {} at LSN {}, where this copy carries {}'.format(
                log.home, against, common, heads[0]),
            '{} projector(s) compared: {}'.format(len(names), ', '.join(names))])
    finally:
        mirror.close()


def nothing_outside_its_seat(log, entitled, judged=None):
    """Every frame judged at its envelope's book or at the firm held the verb its type demanded
    there, and the writer's own voice said only what it says.

    The book half of the writer's decision re-reached (`_judged`); a frame judged at a node below
    a book is `nothing_outside_its_scope`'s. An approval in the writer's own voice is held to the
    record around it as the hub made it: the hub signs only where the tiers policy in force has an
    automatic tier, one declaring no `four_eyes`, and books the ticket after it signs - a tiers
    document nobody can read is one it signs under nothing, and one this copy does not hold leaves
    that approval alone not assessed.

    The ENVELOPE HALF stands without a key: a type no verb declares is a write nobody could be
    scoped for, a type only the writer files under any other name is that voice forged, and the
    writer's own name on anything its voice does not say is a seat wearing it. `judged` is the
    walk where the caller took it already.
    """
    judged = _judged(log, entitled) if judged is None else judged
    if isinstance(judged, str):
        return unasked(judged)
    failures, signed, booked, workflow = [], [], {}, 'no tiers policy in force'
    for row in judged:
        frame, body = row['frame'], row['body']
        event_type, actor = frame['event_type'], frame['actor']
        verb, own = verb_for(event_type), actor == WRITER
        if event_type not in EVENT_VERB:
            failures.append(
                'LSN {}: a {} demands no declared verb, so the append was gated by nothing'.format(
                    frame['lsn'], event_type))
        if (verb == WRITER) != own and not (own and event_type in WRITER_VOICE):
            failures.append(
                'LSN {}: a {} stands under {!r} - the writer\'s own voice files {} and nothing '
                'else, and no seat files a type only it does, so this is that voice forged'.format(
                    frame['lsn'], event_type, actor, ', '.join(WRITER_VOICE)))
        if not entitled:
            continue
        if event_type == 'policy_declared' and body.get('policy') == TIERS_POLICY \
                and is_hash(body.get('blob')):
            workflow = _workflow(log, frame['lsn'], body['blob'])
        if own and event_type == 'approval':
            signed.append((frame['lsn'], body.get('plan_hash'), workflow))
        if event_type in TICKETED:
            booked[body.get('ticket')] = frame['lsn']
        if not row['held'] and not row['node']:
            failures.append(_unscoped(frame, row['scope']))
    failures.extend(
        'LSN {}: an approval of {} stands in the writer\'s own voice with {} - the hub signs a '
        'ticket only under an automatic tier in force and books it after, so this is that voice '
        'forged'.format(lsn, ticket, since if isinstance(since, str)
                        else 'no fill or restrike after it carrying that ticket')
        for lsn, ticket, since in signed
        if isinstance(since, str) or since is not None and booked.get(ticket, 0) < lsn)
    blind = [str(lsn) for lsn, _, since in signed if since is None]
    why = ('the approval at LSN {} stands under tiers this copy does not hold - follow with '
           '`--blobs` and ask again'.format(', '.join(blind))) if blind else None
    if why and not failures:
        return unasked(why)
    return answer(failures, ['{} frame(s) re-adjudicated{}'.format(
        len(judged), '' if entitled else ' by envelope alone')] + (
        ['{}: {}'.format(NOT_ASSESSED, why)] if why else []))


def nothing_outside_its_scope(log, judged=None):
    """Every frame judged at a node below a book held the verb its type demanded there, and sits
    where the record admits it.

    The node half of the writer's decision re-reached (`_judged`): a fill, a restrike, a quote or
    a verdict at the portfolio its body names, a node declared at its parent, and a node admin's
    capabilities declaration through `declarable`. Four places are held to the record around
    them: a portfolio sits in its own book's tree; a restrike is judged at the deepest node
    holding what it moved, off the `positions` fold at the frame before it; once any node of a
    book is declared, a fill books into the book or a declared node; and a verdict at a node
    covers every trade carrying its ticket, so an approver does not sign another node's ticket
    by naming its own. `judged` is the walk where the caller took it already.
    """
    judged = _judged(log, True) if judged is None else judged
    if isinstance(judged, str):
        return unasked(judged)
    failures, positions, declared, carried, verdicts, read = (
        [], POSITIONS.initial(), set(), {}, [], 0)
    for row in judged:
        frame, body = row['frame'], row['body']
        event_type = frame['event_type']
        if row['node']:
            read += 1
            if not row['held']:
                failures.append(_unscoped(frame, row['scope']))
            failures.extend(_placed(frame, body, positions, declared))
            if event_type in ('approval', 'rejection'):
                verdicts.append((frame['lsn'], event_type, body.get('plan_hash'), row['scope']))
        if event_type in POSITIONS.reads:
            POSITIONS.apply(positions, frame, log)
        if event_type == 'portfolio_declared' and is_text(body.get('path')):
            declared.add(body['path'])
        if event_type in PORTFOLIO_TYPES and body.get('ticket') is not None:
            carried.setdefault(body['ticket'], []).append(body.get('portfolio') or frame['book'])
    failures.extend(
        'LSN {}: the {} at {!r} rules on {}, which books at {} - a verdict at a node reaches what '
        'books under it, and an approver naming its own node signs nothing beside it'.format(
            lsn, event_type, node, ticket, ', '.join(sorted(set(carried[ticket]))))
        for lsn, event_type, ticket, node in verdicts
        if any(not under(where, node) for where in carried.get(ticket, ())))
    return answer(failures, ['{} frame(s) judged at a node, {} node(s) declared'.format(
        read, len(declared))])


def _judged(log, entitled):
    """Every frame with the writer's own question re-asked of it - `{frame, body, scope, held,
    node}` in LSN order - or the sentence naming the capabilities document this copy lacks.

    `verb_for` names what a type demanded and `evaluate` answers it at `scope_of` against the
    capability state BEFORE the frame, folded forward by `apply_event`, the writer's own step - the
    same answer at the length of the record rather than its square. A node admin's declaration the
    grants refuse is re-run through `declarable`, as the writer ran it; the writer's own voice is
    never gated, and a home with no document in force answers yes, exactly as the append did.
    `node` is a frame judged at a node below a book - anywhere but its envelope's book or the
    firm - or a node admin's declaration. A copy holding no key answers the envelope alone.

    A COPY THAT HOLDS THE FRAMES AND NOT THE DOCUMENT cannot put the question at all. The
    capabilities fold answers `UNREADABLE` for a blob that is gone - fail-closed, because inside
    the writer's own hook a raise would take every append with it - and re-running the writer
    against THAT would call every frame after the declaration a forgery. So the blob is asked for
    first, and a copy that lacks it is told to follow with `--blobs`.
    """
    state, judged = initial_state(), []
    for frame in log.frames():
        event_type, actor = frame['event_type'], frame['actor']
        if not entitled:
            judged.append({'frame': frame, 'body': None, 'scope': None, 'held': None,
                           'node': False})
            continue
        verb = verb_for(event_type)
        body = log.open_body(frame) if event_type in CAPABILITY_EVENTS + SCOPED_TYPES else None
        doc, genesis = state['doc'], state['genesis']
        scope = scope_of(event_type, body, frame['book'])
        held = actor == WRITER or (doc is None and verb != RECOVERY) or evaluate(
            doc, genesis, actor, verb, scope)
        declaring = event_type == 'policy_declared' and isinstance(body, dict) \
            and body.get('policy') == CAPABILITIES_POLICY
        node = scope not in (None, frame['book']) or declaring and bool(nodes(doc, actor, ADMIN))
        if event_type in CAPABILITY_EVENTS:
            if _undeclared(log, body):
                return ('LSN {} declares the capabilities document {} and this copy does not hold '
                        'it, so every frame after it was judged under a document that is not here '
                        '- follow with `--blobs` and ask again'.format(frame['lsn'], body['blob']))
            apply_event(state, event_type, actor, body, log.store)
            held = held or (declaring and isinstance(doc, dict)
                            and isinstance(state['doc'], dict)
                            and declarable(doc, state['doc'], actor))
        judged.append({'frame': frame, 'body': body, 'scope': scope, 'held': held, 'node': node})
    return judged


def _workflow(log, lsn, blob):
    """What the tiers declared at `lsn` let the hub sign under: that LSN where a tier declares no
    `four_eyes`, else why not - a document nobody can read being one nobody signs under - or None
    where this copy does not hold the blob, the one case it cannot judge."""
    if not log.store.has(blob):
        return None
    try:
        tiers = tiers_in_force(log, lsn)[TIERS_SECTION]
    except SpineRefusal as unreadable:
        return 'the tiers at LSN {} nobody can read ({})'.format(lsn, unreadable)
    return lsn if any(not tier['four_eyes'] for tier in tiers) else 'no automatic tier in force'


def _unscoped(frame, scope):
    """The sentence for a frame whose actor held no scope where the writer judged it."""
    return ('LSN {}: {!r} held no {} scope over {!r} when the {} landed, so this frame is on the '
            'platter and the writer would have refused it'.format(
                frame['lsn'], frame['actor'], verb_for(frame['event_type']), scope or '*',
                frame['event_type']))


def _placed(frame, body, positions, declared):
    """Why a frame judged at a node is not where the writer would have let it stand - the
    sentences, none where it is: under its own book; a restrike covering every position it moves,
    which `positions` - the fold at the frame before it - holds under that book; and a fill into
    the book or a node `declared` under it once any is."""
    event_type, book = frame['event_type'], frame['book']
    named = stray(event_type, body, book)
    if named is not None:
        return ['LSN {}: the {} names the portfolio {!r} and is filed under the book {!r} - a '
                'portfolio outside its own book\'s tree, which the writer refuses'.format(
                    frame['lsn'], event_type, named, book)]
    named = body.get('portfolio') if event_type in ('fill', 'amendment') else None
    if named is None:
        return []
    if event_type == 'fill':
        tree = sorted(path for path in declared if under(path, book))
        return [] if not tree or named in [book] + tree else [
            'LSN {}: the fill books into {!r}, which book {!r} does not declare - it books into '
            'itself or a declared node, {}'.format(frame['lsn'], named, book, ', '.join(tree))]
    held = holders(positions, body['instrument'], book)
    node = deepest(held)
    if node is None or under(node, named):
        return []
    return ['LSN {}: the amendment is judged at {!r} and moves the positions held at {}, the '
            'deepest node holding them being {!r} - a restrike judged narrower than what it '
            'moved'.format(frame['lsn'], named, ', '.join(held), node)]


def _undeclared(log, body):
    """Whether this declaration names a capabilities document the store does not hold.

    The one policy whose blob the capability fold READS, so the one whose absence changes what
    `evaluate` answers; every other reserved name folds somewhere else entirely.
    """
    return (isinstance(body, dict) and body.get('policy') == CAPABILITIES_POLICY
            and is_hash(body.get('blob')) and not log.store.has(body['blob']))


def every_refusal_is_a_denial(log, script):
    """What the script says was refused AT THE WRITER is in the record, and nothing else is.

    A denial is identified by the four fields the writer files and never by its position, because a
    seat refused the same verb twice coalesces onto the LSN it already has.

    A refusal that never REACHED the writer - a tier admitting no ticket, a board already stale, a
    malformed request - mints nothing BY DESIGN, so there is nothing in the record to hold it
    against: it is ECHOED into the evidence as the boundary it is, and what the set equality below
    asserts about it is only that it left no denial behind. A script inventing one is a script
    describing a day nobody played, which no reading of the record can contradict.
    """
    if script is None:
        return unasked('no script says which attempts were refused, so a denial has nothing to be '
                       'held against (--script)')
    acts, rows = script.get(ACTS, ()), fold(log, PROJECTORS['denials'])
    recorded = set(tuple(row[field] for field in DENIAL_FIELDS) for row in rows)
    asked = set(tuple(act[DENIED][field] for field in DENIAL_FIELDS)
                for act in acts if act.get(DENIED))
    failures = ['the script asked for a denial of {} and the record holds none'.format(missing)
                for missing in sorted(asked - recorded)]
    failures.extend('the record holds a denial of {} the script never asked for'.format(surplus)
                    for surplus in sorted(recorded - asked))
    return answer(failures, [
        '{} denial(s) recorded over {} refused append(s) asked for'.format(len(rows), len(asked))] +
        ['refused short of the writer, so nothing was minted: {}'.format(act[REFUSED])
         for act in acts if act.get(REFUSED)])


def an_amended_plan_is_a_new_approval(log):
    """Every ticket is one trade's, and a restrike is a trade of its own.

    A ticket is the plan a trade leaves and the trade's own fields, so every event carrying one
    files one trade - a retry the writer took twice, two seats filing one execution - and one
    carried by two different trades is one signature reaching both. And where the record
    carries a WORKFLOW - a tiers policy in force at the
    position an amendment landed - the amendment carries a ticket of its own, so the position it
    restrikes stands on no approval over the ticket it had: whether one stands over the new one is
    the position's STATUS, derived and never failed. Where no workflow is declared the desk
    declared no second pair of eyes, which is stated rather than failed - and a copy that cannot
    READ the workflow cannot put the question at all, which is the posture of a follower that
    pulled frames and no blobs.
    """
    tickets, failures, restruck = {}, [], 0
    try:
        in_force(log, TIERS_POLICY)
        for frame in log.frames():
            if frame['event_type'] not in TICKETED:
                continue
            body = log.open_body(frame)
            ticket = body.get('ticket')
            trade = (frame['event_type'], {field: value for field, value in body.items()
                                           if field != 'ticket'}, frame['lsn'])
            if ticket is not None and tickets.setdefault(ticket, trade)[:2] != trade[:2]:
                failures.append(
                    'LSN {}: the {} carries the ticket {} that LSN {} carried for another '
                    'trade - one approval would reach two trades'.format(
                        frame['lsn'], frame['event_type'], ticket, tickets[ticket][2]))
            if frame['event_type'] != 'amendment':
                continue
            restruck += 1
            if ticket is None and in_force(log, TIERS_POLICY, frame['lsn'])[1] is not None:
                failures.append(
                    'LSN {}: the amendment restrikes {} under a workflow and carries no ticket - '
                    'the position it moved stands on the approvals over the ticket it had, which '
                    'nobody gave these terms'.format(frame['lsn'], body['instrument']))
    except SpineRefusal as unreadable:
        return unasked(
            'whether a restrike owed a signature is the workflow document\'s question and this '
            'copy cannot read it ({}) - follow with `--blobs` and ask again'.format(unreadable))
    return answer(failures, ['{} ticket(s) filed, {} restrike(s) read'.format(
        len(tickets), restruck)])


def closes_superseded_never_edited(log):
    """A close is a fact this writer's fold reads one of two ways, and the third way is a line no
    writer wrote.

    THE SUPERSESSION HALF IS THE FOLD'S OWN and cannot be broken from a platter. `Markets.apply`
    COMPUTES `supersedes_lsn` off the close of the same day filed before the frame, and stands a
    market on its latest close by day, ties by LSN - so a close restates its own day naming exactly
    the position it stood over, and a past day restated never unseats a later day's. There is no
    third answer to assert, which is why none is asserted: a reading that re-ran the fold to check
    the fold would be one spelling checking itself.

    What a PLATTER can carry that the fold cannot is a close body the closed vocabulary would have
    refused - a copy of a newer hub, or a line somebody put there, since a replica's `accept` asks
    the chain and never the vocabulary - and that is what this reads, before the markets fold is
    taken over a body it would raise on.
    """
    filed = [(frame['lsn'], log.open_body(frame)) for frame in log.frames()
             if frame['event_type'] == 'official_close_declared']
    failures = [
        'LSN {}: the close puts {!r} on the market {!r} - the closed vocabulary would have refused '
        'that body, so no writer that validates wrote this line'.format(
            at, body.get('values_hash'), body.get('market'))
        for at, body in filed
        if not is_text(body.get('market')) or not is_hash(body.get('values_hash'))]
    if failures:
        # the markets fold reads these same bodies, so it is not taken over a line it would raise on
        return answer(failures, [])
    standing = fold(log, PROJECTORS['markets'])['closes']
    return answer(failures, ['{} close(s) over {} market(s): {}'.format(
        len(filed), len(standing), ', '.join(
            '{} at LSN {} over {}'.format(market, row['lsn'], row['supersedes_lsn'])
            for market, row in sorted(standing.items())) or 'none declared')])


def every_numbers_replay_tuple(log, script):
    """The attestations are exactly the standing runs the script asked for, coordinate for
    coordinate.

    A run is recorded IFF its output will be cited by a fact, so the LANE is the whole question and
    the script is where it is written down. Asked as a SET rather than per ask, because content
    addressing dedupes numbers and not standing: an identical what-if after a standing run arrives
    at the same four coordinates and must not read as a second attestation. So a standing ask with
    no row is a number nothing can replay, and a row no standing ask names is a lane that minted
    where it should have minted nothing - which is the whole of what telemetry and curiosity claim.
    The lookup is under the key the fold files rows by, so this reading and the verb's cannot
    disagree about which attestation a number replays from.
    """
    if script is None:
        return unasked('no script says which lane each run was submitted in, and a lane is not a '
                       'property of the record (--script)')
    lanes = [act[LANE] for act in script.get(ACTS, ()) if act.get(LANE)]
    rows = fold(log, PROJECTORS['attestations'])
    asked = set(content_hash(dict((field, act[REPLAY].get(field)) for field in REPLAY_FIELDS))
                for act in script.get(ACTS, ())
                if act.get(LANE) == STANDING and act.get(REPLAY))
    failures = ['a standing run was asked for and the record holds no attestation at {}'.format(
        missing) for missing in sorted(asked - set(rows))]
    failures.extend(
        'the attestation at LSN {} replays from coordinates no standing run asked for - a lane '
        'that mints nothing minted'.format(rows[surplus]['lsn'])
        for surplus in sorted(set(rows) - asked))
    return answer(failures, ['{} attestation(s) over {} run(s): {}'.format(
        len(rows), len(lanes), ', '.join('{} {}'.format(lanes.count(lane), lane)
                                         for lane in sorted(set(lanes))))])


def the_diary_equals_the_filings(log, keys):
    """Every settlement filed against a derived key names a row the diary carries, and every
    confirmation a clip the record holds.

    The diary is a COMPILE and not a fold, so a replica cannot put this question: `keys` is the set
    of cashflow keys an engine answered, handed in as data by a caller that has one. A clip's key is
    the record's own - its instrument and execution reference, off the `positions` fold. A
    transition whose subject is not an address is filed against an instrument or a name and is not
    this question.
    """
    if keys is None:
        return unasked("the diary is a compile of the book and not a fold of the record, so a "
                       "replica cannot put this question - hand in the diary's own keys")
    keys, failures, filed = set(keys), [], 0
    clips = set(fill_key(clip['instrument'], clip['execution_reference'])
                for row in POSITIONS.rows(fold(log, POSITIONS)) for clip in row['tickets'])
    for subject, standing in fold(log, PROJECTORS['lifecycle'])['transitions'].items():
        if not is_hash(subject):
            continue
        filed += 1
        if subject not in keys | clips:
            failures.append(
                'LSN {}: a settlement was filed against {}, and neither the diary nor the record '
                'names a row or a clip under that key - a status moves what the book owes or '
                'holds, or it moves nothing'.format(standing['lsn'], subject))
    return answer(failures, ['{} settlement(s) filed against {} diary key(s) and {} clip(s)'.format(
        filed, len(keys), len(clips))])


def no_delete(log, entitled=True):
    """Nothing left: the positions are dense from genesis, every blob the chain CITES is still
    addressable, the store holds no fewer than it held a moment ago, and neither the store nor the
    vocabulary carries a verb for forgetting.

    REFERENTIAL CLOSURE IS THE DELETION A RECORD CAN ACTUALLY SUFFER - the store has no verb for
    forgetting, so what takes a blob is a file system, and the frame that cited it is still on the
    platter naming an address nothing answers for. So the citing frames are re-read and every
    address asked of the store, over `cited_blobs` - the one spelling the writer, the verifier and
    the follower all cite by, so all four agree about what a citation is.

    A FOLLOWER THAT PULLED NO BLOBS FAILS THIS TOO, and that is the honest answer rather than a
    special case: what it holds and what a shredded hub holds are the same bytes, so "I was never
    given them" and "they were taken" are one state of one store and the sentence says so. An
    entitled verification refuses such a copy for the same reason.

    A citation lives in a SEALED BODY, so a keyless copy is left with the other three arms: the
    positions it can count off the envelope, the store it can walk, and the verbs that do not exist.
    """
    before = set(log.store.walk())
    failures, previous, cited = [], 0, 0
    for frame in log.frames():
        if frame['lsn'] != previous + 1:
            failures.append(
                'LSN {} follows LSN {}: the sequence is dense from genesis, so a position that is '
                'not the next one is a line that left'.format(frame['lsn'], previous))
        previous = frame['lsn']
        if not entitled or frame['event_type'] not in BLOB_FIELDS:
            continue
        for field, digest in cited_blobs(frame['event_type'], log.open_body(frame)):
            cited += 1
            if not log.store.has(digest):
                failures.append(
                    'LSN {}: the {} cites {} as its {} and this home does not hold it - a blob '
                    'leaves a store only through a logged retention event, and a copy that was '
                    'never given one cannot tell that apart from one that lost it'.format(
                        frame['lsn'], frame['event_type'], digest, field))
    gone = before - set(log.store.walk())
    failures.extend('the blob {} was in the store at the start of this walk and is not in it now'
                    .format(digest) for digest in sorted(gone))
    forgetting = sorted(name for name in dir(BlobStore) if 'delete' in name or 'remove' in name)
    forgetting.extend(event_type for event_type in sorted(EVENT_TYPES) if 'delete' in event_type)
    failures.extend('{} is a verb for forgetting, and the record has none'.format(name)
                    for name in forgetting)
    return answer(failures, ['{} position(s) dense from 1, {} blob(s) held throughout, {}'.format(
        previous, len(before), '{} citation(s) resolved'.format(cited) if entitled
        else 'citations {}'.format(NOT_ASSESSED))])


def duplicates_coalesce(log):
    """Every idempotency tag appears once: a retry is the same fact by construction, folded onto the
    position it already has, so two frames under one tag would be one act the record counted twice.

    Read off the ENVELOPE, so a copy holding no key answers it.
    """
    seen, failures, read = {}, [], 0
    for frame in log.frames():
        read += 1
        standing = seen.setdefault(frame['idempotency_tag'], frame['lsn'])
        if standing != frame['lsn']:
            failures.append(
                'LSN {} carries the tag LSN {} already holds - one fact, written twice'.format(
                    frame['lsn'], standing))
    return answer(failures, ['{} tag(s) over {} frame(s)'.format(len(seen), read)])


def the_pnl_is_additive(log, pnl):
    """Consecutive days' P&L sums to the window over them and a window's portfolios to the book,
    figure by figure within the epsilon the deployment declared for `pnl`, and every answer is this
    record's own.

    The P&L is a COMPILE of the book, so it is handed in as data: `{window, days, portfolios}`,
    each `GET /book/pnl`'s own answer - the days tiling the window mark to mark, the portfolios read
    between the window's own two marks and partitioning the book. Every figure but the new-deal
    split is additive, a trade being new against the end of the window it was done in. THE SPINE
    ADMITS NO TOLERANCE OF ITS OWN: one set of floats summed in two orders is one answer only within
    an epsilon, the tolerance policy's for the `pnl` class - none declared, and any difference is a
    departure (`policy.compare`).

    Each answer is held to this record (`_tied`): both ends are marks runs it attested, and every
    row's quantities are the `costs` fold's there and its premiums the fills between them. WHAT NO
    RECORD PROVES is a figure only the engine computes - a mark, a payment the diary determines, a
    spot - so answers whose every such figure was scaled alike still sum: the boundary of what data
    can hold, stated rather than claimed past.
    """
    if pnl is None:
        return unasked('the P&L is a compile of the book and not a fold of the record, so a '
                       'replica cannot put this question - hand in the answers (--pnl)')
    try:
        tolerances = (in_force(log, TOLERANCE_POLICY)[1] or {}).get(TOLERANCE_SECTION, {})
        window, days, parts = pnl['window'], list(pnl.get('days') or ()), list(
            pnl.get('portfolios') or ())
        failures = [] if not days or [window['start']] + [day['end'] for day in days] == [
            day['start'] for day in days] + [window['end']] else [
            'the days run {} and the window {} to {} - days that do not tile a window say nothing '
            'about it'.format(', '.join('{} to {}'.format(day['start']['day'], day['end']['day'])
                                        for day in days),
                              window['start']['day'], window['end']['day'])]
        failures.extend('the {} portfolio is read {} to {} and the window {} to {}'.format(
            part['scope']['portfolio'], part['start']['day'], part['end']['day'],
            window['start']['day'], window['end']['day'])
            for part in parts if (part['start'], part['end']) != (window['start'], window['end']))
        for named, summed, rows in (('days', days, True), ('portfolios', parts, False)):
            if summed and not failures:
                failures.extend('summed over the {}, {}'.format(named, departure) for departure in
                                compare(_figures(window, [window], rows),
                                        _figures(window, summed, rows), tolerances))
        failures.extend(_tied(log, [window] + days + parts, tolerances))
    except SpineRefusal as unreadable:
        return unasked('the tolerance policy and the marks runs are blobs this copy does not hold '
                       '({}) - follow with `--blobs` and ask again'.format(unreadable))
    except (AttributeError, KeyError, TypeError):
        return unasked('the P&L handed in is not {window, days, portfolios} of `GET /book/pnl` '
                       'answers, so there is nothing to sum (--pnl)')
    return answer(failures, ['{} day(s) and {} portfolio(s) held to {} to {} under a tolerance '
                             'of {}; the ends, quantities and premiums held to the record, the '
                             'marks, payments and spots the engine\'s'.format(
                                 len(days), len(parts), window['start']['day'],
                                 window['end']['day'], tolerances.get(PNL, 'none'))])


def _figures(window, answers, rows):
    """`{pnl: {total, rows}}` - the additive figures of `answers` summed as exactly as floats sum
    (`math.fsum`), in total and, where `rows`, per position of `window`; null where any part of a
    sum is."""
    def summed(values):
        values = list(values)
        return None if None in values else math.fsum(values)

    def keyed(answer):
        return {'{portfolio} {agreement} {instrument}'.format(**row): row
                for row in answer['rows']}

    held = [keyed(answer) for answer in answers]
    figures = {'total': {figure: summed(answer['total'][figure] for answer in answers)
                         for figure in ADDITIVE}}
    if rows:
        figures['rows'] = {key: {figure: summed(one[key][figure] for one in held if key in one)
                                 for figure in ADDITIVE} for key in keyed(window)}
    return {PNL: figures}


def _tied(log, answers, tolerances):
    """Why P&L `answers` are not this record's - the sentences, none where they are: each end a
    marks run the record attested (`_marks`), and each row's quantities the `costs` fold's at the
    two ends and its premiums the consideration of the fills between them, as the P&L sums it."""
    attested = set(tuple(row[field] for field in MARKED) for row in _marks(log))
    held = {lsn: {(row['instrument'], row['agreement'], row['portfolio']): row['quantity']
                  for row in COSTS.rows(costs)}
            for lsn, (costs,) in _walked(log, (COSTS,), [
                side['lsn'] for one in answers for side in (one['start'], one['end'])])}
    traded = {}
    for frame in log.frames():
        if frame['event_type'] == 'fill':
            body = log.open_body(frame)
            traded.setdefault((body['instrument'], body.get('agreement') or body['netting_set'],
                               body.get('portfolio') or frame['book'] or ''), []).append(
                (frame['lsn'], body))
    failures = []
    for one in answers:
        start, end = one['start'], one['end']
        failures.extend('the P&L of {} to {} reads the marks of {} at LSN {}, which no run this '
                        'record attested is'.format(start['day'], end['day'], side['day'],
                                                    side['lsn'])
                        for side in (start, end)
                        if tuple(side.get(field) for field in MARKED) not in attested)
        for row in one['rows']:
            key, premiums = (row['instrument'], row['agreement'], row['portfolio']), 0.0
            for lsn, body in traded.get(key, ()):
                if start['lsn'] < lsn <= end['lsn']:
                    premiums = None if premiums is None or body.get('price') is None else (
                        premiums - float(body['quantity']) * float(body['price']) * float(
                            body.get('rate', 1.0)))
            recorded = {'quantity_start': held[start['lsn']].get(key, 0.0),
                        'quantity_end': held[end['lsn']].get(key, 0.0), 'premiums': premiums}
            failures.extend('the P&L of {} to {} at {} {}: {}'.format(
                start['day'], end['day'], row['portfolio'], row['instrument'][:12], departure)
                for departure in compare({PNL: {field: row.get(field) for field in recorded}},
                                         {PNL: recorded}, tolerances))
    return failures


def _marks(log):
    """Every marks run the record attested - `{book, day, lsn, attested, values_hash, job, result}`,
    as `PnL.marked` reads one: `lsn` the position its job's name cuts the book at, which a run
    attested before it cannot have read, else where it was attested. `SpineRefusal` where a job's
    blob is not here."""
    found = []
    for row in PROJECTORS['attestations'].rows(fold(log, PROJECTORS['attestations'])):
        job = json.loads(log.store.get(row['job']).decode('utf-8')).get('Calc') or {}
        named = marks_of((job.get('Deals') or {}).get('Reference'))
        day = (job.get('Calculation') or {}).get('Base_Date')
        day = (day.get('.Timestamp') if isinstance(day, dict) else day) or ''
        if named is not None and (named[1] or 0) <= row['lsn']:
            found.append({'book': named[0], 'day': day[:10], 'attested': row['lsn'],
                          'lsn': row['lsn'] if named[1] is None else named[1],
                          'values_hash': row['values_hash'], 'job': row['job'],
                          'result': row['result']})
    return found


def _walked(log, projectors, positions):
    """Each of `positions` in order, with the states of `projectors` standing there - ONE walk
    forward for all of them, rather than a fold from genesis per position."""
    states, wanted = [projector.initial() for projector in projectors], sorted(set(positions))
    for frame in log.frames():
        while wanted and wanted[0] < frame['lsn']:
            yield wanted.pop(0), states
        if not wanted:
            return
        for projector, state in zip(projectors, states):
            if frame['event_type'] in projector.reads:
                projector.apply(state, frame, log)
    for lsn in wanted:
        yield lsn, states


def cash_reconciles(log, script, keys):
    """Every movement of money resolves where its kind says, a payment moves what the row it settles
    determines and a row is paid once, and every row a settlement file instructed is settled.

    A fee is money a trade cost, so its subject is an instrument a fill booked by then; collateral
    or margin is held under paper, so its subject is an agreement declared by then - the verb asks
    the last when it files, and a copy can carry what the verb never saw. The diary is a COMPILE,
    handed in as `keys`: a list of its keys, or key -> `{amount, currency}` with the amount null
    where the diary does not determine it. The payment standing against a determined row carries
    its amount in its currency within the declared `pnl` tolerance, one against an undetermined
    row any amount, and two standing against one key under two references are one row paid twice.
    An instructed row is settled where its status stands `settled` - a payment, or a bare
    `settled` that moved no money - exactly as the diary reads it.
    """
    positions, declared, failures, moved = POSITIONS.initial(), set(), [], 0
    for frame in log.frames():
        event_type = frame['event_type']
        if event_type in POSITIONS.reads:
            POSITIONS.apply(positions, frame, log)
        elif event_type == 'agreement_declared':
            declared.add(log.open_body(frame).get('agreement'))
        elif event_type == 'status_transition':
            body = log.open_body(frame)
            kind, subject = body.get('kind'), body.get('subject')
            moved += 'amount' in body
            if kind in HELD and subject not in declared or kind == FEE and \
                    subject not in positions:
                failures.append('LSN {}: a {} movement of {} {} names {} - no {} by then'.format(
                    frame['lsn'], kind, body.get('amount'), body.get('asset'), subject,
                    'agreement the record declared' if kind in HELD
                    else 'instrument a fill booked'))
    try:
        tolerances = (in_force(log, TOLERANCE_POLICY)[1] or {}).get(TOLERANCE_SECTION, {})
    except SpineRefusal as unreadable:
        return unasked('the tolerance policy is a blob this copy does not hold ({}) - follow with '
                       '`--blobs` and ask again'.format(unreadable))
    owed, paid = keys if isinstance(keys, dict) else dict.fromkeys(keys or ()), {}
    for row in CASH.rows(fold(log, CASH)):
        if row['kind'] == PAYMENT:
            paid.setdefault(row['subject'], []).append(row)
    for subject, standing in sorted(paid.items()):
        if len(standing) > 1:
            failures.append('{} is paid {} times, under {} - one row is paid once'.format(
                subject, len(standing), ', '.join(row['reference'] for row in standing)))
        due = owed.get(subject) or {}
        failures.extend('{} moved {} {} against {}, which the diary determines at {} {}'.format(
            row['reference'], row['amount'], row['asset'], subject, due['amount'],
            due.get('currency'))
            for row in standing if due.get('amount') is not None and (
                row['asset'] != due.get('currency') or compare(
                    {PNL: due['amount']}, {PNL: row['amount']}, tolerances)))
    instructed = None if script is None else sorted(set(
        key for act in script.get(ACTS, ()) for key in act.get(INSTRUCTED, ())))
    if instructed:
        standing = fold(log, PROJECTORS['lifecycle'])['transitions']
        failures.extend('the settlement file instructed {} and nothing settled it'.format(key)
                        for key in instructed
                        if standing.get(key, {}).get('status') != SETTLED)
    return answer(failures, ['{} movement(s) read; {} row(s) paid, held to {} diary row(s); {}'
                             .format(moved, len(paid), len(owed), NOT_ASSESSED + ' without a '
                                     'script: the instructed rows' if instructed is None else
                                     '{} instructed row(s) settled'.format(len(instructed)))])


def every_close_marked(log):
    """Every day closed on the market the workflow designates for P&L is marked wherever a book held
    anything at its close: a marks run of that book as of the day, attested on the values of a
    close declared for the day at or before it.

    The marks are what every P&L of a day reads, so a day closed without them is one nobody can say
    what the book made on. A book holding nothing at the close - the `positions` fold there - owes
    none, there being nothing to mark. A close restated after its day's marks is a READING and never
    a failure: marks run forward, so the day's P&L carries it as it was marked, and the evidence
    names it.
    """
    designated, ever, days = None, False, {}
    try:
        for frame in log.frames():
            if frame['event_type'] == 'policy_declared' and log.open_body(frame).get(
                    'policy') == TIERS_POLICY:
                designated = (tiers_in_force(log, frame['lsn']) or {}).get(
                    DESIGNATIONS_SECTION, {}).get(PNL)
                ever = ever or designated is not None
            elif frame['event_type'] == 'official_close_declared' and designated is not None:
                body = log.open_body(frame)
                if body.get('market') == designated:
                    days.setdefault((body.get('date') or as_of_key(frame)[0][:10], designated),
                                    []).append((frame['lsn'], body.get('values_hash')))
        if not ever:
            return unasked('no workflow here designates a market for P&L, so no close owes marks')
        marks = _marks(log)
    except SpineRefusal as missing:
        return unasked('a close is held to the workflow and the jobs its marks attested, and this '
                       'copy does not hold them ({}) - follow with `--blobs` and ask again'.format(
                           missing))
    holding = {lsn: set(row['book'] for row in POSITIONS.rows(positions) if row['quantity'])
               for lsn, (positions,) in _walked(log, (POSITIONS,), [
                   lsn for closes in days.values() for lsn, _ in closes])}
    failures, readings = [], []
    for (day, market), closes in sorted(days.items()):
        for book in sorted(set().union(*(holding[lsn] for lsn, _ in closes))):
            ran = [row['attested'] for row in marks if (row['book'], row['day']) == (book, day)
                   and any(values == row['values_hash'] and lsn <= row['attested']
                           for lsn, values in closes)]
            if not ran:
                failures.append('LSN {}: the close of {} on {!r} is not marked - no marks run of '
                                '{!r} as of that day stands on the values of a close declared for '
                                'it, so no P&L can say what the book made on it'.format(
                                    closes[-1][0], day, market, book))
            elif closes[-1][0] > max(ran):
                readings.append('the close of {} on {!r} was restated at LSN {} after the marks of '
                                '{!r} at LSN {}, and the day reads as it was marked'.format(
                                    day, market, closes[-1][0], book, max(ran)))
    return answer(failures, ['{} day(s) closed on the market designated for P&L, {} marks '
                             'run(s)'.format(len(days), len(marks))] + readings)


def the_call_is_the_formula(log, script):
    """Every collateral call a seat read is `CSA.call` over the record's own cash and the
    agreement's declared terms at the position the read names, and the balance it reported is the
    record's.

    The balance is the `cash` fold at the read's `lsn` held as of its `date` and the dials are the
    terms the `agreements` fold stood on there, both advanced ONCE forward through the reads in LSN
    order (`_walked`), so the service's `held` and every figure `CSA.WORKED` names are
    compared as the numbers they are. The exposure and `fx` are compiles - the P&L's value of the
    positions under the agreement at a close, and the close's spots in its currency - so both are
    DATA this trusts as the service answered them. A call nobody could work out carries no numbers
    and is not worked, one that carries them is named, and a day every call of which was unknown
    puts no question.
    """
    called = [] if script is None else [
        call for act in script.get(ACTS, ()) for call in act.get(CALLED, ())]
    if not called:
        return unasked('no script says which collateral calls a seat read (--script)')
    reads, terms, failures, worked = {}, {}, [], 0
    for call in called:
        reads.setdefault(call['lsn'], []).append(call)
    try:
        for lsn, (cash, agreements) in _walked(log, (CASH, AGREEMENTS), reads):
            movements, balances = CASH.rows(cash), {}
            for call in reads[lsn]:
                named = '{} on {} at LSN {}'.format(call['agreement'], call['date'], lsn)
                if call.get('unknown'):
                    carried = [field for field in CSA.WORKED if call.get(field) is not None]
                    if carried:
                        failures.append('{}: the call names what nobody can know and carries {} '
                                        'anyway'.format(named, ', '.join(carried)))
                    continue
                declared = agreements.get(call['agreement'])
                if declared is not None and declared['terms'] not in terms:
                    terms[declared['terms']] = CSA.of(
                        json.loads(log.store.get(declared['terms']).decode('utf-8')))
                if declared is None or terms[declared['terms']] is None:
                    failures.append('{}: the record holds no collateralising terms under it '
                                    'there'.format(named))
                    continue
                if call['date'] not in balances:
                    balances[call['date']] = CSA.held(movements, call['date'])
                held = balances[call['date']].get(call['agreement'], CSA.NOTHING)[
                    'collateral']
                said = dict(CSA.call(call['exposure'], held, terms[declared['terms']], call['fx']),
                            held=held)
                worked += 1
                failures.extend('{}: the service answered {} {!r} and the formula over the record '
                                'says {!r}'.format(named, field, call.get(field), said[field])
                                for field in CSA.WORKED + (('held',) if 'held' in call else ())
                                if call.get(field) != said[field])
    except SpineRefusal as missing:
        return unasked('a call is held to the terms an agreement declared and this copy does not '
                       'hold them ({}) - follow with `--blobs` and ask again'.format(missing))
    if not worked and not failures:
        return unasked('every call the script read was one nobody could work out, so none puts '
                       'the question')
    unknown = sorted(set(call['date'] for call in called) - set(
        call['date'] for call in called if not call.get('unknown')))
    return answer(failures, ['{} call(s) worked again over the record, of {} read; the exposure '
                             'and the spots are the service\'s, taken as data'.format(
                                 worked, len(called))] + [
        '{}: every call unknown, {}'.format(day, NOT_ASSESSED) for day in unknown])


def report(home, script=None, against=None, diary_keys=None, pnl=None):
    """`{invariant: {held, evidence}}` for all fourteen, over `home` and what it was given.

    A home whose class key is gone answers the three that cost no key and the envelope half of
    `nothing_outside_its_seat`, and says by name which it could not assess - the replica posture,
    stated rather than skipped. The capability walk is taken once for both of its halves.
    """
    log = SpineLog(home)
    try:
        entitled = log.keys.has_firm()
        sealed = unasked(
            'this home holds no class key, so its bodies are sealed and this question is inside '
            'them - ask it on a copy that materialized the key')
        judged = _judged(log, entitled)
        return dict(zip(INVARIANTS, (
            copies_agree(log, against, entitled),
            nothing_outside_its_seat(log, entitled, judged),
            every_refusal_is_a_denial(log, script) if entitled else sealed,
            an_amended_plan_is_a_new_approval(log) if entitled else sealed,
            closes_superseded_never_edited(log) if entitled else sealed,
            every_numbers_replay_tuple(log, script) if entitled else sealed,
            the_diary_equals_the_filings(log, diary_keys) if entitled else sealed,
            no_delete(log, entitled),
            duplicates_coalesce(log),
            nothing_outside_its_scope(log, judged) if entitled else sealed,
            the_pnl_is_additive(log, pnl) if entitled else sealed,
            cash_reconciles(log, script, diary_keys) if entitled else sealed,
            every_close_marked(log) if entitled else sealed,
            the_call_is_the_formula(log, script) if entitled else sealed)))
    finally:
        log.close()


def failed(answers):
    """Every invariant that did NOT hold. Empty is the day passing; a null holds nothing against
    anybody, since a question nobody put is not a question that failed."""
    return sorted(name for name, found in answers.items() if found['held'] is False)
