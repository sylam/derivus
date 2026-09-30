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

"""What a copy of the record proves about a day on the desk - nine invariants and their evidence.

Pure over a HOME and, where one is given, the SCRIPT of what was asked. The oracle folds the record
the way every other reader does and re-runs the writer's own decisions through the writer's own
functions, so an answer here is a property of the bytes rather than of the process that made them.
It is read as a REPLICA, which is the honest posture - verification is always local - and a
chain-only home still answers the four questions that cost no key while naming the five it cannot.

Two invariants ask something the record cannot hold and take it as data. The SCRIPT says what was
ASKED: which attempts the writer refused, which were turned away short of it by a tier or a
staleness window, and which attestation lane each run was submitted in - so `asked x recorded` is
checkable where `recorded` alone is only half the question. The DIARY's keys are the engine's, so a
replica reports that one as not assessed and the caller holding an engine hands them in.

Each invariant is one function answering `{held, evidence}`, where `held` is null for a question
this home or this script could not put, and `evidence` is the rows that decided it.
"""
from .canon import canonical_bytes, content_hash
from .capability import (
    CAPABILITIES_POLICY, CAPABILITY_EVENTS, SCOPED_TYPES, apply_event, declarable, deepest,
    evaluate, holders, initial_state, scope_of, stray, under, verb_for)
from .errors import SpineRefusal
from .log import GENESIS_PREV, SpineLog
from .policy import TIERS_POLICY, in_force
from .projections import PROJECTORS, fold
from .store import BlobStore
from .vocabulary import (
    BLOB_FIELDS, EVENT_TYPES, EVENT_VERB, RECOVERY, WRITER, WRITER_VOICE, cited_blobs, is_hash,
    is_text)
from .verbs import REPLAY_FIELDS, STANDING
from .verify import NOT_ASSESSED, verify_home

#: The nine questions, in the order a report answers them. The names are the report's keys.
INVARIANTS = ('copies_agree', 'nothing_outside_its_seat', 'every_refusal_is_a_denial',
              'an_amended_plan_is_a_new_approval', 'closes_superseded_never_edited',
              'every_numbers_replay_tuple', 'the_diary_equals_the_filings', 'no_delete',
              'duplicates_coalesce')

#: The one projector that opens no body, which is the whole of what a keyless copy can compare.
KEYLESS_PROJECTORS = ('activity',)
#: The fold an amendment is held against: where the terms it restrikes were held.
POSITIONS = PROJECTORS['positions']
#: What carries a ticket - a trade, and a restrike of one.
TICKETED = ('fill', 'amendment')

#: The script's one section, and the keys an act of it may carry - each read by one invariant. An
#: act naming none of them is a step the oracle has no question about.
ACTS = 'acts'
DENIED, REFUSED, LANE, REPLAY = 'denied', 'refused', 'lane', 'replay'

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


def nothing_outside_its_seat(log, entitled):
    """Every frame's own append re-adjudicated, and the writer's decision re-reached.

    `verb_for` names what a type demands and `evaluate` answers it at `scope_of` - the firm for the
    firm's own facts, the portfolio a body names, a declared node's parent, else the envelope's
    book - against the capability state BEFORE the frame, folded forward by `apply_event`, the
    writer's own step, rather than by one `state_at` per position, which is the same answer at the
    length of the record instead of its square. A node admin's capabilities declaration the grants
    refuse is re-run through `declarable`, as the writer ran it; the writer's own voice is never
    gated, and a home with no document in force answers yes, exactly as the append did.

    Three things a frame says are held to the record around it, as the writer or the hub made
    them: a portfolio sits under its own book; an amendment is judged at a node covering every
    position it moves, the deepest one holding its old terms at the frame before it - the
    `positions` fold advanced beside the capability state; and an approval in the writer's own
    voice is the hub's, which signs only where a tiers policy is in force and books the ticket
    after it signs.

    The ENVELOPE HALF stands without a key: a type no verb declares is a write nobody could be
    scoped for, a type only the writer files under any other name is that voice forged, and the
    writer's own name on anything its voice does not say is a seat wearing it.

    A COPY THAT HOLDS THE FRAMES AND NOT THE DOCUMENT cannot put the entitled half at all. The
    capabilities fold answers `UNREADABLE` for a blob that is gone - fail-closed, because inside
    the writer's own hook a raise would take every append with it - and re-running the writer
    against THAT would call every frame after the declaration a forgery. So the blob is asked for
    first, and a copy that lacks it is told to follow with `--blobs`.
    """
    state, failures, read = initial_state(), [], 0
    positions, signed, booked, workflow = POSITIONS.initial(), [], {}, None
    for frame in log.frames():
        read += 1
        event_type, actor, verb = frame['event_type'], frame['actor'], verb_for(frame['event_type'])
        own = actor == WRITER
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
        body = log.open_body(frame) if event_type in CAPABILITY_EVENTS + SCOPED_TYPES else None
        doc, genesis = state['doc'], state['genesis']
        scope = scope_of(event_type, body, frame['book'])
        held = own or (doc is None and verb != RECOVERY) or evaluate(
            doc, genesis, actor, verb, scope)
        failures.extend(_placed(frame, body, positions))
        if event_type in POSITIONS.reads:
            POSITIONS.apply(positions, frame, log)
        if event_type == 'policy_declared' and body.get('policy') == TIERS_POLICY \
                and is_hash(body.get('blob')):
            workflow = frame['lsn']
        if own and event_type == 'approval':
            signed.append((frame['lsn'], body.get('plan_hash'), workflow))
        if event_type in TICKETED:
            booked[body.get('ticket')] = frame['lsn']
        if event_type in CAPABILITY_EVENTS:
            if _undeclared(log, body):
                return unasked(
                    'LSN {} declares the capabilities document {} and this copy does not hold it, '
                    'so every frame after it was judged under a document that is not here - follow '
                    'with `--blobs` and ask again'.format(frame['lsn'], body['blob']))
            apply_event(state, event_type, actor, body, log.store)
            held = held or (body.get('policy') == CAPABILITIES_POLICY and isinstance(doc, dict)
                            and isinstance(state['doc'], dict)
                            and declarable(doc, state['doc'], actor))
        if not held:
            failures.append(
                'LSN {}: {!r} held no {} scope over {!r} when the {} landed, so this frame is on '
                'the platter and the writer would have refused it'.format(
                    frame['lsn'], actor, verb, scope or '*', event_type))
    failures.extend(
        'LSN {}: an approval of {} stands in the writer\'s own voice with {} - the hub signs a '
        'ticket only under a tiers policy in force and books it after, so this is that voice '
        'forged'.format(lsn, ticket, 'no tiers policy in force' if since is None
                        else 'no fill or restrike after it carrying that ticket')
        for lsn, ticket, since in signed if since is None or booked.get(ticket, 0) < lsn)
    return answer(failures, [
        '{} frame(s) re-adjudicated{}'.format(read, '' if entitled else ' by envelope alone')])


def _placed(frame, body, positions):
    """Why a frame's portfolio is not where the writer would have let it stand - the sentence,
    none where it is: under its own book, and for an amendment covering every position it moves,
    which `positions` - the fold at the frame before it - holds under that book."""
    event_type, book = frame['event_type'], frame['book']
    named = stray(event_type, body, book)
    if named is not None:
        return ['LSN {}: the {} names the portfolio {!r} and is filed under the book {!r} - a '
                'portfolio outside its own book\'s tree, which the writer refuses'.format(
                    frame['lsn'], event_type, named, book)]
    named = body.get('portfolio') if event_type == 'amendment' else None
    held = [] if named is None else holders(positions, body['instrument'], book)
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
    COMPUTES `supersedes_lsn` off the row standing before the frame and files it only where the
    as-of key is later, so a close either does not displace - it is behind the one in force, on the
    platter and standing over nothing - or displaces naming exactly the position it stood over.
    There is no third answer to assert, which is why none is asserted: a reading that re-ran the
    fold to check the fold would be one spelling checking itself.

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
    """Every settlement filed against a derived key names a row the diary carries.

    The diary is a COMPILE and not a fold, so a replica cannot put this question: `keys` is the set
    of cashflow keys an engine answered, handed in as data by a caller that has one. A transition
    whose subject is not an address is filed against an instrument or a name and is not this
    question.
    """
    if keys is None:
        return unasked("the diary is a compile of the book and not a fold of the record, so a "
                       "replica cannot put this question - hand in the diary's own keys")
    keys, failures, filed = set(keys), [], 0
    for subject, standing in fold(log, PROJECTORS['lifecycle'])['transitions'].items():
        if not is_hash(subject):
            continue
        filed += 1
        if subject not in keys:
            failures.append(
                'LSN {}: a settlement was filed against {}, and the diary announces no row under '
                'that key - a status moves a payment the book owes or it moves nothing'.format(
                    standing['lsn'], subject))
    return answer(failures, ['{} settlement(s) filed against {} diary key(s)'.format(
        filed, len(keys))])


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


def report(home, script=None, against=None, diary_keys=None):
    """`{invariant: {held, evidence}}` for all nine, over `home` and what it was given.

    A home whose class key is gone answers the four that cost no key and the envelope half of the
    fifth, and says by name which it could not assess - the replica posture, stated rather than
    skipped.
    """
    log = SpineLog(home)
    try:
        entitled = log.keys.has_firm()
        sealed = unasked(
            'this home holds no class key, so its bodies are sealed and this question is inside '
            'them - ask it on a copy that materialized the key')
        return dict(zip(INVARIANTS, (
            copies_agree(log, against, entitled),
            nothing_outside_its_seat(log, entitled),
            every_refusal_is_a_denial(log, script) if entitled else sealed,
            an_amended_plan_is_a_new_approval(log) if entitled else sealed,
            closes_superseded_never_edited(log) if entitled else sealed,
            every_numbers_replay_tuple(log, script) if entitled else sealed,
            the_diary_equals_the_filings(log, diary_keys) if entitled else sealed,
            no_delete(log, entitled),
            duplicates_coalesce(log))))
    finally:
        log.close()


def failed(answers):
    """Every invariant that did NOT hold. Empty is the day passing; a null holds nothing against
    anybody, since a question nobody put is not a question that failed."""
    return sorted(name for name, found in answers.items() if found['held'] is False)
