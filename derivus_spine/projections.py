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

"""The projections - positions, the blotter, lifecycle state and the strip, folded out of the log.

A projection is a pure fold. `fold` streams the frames a projector names, applies them in LSN order
and answers state nobody edited; a knock, an expiry, an accrual, a position and the seat that struck
a quote are READ off that state and stored nowhere, since storing one would be a second source of
truth about whether the barrier fired. The vocabulary holds facts and this module holds every
consequence of them.

Frames are located by ENVELOPE and only the hits are opened - the rule `capability.build_state` and
`policy.in_force` already state - so a fold costs the length of the history rather than the price of
a key, and `activity` opens nothing at all, which is what lets a strip render on a replica holding
no key. Nothing here caches: the caller holds `(head_lsn, state)` and advances it, and a seed is
that pair written down at an official close so a cold reader starts there instead of at genesis.

A seed is derivable, disposable and VERIFIED rather than trusted: it carries the event hash of the
close it summarises, the projector and version whose row shape its state is in, and the address of
that state, so a seed from another home, another projector or an edited file refuses by name at
the fold that would consume it. Deleting `seeds/` costs one refold.
"""
import json
import os

from .canon import canonical_bytes, content_hash
from .errors import SpineRefusal
from .log import as_of_key, parse_number
from .policy import FIXINGS_POLICY, FIXINGS_SECTION, in_force
from .verbs import REPLAY_FIELDS

#: Where a seed is filed, beside `log/` and `blobs/`. Not a blob: a blob is write-once and never
#: forgotten, and a seed is a file an operator may delete at the cost of one refold.
SEEDS = 'seeds'
#: The one position a seed may be minted at - where the desk already agrees what the day was.
SEED_EVENT = 'official_close_declared'
#: The election choice the blotter reads as exercised. What the rest of a choice means is the
#: deal's business and not the record's.
EXERCISE = 'exercise'

#: event type -> the one sentence the strip renders. A type absent from this table renders its own
#: name rather than dropping out of the sequence, so a replica of a hub running a newer vocabulary
#: still shows every LSN; the gate holds the table against the vocabulary this hub has.
SUMMARIES = {
    'fill': 'a clip was booked',
    'amendment': 'terms were restruck under a new instrument',
    'election': 'a holder made its choice',
    'fixing_observed': 'an observation was printed',
    'determination': 'a ruling was made',
    'status_transition': 'a state was moved',
    'market_declared': 'a market name was pointed at a values vector',
    'official_close_declared': 'the official close was declared',
    'approval': 'a plan was approved',
    'rejection': 'a plan was rejected',
    'snapshot_registered': 'a snapshot was registered',
    'retention_declared': 'a retention policy was declared',
    'rehash_declared': 'a hash algorithm was declared',
    'break_glass_used': 'the recovery seat was used',
    'policy_declared': 'a policy document was declared',
    'checkpoint': 'the head was signed',
    'run_completed': 'a standing run attested its numbers',
    'result_pinned': 'a replayed result was pinned',
    'quote_filed': 'a quote was filed',
    'seat_enrolled': 'a seat was enrolled',
    'key_wrapped': 'a class key was wrapped to a seat',
    'capability_denied': 'the writer refused an append',
}


class Projector:
    """A fold's three members and no fourth: what it is called, what version its rows are, and the
    envelope types it reads - `None` for every type.

    `initial` mints empty state, `apply` folds one frame into it, `rows` answers the JSON a reader
    sees - sorted, canonicalisable, and carrying no clock of its own. `version` is bumped when a row
    shape moves, and a seed minted by another version refuses.
    """

    name = None
    version = 1
    reads = ()


class Positions(Projector):
    """The net position per instrument, the clips behind it, and the amendment that moved it.

    An amendment carries the position FORWARD: the file's deal node hashes to the amended terms, so
    the position lives on the new instrument and the old row stands at zero naming where it went. A
    row is never dropped - a position closed out is a fact about the book, not an absence.
    """

    name = 'positions'
    reads = ('fill', 'amendment')

    def initial(self):
        return {}

    def apply(self, state, frame, log):
        body = log.open_body(frame)
        row = state.get(body['instrument'])
        if row is None:
            row = state[body['instrument']] = {
                'book': frame['book'], 'netting_set': None, 'counterparty': None,
                'quantity': 0.0, 'clips': 0, 'amended_to': None,
                'first_lsn': frame['lsn'], 'last_lsn': frame['lsn']}
        if frame['event_type'] == 'fill':
            row['netting_set'] = body['netting_set']
            row['counterparty'] = body['counterparty']
            row['quantity'] += float(body['quantity'])
            row['clips'] += 1
        else:
            row['amended_to'] = body['amended_to']
            head = state.setdefault(body['amended_to'], dict(
                row, quantity=0.0, clips=0, amended_to=None, first_lsn=frame['lsn']))
            head['quantity'] += row['quantity']
            head['clips'] += row['clips']
            head['last_lsn'] = frame['lsn']
            row['quantity'], row['clips'] = 0.0, 0
        row['last_lsn'] = frame['lsn']

    def rows(self, state):
        return [_shown(row, instrument=instrument) for instrument, row in sorted(state.items())]


class Lifecycle(Projector):
    """Every print under its `(index, date, source)` key with the prints it superseded, the
    elections and rulings filed against each instrument, and the status standing under each subject
    a transition names.

    Supersession is by `(effective_time, lsn)`, so a backdated republication does not win merely by
    arriving last, and the print it beat stays on the row - the record holds it, so the projection
    may not hide it.
    """

    name = 'lifecycle'
    reads = ('fixing_observed', 'election', 'determination', 'status_transition')

    def initial(self):
        return {'fixings': {}, 'instruments': {}, 'transitions': {}}

    def apply(self, state, frame, log):
        body = log.open_body(frame)
        if frame['event_type'] == 'fixing_observed':
            self._observe(state['fixings'], frame, body)
            return
        if frame['event_type'] == 'status_transition':
            # A subject is an instrument here and a cashflow key later, so it is filed as given.
            _stand(state['transitions'], body['subject'], frame, {'status': body['status']})
            return
        row = state['instruments'].setdefault(
            body.get('instrument') or body['subject'], {'elections': [], 'determinations': []})
        if frame['event_type'] == 'election':
            row['elections'].append({'choice': body['choice'], 'lsn': frame['lsn']})
        else:
            row['determinations'].append(
                {'ruling': body['ruling'], 'actor': frame['actor'], 'lsn': frame['lsn']})

    def rows(self, state):
        fixings = state['fixings']
        return {
            'fixings': [_shown(fixings[index][date][source],
                               index=index, date=date, source=source)
                        for index in sorted(fixings) for date in sorted(fixings[index])
                        for source in sorted(fixings[index][date])],
            'instruments': [_shown(row, instrument=instrument)
                            for instrument, row in sorted(state['instruments'].items())],
            'transitions': [_shown(row, subject=subject)
                            for subject, row in sorted(state['transitions'].items())]}

    @staticmethod
    def _observe(fixings, frame, body):
        """File one print under `(index, date, source)`, the later as-of key standing and every
        print it beat kept beside it in LSN order."""
        under = fixings.setdefault(body['index'], {}).setdefault(body['date'], {})
        filed = {'value': float(body['value']), 'effective_time': frame['effective_time'],
                 'as_of': as_of_key(frame)[0], 'lsn': frame['lsn'], 'supersedes': []}
        standing = under.get(body['source'])
        if standing is None:
            under[body['source']] = filed
        elif _later(filed, standing):
            filed['supersedes'] = standing['supersedes'] + [
                {'value': standing['value'], 'lsn': standing['lsn']}]
            under[body['source']] = filed
        else:
            standing['supersedes'].append({'value': filed['value'], 'lsn': filed['lsn']})


class Blotter(Projector):
    """One row per instrument any fact names: its size, the state the facts put it in, and the last
    fact that touched it.

    A row exists for an instrument an election or a ruling names and no fill does, so an act on an
    amended-to hash is visible rather than silently dropped.
    """

    name = 'blotter'
    reads = ('fill', 'amendment', 'election', 'status_transition', 'determination')

    def initial(self):
        return {}

    def apply(self, state, frame, log):
        body = log.open_body(frame)
        row = state.setdefault(body.get('instrument') or body['subject'],
                               {'netting_set': None, 'quantity': 0.0, 'state': 'live'})
        if frame['event_type'] == 'fill':
            row['netting_set'] = body['netting_set']
            row['quantity'] += float(body['quantity'])
        elif frame['event_type'] == 'amendment':
            row['state'] = 'amended'
        elif frame['event_type'] == 'election' and body['choice'] == EXERCISE:
            row['state'] = 'exercised'
        row['last_fact'] = {'type': frame['event_type'], 'lsn': frame['lsn'],
                            'effective_time': frame['effective_time'], 'actor': frame['actor']}

    def rows(self, state):
        return [_shown(row, instrument=instrument) for instrument, row in sorted(state.items())]


class Markets(Projector):
    """The market names, the official close standing per market with the close it restated, and the
    snapshots.

    A close is superseded by a NEW close rather than corrected in place, so the row names the LSN it
    stands over.
    """

    name = 'markets'
    reads = ('market_declared', 'official_close_declared', 'snapshot_registered')

    def initial(self):
        return {'names': {}, 'closes': {}, 'snapshots': []}

    def apply(self, state, frame, log):
        body = log.open_body(frame)
        if frame['event_type'] == 'snapshot_registered':
            state['snapshots'].append(
                {'blob': body['blob'], 'lsn': frame['lsn'], 'book': frame['book']})
        elif frame['event_type'] == 'market_declared':
            _stand(state['names'], body['name'], frame,
                   {'values_hash': body['values_hash'], 'actor': frame['actor']})
        else:
            standing = state['closes'].get(body['market'])
            _stand(state['closes'], body['market'], frame,
                   {'values_hash': body['values_hash'],
                    'supersedes_lsn': standing['lsn'] if standing else None})

    def rows(self, state):
        return {'names': [_shown(row, name=name) for name, row in sorted(state['names'].items())],
                'closes': [_shown(row, market=market)
                           for market, row in sorted(state['closes'].items())],
                'snapshots': list(state['snapshots'])}


class Attestations(Projector):
    """Every standing attestation, under the four coordinates a reported number replays from.

    The FIRST row under a tuple stands, which is what `verbs.attestation` answers, so this reading
    and that one cannot disagree about which attestation a number replays from. `result_pinned` is
    not read: a pin is by definition a claim this hub did not witness, and reading one would let a
    promotion become the evidence for the next.
    """

    name = 'attestations'
    reads = ('run_completed',)

    def initial(self):
        return {}

    def apply(self, state, frame, log):
        body = log.open_body(frame)
        state.setdefault(content_hash(dict((field, body[field]) for field in REPLAY_FIELDS)),
                         dict(body, lsn=frame['lsn']))

    def rows(self, state):
        return sorted(state.values(), key=lambda row: row['lsn'])


class Decisions(Projector):
    """The verdicts filed against each plan hash, in the order they were filed, and the policy
    document standing under each name.

    A verdict is never withdrawn: a second seat's rejection sits beside the first's approval, and
    what a deployment does with two verdicts is the deployment's rule. A policy is the opposite -
    the LAST declaration stands, which is what `policy.in_force` answers.
    """

    name = 'decisions'
    reads = ('approval', 'rejection', 'policy_declared')

    def initial(self):
        return {'plans': {}, 'policies': {}}

    def apply(self, state, frame, log):
        body = log.open_body(frame)
        if frame['event_type'] == 'policy_declared':
            state['policies'][body['policy']] = {'blob': body.get('blob'), 'lsn': frame['lsn']}
            return
        state['plans'].setdefault(body['plan_hash'], []).append(
            {'verdict': frame['event_type'], 'actor': frame['actor'],
             'reason': body.get('reason'), 'lsn': frame['lsn']})

    def rows(self, state):
        return {'plans': [{'plan_hash': plan, 'verdicts': verdicts}
                          for plan, verdicts in sorted(state['plans'].items())],
                'policies': [_shown(row, policy=policy)
                             for policy, row in sorted(state['policies'].items())]}


class Quotes(Projector):
    """One row per quote filed: who struck it, the two hashes it pinned, the ticket an approval
    would sign where one was computed, and what was solved for what edge.

    The BOOKER is the frame's own actor, since no body carries one - which is what makes "may this
    seat approve this quote" a fold rather than a second field somebody has to fill in. A quote id
    is minted per structure, so a second filing under one id is a RESTATEMENT and stands by the
    as-of key the whole record files keyed rows under - a backdated one does not win by arriving
    last. Rows read in LSN order, which is the order they were struck in and not their ids'.
    """

    name = 'quotes'
    reads = ('quote_filed',)

    def initial(self):
        return {}

    def apply(self, state, frame, log):
        body = log.open_body(frame)
        _stand(state, body['quote_id'], frame, {
            'booker': frame['actor'], 'book': frame['book'], 'plan_hash': body['plan_hash'],
            'values_hash': body['values_hash'], 'ticket': body.get('ticket'),
            'structure': body['structure'], 'solved': body['solved'], 'edge': body['edge']})

    def rows(self, state):
        return sorted((_shown(row, quote_id=quote_id) for quote_id, row in state.items()),
                      key=lambda row: row['lsn'])


class Denials(Projector):
    """Every append the writer refused, in the order it refused them: the seat, the verb it lacked,
    the scope the verb was wanted over, and the type the refused act would have said.

    The one reading that says WHAT was refused: the envelope carries a type and the strip renders
    one declared sentence for every denial alike, so who was turned away from what is inside the
    body and nowhere else. Its reader is the acceptance game's oracle, which holds the script's
    attempts against the refusals the record kept. A refusal that never reached the writer - a tier
    that admits no ticket, a stale board, a malformed request - mints nothing and is not here.
    """

    name = 'denials'
    reads = ('capability_denied',)

    def initial(self):
        return []

    def apply(self, state, frame, log):
        body = log.open_body(frame)
        state.append({'lsn': frame['lsn'], 'subject': body['subject'], 'verb': body['verb'],
                      'book': body['book'], 'attempted_type': body['attempted_type']})

    def rows(self, state):
        return list(state)


class Activity(Projector):
    """One line per event, envelope only: where it sits, when it was recorded and when it is true,
    who said it, and the declared sentence about what it was.

    Every type, so the sequence a replica renders has no gaps in it, and the only projector that
    opens no body, so that replica needs no key.
    """

    name = 'activity'
    reads = None

    def initial(self):
        return []

    def apply(self, state, frame, log):
        state.append({'lsn': frame['lsn'], 'record_time': frame['record_time'],
                      'effective_time': frame['effective_time'], 'actor': frame['actor'],
                      'event_type': frame['event_type'], 'book': frame['book'],
                      'summary': SUMMARIES.get(frame['event_type'], frame['event_type'])})

    def rows(self, state):
        return list(state)


#: The projectors this module ships, by name. A reader picks one; nothing here is a default.
PROJECTORS = dict((projector.name, projector) for projector in (
    Positions(), Lifecycle(), Blotter(), Markets(), Attestations(), Decisions(), Quotes(),
    Denials(), Activity()))


def fold(log, projector, lsn=None, seed=None):
    """`projector`'s state folded out of `log` at or before `lsn`, from genesis or from `seed`.

    The frames the projector does not name are skipped by envelope, so nothing the fold ignores
    costs a key. PURE in its seed: the state is copied before it is advanced and the seed is left
    where it was, so one seed folds to as many positions as a caller wants and a fold behind it
    refuses rather than answering the state in front.
    """
    state, start = ((projector.initial(), 1) if seed is None
                    else (_from_seed(seed, projector, lsn), seed['lsn'] + 1))
    for frame in log.frames(start_lsn=start, end_lsn=lsn):
        if projector.reads is None or frame['event_type'] in projector.reads:
            projector.apply(state, frame, log)
    return state


def seed_at(log, projector, close_lsn):
    """Mint `projector`'s seed at `close_lsn`, file it under `seeds/`, and answer it.

    A seed is minted only AT an official close, the one position where the desk already agrees what
    the day was, and it carries that close's own event hash and its state's own address, so a
    reader can tell whether the history it summarises is this one and whether the state is the
    one minted over it. Written the way the store writes: scratch, fsync, rename.
    """
    frame = _close_frame(log, close_lsn, 'a seed at LSN {}'.format(close_lsn))
    state = fold(log, projector, lsn=close_lsn)
    seed = {'projector': projector.name, 'version': projector.version, 'lsn': close_lsn,
            'head': frame['event_hash'], 'state_hash': content_hash(state),
            'state': state}
    path = _seed_path(log, projector, close_lsn)
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.parent / (path.name + '.new')
    with scratch.open('wb') as handle:
        handle.write(canonical_bytes(seed))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(scratch), str(path))
    return seed


def read_seed(log, projector, close_lsn):
    """The seed filed for `projector` at `close_lsn`, or None where none is filed.

    Verified, never trusted: the file must parse, say which projector and version minted it, carry
    the event hash LSN `close_lsn` has IN THIS LOG, and hold state that hashes to its own stamp -
    so a seed carried in from another home, or edited under its own name, refuses by name rather
    than folding a fiction nothing else can detect. Every refusal here is cured by deleting the
    file and refolding.
    """
    path = _seed_path(log, projector, close_lsn)
    if not path.is_file():
        others = sorted(other.name for other in (log.home / SEEDS).glob(
            '{}-*-{}.json'.format(projector.name, close_lsn)))
        if not others:
            return None
        raise SpineRefusal(
            'the seed filed for {} at LSN {} is {} and this projector is version {}: a row shape '
            'that moved is not state this fold can advance from. Delete it and refold - a seed is '
            'derivable, and the pass that replaces it is the one it saved'.format(
                projector.name, close_lsn, ', '.join(others), projector.version))
    where = 'the seed at {}'.format(path)
    try:
        seed = json.loads(path.read_text(encoding='utf-8'), parse_int=parse_number)
    except (UnicodeDecodeError, ValueError) as unreadable:
        raise SpineRefusal(
            '{} is not JSON ({}): a seed is written whole or not at all, so a torn one is a file to '
            'delete and refold, never a state to advance'.format(where, unreadable))
    if not isinstance(seed, dict):
        raise SpineRefusal(
            '{} holds {}, not the object a seed is - delete it and refold'.format(
                where, type(seed).__name__))
    _minted_by(seed, projector, where)
    head = _close_frame(log, close_lsn, where)['event_hash']
    if seed.get('lsn') != close_lsn or seed.get('head') != head:
        raise SpineRefusal(
            '{} summarises LSN {} under the event hash {}, and LSN {} of THIS log is {}: a seed is '
            'one history at one position, and this is not that history. Delete it and refold'
            .format(where, seed.get('lsn'), seed.get('head'), close_lsn, head))
    if content_hash(seed.get('state')) != seed.get('state_hash'):
        raise SpineRefusal(
            '{} carries state hashing to {} under the stamp {}: the file was edited beneath its '
            'own close, and a state nobody can address is not one this fold will advance. Delete '
            'it and refold'.format(
                where, content_hash(seed.get('state')), seed.get('state_hash')))
    return seed


def fixings_at(log, lsn=None, sources=None, indices=None):
    """`{(index, date): {value, source, effective_time, lsn}}` - the fixing in force per index and
    date, resolved across sources by the declared order.

    One fold: the prints are the `lifecycle` projection's, and `sources` is the `fixings` policy in
    force unless the caller states one. The first named source holding a print wins. `indices` is
    what the caller needs resolved (default: every index the log holds prints of), and an index
    among them that the document does not name REFUSES BY NAME - a fixing whose authority nobody
    declared is not a fixing a plan may use - while an index nobody asked about is left alone. A
    home declaring no policy at all has named an authority for nothing and answers nothing.
    """
    if sources is None:
        blob, document, _ = in_force(log, FIXINGS_POLICY, lsn)
        if blob is None:
            return {}
        sources = document[FIXINGS_SECTION]
    observed = fold(log, PROJECTORS['lifecycle'], lsn=lsn)['fixings']
    resolved = {}
    for index in sorted(observed if indices is None else set(indices) & set(observed)):
        order = sources.get(index)
        if not order:
            raise SpineRefusal(
                'the {} policy in force here names no source order for {!r}, which this log holds '
                'prints of from {}: a fixing whose authority nobody declared is not a fixing a plan '
                'may use. Declare the order ({{"{}": {{{!r}: [administrator, ...]}}}}) and ask '
                'again'.format(FIXINGS_POLICY, index, ', '.join(sorted(set(
                    source for under in observed[index].values() for source in under))),
                    FIXINGS_SECTION, index))
        for date in sorted(observed[index]):
            for source in order:
                print_ = observed[index][date].get(source)
                if print_ is not None:
                    resolved[(index, date)] = {
                        'value': print_['value'], 'source': source,
                        'effective_time': print_['effective_time'], 'lsn': print_['lsn']}
                    break
    return resolved


def knocked(observations, index, level, rising=True):
    """`{knocked, knocked_on}` - the first date `index`'s fixing in force crosses `level`, read off
    `fixings_at`'s answer.

    The terms are the caller's and the observations are the record's, so a knock is derived where it
    is asked for and stored nowhere; the record would have to hold two answers to hold this one.
    """
    for name, date in sorted(observations):
        if name != index:
            continue
        value = observations[(name, date)]['value']
        if value >= level if rising else value <= level:
            return {'knocked': True, 'knocked_on': date}
    return {'knocked': False, 'knocked_on': None}


# ------------------------------------------------------------------------------------------------
# The pieces the projectors and the fold above are made of.

def _shown(row, **named):
    """`row` as the reader sees it, under the key it was filed by.

    The as-of key the fold compared on is dropped: where a fact carried no truth-time of its own it
    is the writer's clock, which is a property of the recording rather than of the fact.
    """
    shown = dict((field, value) for field, value in row.items() if field != 'as_of')
    shown.update(named)
    return shown


def _later(filed, standing):
    """Whether `filed` supersedes `standing` - the `(effective_time, lsn)` order, on rows that carry
    their own as-of key."""
    return (filed['as_of'], filed['lsn']) > (standing['as_of'], standing['lsn'])


def _stand(under, name, frame, row):
    """File `row` under `name` where its as-of key is later than the one standing there.

    A declaration backdated behind the one in force is on the platter and does not displace it.
    """
    filed = dict(row, effective_time=frame['effective_time'], lsn=frame['lsn'],
                 as_of=as_of_key(frame)[0])
    standing = under.get(name)
    if standing is None or _later(filed, standing):
        under[name] = filed


def _seed_path(log, projector, close_lsn):
    """`seeds/<projector>-<version>-<lsn>.json` - the name carries what the file must say."""
    return log.home / SEEDS / '{}-{}-{}.json'.format(
        projector.name, projector.version, close_lsn)


def _close_frame(log, close_lsn, where):
    """The frame at `close_lsn`, read off the PLATTER and asserted to be an official close.

    Never `frame_at`: that answers out of the index this handle built when it opened, and a handle
    routinely outlives someone else's append, so a close another process just declared would read
    as a log that has no such position.
    """
    frame = next(iter(log.frames(start_lsn=close_lsn, end_lsn=close_lsn)), None)
    if frame is None:
        raise SpineRefusal(
            '{}: this log holds no LSN {} - seed at a close this home actually carries'.format(
                where, close_lsn))
    if frame['event_type'] != SEED_EVENT:
        raise SpineRefusal(
            '{}: LSN {} is a {}, and a seed is pinned to an {} - the one position where the desk '
            'already agrees what the day was'.format(
                where, close_lsn, frame['event_type'], SEED_EVENT))
    return frame


def _minted_by(seed, projector, where):
    """`seed` asserted to be this projector's at this version - the two facts its state's shape
    depends on, checked where the state is consumed rather than only where the file is read."""
    if seed.get('projector') != projector.name:
        raise SpineRefusal(
            '{}: it was minted by the {!r} projector and this is {!r} - one projector\'s state is '
            'not another\'s, whatever the file it arrived in was called'.format(
                where, seed.get('projector'), projector.name))
    if seed.get('version') != projector.version:
        raise SpineRefusal(
            '{}: it was minted by version {} of {} and this projector is version {} - a row shape '
            'that moved is not state this fold can advance from. Refold from genesis'.format(
                where, seed.get('version'), projector.name, projector.version))


def _from_seed(seed, projector, lsn):
    """`seed`'s state, checked and COPIED - what the fold advances instead of the caller's object.

    The copy is the canonical round trip, which is also the check that the state is the JSON a seed
    file holds. A position behind the seed refuses: a fold walks forward, and the frames that would
    undo a close are not on the platter to walk.
    """
    _minted_by(seed, projector, 'this fold\'s seed')
    if lsn is not None and lsn < seed['lsn']:
        raise SpineRefusal(
            'this seed holds {} as of LSN {} and the fold was asked for LSN {}: a fold walks '
            'forward, so a position behind a seed is not one it can reach - fold from genesis for '
            'that position, or seed at an earlier close'.format(
                projector.name, seed['lsn'], lsn))
    return json.loads(canonical_bytes(seed['state']).decode('utf-8'), parse_int=parse_number)
