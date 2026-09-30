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

"""The projections - positions, the blotter, lifecycle state, cash and the strip, folded out of the
log - and `CSA`, the credit support annex's arithmetic read over the cash fold.

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
import pathlib

from .canon import canonical_bytes, content_hash
from .errors import SpineRefusal
from .log import as_of_key, parse_number
from .policy import FIXINGS_POLICY, FIXINGS_SECTION, in_force
from .verbs import HELD, PAYMENT, REPLAY_FIELDS
from .vocabulary import is_hash

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
    'portfolio_declared': 'a portfolio was declared',
    'checkpoint': 'the head was signed',
    'run_completed': 'a standing run attested its numbers',
    'result_pinned': 'a replayed result was pinned',
    'quote_filed': 'a quote was filed',
    'seat_enrolled': 'a seat was enrolled',
    'key_wrapped': 'a class key was wrapped to a seat',
    'entity_declared': 'a legal entity was declared',
    'agreement_declared': 'an agreement was declared',
    'capability_denied': 'the writer refused an append',
}

#: The projectors a seed is never minted for. The strip's state IS its history, so copying it out of
#: a seed costs more than folding it (219 ms against 38 at two thousand events).
UNSEEDED = ('activity',)


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
    """The net position per instrument, agreement and portfolio, the clips behind it, and the
    amendment that moved it.

    A position is keyed WHERE IT SITS: one instrument dealt under two agreements is two positions,
    credit exposure being per agreement, and one instrument in two portfolios is two, a portfolio
    being where risk is owned. A fill filed before the key existed sits under its netting set and
    its book. An amendment carries every position FORWARD onto the instrument the amended terms
    hash to, each under its own key, the old row standing at zero naming where it went; a row is
    never dropped - a position closed out is a fact about the book, not an absence.

    `tickets` are its clips under the ticket that books each - its fill's, or the restrike's that
    restruck it - every clip keeping the instrument and execution reference it was filled under,
    the key its confirmation is filed against, and the LSN of that fill: `{ticket, lsn, actor,
    at, quantity, instrument, execution_reference, fill}`. So what a position reads is a fold
    and no fill is reopened.
    """

    name = 'positions'
    version = 3
    reads = ('fill', 'amendment')

    def initial(self):
        return {}

    def apply(self, state, frame, log):
        body = log.open_body(frame)
        if frame['event_type'] == 'fill':
            row = _position(state, body['instrument'], body.get('agreement') or body['netting_set'],
                            body.get('portfolio') or frame['book'] or '', frame)
            row['counterparty'] = body['counterparty']
            row['quantity'] += float(body['quantity'])
            row['clips'] += 1
            row['last_lsn'] = frame['lsn']
            row['tickets'].append(_ticketed(frame, body, {
                'quantity': float(body['quantity']), 'instrument': body['instrument'],
                'execution_reference': body['execution_reference'], 'fill': frame['lsn']}))
            return
        for agreement, under in sorted(state.get(body['instrument'], {}).items()):
            for portfolio, row in sorted(under.items()):
                head = _position(state, body['amended_to'], agreement, portfolio, frame, row)
                head['quantity'] += row['quantity']
                head['clips'] += row['clips']
                head['last_lsn'] = frame['lsn']
                head['tickets'].extend(_ticketed(frame, body, clip) for clip in row['tickets'])
                row['amended_to'] = body['amended_to']
                row['quantity'], row['clips'], row['tickets'] = 0.0, 0, []
                row['last_lsn'] = frame['lsn']

    def rows(self, state):
        return [_shown(row, instrument=instrument, agreement=agreement, portfolio=portfolio)
                for instrument, agreements in sorted(state.items())
                for agreement, portfolios in sorted(agreements.items())
                for portfolio, row in sorted(portfolios.items())]


class Costs(Projector):
    """What every position cost, at AVERAGE COST, and what its reductions realised - keyed where
    the position sits, as `positions` keys it.

    `basis` is the open quantity times its average price, signed with the position: a fill adding
    to a position adds its quantity times its price, and one reducing it realises the difference
    between the price and the average on the part it closes, the rest opening the other way at its
    own price. The price is the fill's, per unit of the instrument as written, crossed into the
    book's currency at the rate the booking filed beside it. A fill booked with no price leaves
    what it touched UNKNOWN - `basis` null while that lot is open - and counts under `unpriced`;
    `realised` is what every reduction it COULD price realised, and `unpriced_reductions` counts
    the ones it could not, closing at no price or against an average nobody priced, so a realised
    figure is whole only while that count is nothing and a later priced trade is still known. An
    amendment carries the open basis onto the instrument the terms became; what the old terms
    realised stays on their row.
    """

    name = 'costs'
    version = 2
    reads = ('fill', 'amendment')

    def initial(self):
        return {}

    def apply(self, state, frame, log):
        body = log.open_body(frame)
        if frame['event_type'] == 'amendment':
            for agreement, under in sorted(state.get(body['instrument'], {}).items()):
                for portfolio, row in sorted(under.items()):
                    head = _cost_row(state, body['amended_to'], agreement, portfolio)
                    head['quantity'] += row['quantity']
                    head['basis'] = _plus(head['basis'], row['basis'])
                    head['unpriced'] += row['unpriced']
                    row['quantity'], row['basis'] = 0.0, 0.0
            return
        row = _cost_row(state, body['instrument'], body.get('agreement') or body['netting_set'],
                        body.get('portfolio') or frame['book'] or '')
        quantity, held = float(body['quantity']), row['quantity']
        price = None if body.get('price') is None else float(body['price']) * float(
            body.get('rate', 1.0))
        # the part of this fill that closes what is held, and the part that opens beyond it
        closing = 0.0 if held * quantity >= 0 else (
            quantity if abs(quantity) <= abs(held) else -held)
        opening = quantity - closing
        if closing:
            average = None if row['basis'] is None else row['basis'] / held
            if average is None or price is None:
                row['unpriced_reductions'] += 1
            else:
                row['realised'] += closing * (average - price)
            row['basis'] = None if average is None else row['basis'] + closing * average
        if opening:
            lot = None if price is None else opening * price
            row['basis'] = lot if held + closing == 0 else _plus(row['basis'], lot)
        if price is None:
            row['unpriced'] += 1
        row['quantity'] = held + quantity
        if not row['quantity']:
            row['basis'] = 0.0

    def rows(self, state):
        return [_shown(row, instrument=instrument, agreement=agreement, portfolio=portfolio)
                for instrument, agreements in sorted(state.items())
                for agreement, portfolios in sorted(agreements.items())
                for portfolio, row in sorted(portfolios.items())]


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
            # Only a payment moves a state; a fee or a balance moved money and nothing else.
            if body.get('kind', PAYMENT) == PAYMENT:
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
        if body.get('kind') in HELD:
            return  # a balance held under an agreement, and an agreement is not a trade
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
    """The market names, the official close standing per market with the close of its day it
    restated, and the snapshots.

    A close is FOR a day - its `date`, else the day it is true on - and a market stands on its
    LATEST close by day, ties by LSN, so a past day restated after a later day's close supersedes
    the declaration of its own day and never the later one. A close is superseded by a NEW close
    rather than corrected in place, so the row names the LSN of its day's close it stands over.
    """

    name = 'markets'
    version = 2
    reads = ('market_declared', 'official_close_declared', 'snapshot_registered')

    def initial(self):
        return {'names': {}, 'closes': {}, 'days': {}, 'snapshots': []}

    def apply(self, state, frame, log):
        body = log.open_body(frame)
        if frame['event_type'] == 'snapshot_registered':
            state['snapshots'].append(
                {'blob': body['blob'], 'lsn': frame['lsn'], 'book': frame['book']})
        elif frame['event_type'] == 'market_declared':
            _stand(state['names'], body['name'], frame,
                   {'values_hash': body['values_hash'], 'actor': frame['actor']})
        else:
            day = body.get('date') or as_of_key(frame)[0][:10]
            days, standing = state['days'].setdefault(body['market'], {}), state['closes'].get(
                body['market'])
            filed = {'values_hash': body['values_hash'], 'date': day,
                     'supersedes_lsn': days.get(day), 'effective_time': frame['effective_time'],
                     'lsn': frame['lsn'], 'as_of': as_of_key(frame)[0]}
            days[day] = frame['lsn']
            if standing is None or (day, frame['lsn']) > (standing['date'], standing['lsn']):
                state['closes'][body['market']] = filed

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
    document standing under each name with the position the first one that names a document
    stood from.

    A verdict is never withdrawn: a second seat's rejection sits beside the first's approval, and
    what a deployment does with two verdicts is the deployment's rule. A policy is the opposite -
    the LAST declaration stands, which is what `policy.in_force` answers.
    """

    name = 'decisions'
    version = 2
    reads = ('approval', 'rejection', 'policy_declared')

    def initial(self):
        return {'plans': {}, 'policies': {}}

    def apply(self, state, frame, log):
        body = log.open_body(frame)
        if frame['event_type'] == 'policy_declared':
            since = state['policies'].get(body['policy'], {}).get('since')
            state['policies'][body['policy']] = {
                'blob': body.get('blob'), 'lsn': frame['lsn'],
                'since': since or (frame['lsn'] if is_hash(body.get('blob')) else None)}
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


class Entities(Projector):
    """Every legal entity declared, under its id: its name and the parent it is grouped under.

    A second declaration under one id restates the entity and stands by the as-of key, so a
    backdated one does not win by arriving last.
    """

    name = 'entities'
    reads = ('entity_declared',)

    def initial(self):
        return {}

    def apply(self, state, frame, log):
        body = log.open_body(frame)
        _stand(state, body['entity'], frame,
               {'name': body['name'], 'parent': body.get('parent'), 'actor': frame['actor']})

    def rows(self, state):
        return [_shown(row, entity=entity) for entity, row in sorted(state.items())]


class Agreements(Projector):
    """Every agreement declared, under its id: the entity it is with, the document's kind, and the
    address of the terms its netting set is compiled from.

    A restatement stands by the as-of key and names the declaration it stood over, as a close does,
    so an as-at read before it still answers the terms it answered.
    """

    name = 'agreements'
    reads = ('agreement_declared',)

    def initial(self):
        return {}

    def apply(self, state, frame, log):
        body = log.open_body(frame)
        standing = state.get(body['agreement'])
        _stand(state, body['agreement'], frame,
               {'entity': body['entity'], 'kind': body['kind'], 'terms': body['terms'],
                'actor': frame['actor'],
                'supersedes_lsn': standing['lsn'] if standing else None})

    def rows(self, state):
        return [_shown(row, agreement=agreement) for agreement, row in sorted(state.items())]


class Portfolios(Projector):
    """Every node of the desk's tree declared, under its path: who declared it, and when.

    A path declared again stands by the as-of key, as every keyed row here does.
    """

    name = 'portfolios'
    reads = ('portfolio_declared',)

    def initial(self):
        return {}

    def apply(self, state, frame, log):
        _stand(state, log.open_body(frame)['path'], frame, {'actor': frame['actor']})

    def rows(self, state):
        return [_shown(row, path=path) for path, row in sorted(state.items())]


class Cash(Projector):
    """Every movement of money a settlement filed, under the settlement system's own reference: what
    it settled, its kind, the asset and the signed amount - received positive, paid or posted
    negative - on the value date it states.

    A second filing under one reference RESTATES the movement and stands by the as-of key, naming
    the filing it stood over, so a corrected amount replaces the one it corrects rather than
    counting beside it. A balance - collateral held under an agreement, cash in a currency - is a
    sum over these rows and stored nowhere.
    """

    name = 'cash'
    reads = ('status_transition',)

    def initial(self):
        return {}

    def apply(self, state, frame, log):
        body = log.open_body(frame)
        if 'amount' not in body:
            return
        standing = state.get(body['reference'])
        _stand(state, body['reference'], frame,
               {'subject': body['subject'], 'kind': body['kind'], 'asset': body['asset'],
                'amount': float(body['amount']), 'actor': frame['actor'], 'book': frame['book'],
                'supersedes_lsn': standing['lsn'] if standing else None})

    def rows(self, state):
        return sorted((_shown(row, reference=reference) for reference, row in state.items()),
                      key=lambda row: row['lsn'])


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


class CSA:
    """The collateral call at one date - the engine's own recursion, read on a close.

    A CSA asks for the credit support its dials make of the exposure: the independent amount, plus
    the exposure's excess over the received threshold where it is above it, plus its excess below
    the posted threshold where it is below it - the engine's `At`, thresholds and independent amount
    stated in the agreement currency and crossed from it. The CALL is that less the balance held,
    each asset valued after its `Haircut_Posted`, the one haircut the engine reads, and it moves
    only where it clears the minimum transfer on its side STRICTLY, as `scan_collateral_balance`
    transfers: the bank calls where it is short by more than `Minimum_Received`, and posts where it
    holds more than `Minimum_Posted` over what is required. Nothing is rounded, the netting set
    declaring no rounding. COLLATERAL AND MARGIN ARE TWO BALANCES: the call reads the collateral
    alone. Pure functions over plain data, so the service and the oracle run one spelling.
    """

    #: The two sides of a call, from the bank's: it calls collateral in, or posts it out.
    CALL, POST = 'call', 'post'
    #: What `call` answers - the fields a call nobody can work out carries as nulls.
    WORKED = ('required', 'balance', 'call', 'direction', 'minimum_transfer')
    #: The dials a CSA states under `Credit_Support_Amounts`, each a `CreditSupportList`.
    DIALS = ('Independent_Amount', 'Received_Threshold', 'Posted_Threshold', 'Minimum_Received',
             'Minimum_Posted')
    #: What an agreement nothing moved under holds: two empty balances.
    NOTHING = dict.fromkeys(HELD, {})

    @classmethod
    def of(cls, terms):
        """The CSA a netting set's `terms` declare, as the engine reads them - or None where they
        collateralise nothing, the engine then holding no collateral at all.

        `{Agreement_Currency, Balance_Currency, <each of DIALS>, haircuts}`: every
        `CreditSupportList` at its first value, None where the terms state none - the independent
        amount excepted, which the engine reads as nothing - the balance currency the agreement's
        where unstated, and `haircuts` `{currency: Haircut_Posted}` off the cash rows of
        `Collateral_Assets`, a currency being the one asset a close's spots value.
        """
        if terms.get('Collateralized', 'False') != 'True':
            return None
        support = terms.get('Credit_Support_Amounts') or {}
        dials = {dial: cls._first(support.get(dial)) for dial in cls.DIALS}
        rows = (terms.get('Collateral_Assets') or {}).get('Cash_Collateral') or []
        return dict(dials, Independent_Amount=dials['Independent_Amount'] or 0.0,
                    Agreement_Currency=terms.get('Agreement_Currency'),
                    Balance_Currency=terms.get('Balance_Currency') or terms.get(
                        'Agreement_Currency'),
                    haircuts={row['Currency']: cls._percent(row.get('Haircut_Posted'))
                              for row in rows})

    @staticmethod
    def held(movements, until=None):
        """`{agreement: {kind: {asset: amount}}}` - the collateral and the margin held under every
        agreement, kept apart, in one pass over the `cash` fold's standing movements: each held kind
        summed per asset, as of the value date `until` where one is named. Signed from the bank's
        side, so what it posted is negative; an agreement nothing moved under holds `NOTHING`."""
        balances = {}
        for row in movements:
            if row['kind'] in HELD and (until is None or row['effective_time'][:10] <= until):
                kinds = balances.setdefault(row['subject'], {kind: {} for kind in HELD})
                kinds[row['kind']][row['asset']] = (kinds[row['kind']].get(row['asset'], 0.0)
                                                    + row['amount'])
        return balances

    @staticmethod
    def valued(holding, csa, fx):
        """What `holding` (`{asset: amount}`) is worth to the CSA in the currency `fx` values every
        asset in: each amount crossed, less its `Haircut_Posted` on whichever side it is held - the
        engine reads no other - and an asset held at nothing worth nothing, crossed or not."""
        return sum((amount * fx[asset] * (1.0 - csa['haircuts'].get(asset, 0.0))
                    for asset, amount in sorted(holding.items()) if amount), 0.0)

    @staticmethod
    def required(exposure, csa, fx):
        """The credit support the CSA asks for at `exposure`, in the currency `exposure` is in and
        `fx` values every currency in - the engine's `At`, term for term: positive is support the
        bank should hold, negative support it should have posted."""
        agreement = fx[csa['Agreement_Currency']]
        above, below = csa['Received_Threshold'] * agreement, csa['Posted_Threshold'] * agreement
        return (csa['Independent_Amount'] * agreement + (exposure - above) * (exposure > above)
                + (exposure - below) * (exposure < below))

    @classmethod
    def call(cls, exposure, holding, csa, fx):
        """The call at `exposure` with the collateral `holding` held: `{required, balance, call,
        direction, minimum_transfer}` in the currency `fx` values in. `call` is the signed amount
        the settlement moves - received positive - and nothing where the difference clears neither
        minimum, `direction` saying which the bank does, and `minimum_transfer` the minimum on the
        side the difference falls."""
        needed, balance = cls.required(exposure, csa, fx), cls.valued(holding, csa, fx)
        agreement, short = fx[csa['Agreement_Currency']], needed - balance
        received = csa['Minimum_Received'] * agreement
        posted = csa['Minimum_Posted'] * agreement
        direction = cls.CALL if short > received else cls.POST if -short > posted else None
        return {'required': needed, 'balance': balance, 'call': short if direction else 0.0,
                'direction': direction, 'minimum_transfer': received if short >= 0 else posted}

    @staticmethod
    def _first(table):
        """A `CreditSupportList` at its first value - a dict over its `[rating, amount]` rows, as
        the engine reads one - or None where it states none."""
        rows = table.get('.CreditSupportList') if isinstance(table, dict) else None
        return float(next(iter(dict(rows).values()))) if rows else None

    @staticmethod
    def _percent(value):
        """A haircut as the fraction the engine reads its `Percent` as, nothing where unstated."""
        return value['.Percent'] / 100.0 if isinstance(value, dict) else float(value or 0.0)


#: The projectors this module ships, by name. A reader picks one; nothing here is a default.
PROJECTORS = dict((projector.name, projector) for projector in (
    Positions(), Costs(), Lifecycle(), Blotter(), Markets(), Attestations(), Decisions(), Quotes(),
    Denials(), Entities(), Agreements(), Portfolios(), Cash(), Activity()))


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


def seed_at(log, projector, close_lsn, folder=None):
    """Mint `projector`'s seed at `close_lsn`, file it in `folder` (the home's `seeds/` where none
    is named), and answer it.

    A seed is minted only AT an official close, the one position where the desk already agrees what
    the day was, and it carries that close's own event hash and its state's own address, so a
    reader can tell whether the history it summarises is this one and whether the state is the
    one minted over it. WHEN one is minted and WHERE it is shared is the deployment's to decide - a
    folder every seat reads, or a seat's own for a day it wants to stand at. Written the way the
    store writes: scratch, fsync, rename.
    """
    frame = _close_frame(log, close_lsn, 'a seed at LSN {}'.format(close_lsn))
    state = fold(log, projector, lsn=close_lsn)
    seed = {'projector': projector.name, 'version': projector.version, 'lsn': close_lsn,
            'head': frame['event_hash'], 'state_hash': content_hash(state),
            'state': state}
    path = _seed_path(log, projector, close_lsn, folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.parent / (path.name + '.new')
    with scratch.open('wb') as handle:
        handle.write(canonical_bytes(seed))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(scratch), str(path))
    return seed


def read_seed(log, projector, close_lsn, folder=None):
    """The seed filed for `projector` at `close_lsn` in `folder`, or None where none is filed.

    Verified, never trusted: the file must parse, say which projector and version minted it, carry
    the event hash LSN `close_lsn` has IN THIS LOG, and hold state that hashes to its own stamp -
    so a seed carried in from another home, or edited under its own name, refuses by name rather
    than folding a fiction nothing else can detect. Every refusal here is cured by deleting the
    file and refolding.
    """
    path = _seed_path(log, projector, close_lsn, folder)
    if not path.is_file():
        others = sorted(other.name for other in _seeds(log, folder).glob(
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


def latest_seed(log, projector, folder=None, lsn=None):
    """The newest seed filed for `projector` in `folder` at or behind `lsn` - this log's head where
    none is named - verified as `read_seed` verifies one, or None where there is none to start from.

    A seed of another VERSION is not a candidate, since a folder several seats share may hold two
    releases' seeds while one rolls out; a seed of this version that does not verify refuses by
    name rather than being stepped over. A close past this log's head - a copy behind the one that
    minted the seed - is not one it can stand at yet.
    """
    at = log.head()[0] if lsn is None else lsn
    prefix = '{}-{}-'.format(projector.name, projector.version)
    closes = sorted((int(path.stem[len(prefix):])
                     for path in _seeds(log, folder).glob(prefix + '*.json')
                     if path.stem[len(prefix):].isdigit()), reverse=True)
    for close_lsn in closes:
        if close_lsn <= at:
            return read_seed(log, projector, close_lsn, folder)
    return None


def fold_from(log, projector, lsn=None, folder=None):
    """`fold` started at the newest verified seed in `folder` at or behind `lsn`, else at genesis.

    The same state either way - which the seed-equivalence gate holds - at the cost of the frames
    since the close rather than of the whole history, so a reader pays for its day.
    """
    return fold(log, projector, lsn=lsn, seed=latest_seed(log, projector, folder, lsn))


def close_on(log, date=None):
    """The LSN of the official close standing for the last day on or before `date` (`YYYY-MM-DD`)
    - the last there is where none is named - or None where there is no such close.

    A close is FOR its `date`, else the day it is true on, and the later day stands, ties by LSN -
    the markets fold's order - so a past day restated stands for that day and never for a later one.
    """
    found = None
    for frame in log.frames():
        if frame['event_type'] != SEED_EVENT:
            continue
        key = (log.open_body(frame).get('date') or as_of_key(frame)[0][:10], frame['lsn'])
        if (date is None or key[0] <= date) and (found is None or key > found):
            found = key
    return None if found is None else found[1]


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


def _position(state, instrument, agreement, portfolio, frame, moved=None):
    """The positions row under `(instrument, agreement, portfolio)`, opened at this frame where none
    stands. A row an amendment opens carries the book and counterparty of the row `moved` onto it."""
    under = state.setdefault(instrument, {}).setdefault(agreement, {})
    if portfolio not in under:
        under[portfolio] = {'book': frame['book'] if moved is None else moved['book'],
                            'counterparty': None if moved is None else moved['counterparty'],
                            'quantity': 0.0, 'clips': 0, 'amended_to': None,
                            'first_lsn': frame['lsn'], 'last_lsn': frame['lsn'], 'tickets': []}
    return under[portfolio]


def _ticketed(frame, body, clip):
    """One entry of a position's `tickets`: a clip under the ticket the fill or restrike `body`
    carries, who filed that and when it is true."""
    return dict(clip, ticket=body.get('ticket'), lsn=frame['lsn'], actor=frame['actor'],
                at=as_of_key(frame)[0])


def _cost_row(state, instrument, agreement, portfolio):
    """The costs row under `(instrument, agreement, portfolio)`, opened flat where none stands."""
    return state.setdefault(instrument, {}).setdefault(agreement, {}).setdefault(
        portfolio, {'quantity': 0.0, 'basis': 0.0, 'realised': 0.0, 'unpriced': 0,
                    'unpriced_reductions': 0})


def _plus(held, more):
    """A sum that stays UNKNOWN once either side is - a cost nobody priced is never a zero."""
    return None if held is None or more is None else held + more


def _seeds(log, folder):
    """The folder seeds are filed in: `folder`, else the home's own `seeds/`."""
    return log.home / SEEDS if folder is None else pathlib.Path(folder)


def _seed_path(log, projector, close_lsn, folder=None):
    """`<folder>/<projector>-<version>-<lsn>.json` - the name carries what the file must say."""
    return _seeds(log, folder) / '{}-{}-{}.json'.format(
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
