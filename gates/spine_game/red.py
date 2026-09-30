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

"""What an adversary tries at this bank, and what the record answers.

Each objective is a scripted ATTEMPT with the answer it expects, played by the bank's own seats
against the same hub the days were played against: nothing is monkeypatched, and an attempt that
must reach a platter reaches a COPY - the hub is the single writer, so an adversary with a second
one is not an objective but a divergence `/book/reconcile` names.

The answers come in three shapes and the difference is the point. A DENIAL is the writer refusing
an append and filing the refusal as a fact, so the script says which four fields it expects. A
REFUSAL is a validator, the tree or the seam turning an act away before any append, which mints
nothing and is readable in the script and in the absence it leaves. And some attempts are answered
by the record simply CARRYING what happened: a booker's own approval is a fact four eyes reads
through under any name the booker wears, an approval sent again is the fact it repeats, and a
restrike is a ticket at the head that reads pending until another seat signs.

`play` answers one row per objective - what was tried, what was expected and what came back - and
every attempt is written into the script too, where the oracle holds it against the record.
"""
import base64
import hashlib
import json
import shutil
import threading

from derivus_spine import (
    ChainBroken, CheckpointInvalid, MissingBlobRefusal, SpineLog, verify_home)
from derivus_spine.canon import canonical_bytes
from derivus_spine.log import EVENT_VERSION, aad_bytes, event_hash, now_stamp, semantic_tuple
from derivus_spine.projections import PROJECTORS, fold
from derivus_spine.verbs import SETTLED
from derivus_spine.vocabulary import (
    ADMIN, APPROVE, BOOK, FIRM_CLASS, SETTLE, VALIDATE, WRITER, cited_blobs)
from derivus_mcp import server as binding

from gates.spine_game import roles
from gates.spine_game.roles import node, seat

#: A signature of the right width that is nobody's - what a forged checkpoint carries.
UNSIGNED = '00' * 64
#: The two display names one colluding subject wears. The record reads neither.
MASKS = ('A. Trader', 'The Second Seat')
#: What a curiosity submission names as the type it would have filed - which is nothing, so the
#: denial records the lane. Spelled here because the script must say what it asked for.
WHAT_IF = 'curiosity'
#: The desks the adversary works: FX's trader and approver, rates' swaps book.
FX, SWAPS = seat('FX', 'trader'), node('RATES', roles.BOOKS['RATES'])


def play(table, out):
    """Every objective in turn, each answering `{objective, expected, answer}`."""
    return [objective(table, out) for objective in OBJECTIVES]


def the_back_office_books_a_trade(table, out):
    """Settlements, which settles everything and books nothing, books a forward: the writer
    refuses the fill at the desk node it names and files the refusal."""
    return _denied(table, 'a settlements seat books a trade', roles.SETTLEMENTS, BOOK,
                   node('FX'), 'fill', lambda: roles.book(
                       table, 'FWD-1', actor=roles.SETTLEMENTS,
                       execution_reference='EXEC-RED-SETTLEMENTS'))


def a_trader_settles_its_own_trade(table, out):
    """The FX trader files its own forward's payment as settled: `settle` is the back office's,
    so the status transition is refused at the book."""
    key = next(row['key'] for row in binding.book_diary(actor=FX)['rows'] if row['key'])
    return _denied(table, 'a trader settles its own trade', FX, SETTLE, roles.BANK,
                   'status_transition', lambda: binding.file_status(key, SETTLED, actor=FX))


def an_approver_signs_beside_its_node(table, out):
    """The FX approver signs the rates desk's rejected swap: a verdict is filed where its ticket
    books, read off the record, so it is refused at the swaps node and not at the FX one."""
    ticket = next(act['ticket'] for act in table.acts if act['act'] == 'book_deal SWAP-1')
    return _denied(table, 'an approver signs outside its node', seat('FX', 'approver'), APPROVE,
                   SWAPS, 'approval',
                   lambda: binding.approve_ticket(ticket, seat('FX', 'approver')))


def the_fx_seat_books_into_the_rates_path(table, out):
    """The FX trader books a trade into the rates desk's node: refused at that node."""
    return _denied(table, 'the FX seat books into the rates path', FX, BOOK, SWAPS, 'fill',
                   lambda: roles.book(table, 'FWD-1', actor=FX, portfolio=SWAPS,
                                      agreement='CSA-B', execution_reference='EXEC-RED-PATH'))


def a_head_grants_beyond_its_node(table, out):
    """The FX head declares the document again with a trader seated on the rates desk, then with
    itself `admin` over `*`: a node admin moves nothing beyond its node, so both are refused and
    the one denial they share is filed."""
    head, standing = seat('FX', 'head'), roles.staffed(roles.DESKS)
    answers = []
    for name, extra in (('beside', (FX, (BOOK,), SWAPS)),
                        ('firm', (head, (ADMIN,), '*'))):
        path = out / 'red-grant-{}.json'.format(name)
        path.write_text(json.dumps(roles.grants(*standing + [extra]), indent=2), newline='\n')
        code, said = roles.spine('grant', '--home', table.home, '--file', path, '--actor', head)
        answers.append(said if code else 'NOT REFUSED: {}'.format(said))
        table.did(head, 'DV_Spine grant ({})'.format(name), denied={
            'subject': head, 'verb': ADMIN, 'book': '*', 'attempted_type': 'policy_declared'})
    return _row('a node admin grants beyond its node, and itself the firm',
                'both refused and one denial filed', ' | '.join(answers))


def race_two_acceptances(table, out):
    """Two acceptances of one quote at once: one fill, at one LSN.

    Both threads meet a bookable quote - what orders them is the book's own lock, and the loser
    meets the book the winner moved, or the pending file's `booked`.
    """
    table.standing['red_quote'] = quote = roles.quote(table, seat('FX', 'sales'),
                                                      'the adversary, racing')
    answers = []

    def accept():
        answers.append(roles.refused(lambda: binding.book_quote(quote, actor=FX)))

    racing = [threading.Thread(target=accept) for _ in range(2)]
    for hand in racing:
        hand.start()
    for hand in racing:
        hand.join()
    fills = _fills(table, quote)
    table.did(FX, 'book_quote x2 (raced)', fills=fills,
              refused=[found for found in answers if not found.startswith('NOT REFUSED')])
    return _row('race two acceptances of one quote',
                'one fill at one LSN, the loser refused and never a second clip',
                '{} fill(s) at {}'.format(len(fills), fills))


def a_trader_approves_its_own_ticket(table, out):
    """The FX trader signs the ticket it just booked. It holds `approve` at its node, so the
    approval APPENDS - a seat signing a plan is a fact - and the trade still reads pending,
    because four eyes asks WHO signed rather than whether anybody did."""
    signed = binding.approve_quote(table.standing['red_quote'], FX)
    table.standing['red_ticket'] = signed['ticket']
    table.did(FX, 'approve_quote (own ticket)', recorded=signed['recorded']['lsn'],
              status=signed['status'])
    return _row('a trader approves its own ticket',
                'the booker and the approver are one seat, so the trade stays pending',
                'the ticket reads {}'.format(signed['status']))


def collude_under_two_names(table, out):
    """The FX trader wears two display names and signs its waiting ticket again under the second.
    The side table outside the log is mutable by design - it is where the name a person answers to
    lives - and the record reads neither: a verdict stands under its SUBJECT, so the second
    signature is the first fact again and four eyes still counts one signer."""
    for mask in MASKS:
        roles.spine('name', '--home', table.home, FX, '--display', mask)
    signed = binding.approve_quote(table.standing['red_quote'], FX)
    signers = table.read(lambda log: sorted(set(
        frame['actor'] for frame in log.frames() if frame['event_type'] == 'approval'
        and log.open_body(frame)['plan_hash'] == signed['ticket'])))
    table.did(FX, 'approve_quote under a second display name',
              recorded=signed['recorded']['lsn'], status=signed['status'])
    return _row('collude under two display names',
                'the verdicts name the subject and neither mask, so the trade stays pending',
                'the ticket reads {} under {}'.format(signed['status'], signers))


def replay_an_old_approval(table, out):
    """The FX approver's signature over the day's collar, sent again while the adversary's fill
    waits on a ticket of its own: an approval sent again is the fact it repeats, so it lands on the
    LSN it already has, and a ticket is one trade's - so the old approval reaches nothing new and
    the waiting fill still waits."""
    approver = seat('FX', 'approver')
    old = next(act for act in table.acts
               if act['seat'] == approver and act['act'] == 'approve_ticket')
    again = binding.approve_ticket(old['ticket'], approver)
    waiting = [row['key'] for row in table.work(approver, 'pending')]
    table.did(approver, 'approve_ticket (the collar\'s approval, again)',
              recorded=again['recorded']['lsn'], stood=old['recorded'], ticket=old['ticket'],
              waiting=waiting)
    return _row('replay an old approval against a new ticket',
                'the approval lands where it stood and the new ticket still waits',
                'the approval of {} lands at LSN {} where it stood at {}, and ticket {} still '
                'waits: {}'.format(old['ticket'][:12], again['recorded']['lsn'], old['recorded'],
                                   table.standing['red_ticket'][:12],
                                   table.standing['red_ticket'] in waiting))


def book_a_deleted_trade_a_hundredfold(table, out):
    """The rates trader deletes its deposit from the book file - a delete records nothing - and
    books it again a hundred times over under its own execution reference: a trade is taken once,
    so the second booking is refused naming what moved, and the trade as it was is taken back."""
    trader = seat('RATES', 'trader')
    deleted = binding.delete_deal(roles.path_of('DEP-1'), reference='DEP-1')
    table.did(trader, 'delete_deal DEP-1', deleted=deleted.get('deleted'))
    answer = roles.refused(lambda: roles.book(table, 'DEP-1', quantity=100.0))
    table.did(trader, 'book_deal DEP-1 x100 under its own reference', refused=answer)
    back = roles.book(table, 'DEP-1')
    return _row('a deleted trade booked again a hundred times its size',
                'refused by name, and the trade as it was taken back with nothing filed',
                '{} | taken back: written {}, booked at LSN {}'.format(
                    answer, back['written'], (back.get('booked') or {}).get('lsn')))


def restrike_to_hide_a_loss(table, out):
    """The commodities trader restrikes its losing forward and states no notional, hoping the
    hub signs it: a cap reads what is stated, so no automatic tier admits it and the restrike
    lands PENDING, unsigned - a fact at the head, with the fold behind it unchanged."""
    trader = seat('COMMODITIES', 'trader')
    before = table.read(lambda log: PROJECTORS['positions'].rows(
        fold(log, PROJECTORS['positions'])))
    amended = binding.amend_deal(roles.path_of('METAL-1'), {'Units': 300.0},
                                 reference='METAL-1', actor=trader)
    at = amended['recorded']['lsn']
    behind = table.read(lambda log: PROJECTORS['positions'].rows(
        fold(log, PROJECTORS['positions'], lsn=at - 1)))
    table.did(trader, 'amend_deal METAL-1 (no notional)', recorded=at, ticket=amended['ticket'],
              waits_on=amended.get('waits_on'))
    return _row('restrike a losing trade to hide it',
                'the restrike lands pending on a ticket of its own, the fold behind it unchanged',
                'pending: {} | as-at LSN {} unchanged: {}'.format(
                    amended.get('waits_on'), at - 1,
                    canonical_bytes(behind) == canonical_bytes(before)))


def a_strangers_what_if(table, out):
    """A seat nobody scoped asks this box for arithmetic: refused at the QUEUE, before a solve is
    paid for, with the denial landed as a fact in the writer's own voice."""
    return _denied(table, 'a stranger\'s what-if', roles.STRANGER, VALIDATE, roles.BANK,
                   WHAT_IF, lambda: roles.quote(table, roles.STRANGER, 'nobody'))


def the_writers_name_on_a_request(table, out):
    """A request names the hub's own reserved voice as its seat: the seam refuses the name
    before anything is asked of the record, so nothing is minted."""
    answer = roles.refused(lambda: roles.book(table, 'FWD-1', actor=WRITER,
                                              execution_reference='EXEC-RED-WRITER'))
    table.did(WRITER, 'book_deal under the writer\'s name', refused=answer)
    return _row('the reserved actor named on a request', 'refused by name, nothing appended',
                answer)


def delete_the_evidence(table, out):
    """Take a blob the chain cites off a copy's disk. There is no verb for it, so it takes a file
    system: the copy refuses BY NAME at the citation it can no longer resolve, and the hub's own
    home is untouched - the record survives a reader losing bytes."""
    home = _copy(table, out, 'red-shredded')
    log = SpineLog(home)
    try:
        taken = next(digest for frame in log.frames() if frame['event_type'] == 'fill'
                     for _, digest in cited_blobs('fill', log.open_body(frame)))
    finally:
        log.close()
    (home / 'blobs' / taken[:2] / taken[2:4] / taken).unlink()
    return _row('delete the evidence', 'the copy refuses by name and the hub is intact',
                '{} | hub at LSN {}'.format(
                    roles.refused(lambda: verify_home(home), MissingBlobRefusal),
                    verify_home(table.home)['head_lsn']))


def forge_a_checkpoint(table, out):
    """Sign a head with a signature that is nobody's, on a copy.

    A replica's `accept` asks four things and a signature is not one of them - authenticity is the
    hub's question, answered where the fact was made - so the frame CHAINS and the verification
    refuses it: a copy proves authenticity up to its last pulled checkpoint.
    """
    home = _copy(table, out, 'red-forged')
    log = SpineLog(home)
    lsn, head = log.head()
    try:
        log.accept(forged(log, 'checkpoint',
                           {'event_hash': head, 'lsn': lsn, 'signature': UNSIGNED},
                           roles.FOUNDER))
    finally:
        log.close()
    return _row('forge a checkpoint', 'the copy chains it and the verification refuses',
                roles.refused(lambda: verify_home(home), CheckpointInvalid))


def tamper_with_a_copy(table, out):
    """Edit one byte of one line on a copy's platter: the line parts company with the chain at the
    position it was altered at, and the refusal names that position and the remedy."""
    home = _copy(table, out, 'red-tampered')
    segment = sorted((home / 'log').glob('segment-*.jsonl'))[0]
    lines = segment.read_bytes().split(b'\n')
    lines[1] = lines[1].replace(b'"record_time":"2', b'"record_time":"3', 1)
    segment.write_bytes(b'\n'.join(lines))
    return _row('tamper with a copy', 'the copy is ChainBroken at the line and the hub is intact',
                '{} | hub at LSN {}'.format(
                    roles.refused(lambda: verify_home(home, entitled=False), ChainBroken),
                    verify_home(table.home)['head_lsn']))


#: The objectives, in the order they are played - the race's pending fill is the one its trader
#: then signs, under one name and then another, and the one an old approval is sent against.
OBJECTIVES = (the_back_office_books_a_trade, a_trader_settles_its_own_trade,
              an_approver_signs_beside_its_node, the_fx_seat_books_into_the_rates_path,
              a_head_grants_beyond_its_node, race_two_acceptances,
              a_trader_approves_its_own_ticket, collude_under_two_names, replay_an_old_approval,
              book_a_deleted_trade_a_hundredfold, restrike_to_hide_a_loss, a_strangers_what_if,
              the_writers_name_on_a_request, delete_the_evidence, forge_a_checkpoint,
              tamper_with_a_copy)


# ------------------------------------------------------------------------------------------------
# The pieces the objectives above are made of.

def _row(objective, expected, answer):
    """One objective's row: what was tried, what the record was expected to answer, and what it
    did."""
    return {'objective': objective, 'expected': expected, 'answer': answer}


def _denied(table, objective, subject, verb, book, attempted_type, act):
    """An attempt the writer refuses with a denial: played, written into the script as the denial
    it expects, and answered with the record's own sentence."""
    answer = roles.refused(act)
    table.did(subject, objective, denied={'subject': subject, 'verb': verb, 'book': book,
                                          'attempted_type': attempted_type})
    return _row(objective, 'refused with a denial of {} over {!r} filed'.format(verb, book),
                answer)


def _fills(table, quote):
    """The LSNs of every fill booked against `quote` - one is the record's answer to a race."""
    def read(log):
        return [frame['lsn'] for frame in log.frames() if frame['event_type'] == 'fill'
                and log.open_body(frame)['execution_reference'] == quote]

    return table.read(read)


def _copy(table, out, name):
    """A copy of the hub's home on disk, for an objective that must reach a platter: what is under
    test is what a verification says about bytes that moved, and the hub is the single writer
    either way."""
    home = out / name
    shutil.copytree(str(table.home), str(home), dirs_exist_ok=True)
    return home


def forged(log, event_type, body, actor, book=None, tag=None):
    """A frame sealed, tagged and chained under this home's own keys onto its head.

    Everything a writer here would produce except the one thing under test, so a refusal is about
    what was forged rather than about the forgery being clumsy. `tag` takes an idempotency tag
    already on the platter, which is the only way to ask whether a copy coalesces: the hub's writer
    folds a repeat onto the LSN it has, and a replica takes the order it was given.
    """
    semantic = canonical_bytes(semantic_tuple(event_type, body, actor, book, None))
    envelope = {'actor': actor, 'book': book, 'effective_time': None,
                'entitlement_class': FIRM_CLASS, 'event_type': event_type,
                'event_version': EVENT_VERSION,
                'idempotency_tag': tag or log.keys.blind_tag(semantic),
                'prev_hash': log.head()[1], 'record_time': now_stamp()}
    sealed = log.keys.seal(
        canonical_bytes({'content_hash': hashlib.sha256(semantic).hexdigest(), 'payload': body}),
        aad_bytes(envelope))
    return dict(envelope, body=base64.b64encode(sealed).decode('ascii'), lsn=log.head()[0] + 1,
                event_hash=event_hash(hashlib.sha256(sealed).hexdigest(),
                                      envelope['idempotency_tag'], envelope['prev_hash'],
                                      envelope['record_time']))
