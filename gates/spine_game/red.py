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

"""What an adversary tries at this desk, and what the record answers.

Each objective is a scripted ATTEMPT with the answer it expects, played against the same hub the
day is played against: nothing is monkeypatched, the seats are the day's own, and an attempt that
must reach a platter reaches a COPY - the hub is the single writer, so an adversary with a second
one is not an objective but a divergence `/book/reconcile` names.

The answers come in three shapes and the difference is the point. A DENIAL is the writer refusing
an append and filing the refusal as a fact. A REFUSAL is a tier, a staleness window or a validator
turning an act away before any append, which mints nothing and is readable in the script and in the
absence it leaves. And some attempts are answered by the record simply CARRYING what happened: an
amendment is a fact at the head and the fold as at the position before it is unchanged, which is
the whole of what "it cannot be rewritten" means.

`play` answers one row per objective - what was tried, what was expected and what came back - and
every attempt is written into the day's script too, where the oracle holds it against the record.
"""
import base64
import hashlib
import shutil
import threading

from derivus_spine import (
    ChainBroken, CheckpointInvalid, MissingBlobRefusal, SpineLog, verify_home)
from derivus_spine.canon import canonical_bytes
from derivus_spine.identity import set_display_name
from derivus_spine.log import EVENT_VERSION, aad_bytes, event_hash, now_stamp, semantic_tuple
from derivus_spine.projections import PROJECTORS, fold
from derivus_spine.vocabulary import FIRM_CLASS, VALIDATE, cited_blobs
from derivus_mcp import server as binding
from mcp.server.mcpserver.exceptions import ToolError

from gates.spine_game import roles

#: A signature of the right width that is nobody's - what a forged checkpoint carries.
UNSIGNED = '00' * 64
#: The two display names one colluding subject wears. The record reads neither.
MASKS = ('A. Trader', 'The Second Seat')
#: The trade the adversary books so it has one of its own to restate.
RESTATED = {'Object': 'FixedCashflowDeal', 'Reference': 'CF-RED', 'Currency': 'ZAR',
            'Discount_Rate': 'ZAR', 'Amount': 250_000.0,
            'Payment_Date': {'.Timestamp': '2027-06-28'}}
#: What a curiosity submission names as the type it would have filed - which is nothing, so the
#: denial records the lane. Spelled here because the script must say what it asked for.
WHAT_IF = 'curiosity'


def play(table, out):
    """Every objective in turn, each answering `{objective, expected, answer}`.

    The first three share one quote: an objective is an ATTEMPT, and striking a fresh structure for
    each would be arithmetic rather than adversary.
    """
    return [objective(table, out) for objective in OBJECTIVES]


def approve_your_own_ticket(table, out):
    """The booker signs their own ticket, and four eyes is what refuses.

    The approval APPENDS - the trader is scoped to approve, and a seat signing a plan is a fact -
    and the booking still waits, because what four eyes asks is WHO signed rather than whether
    anybody did.
    """
    table.standing['red_quote'] = quote = roles.quote(table, roles.SALES, 'the adversary')
    binding.book_quote(quote, actor=roles.TRADER)
    signed = binding.approve_quote(quote, roles.TRADER)
    answer = binding.book_quote(quote, actor=roles.TRADER)
    table.did(roles.TRADER, 'approve_quote (own ticket)', recorded=signed['recorded']['lsn'],
              refused=answer.get('waits_on'))
    return _row('approve your own ticket',
                'the tier refuses - the booker and the approver are one seat - and no fill lands',
                answer.get('waits_on'))


def wear_a_second_name(table, out):
    """One subject under two display names, neither of which the record reads.

    The side table outside the log is mutable by design - it is where a name a person answers to
    lives - so moving it is not an attack on the record, and the refusal naming the SUBJECT is what
    says the record was never reading it.
    """
    for mask in MASKS:
        set_display_name(table.home, roles.TRADER, mask)
    answer = binding.book_quote(table.standing['red_quote'], actor=roles.TRADER)
    table.did(roles.TRADER, 'book_quote under a second display name',
              refused=answer.get('waits_on'))
    return _row('collude under two display names',
                'the refusal names the subject {!r} and neither mask'.format(roles.TRADER),
                answer.get('waits_on'))


def race_two_acceptances(table, out):
    """Two acceptances of one quote at once: one fill, at one LSN.

    The second seat signs first, so both threads meet a bookable quote - what orders them is the
    book's own lock, and the pending file's `booked` is what the loser reads.
    """
    quote = table.standing['red_quote']
    binding.approve_quote(quote, roles.RISK)
    answers = []

    def accept():
        answers.append(_refused(ToolError, lambda: binding.book_quote(quote, actor=roles.TRADER)))

    racing = [threading.Thread(target=accept) for _ in range(2)]
    for hand in racing:
        hand.start()
    for hand in racing:
        hand.join()
    fills = _fills(table, quote)
    table.did(roles.TRADER, 'book_quote x2 (raced)', fills=fills,
              refused=[found for found in answers if not found.startswith('NOT REFUSED')])
    return _row('race two acceptances of one quote',
                'one fill at one LSN, the loser refused and never a second clip',
                '{} fill(s) at {}'.format(len(fills), fills))


def replay_an_old_approval(table, out):
    """A signature over one plan held against another.

    A ticket is the plan an acceptance leaves the book at, so a re-quote is a new hash by
    construction and a standing approval reaches the quote it was filed over and no other.
    """
    quote = roles.quote(table, roles.SALES, 'the adversary, again')
    answer = binding.book_quote(quote, actor=roles.TRADER)
    table.did(roles.TRADER, 'book_quote on another ticket\'s approval',
              refused=answer.get('waits_on') or answer.get('refused'))
    return _row('replay an old approval against an amended ticket',
                'the new ticket carries no verdict, so the booking waits and no fill lands',
                answer.get('waits_on') or answer.get('refused'))


def backdate_an_amendment(table, out):
    """Book a trade, then restate its terms and try to make the record read as though they always
    were.

    There is nothing to attempt: economics are never edited, so an amendment is a NEW instrument
    hash linked to the old one, the attempt lands as a fact at the head, and the fold as at the
    position before it answers exactly what it answered.
    """
    booked = binding.book_deal(RESTATED, parent_reference=roles.CLIENT, quantity=-250_000.0,
                               execution_reference='EXEC-RED', actor=roles.TRADER)
    before = table.read(lambda log: PROJECTORS['positions'].rows(
        fold(log, PROJECTORS['positions'])))
    amended = binding.amend_deal(_path_of(RESTATED['Reference']), {'Amount': 1.0},
                                 reference=RESTATED['Reference'], actor=roles.TRADER)
    at = amended['recorded']['lsn']
    behind = table.read(lambda log: PROJECTORS['positions'].rows(
        fold(log, PROJECTORS['positions'], lsn=at - 1)))
    table.did(roles.TRADER, 'amend_deal (restating a booked trade)', booked=booked['recorded'],
              recorded=at)
    return _row('backdate an amendment to hide a loss',
                'the amendment is a fact at the head and the fold behind it is unchanged',
                'as-at LSN {} unchanged: {}'.format(
                    at - 1, canonical_bytes(behind) == canonical_bytes(before)))


def delete_the_evidence(table, out):
    """Take a blob off a copy's disk. There is no verb for it, so it takes a file system.

    The copy refuses BY NAME at the citation it can no longer resolve - referential closure is a
    property of the whole history - and the hub's own home is untouched, which is what says the
    record survives a reader losing bytes.
    """
    home = _copy(table, out, 'red-shredded')
    log = SpineLog(home)
    try:
        taken = _cited(log)
    finally:
        log.close()
    (home / 'blobs' / taken[:2] / taken[2:4] / taken).unlink()
    return _row('delete the evidence', 'the copy refuses by name and the hub is intact',
                '{} | hub at LSN {}'.format(_refused(MissingBlobRefusal, lambda: verify_home(home)),
                                            verify_home(table.home)['head_lsn']))


def forge_a_checkpoint(table, out):
    """Sign a head with a signature that is nobody's.

    A replica's `accept` asks four things and a signature is not one of them - authenticity is the
    hub's question, answered where the fact was made - so the frame CHAINS and the verification
    refuses it, which is the boundary the page states: a copy proves authenticity up to its last
    pulled checkpoint and the frames past it are chained and unsigned.
    """
    home = _copy(table, out, 'red-forged')
    log = SpineLog(home)
    lsn, head = log.head()
    try:
        log.accept(forged(log, 'checkpoint',
                           {'event_hash': head, 'lsn': lsn, 'signature': UNSIGNED}, roles.CONTROL))
    finally:
        log.close()
    return _row('forge a checkpoint', 'the copy chains it and the verification refuses',
                _refused(CheckpointInvalid, lambda: verify_home(home)))


def tamper_with_a_copy(table, out):
    """Edit one byte of one line on a copy's platter.

    Every hash is recomputed from the bytes, so the line parts company with the chain at the
    position it was altered at, and the refusal names that position and the remedy.
    """
    home = _copy(table, out, 'red-tampered')
    segment = sorted((home / 'log').glob('segment-*.jsonl'))[0]
    lines = segment.read_bytes().split(b'\n')
    lines[1] = lines[1].replace(b'"record_time":"2', b'"record_time":"3', 1)
    segment.write_bytes(b'\n'.join(lines))
    return _row('tamper with a copy', 'the copy is ChainBroken at the line and the hub is intact',
                '{} | hub at LSN {}'.format(
                    _refused(ChainBroken, lambda: verify_home(home, entitled=False)),
                    verify_home(table.home)['head_lsn']))


def a_strangers_what_if(table, out):
    """A seat nobody scoped asks this box for arithmetic.

    Refused at the QUEUE, before a Monte Carlo is paid for, with the denial landed as a fact in the
    writer's own voice - which is what makes who asked a ROW rather than a log line.
    """
    answer = _refused(ToolError, lambda: roles.quote(table, roles.STRANGER, 'nobody'))
    table.did(roles.STRANGER, 'solve_structure {}'.format(roles.STRUCTURE),
              denied={'subject': roles.STRANGER, 'verb': VALIDATE, 'book': roles.DESK_BOOK,
                      'attempted_type': WHAT_IF})
    return _row('a stranger\'s what-if', 'refused at the queue with the denial recorded', answer)


#: The objectives, in the order they are played.
OBJECTIVES = (approve_your_own_ticket, wear_a_second_name, race_two_acceptances,
              replay_an_old_approval, backdate_an_amendment, delete_the_evidence,
              forge_a_checkpoint, tamper_with_a_copy, a_strangers_what_if)


# ------------------------------------------------------------------------------------------------
# The pieces the objectives above are made of.

def _row(objective, expected, answer):
    """One objective's row: what was tried, what the record was expected to answer, and what it
    did."""
    return {'objective': objective, 'expected': expected, 'answer': answer}


def _refused(refusal, act):
    """`act()`'s refusal as its own sentence, or what it answered instead of refusing."""
    try:
        return 'NOT REFUSED: {!r}'.format(act())
    except refusal as refused:
        return str(refused)


def _path_of(reference):
    """The positional path the live book carries `reference` at - deals are addressed by position,
    so a booking's own path is read back rather than assumed."""
    return next(deal['deal_path'] for deal in binding.read_book()['deals']
                if deal['reference'] == reference)


def _fills(table, quote):
    """The LSNs of every fill booked against `quote` - one is the record's answer to a race."""
    def read(log):
        return [frame['lsn'] for frame in log.frames() if frame['event_type'] == 'fill'
                and log.open_body(frame)['execution_reference'] == quote]

    return table.read(read)


def _cited(log):
    """One blob the chain CITES, other than a policy's - what taking a file off a platter has to
    reach for a verification to have anything to say about it.

    A policy is skipped because the published verifying key is one, and losing THAT is answered by
    the checkpoint ladder before referential closure is ever asked.
    """
    for frame in log.frames():
        if frame['event_type'] == 'policy_declared':
            continue
        for _, digest in cited_blobs(frame['event_type'], log.open_body(frame)):
            return digest
    raise AssertionError('this record cites no blob, so there is no evidence to take')


def _copy(table, out, name):
    """A copy of the hub's home on disk, for an objective that must reach a platter.

    A file copy rather than a pull: what is under test is what a verification says about bytes that
    moved, and the hub is the single writer either way.
    """
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
