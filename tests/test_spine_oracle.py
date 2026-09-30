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

"""The oracle's fourteen, each made to FAIL, and each named where it did.

A gate that only ever sees a green report is a gate that cannot tell an invariant from a constant,
so every one of these doctors a home until the question it asks has the wrong answer and then reads
the report back. The doctoring is DATA every time: a second home that is not a copy, a frame put on
a platter by hand, a policy declared and a seat that is not in it, a fill booked against a ticket
nobody signed, a close body no validating writer would have taken, a script asking for a run the
record never attested, a settlement against a key the book has no row for, a line removed, one
fact written twice under one tag, a verdict naming a node its ticket does not book at, a P&L that
does not sum or that this record did not make, money moved under paper nobody declared or past what
the diary determines, a close nobody marked, and a call the formula does not give.

The frames put on a platter by hand go through `SpineLog.accept`, which is what a replica writes
with: it asks the chain and never the vocabulary, never scope and never the tag, so it is exactly
the door a copy of a newer hub - or somebody with a file handle - comes through. That is the door
the oracle stands behind.
"""
import hashlib
import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from derivus_spine import (
    CapabilityDenied, ChainBroken, SpineLog, oracle, policy, verify_home)
from derivus_spine.canon import canonical_bytes, content_hash
from derivus_spine.capability import CAPABILITIES_POLICY, canonical_document
from derivus_spine.projections import CSA, PROJECTORS, fold
from derivus_spine.verbs import (
    STANDING, amendment, approve, book, declare_agreement, declare_close, declare_entity,
    declare_portfolio, file_quote, transition)
from derivus_spine.vocabulary import cited_blobs

from gates.spine_game.red import forged

from test_spine import (
    ACTOR, BOOK, INSTRUMENT, OTHER, WHEN, fill, seeded, segments, synthetic_book)
from test_spine_doorbell import STRANGER, granted
from test_spine_imports import spine

#: A plan hash and a ticket the gates below sign, book against and collide on. Sixty-four hex, as
#: every address in the record is.
TICKET = 'a' * 64
PLAN = 'b' * 64
#: The workflow that makes an unsigned fill a failure: with no tiers policy in force the desk has
#: declared no second pair of eyes, and an unsigned booking is what it asked for.
TIERS = {'tiers': [{'name': 'desk', 'four_eyes': True}]}


def copied(home, out, name):
    """A file copy of `home` a forged frame can be accepted onto - what a replica looks like on
    disk, keys included, so the copy can open its own bodies."""
    import shutil

    shutil.copytree(str(home), str(out / name))
    return out / name


def planted(home, out, name, event_type, body, actor=ACTOR, book_name=None, tag=None):
    """A copy of `home` carrying one frame this writer would not have written, and its path."""
    where = copied(home, out, name)
    log = SpineLog(where)
    try:
        log.accept(forged(log, event_type, body, actor, book=book_name, tag=tag))
    finally:
        log.close()
    return where


def traded(tmp_path, name, tickets, restrike=(), workflow=True):
    """A home - under a workflow where `workflow` - carrying one fill per ticket in `tickets`,
    then a restrike of the terms they hold per entry of `restrike`, each carrying that ticket.

    The spine's own verbs and no engine anywhere near them - what a booking puts on the record is
    plain data, so the shape invariant four reads is buildable without a pricer.
    """
    home = seeded(tmp_path, name, clips=())
    log = SpineLog(home)
    try:
        if workflow:
            policy.declare(log, ACTOR, policy.TIERS_POLICY, TIERS)
        terms = b'{"Reference":"CF1"}'
        for position, ticket in enumerate(tickets):
            book(log, ACTOR, terms, 1.0, 'LEI-549300', 'CSA-0007', 'EXEC-{}'.format(position),
                 book=BOOK, ticket=ticket)
        for position, ticket in enumerate(restrike):
            became = '{{"Reference":"CF1","Amount":{}}}'.format(position + 2).encode()
            body = amendment(log, terms, became)
            log.append('amendment', body if ticket is None else dict(body, ticket=ticket),
                       actor=ACTOR, book=BOOK)
            terms = became
    finally:
        log.close()
    return home


def script(*acts):
    """The day's script as the oracle takes it: a document with an `acts` list and nothing else."""
    return {'acts': list(acts)}


def test_the_report_answers_fourteen_and_names_what_it_could_not_assess(tmp_path):
    """THE GREEN READING, so the reds below are differences rather than the only thing seen. The
    design's synthetic book holds every fact type this vocabulary has, including a close restated
    by a second close, and every invariant the record alone can answer holds over it.

    The six a home with no script, no P&L, no workflow designating one and no second copy cannot
    put are NULLS carrying a sentence, not passes: a question nobody asked is not a question that
    held.

    Killing mutation: `unasked` answering `held: true`, which turns a report about a record nobody
    compared into a clean bill.
    """
    home = synthetic_book(tmp_path)[:2][0]
    answers = oracle.report(home, diary_keys=[INSTRUMENT])

    assert sorted(answers) == sorted(oracle.INVARIANTS)
    assert oracle.failed(answers) == [], answers
    assert sorted(name for name, found in answers.items() if found['held'] is None) == sorted(
        ('copies_agree', 'every_numbers_replay_tuple', 'every_refusal_is_a_denial',
         'the_pnl_is_additive', 'every_close_marked', 'the_call_is_the_formula'))
    assert '2 close(s) over 1 market(s)' in answers['closes_superseded_never_edited']['evidence'][0]
    assert answers['duplicates_coalesce']['evidence'] == ['22 tag(s) over 22 frame(s)']


def test_two_homes_that_are_not_copies_of_one_history_are_named(tmp_path):
    """COPIES AGREE. Two homes minted separately are two histories, whatever they hold: the
    comparison is taken at the SHALLOWER head, so a copy that is a PREFIX of the hub agrees - a
    replica behind its hub has not pulled yet rather than disagreed - and two mints do not.

    Killing mutation: the comparison taken at the deeper of the two heads, which asks the copy for
    a position it does not hold and calls every follower mid-pull a divergence.
    """
    answers = oracle.report(seeded(tmp_path, 'one'), against=seeded(tmp_path, 'two'))

    assert oracle.failed(answers) == ['copies_agree'], answers
    assert 'copies of different histories' in answers['copies_agree']['evidence'][0]

    behind = copied(seeded(tmp_path, 'three'), tmp_path, 'four')
    agreed = oracle.report(tmp_path / 'three', against=behind)
    assert agreed['copies_agree']['held'] is True, agreed['copies_agree']

    log = SpineLog(tmp_path / 'three')
    try:
        log.append('fill', fill('EXEC-AHEAD'), actor=ACTOR, book=BOOK)
    finally:
        log.close()
    ahead = oracle.report(tmp_path / 'three', against=behind)
    assert ahead['copies_agree']['held'] is True, ahead['copies_agree']
    assert 'at LSN {}'.format(SpineLog(behind).head()[0]) in ahead['copies_agree']['evidence'][0]


def test_a_frame_under_a_seat_the_document_never_scoped_is_named(tmp_path):
    """NOTHING OUTSIDE ITS SEAT. A replica's `accept` does not authorize - scope is the hub's
    question, answered where the fact was made - so a frame under a stranger chains perfectly well
    and the oracle re-runs the decision the writer would have made.

    Killing mutation: the capability state folded at the head rather than at the frame's own
    position, which judges every append under whatever document is in force today.
    """
    home = seeded(tmp_path, 'scoped', clips=())
    log = SpineLog(home)
    try:
        # booked BEFORE any document, so it is a legal append that the document in force today
        # would refuse - which is what says the state is read at the frame and not at the head
        log.append('fill', fill('EXEC-EARLY'), actor='subject-desk-two', book=BOOK)
        granted(log, ACTOR)
    finally:
        log.close()
    assert oracle.report(home)['nothing_outside_its_seat']['held'] is True, \
        'a fact filed before the document was judged under it'

    answers = oracle.report(planted(home, tmp_path, 'stranger', 'fill', fill('EXEC-FORGED'),
                                    actor=STRANGER, book_name=BOOK))
    assert oracle.failed(answers) == ['nothing_outside_its_seat'], answers
    said = answers['nothing_outside_its_seat']['evidence'][0]
    assert STRANGER in said and 'no book scope' in said and 'the writer would have refused' in said


def test_a_node_seat_and_a_node_admin_are_judged_where_the_writer_judged_them(tmp_path):
    """NOTHING OUTSIDE ITS SCOPE: A FILL AT A NODE ITS ACTOR HOLDS NO GRANT AT. The writer judges a
    fill at the portfolio its body names and a node admin's capabilities declaration by what it
    moves beyond its node, so the oracle opens the same bodies and asks the same two functions: a
    desk seat's fill into its own node of a book it holds nothing over, and a node admin's
    declaration moving only rows under its node, both hold - and the same fill forged into the
    node beside it is named by the node half of the writer's decision, the book half holding.

    Killing mutations: the oracle judging at the envelope's book, which calls the node fill a
    forgery; the oracle not re-running `declarable`, which calls the node admin's legal
    declaration one; and a node-scoped frame judged by the book half, which names it twice.
    """
    desk, node = 'subject-desk-two', BOOK + '/FX'
    home = seeded(tmp_path, 'nodes', clips=())
    log = SpineLog(home)
    try:
        granted(log, ACTOR)
        rows = [{'subject': desk, 'verb': verb, 'book': node} for verb in ('admin', 'book')]
        rows += [{'subject': ACTOR, 'verb': 'admin', 'book': '*'}]
        for grants in (rows, rows + [{'subject': 'subject-desk-three', 'verb': 'book',
                                      'book': node + '/Options'}]):
            blob = log.store.put(canonical_document({'grants': grants, 'read': []}))
            log.append('policy_declared', {'policy': CAPABILITIES_POLICY, 'blob': blob},
                       actor=ACTOR if grants is rows else desk, blob_refs=(blob,))
        log.append('fill', dict(fill('EXEC-NODE'), portfolio=node), actor=desk, book=BOOK)
        with pytest.raises(CapabilityDenied):
            log.append('fill', dict(fill('EXEC-BESIDE'), portfolio=BOOK + '/Rates'), actor=desk,
                       book=BOOK)
    finally:
        log.close()
    held = oracle.report(home)
    assert oracle.failed(held) == [], held
    assert held['nothing_outside_its_scope']['evidence'] == [
        '2 frame(s) judged at a node, 0 node(s) declared']

    answers = oracle.report(planted(
        home, tmp_path, 'beside', 'fill', dict(fill('EXEC-FORGED'), portfolio=BOOK + '/Rates'),
        actor=desk, book_name=BOOK))
    assert oracle.failed(answers) == ['nothing_outside_its_scope'], answers
    assert "over '{}/Rates'".format(BOOK) in answers['nothing_outside_its_scope']['evidence'][0]


def test_the_writers_own_voice_is_forged_under_a_seat_and_a_seat_is_forged_under_it(tmp_path):
    """THE WRITER'S OWN VOICE, read off the envelope. A run the hub attests is the writer's alone,
    so one standing under a seat is that voice forged; and the writer's name on a type its voice
    does not say is a seat wearing it.

    Killing mutation: the voice read off `capability_denied` alone, which leaves an attestation
    filed by a desk seat - numbers the hub never ran - reading as a fact of the record.
    """
    home = seeded(tmp_path, 'voice', clips=())
    log = SpineLog(home)
    try:
        cited = dict(values_hash=log.store.put(b'{"EURUSD":1.09}'),
                     job=log.store.put(b'{"Calc":{}}'), result=log.store.put(b'{"mtm":1.0}'))
    finally:
        log.close()
    assert oracle.report(home)['nothing_outside_its_seat']['held'] is True

    attested = dict(plan_hash=PLAN, engine_version='2.0', seed=1, lane=STANDING, **cited)
    for name, event_type, body, actor in (
            ('seat-attests', 'run_completed', attested, ACTOR),
            ('writer-books', 'fill', fill('EXEC-WRITER'), 'writer')):
        answers = oracle.report(planted(home, tmp_path, name, event_type, body, actor=actor,
                                        book_name=BOOK))
        assert oracle.failed(answers) == ['nothing_outside_its_seat'], (name, answers)
        said = answers['nothing_outside_its_seat']['evidence'][0]
        assert 'that voice forged' in said and repr(actor) in said, said


def test_an_approval_in_the_writers_voice_stands_only_where_the_hub_signs(tmp_path):
    """THE HUB'S OWN ACT, held to the record around it. The hub signs a ticket in its own voice
    only under a tiers policy in force, then books it: under a document granting nobody `approve`,
    an automatic tier's approval followed by the fill carrying its ticket holds - and the same
    approval put on a copy after a four-eyes ticket was booked unsigned, or where no tiers policy
    stands, is that voice forged, and `DV_Spine oracle` exits 1 on it.

    Killing mutations: the writer's own frames re-adjudicated against the document, which calls
    the hub's approval a seat's; and the writer's approval admitted by its voice alone, which lets
    a forgery clear a four-eyes ticket.
    """
    def booked(name, rows, signed):
        home = seeded(tmp_path, name, clips=())
        log = SpineLog(home)
        try:
            granted(log, ACTOR)
            policy.declare(log, ACTOR, policy.TIERS_POLICY, {'tiers': rows})
            file_quote(log, ACTOR, 'Q-1', 'ZeroCostCollar', PLAN, b'{"EURUSD":1.0851}',
                       {'floor': 1.07}, 4100.0, ticket=TICKET, book=BOOK)
            if signed:
                log.own('approval', {'plan_hash': TICKET}, book=BOOK)
            book(log, ACTOR, b'{"Reference":"CF1"}', -1.0, 'LEI-549300', 'CSA-0007', 'Q-1',
                 book=BOOK, ticket=TICKET)
        finally:
            log.close()
        return home

    held = oracle.report(booked('automatic', [{'name': 'auto'}], signed=True))
    assert oracle.failed(held) == [], held
    for name, where in (('four-eyes', booked('unsigned', TIERS['tiers'], signed=False)),
                        ('no-workflow', seeded(tmp_path, 'bare', clips=()))):
        copy = planted(where, tmp_path, name, 'approval', {'plan_hash': TICKET}, actor='writer',
                       book_name=BOOK)
        answers = oracle.report(copy)
        assert oracle.failed(answers) == ['nothing_outside_its_seat'], (name, answers)
        assert 'that voice forged' in answers['nothing_outside_its_seat']['evidence'][0]
        assert spine('oracle', '--home', str(copy)).returncode == 1, name


def test_a_restrike_judged_narrower_than_what_it_moved_is_named(tmp_path):
    """AN AMENDMENT IS JUDGED WHERE ITS TERMS ARE HELD, and a portfolio sits in its own book. The
    oracle re-derives the deepest node holding the terms at the frame before an amendment, off the
    positions fold, and holds the body's portfolio to it: a restrike judged at the book holding
    `/FX` and `/Rates` holds, while a copy carrying the FX seat's restrike judged at its own node -
    the Rates position moved too - is named, and so is a fill filed into another book's tree.

    Killing mutations: the amendment's portfolio trusted as written, which lets a node seat move
    every holder's position under a judgment at its own node; and the envelope's book not asked of
    a body's portfolio, which lets a copy carry rows in another book's tree.
    """
    terms, restruck = b'{"Reference":"CF-R"}', b'{"Reference":"CF-R","Amount":2}'
    home = seeded(tmp_path, 'restrike', clips=())
    log = SpineLog(home)
    try:
        blob = log.store.put(canonical_document({'grants': [
            {'subject': ACTOR, 'verb': verb, 'book': '*'} for verb in ('admin', 'book')] + [
            {'subject': 'subject-fx', 'verb': 'book', 'book': BOOK + '/FX'}], 'read': []}))
        log.append('policy_declared', {'policy': CAPABILITIES_POLICY, 'blob': blob}, actor=ACTOR,
                   blob_refs=(blob,))
        for execution, node in (('E-FX', '/FX'), ('E-RT', '/Rates')):
            book(log, ACTOR, terms, 5.0, 'LEI-1', 'CSA-1', execution, book=BOOK,
                 portfolio=BOOK + node)
        log.store.put(restruck)
    finally:
        log.close()
    moved = {'instrument': hashlib.sha256(terms).hexdigest(),
             'amended_to': hashlib.sha256(restruck).hexdigest()}
    wide = planted(home, tmp_path, 'wide', 'amendment', dict(moved, portfolio=BOOK),
                   book_name=BOOK)
    assert oracle.failed(oracle.report(wide)) == []
    for name, event_type, body, actor, said in (
            ('narrow', 'amendment', dict(moved, portfolio=BOOK + '/FX'), 'subject-fx',
             'judged narrower than what it moved'),
            ('stray', 'fill', dict(fill('E-STRAY'), portfolio='ELSEWHERE/FX'), ACTOR,
             "outside its own book's tree")):
        answers = oracle.report(planted(home, tmp_path, name, event_type, body, actor=actor,
                                        book_name=BOOK))
        assert oracle.failed(answers) == ['nothing_outside_its_scope'], (name, answers)
        assert said in answers['nothing_outside_its_scope']['evidence'][0], answers


def test_a_verdict_beside_its_ticket_and_a_fill_beside_the_tree_are_named(tmp_path):
    """A VERDICT REACHES WHAT BOOKS UNDER ITS NODE, AND A FILL BOOKS INTO THE TREE. An approver at
    the FX node holds `approve` there, so the writer admits its verdict on any ticket as long as
    the body names that node - the service derives the node off the record, a library caller need
    not - and a fill's node is checked against the tree by the seam, not the writer: a copy
    carrying the FX approver's signature on a Rates ticket, or a fill into a node the book never
    declared, is named, while the approval of an FX ticket and fills into declared nodes hold.

    Killing mutations: the verdicts' nodes never held to the tickets they rule on, which lets one
    desk clear another's trade by naming itself; and the tree read at the head rather than at the
    fill, which names a fill booked before the node it sits in was declared.
    """
    fx, rates, approver = BOOK + '/FX', BOOK + '/Rates', 'subject-fx-approver'
    home = seeded(tmp_path, 'placed', clips=())
    log = SpineLog(home)
    try:
        blob = log.store.put(canonical_document({'grants': [
            {'subject': ACTOR, 'verb': verb, 'book': '*'} for verb in ('admin', 'book')] + [
            {'subject': approver, 'verb': 'approve', 'book': fx}], 'read': []}))
        log.append('policy_declared', {'policy': CAPABILITIES_POLICY, 'blob': blob}, actor=ACTOR,
                   blob_refs=(blob,))
        book(log, ACTOR, b'{"Reference":"CF-EARLY"}', 1.0, 'LEI-1', 'CSA-1', 'EXEC-EARLY',
             book=BOOK, portfolio=fx + '/Early')
        for node in (fx, rates):
            declare_portfolio(log, ACTOR, node)
        for execution, node, ticket in (('EXEC-FX', fx, TICKET), ('EXEC-RT', rates, PLAN)):
            book(log, ACTOR, b'{"Reference":"CF1"}', 1.0, 'LEI-1', 'CSA-1', execution, book=BOOK,
                 portfolio=node, ticket=ticket)
        approve(log, approver, TICKET, book=BOOK, portfolio=fx)
    finally:
        log.close()
    held = oracle.report(home)
    assert oracle.failed(held) == [], held

    for name, event_type, body, actor, said in (
            ('beside', 'approval', {'plan_hash': PLAN, 'portfolio': fx}, approver,
             'reaches what books under it'),
            ('untreed', 'fill', dict(fill('EXEC-NOWHERE'), portfolio=fx + '/Exotics'), ACTOR,
             'does not declare')):
        answers = oracle.report(planted(home, tmp_path, name, event_type, body, actor=actor,
                                        book_name=BOOK))
        assert oracle.failed(answers) == ['nothing_outside_its_scope'], (name, answers)
        assert said in answers['nothing_outside_its_scope']['evidence'][0], answers


def test_the_firms_own_facts_are_judged_at_the_firm_whatever_their_envelope_names(tmp_path):
    """A FIRM FACT IS THE FIRM'S. The capabilities document and the official close are read by
    folds that ignore the book, so a node's admin granting itself admin over `*` and a node's mark
    seat filing the firm's close are refused whatever book the envelope names, each denial filing
    `*` - and the same two frames put on a copy are named by the oracle.

    Killing mutation: a firm type judged at its envelope's book, which lets a node seat reach the
    firm by naming its node.
    """
    governor, marks, node = 'subject-gov', 'subject-fx-marks', BOOK + '/FX'
    home = seeded(tmp_path, 'firm', clips=())
    log = SpineLog(home)
    try:
        rows = [{'subject': ACTOR, 'verb': 'admin', 'book': '*'},
                {'subject': governor, 'verb': 'admin', 'book': node},
                {'subject': marks, 'verb': 'mark', 'book': node}]
        declared = log.store.put(canonical_document({'grants': rows, 'read': []}))
        log.append('policy_declared', {'policy': CAPABILITIES_POLICY, 'blob': declared},
                   actor=ACTOR, blob_refs=(declared,))
        grab = log.store.put(canonical_document({'grants': rows + [
            {'subject': governor, 'verb': 'admin', 'book': '*'}], 'read': []}))
        attempts = ((governor, 'policy_declared', {'policy': CAPABILITIES_POLICY, 'blob': grab}),
                    (marks, 'official_close_declared', {
                        'market': 'official', 'values_hash': log.store.put(b'{"USDZAR":18.5}')}))
        for actor, event_type, body in attempts:
            with pytest.raises(CapabilityDenied):
                log.append(event_type, body, actor=actor, book=node)
        denied = [log.open_body(frame) for frame in log.frames()
                  if frame['event_type'] == 'capability_denied']
    finally:
        log.close()
    assert denied == [{'subject': actor, 'verb': verb, 'book': '*', 'attempted_type': event_type}
                      for (actor, event_type, _), verb in zip(attempts, ('admin', 'mark'))]
    # the node admin's grab is a scoped declaration, judged by the node half of the decision
    for (actor, event_type, body), half in zip(attempts, ('nothing_outside_its_scope',
                                                          'nothing_outside_its_seat')):
        answers = oracle.report(planted(home, tmp_path, event_type, event_type, body, actor=actor,
                                        book_name=node))
        assert oracle.failed(answers) == [half], answers
        assert "over '*'" in answers[half]['evidence'][0], answers


def test_a_refusal_the_script_asked_for_and_a_denial_it_did_not_are_both_named(tmp_path):
    """EVERY REFUSAL IS A DENIAL, and every denial a refusal somebody asked for. The two directions
    are one set equality, because a denial nobody asked for is a seat this script never mentions
    reaching this record.

    Killing mutation: the asked denials checked for presence alone, which lets a refusal the day
    never made sit in the record unremarked.
    """
    from derivus_spine import CapabilityDenied

    home = seeded(tmp_path, 'refused', clips=())
    log = SpineLog(home)
    try:
        granted(log, ACTOR)
        with pytest.raises(CapabilityDenied):
            log.append('fill', fill('EXEC-STRANGER'), actor=STRANGER, book=BOOK)
    finally:
        log.close()

    denial = {'subject': STRANGER, 'verb': 'book', 'book': BOOK, 'attempted_type': 'fill'}
    assert oracle.report(home, script=script({'denied': denial}))[
        'every_refusal_is_a_denial']['held'] is True

    surplus = oracle.report(home, script=script())['every_refusal_is_a_denial']
    assert surplus['held'] is False and 'never asked for' in surplus['evidence'][0]
    missing = oracle.report(home, script=script(
        {'denied': dict(denial, verb='mark')}))['every_refusal_is_a_denial']
    assert missing['held'] is False and 'the record holds none' in missing['evidence'][0]


def test_a_ticket_filed_twice_and_a_restrike_nobody_routed_are_named(tmp_path):
    """AN AMENDED PLAN IS A NEW APPROVAL: every ticket is one trade's. Two different trades
    carrying one ticket - a fill and another fill, or a fill and a restrike - are named, one
    approval reaching both, while one trade two seats filed is one trade and holds; and under a
    workflow a restrike carrying no ticket is named, the position it moved standing on the
    approvals over the ticket it had. Tickets of their own hold, pending or not - a status and
    never a failure - and with no workflow a restrike owes none.

    Killing mutations: a second filing of one ticket let through, which lets one approval clear a
    clip nobody signed; a ticket read as one filing, which names a trade two seats filed; and a
    restrike carrying no ticket let through under a workflow, which reads the restruck position on
    approvals nobody gave its terms.
    """
    held = oracle.report(traded(tmp_path, 'held', [TICKET, PLAN], restrike=['c' * 64]))
    assert oracle.failed(held) == [], held
    retold = traded(tmp_path, 'retold', [TICKET])
    log = SpineLog(retold)
    try:
        log.append('fill', next(log.open_body(frame) for frame in log.frames()
                                if frame['event_type'] == 'fill'), actor=STRANGER, book=BOOK)
    finally:
        log.close()
    assert oracle.failed(oracle.report(retold)) == [], 'one trade two seats filed was named'

    twice = oracle.report(traded(tmp_path, 'twice', [TICKET, TICKET]))
    assert oracle.failed(twice) == ['an_amended_plan_is_a_new_approval'], twice
    assert 'for another trade' in twice['an_amended_plan_is_a_new_approval']['evidence'][0]
    reused = oracle.report(traded(tmp_path, 'reused', [TICKET], restrike=[TICKET]))
    assert oracle.failed(reused) == ['an_amended_plan_is_a_new_approval'], reused

    unrouted = oracle.report(traded(tmp_path, 'unrouted', [TICKET], restrike=[None]))
    assert oracle.failed(unrouted) == ['an_amended_plan_is_a_new_approval'], unrouted
    assert 'carries no ticket' in unrouted['an_amended_plan_is_a_new_approval']['evidence'][0]
    free = oracle.report(traded(tmp_path, 'free', [None], restrike=[None], workflow=False))
    assert oracle.failed(free) == [], free


def test_a_close_no_validating_writer_wrote_is_named(tmp_path):
    """CLOSES SUPERSEDED, NEVER EDITED. Supersession is computed by the fold, so what a platter can
    carry that the fold cannot is a close body the closed vocabulary would have refused - which is
    what a copy of a newer hub, or a hand on a file, puts there.

    Killing mutation: the markets fold taken before the bodies are read, which raises out of the
    projector instead of reporting the line.
    """
    home = synthetic_book(tmp_path)[:2][0]
    where = planted(home, tmp_path, 'edited', 'official_close_declared',
                    {'values_hash': 'not an address'})

    answers = oracle.report(where)
    assert oracle.failed(answers) == ['closes_superseded_never_edited'], answers
    assert answers['closes_superseded_never_edited']['evidence'][0].endswith(
        "no writer that validates wrote this line")


def test_a_standing_run_without_its_attestation_is_named(tmp_path):
    """EVERY NUMBER'S REPLAY TUPLE. The attestations are exactly the standing runs the script asked
    for: a standing ask with no row is a number nothing can replay, and a row no standing ask names
    is a lane that mints nothing having minted.

    Killing mutation: the asks counted rather than matched, which passes any script with as many
    standing runs in it as the record has attestations.
    """
    home, log = synthetic_book(tmp_path)[:2]
    try:
        cited = dict(values_hash=log.store.put(b'{"EURUSD":1.09}'),
                     job=log.store.put(b'{"Calc":{}}'), result=log.store.put(b'{"mtm":1.0}'))
        tuple_ = dict(plan_hash=PLAN, engine_version='2.0', seed=1,
                      values_hash=cited['values_hash'])
        log.own('run_completed', dict(tuple_, lane=STANDING, **cited),
                blob_refs=tuple(cited.values()))
    finally:
        log.close()

    assert oracle.report(home, script=script(
        {'lane': STANDING, 'replay': tuple_}))['every_numbers_replay_tuple']['held'] is True

    missing = oracle.report(home, script=script(
        {'lane': STANDING, 'replay': dict(tuple_, seed=99)}))['every_numbers_replay_tuple']
    assert missing['held'] is False
    assert 'the record holds no attestation' in missing['evidence'][0]
    assert 'no standing run asked for' in missing['evidence'][1], 'the other direction went unread'

    quiet = oracle.report(home, script=script({'lane': 'curiosity', 'replay': tuple_}))
    assert 'a lane that mints nothing minted' in quiet['every_numbers_replay_tuple']['evidence'][0]


def test_a_settlement_against_a_key_the_diary_has_no_row_for_is_named(tmp_path):
    """THE DIARY EQUALS THE FILINGS - the one invariant a replica cannot put, because the diary is a
    COMPILE of the book and not a fold of the record. The keys arrive as data from a caller that
    holds an engine, and a transition against a key no row carries is a status moving nothing; a
    confirmation is filed against a clip's own key, which the record derives, and holds.

    Killing mutations: a transition against a name rather than an address counted as a settlement,
    which reads every lifecycle row as a payment; and the clips not read, which names every
    confirmation the back office files.
    """
    home = synthetic_book(tmp_path)[:2][0]

    assert oracle.report(home)['the_diary_equals_the_filings']['held'] is None
    orphaned = oracle.report(home, diary_keys=[])['the_diary_equals_the_filings']
    assert orphaned['held'] is False and INSTRUMENT in orphaned['evidence'][0]
    log = SpineLog(home)
    try:
        clip = next(entry for row in PROJECTORS['positions'].rows(fold(
            log, PROJECTORS['positions'])) for entry in row['tickets'])
    finally:
        log.close()
    confirmed = planted(home, tmp_path, 'confirmed', 'status_transition', {
        'subject': content_hash({'instrument': clip['instrument'],
                                 'execution_reference': clip['execution_reference']}),
        'status': 'confirmed'}, book_name=BOOK)
    assert oracle.report(confirmed, diary_keys=[INSTRUMENT])[
        'the_diary_equals_the_filings']['held'] is True
    assert oracle.report(home, diary_keys=[INSTRUMENT])[
        'the_diary_equals_the_filings']['evidence'] == ['1 settlement(s) filed against 1 diary '
                                                        'key(s) and 2 clip(s)']


def test_a_blob_taken_off_a_copy_is_named_and_a_line_taken_off_it_is_refused(tmp_path):
    """NO DELETE, both ways a record can suffer one. The store has NO VERB for forgetting, so what
    takes a blob is a file system - and the frame that cited it is still on the platter naming an
    address nothing answers for, which is the deletion that actually happens. Every citing frame is
    re-read and every address asked of the store, so a copy a blob was taken off is named by it.

    A LINE that left is met earlier and harder: the sequence is dense from genesis, so the home
    refuses when it is OPENED and there is no report to write. The hub the copies were taken from
    is untouched either way.

    Killing mutations: the closure arm dropped, which leaves every arm of this invariant a constant
    - the blob walk compares the store with itself microseconds apart on a home nobody is writing,
    and the verbs that do not exist never will; and the removed line healed as a torn tail, which
    is the one repair this package has and is reserved for an unterminated FINAL line.
    """
    hub = seeded(tmp_path, 'whole')
    log = SpineLog(hub)
    try:
        taken = next(digest for frame in log.frames() if frame['event_type'] == 'fill'
                     for _, digest in cited_blobs('fill', log.open_body(frame)))
    finally:
        log.close()
    shredded = copied(hub, tmp_path, 'shredded')
    (shredded / 'blobs' / taken[:2] / taken[2:4] / taken).unlink()

    answers = oracle.report(shredded)
    assert oracle.failed(answers) == ['no_delete'], answers
    said = answers['no_delete']['evidence'][0]
    assert taken in said and 'instrument' in said and 'logged retention event' in said
    assert oracle.report(hub)['no_delete']['held'] is True, 'the hub was touched'

    # a copy holding no key cannot open a body to find a citation, so it keeps the other three arms
    (shredded / 'keys' / 'class_firm.key').unlink()
    assert oracle.report(shredded)['no_delete']['evidence'][0].endswith('citations not assessed')

    cut = copied(hub, tmp_path, 'cut')
    segment = segments(cut)[0]
    lines = [raw for raw in segment.read_bytes().split(b'\n') if raw.strip()]
    segment.write_bytes(b'\n'.join(lines[:1] + lines[2:]) + b'\n')
    with pytest.raises(ChainBroken, match='dense'):
        oracle.report(cut)


def test_one_fact_written_twice_under_one_tag_is_named(tmp_path):
    """DUPLICATES COALESCE. The writer folds a repeat onto the position it already has; a replica
    takes the order the hub gave it and coalesces nothing, so a tag on two lines is one act the
    record counted twice.

    Read off the ENVELOPE, so the copy below answers it holding no key at all - which is asserted
    here by deleting the class key and asking again.

    Killing mutation: the tags counted rather than located, which cannot say which two positions
    share one.
    """
    home = seeded(tmp_path, 'once')
    log = SpineLog(home)
    try:
        repeated = next(frame for frame in log.frames()
                        if frame['event_type'] == 'fill')['idempotency_tag']
    finally:
        log.close()

    where = planted(home, tmp_path, 'twice', 'fill', fill('EXEC-1'), book_name=BOOK,
                    tag=repeated)
    answers = oracle.report(where)
    assert oracle.failed(answers) == ['duplicates_coalesce'], answers
    assert 'one fact, written twice' in answers['duplicates_coalesce']['evidence'][0]

    (where / 'keys' / 'class_firm.key').unlink()
    sealed = oracle.report(where)
    assert oracle.failed(sealed) == ['duplicates_coalesce'], sealed
    assert verify_home(where, entitled=False)['mode'] == 'chain-only'


#: The desk's two portfolios, which the P&L gates partition the book into, and what the record
#: holds of each at the desk's three marks - one clip into FX before the first, two into Rates
#: at a quarter between the first two.
FX, RATES = BOOK + '/FX', BOOK + '/Rates'
HOLDS = {FX: (1.0, 1.0, 1.0), RATES: (0.0, 2.0, 2.0)}
CF1 = hashlib.sha256(b'{"Reference":"CF1"}').hexdigest()


def marked(log, day, rate, reference=None, market='official'):
    """Close `market` for `day` at an EURUSD of `rate` and attest a run of the book's marks job on
    it - cut at that close, unless `reference` names the job otherwise - answered as a P&L names
    one of its ends."""
    closed = declare_close(log, ACTOR, market, '{{"EURUSD":{}}}'.format(rate).encode(), date=day)
    cited = dict(values_hash=closed['values_hash'], result=log.store.put(b'{"mtm":1.0}'),
                 job=log.store.put(canonical_bytes({'Calc': {
                     'Deals': {'Reference': reference or 'marks:{}@{}'.format(BOOK, closed['lsn'])},
                     'Calculation': {'Base_Date': {'.Timestamp': day}}}})))
    log.own('run_completed', dict(cited, plan_hash=PLAN, engine_version='2.0', seed=1,
                                  lane=STANDING), blob_refs=tuple(cited.values()))
    return dict(cited, day=day, lsn=closed['lsn'])


def desk(tmp_path, name):
    """The desk's home, marked on three days as `HOLDS` says, and its three marks."""
    home = seeded(tmp_path, name, clips=())
    log = SpineLog(home)
    try:
        policy.declare(log, ACTOR, policy.TOLERANCE_POLICY, {'tolerances': {'pnl': 1e-9}})
        book(log, ACTOR, b'{"Reference":"CF1"}', 1.0, 'LEI-1', 'CSA-1', 'EXEC-FX', book=BOOK,
             price=0.5, portfolio=FX)
        marks = [marked(log, '2026-08-26', 1.0851)]
        book(log, ACTOR, b'{"Reference":"CF1"}', 2.0, 'LEI-1', 'CSA-1', 'EXEC-RATES', book=BOOK,
             price=0.25, portfolio=RATES)
        marks += [marked(log, day, rate) for day, rate in (('2026-08-27', 1.0853),
                                                           ('2026-08-28', 1.0855))]
    finally:
        log.close()
    return home, marks


def answered(marks, start, end, *made):
    """A `GET /book/pnl` answer between two of the desk's `marks`: one position per `(portfolio,
    pnl)` holding what the record holds and paying what its fills cost, every other additive
    figure nothing, and the total their sum."""
    rows = [dict({figure: 0.0 for figure in oracle.ADDITIVE}, portfolio=portfolio,
                 agreement='CSA-1', instrument=CF1, pnl=pnl, unrealised=pnl,
                 quantity_start=HOLDS[portfolio][start], quantity_end=HOLDS[portfolio][end],
                 premiums=-0.5 if portfolio == RATES and start == 0 else 0.0)
            for portfolio, pnl in made]
    return {'start': marks[start], 'end': marks[end], 'rows': rows,
            'scope': {'portfolio': rows[0]['portfolio'] if len(rows) == 1 else BOOK},
            'total': {figure: math.fsum(row[figure] for row in rows)
                      for figure in oracle.ADDITIVE}}


def test_a_pnl_that_does_not_sum_is_named(tmp_path):
    """THE P&L IS ADDITIVE: two days sum to the window over both and the portfolios of a window to
    the book, figure by figure, within the epsilon the deployment declared for `pnl` - never one
    of the spine's own. A window whose Rates row made a tenth more than its two days did is named
    at that row, portfolios summing past their book are named, days that do not tile the window
    are named before any figure is read, and answers that are no answers put no question.

    Killing mutations: the days' figures summed per total alone, which misses a row two errors
    cancel in; a default epsilon in place of the declared one; and the portfolios never summed.
    """
    home, marks = desk(tmp_path, 'additive')
    days = [answered(marks, 0, 1, (FX, 0.1), (RATES, 0.2)),
            answered(marks, 1, 2, (FX, 0.2), (RATES, 0.3))]
    sums = {'window': answered(marks, 0, 2, (FX, 0.1 + 0.2), (RATES, 0.2 + 0.3)), 'days': days,
            'portfolios': [answered(marks, 0, 2, (FX, 0.1 + 0.2)),
                           answered(marks, 0, 2, (RATES, 0.5))]}
    assert oracle.report(home, pnl=sums)['the_pnl_is_additive']['held'] is True

    cancelled = dict(sums, window=answered(marks, 0, 2, (FX, 0.4), (RATES, 0.4)), portfolios=[])
    answers = oracle.report(home, pnl=cancelled)
    assert oracle.failed(answers) == ['the_pnl_is_additive'], answers
    said = answers['the_pnl_is_additive']['evidence']
    assert said[0].startswith('summed over the days, pnl.rows.{} CSA-1'.format(FX)), said
    apart = oracle.report(home, pnl=dict(sums, portfolios=sums['portfolios'][:1] + [
        answered(marks, 0, 2, (RATES, 0.6))]))['the_pnl_is_additive']
    assert apart['evidence'][0].startswith('summed over the portfolios, pnl.total.pnl'), apart
    untiled = oracle.report(home, pnl=dict(sums, days=days[1:]))['the_pnl_is_additive']
    assert untiled['held'] is False and 'do not tile' in untiled['evidence'][0], untiled
    assert oracle.report(home, pnl={})['the_pnl_is_additive']['held'] is None


def test_a_pnl_this_record_did_not_make_is_named(tmp_path):
    """Every P&L answer is THIS record's: each end a marks run it attested - the day, the cut, the
    values, the job and the result - each row's quantities the `costs` fold's at the two ends and
    its premiums what the fills between them cost. Answers read at marks nobody ran, or whose
    premiums were doubled, or whose quantities moved, are named, however well they sum.

    Killing mutations: the ends not held to the attestations; the premiums read off the answer
    rather than the fills; and the quantities never read off the `costs` fold.
    """
    home, marks = desk(tmp_path, 'tied')
    window = answered(marks, 0, 1, (FX, 0.1), (RATES, 0.2))
    for pnl, said in ((dict(window, start=dict(marks[0], job='0' * 64)), 'which no run this '
                       'record attested is'),
                      (dict(window, rows=[dict(row, premiums=2 * row['premiums'])
                                          for row in window['rows']]), 'pnl.premiums'),
                      (dict(window, rows=[dict(row, quantity_end=3.0)
                                          for row in window['rows']]), 'pnl.quantity_end')):
        found = oracle.report(home, pnl={'window': pnl})['the_pnl_is_additive']
        assert found['held'] is False and said in found['evidence'][0], found


def test_money_moved_under_paper_nobody_declared_is_named(tmp_path):
    """CASH RECONCILES: a fee is paid on an instrument a fill booked, collateral moves under a
    declared agreement, a payment moves what the diary determines for its row and a row is paid
    once, and every row a settlement file instructed is settled - by a payment, or by a bare
    `settled` as the diary reads it. A copy carrying money the verb never saw is named at each.

    Killing mutations: the paper not asked of collateral; the fee's instrument not asked; the
    payment not held to the diary's amount and currency; a row paid under two references counted
    once; and a bare `settled` read as unsettled, which names every row closed without money.
    """
    key, owed, lost = 'c' * 64, 'd' * 64, 'e' * 64
    home = seeded(tmp_path, 'cash', clips=('EXEC-1',))
    log = SpineLog(home)
    try:
        declare_entity(log, ACTOR, 'LEI-1', 'Client One')
        declare_agreement(log, ACTOR, 'CSA-1', 'LEI-1', 'ISDA with CSA', b'{"Reference":"CSA-1"}')
        for subject, amount, kind in (('CSA-1', 250_000.0, 'collateral'), (INSTRUMENT, -50.0, 'fee'),
                                      (key, 1_000.0, 'payment')):
            transition(log, ACTOR, subject, 'settled', amount=amount, asset='USD', kind=kind,
                       reference='{}-1'.format(kind), effective_time=WHEN)
        transition(log, ACTOR, owed, 'settled')
        head = log.head()[0]
    finally:
        log.close()
    diary = {key: {'amount': 1_000.0, 'currency': 'USD'}, owed: {'amount': None, 'currency': 'USD'}}
    for keys in (diary, sorted(diary)):
        assert oracle.report(home, script=script({'instructed': [key, owed]}), diary_keys=keys)[
            'cash_reconciles']['held'] is True

    copy = home
    for name, subject, kind, asset in (('unpapered', 'CSA-NOBODY', 'collateral', 'USD'),
                                       ('unbooked', OTHER, 'fee', 'USD'),
                                       ('twice', key, 'payment', 'JPY')):
        copy = planted(copy, tmp_path, name, 'status_transition', {
            'subject': subject, 'status': 'settled', 'amount': 1.0, 'asset': asset, 'kind': kind,
            'reference': name}, book_name=BOOK)
    answers = oracle.report(copy, script=script({'instructed': [key, owed, lost]}),
                            diary_keys=diary)
    assert oracle.failed(answers) == ['cash_reconciles'], answers
    assert answers['cash_reconciles']['evidence'][:5] == [
        'LSN {}: a collateral movement of 1 USD names CSA-NOBODY - no agreement the record '
        'declared by then'.format(head + 1),
        'LSN {}: a fee movement of 1 USD names {} - no instrument a fill booked by '
        'then'.format(head + 2, OTHER),
        '{} is paid 2 times, under payment-1, twice - one row is paid once'.format(key),
        'twice moved 1.0 JPY against {}, which the diary determines at 1000.0 USD'.format(key),
        'the settlement file instructed {} and nothing settled it'.format(lost)]


def test_a_close_nobody_marked_is_named(tmp_path):
    """EVERY CLOSE IS MARKED: a day closed on the market the workflow designates for P&L owes, while
    the book holds anything, a marks run of the book as of that day on the values of a close
    declared for it, named `marks:<book>@<cut>` with the cut at or before the run. The next day's
    close nobody marked is named, and so is one marked under a name that cuts nothing or cuts past
    its own run, or only by a run of another day on its values; a close on a market nothing
    designates, a day the book held nothing and a day restated after its marks owe nothing more -
    the last a reading.

    Killing mutations: the marks matched by values alone; the designation read as `official`
    whatever the workflow says; marks owed whatever the book held; a name with no digits after its
    last '@' read as a cut; and a cut past the run that attested it believed.
    """
    home = seeded(tmp_path, 'marked', clips=('EXEC-1',))
    log = SpineLog(home)
    try:
        policy.declare(log, ACTOR, policy.TIERS_POLICY, dict(TIERS, designations={'pnl': 'eod'}))
        marked(log, '2026-08-26', 1.0851, market='eod')
        head = log.head()[0]
    finally:
        log.close()
    assert oracle.report(home)['every_close_marked']['held'] is True

    def closed(name, day='2026-08-27', market='eod', rate=1.0851, reference=None, flat=False,
               marking=None):
        where = copied(home, tmp_path, name)
        log = SpineLog(where)
        try:
            if flat:
                log.append('fill', fill('EXEC-FLAT', -1000000.0), actor=ACTOR, book=BOOK)
            if reference is None:
                declare_close(log, ACTOR, market, '{{"EURUSD":{}}}'.format(rate).encode(), date=day)
            else:
                marked(log, day, rate, reference, market)
            if marking:
                marked(log, marking, rate, market=market)
        finally:
            log.close()
        return oracle.report(where)['every_close_marked']

    for name, reference, rate, marking in (
            ('unmarked', None, 1.0851, None), ('uncut', 'marks:{}@x'.format(BOOK), 1.0852, None),
            ('ahead', 'marks:{}@{}'.format(BOOK, head + 10), 1.0852, None),
            ('yesterday', None, 1.0858, '2026-08-26')):
        found = closed(name, rate=rate, reference=reference, marking=marking)
        assert found['held'] is False and found['evidence'][0].startswith(
            "LSN {}: the close of 2026-08-27 on 'eod' is not marked".format(head + 1)), found
    assert closed('elsewhere', market='official')['held'] is True
    assert closed('flat', flat=True)['held'] is True
    restated = closed('restated', day='2026-08-26', rate=1.0859)
    assert restated['held'] is True and restated['evidence'][1] == (
        "the close of 2026-08-26 on 'eod' was restated at LSN {} after the marks of {!r} at LSN "
        "{}, and the day reads as it was marked".format(head + 1, BOOK, head)), restated


def test_a_call_the_formula_does_not_give_is_named(tmp_path):
    """THE CALL IS THE FORMULA: every call a seat read is `CSA.call` over the exposure and
    the spots it was read at, the `cash` fold at the read's position held as of its day and the
    terms the agreement stood on there - so terms restated after the read do not reach back into
    it. A read a dollar off the formula, or reporting a balance the record does not hold, is named;
    a call nobody could work out carries no numbers, and a script of nothing else puts no question.

    Killing mutations: the balance read at the head rather than at the read's own position; the
    terms read there too; the balance the service reported never compared; and an unknown call's
    numbers never read.
    """
    terms = {'Object': 'NettingCollateralSet', 'Reference': 'CSA-1', 'Collateralized': 'True',
             'Agreement_Currency': 'USD', 'Credit_Support_Amounts': {
                 dial: {'.CreditSupportList': [[1, amount]]} for dial, amount in (
                     ('Received_Threshold', 100.0), ('Posted_Threshold', -100.0),
                     ('Minimum_Received', 10.0), ('Minimum_Posted', 10.0))}}
    home = seeded(tmp_path, 'called', clips=())
    log = SpineLog(home)
    try:
        declare_entity(log, ACTOR, 'LEI-1', 'Client One')
        declare_agreement(log, ACTOR, 'CSA-1', 'LEI-1', 'ISDA with CSA', canonical_bytes(terms))
        transition(log, ACTOR, 'CSA-1', 'settled', amount=40.0, asset='USD', kind='collateral',
                   reference='COL-1', effective_time='2026-08-26T00:00:00.000000Z')
        at = log.head()[0]
        transition(log, ACTOR, 'CSA-1', 'settled', amount=360.0, asset='USD', kind='collateral',
                   reference='COL-2', effective_time='2026-08-26T00:00:00.000000Z')
        declare_agreement(log, ACTOR, 'CSA-1', 'LEI-1', 'ISDA with CSA', canonical_bytes(dict(
            terms, Credit_Support_Amounts=dict(terms['Credit_Support_Amounts'], Received_Threshold={
                '.CreditSupportList': [[1, 200.0]]}))))
    finally:
        log.close()
    fx = {'USD': 1.0}
    read = dict(CSA.call(500.0, {'USD': 40.0}, CSA.of(terms), fx),
                agreement='CSA-1', date='2026-08-26', lsn=at, exposure=500.0, fx=fx,
                held={'USD': 40.0}, unknown=[])
    assert read['call'] == 360.0
    assert oracle.report(home, script=script({'called': [read]}))[
        'the_call_is_the_formula']['held'] is True

    named = 'CSA-1 on 2026-08-26 at LSN {}: '.format(at)
    unknown = [{'instrument': None, 'what': 'no mark at the end'}]
    for wrong, said in (
            (dict(read, call=361.0), 'the service answered call 361.0 and the formula over the '
                                     'record says 360.0'),
            (dict(read, held={'USD': 41.0}), "the service answered held {'USD': 41.0} and the "
                                             "formula over the record says {'USD': 40.0}"),
            (dict(read, unknown=unknown), 'the call names what nobody can know and carries '
                                          'required, balance, call, direction, minimum_transfer '
                                          'anyway')):
        answers = oracle.report(home, script=script({'called': [wrong]}))
        assert oracle.failed(answers) == ['the_call_is_the_formula'], answers
        assert answers['the_call_is_the_formula']['evidence'][0] == named + said
    blank = dict(read, unknown=unknown, **dict.fromkeys(CSA.WORKED))
    assert oracle.report(home, script=script({'called': [blank]}))[
        'the_call_is_the_formula']['held'] is None


def test_the_cli_answers_the_fourteen_and_exits_on_one_that_did_not_hold(tmp_path):
    """The mouth: `DV_Spine oracle` prints the report and exits 1 where an invariant did not hold,
    which is what makes it usable from a cron and from a gate that is not this suite; the diary's
    keys and the P&L are files it takes as data.

    A subprocess, which is the only honest way to gate an exit code.
    """
    home = str(synthetic_book(tmp_path)[:2][0])
    asked = spine('oracle', '--home', home)
    assert asked.returncode == 0, asked.stderr
    assert sorted(json.loads(asked.stdout)) == sorted(oracle.INVARIANTS)
    keys = tmp_path / 'diary.json'
    keys.write_text(json.dumps([INSTRUMENT]), newline='\n')
    diary = json.loads(spine('oracle', '--home', home, '--diary', str(keys)).stdout)
    assert diary['the_diary_equals_the_filings']['held'] is True

    against = str(seeded(tmp_path, 'elsewhere'))
    diverged = spine('oracle', '--home', home, '--against', against)
    assert diverged.returncode == 1, diverged.stdout
    assert json.loads(diverged.stdout)['copies_agree']['held'] is False

    where = tmp_path / 'script.json'
    where.write_text(json.dumps({'acts': [{'denied': {
        'subject': STRANGER, 'verb': 'book', 'book': BOOK, 'attempted_type': 'fill'}}]}),
        newline='\n')
    scripted = spine('oracle', '--home', home, '--script', str(where))
    assert scripted.returncode == 1
    assert 'the record holds none' in json.dumps(
        json.loads(scripted.stdout)['every_refusal_is_a_denial'])


def test_the_oracle_holds_the_package_to_its_one_dependency():
    """The oracle is a module of the record, so it reaches stdlib and `cryptography` and nothing
    else - the import gate's glob already covers it, and this says out loud that it MUST.

    Killing mutation: the oracle importing the engine for the diary or the P&L invariant, which is
    exactly the temptation the `diary_keys` and `pnl` arguments exist to remove.
    """
    import ast

    with open(oracle.__file__, encoding='utf-8') as handle:
        tree = ast.parse(handle.read(), filename=oracle.__file__)
    absolute = set((node.module or '').split('.')[0] for node in ast.walk(tree)
                   if isinstance(node, ast.ImportFrom) and not node.level) | set(
        alias.name.split('.')[0] for node in ast.walk(tree)
        if isinstance(node, ast.Import) for alias in node.names)

    assert absolute <= set(sys.stdlib_module_names), \
        'the oracle reaches outside the package: {}'.format(absolute)
    assert {'diary_keys', 'pnl'} <= set(oracle.report.__code__.co_varnames)
