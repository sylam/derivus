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

"""The tiers policy and the evaluator that routes a ticket through it.

The document half runs on real homes in temp directories, declared through the ordinary writer, so
every refusal is met WHERE IT IS DECLARED - while the operator still has the file open - rather than
later at an approval that could not act on it. The evaluator half runs on nothing at all: it holds
no log, clock, store or home, and the gate that says so reads its signatures and its imports rather
than believing the docstring.

  * THE FIRST TIER WHOSE CHECKS PASS applies, the catch-all is a tier declaring none, and the
    verdict names every check it read with the value measured and the bound declared.
  * A CHECK THAT COULD NOT BE MADE IS A FAILURE, never a pass: a notional the cap's currency cannot
    see and a market nobody declared each fail their tier by name.
  * THE LATEST VERDICT STANDS, and under four eyes the approver may not be the booker.
"""
import ast
import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from derivus_spine import (
    MalformedEvent, SpineLog, TierRefused, init_home, policy, tiers)

MINT = 'subject-deployment'
BOOKER = 'subject-desk-one'
APPROVER = 'subject-desk-two'

#: The board the quote was struck on, and a board it was not.
VALUES = 'a' * 64
MOVED = 'b' * 64

#: Three tiers in the order they are read: an automatic tier with all three checks, a four-eyes
#: tier capping size alone, and the catch-all that declares no check at all.
POLICY = {
    'tiers': [
        {'name': 'auto', 'market': 'official',
         'max_notional': {'amount': 5_000_000.0, 'currency': 'USD'}, 'max_tenor_years': 2.0},
        {'name': 'senior', 'four_eyes': True,
         'max_notional': {'amount': 25_000_000.0, 'currency': 'USD'}},
        {'name': 'board'},
    ],
    'designations': {'settlement_export': 'official'},
}
STANDING = {'official': VALUES}
#: The same three tiers as the parser COMPLETES them - the one default this shape has, written in
#: before the bytes are hashed: a tier is automatic unless it says otherwise.
COMPLETED = {'tiers': [dict(POLICY['tiers'][0], four_eyes=False),
                       POLICY['tiers'][1],
                       {'name': 'board', 'four_eyes': False}],
             'designations': POLICY['designations']}
CAP = {'amount': 5_000_000.0, 'currency': 'USD'}


def minted(tmp_path, name='home'):
    """A home mid-genesis, handed back with the writer open on it."""
    home = tmp_path / name
    init_home(home, MINT)
    return home, SpineLog(home)


def ticket(usd=1_000_000.0, tenor=1.0, values_hash=VALUES, **notionals):
    """A ticket as the caller states it: the notional in every currency it can name, the tenor in
    years, and the values vector the quote pinned."""
    stated = dict(notionals)
    if usd is not None:
        stated['USD'] = usd
    return {'notional_in': stated, 'tenor_years': tenor, 'values_hash': values_hash}


def verdict(kind, actor, lsn, reason=None):
    """One row of the `decisions` fold's list for a ticket."""
    return {'verdict': kind, 'actor': actor, 'reason': reason, 'lsn': lsn}


def read(verdict_, tier, check):
    """The one check row `tier` read for `check`, so a gate asserts on it by name rather than by
    position in a list whose order is the policy's."""
    return next(row for row in verdict_['checks']
                if row['tier'] == tier and row['check'] == check)


# --------------------------------------------------------------------------------------------
# The document, closed at the field level and refused where it is declared.

def test_a_tiers_policy_is_closed_at_the_field_level_and_refuses_where_it_is_declared(tmp_path):
    """Every way the document can be wrong, met at the DECLARATION and named there.

    The two that are not merely shape: a tier restating `pillar_seconds`, `firm` or either of the
    two windows those replaced is refused because how stale a board may be is the firmness policy's
    one window and is checked on every booking before a tier is read - a second spelling would be
    two standards for one question; and a designation naming a `private/` market is refused because
    a designated process resolves the market the firm declared and never one seat's own. What is
    stored is the COMPLETED document - `four_eyes` false and an empty `designations` written in
    before the bytes are hashed - so one workflow is one blob however it was spelled.

    Killing mutation: the restated staleness window let through to the tier's field check.
    """
    home, log = minted(tmp_path)
    broken = (
        ('note', dict(POLICY, note='why')),
        ('tiers', {'tiers': []}),
        ('tiers', {'tiers': {'auto': {}}}),
        ('tier 1', {'tiers': [{'four_eyes': True}]}),
        ('declared twice', {'tiers': [{'name': 'auto'}, {'name': 'auto'}]}),
        ('currency', {'tiers': [{'name': 'auto', 'max_notional': {'amount': 1.0}}]}),
        ('max_notional', {'tiers': [{'name': 'auto', 'max_notional': 5_000_000.0}]}),
        ('max_tenor_years', {'tiers': [{'name': 'auto', 'max_tenor_years': -1.0}]}),
        ('escalates_to', {'tiers': [{'name': 'auto', 'escalates_to': 'desk'}]}),
        ('four_eyes', {'tiers': [{'name': 'auto', 'four_eyes': 'yes'}]}),
        # an automatic approval is the hub's own act, so a seat to sign it under is refused by name
        ('names the seat', {'tiers': [{'name': 'auto', 'seat': 'policy/tiers/auto'}]}),
        ('scope', {'tiers': [{'name': 'fx', 'scope': 'BANK//FX'}]}),
        ('firmness', {'tiers': [{'name': 'auto', 'pillar_seconds': 900}]}),
        ('firmness', {'tiers': [{'name': 'auto', 'values_seconds': 30}]}),
        ('firmness', {'tiers': [{'name': 'auto', 'firm': True}]}),
        ('firmness', dict(POLICY, plan_seconds=600)),
        ('settlement_export', dict(POLICY, designations={'official_pnl': 'official'})),
        ('private/', dict(POLICY, designations={'settlement_export': 'private/desk-one/screen'})),
        ('private/', {'tiers': [{'name': 'auto', 'market': 'private/desk-one/screen'}]}),
        ('designations', dict(POLICY, designations=None)),
    )
    for named, document in broken:
        with pytest.raises(MalformedEvent) as refusal:
            policy.declare(log, MINT, policy.TIERS_POLICY, document)
        assert named in str(refusal.value), (named, str(refusal.value))
    assert log.head()[0] == 4, 'a refused declaration wrote something'

    declared = policy.declare(log, MINT, policy.TIERS_POLICY, POLICY)
    assert declared['lsn'] == 5
    # what is stored is the COMPLETED document: both defaults written in before it was hashed
    assert policy.tiers_in_force(log) == COMPLETED
    assert policy.tiers_in_force(log, declared['lsn'] - 1) is None
    assert policy.tiers_in_force(
        log, policy.declare(log, MINT, policy.TIERS_POLICY,
                            {'tiers': [{'name': 'board'}]})['lsn']) == {
        'tiers': [{'name': 'board', 'four_eyes': False}], 'designations': {}}
    log.close()


# --------------------------------------------------------------------------------------------
# The evaluator: the first tier whose checks pass.

def test_the_first_tier_whose_checks_pass_is_the_one_that_applies():
    """The rule in one sentence, gated as three answers to it. The order IS the policy: a ticket
    every tier would admit falls in the FIRST, and one no tier before the catch-all admits falls in
    the catch-all - which is a tier declaring no check rather than a default this module holds.

    The verdict reports what it read, per tier, with the value measured and the bound declared, so
    a desk shown a route never has to re-derive the comparison to believe it.

    Killing mutation: the scan run on past a tier that passed, so the last passing tier applies.
    """
    auto = tiers.assess(POLICY, ticket(), STANDING)
    assert sorted(auto) == ['checks', 'refusals', 'tier'] and auto['tier'] == 'auto'
    assert auto['refusals'] == []
    assert [(row['check'], row['value'], row['bound']) for row in auto['checks']] == [
        ('max_notional', 1_000_000.0, CAP), ('max_tenor_years', 1.0, 2.0),
        ('market', VALUES, 'official')]
    assert all(row['passed'] and row['tier'] == 'auto' for row in auto['checks'])

    senior = tiers.assess(POLICY, ticket(usd=10_000_000.0), STANDING)
    assert senior['tier'] == 'senior'
    assert read(senior, 'auto', 'max_notional') == {
        'tier': 'auto', 'check': 'max_notional', 'value': 10_000_000.0, 'bound': CAP,
        'passed': False}
    assert len(senior['refusals']) == 1 and "'auto'" in senior['refusals'][0]
    assert '5000000.0 USD' in senior['refusals'][0] and '10000000.0' in senior['refusals'][0]
    # the tier that failed is read whole: a check after the one that failed is still reported
    assert [row['tier'] for row in senior['checks']] == ['auto', 'auto', 'auto', 'senior']

    board = tiers.assess(POLICY, ticket(usd=100_000_000.0, tenor=5.0), STANDING)
    assert board['tier'] == 'board'
    assert len(board['refusals']) == 3, board['refusals']
    assert not any(row['tier'] == 'board' for row in board['checks']), \
        'the catch-all declares no check, so there was nothing to read'


def test_a_cap_is_a_maximum_and_a_ticket_sitting_on_one_passes_it():
    """WHICH SIDE THE LINE IS ON, said out loud on every bound the evaluator has. A cap is a
    MAXIMUM: a ticket exactly at it passes and the smallest step over it does not, on the notional
    and on the tenor alike, so a later hand cannot move a `<=` to a `<` without this going red.

    A notional is an AMOUNT of a currency and never a sign: a zero or a negative one is refused by
    name rather than clearing every cap there is, which is what a comparison against a magnitude
    the caller got backwards would otherwise do.

    Killing mutation: the notional cap compared with `<`.
    """
    assert tiers.assess(POLICY, ticket(usd=5_000_000.0), STANDING)['tier'] == 'auto'
    assert tiers.assess(POLICY, ticket(usd=5_000_000.01), STANDING)['tier'] == 'senior'
    assert tiers.assess(POLICY, ticket(tenor=2.0), STANDING)['tier'] == 'auto'
    assert tiers.assess(POLICY, ticket(tenor=2.01), STANDING)['tier'] == 'senior'
    # a tier capping the tenor at nothing admits a tenor of nothing, by the same rule
    assert tiers.assess({'tiers': [{'name': 'spot', 'max_tenor_years': 0.0}]},
                        ticket(tenor=0.0), STANDING)['tier'] == 'spot'

    for wrong in (0.0, -1.0, -5_000_000.0):
        with pytest.raises(MalformedEvent) as refusal:
            tiers.assess(POLICY, ticket(usd=wrong), STANDING)
        assert 'never a sign' in str(refusal.value) and 'USD' in str(refusal.value), wrong

    # and the hash comparison is a comparison of HASHES: two strings that are not addresses do not
    # agree their way past a market check
    assert tiers.assess(POLICY, ticket(values_hash='x'), {'official': 'x'})['tier'] == 'senior'


def test_a_notional_the_caps_currency_cannot_see_fails_that_tier_and_says_so():
    """The safe direction, and the reason it is safe. A cap is an amount OF a currency, so a ticket
    that does not state its notional in that currency is a ticket this check cannot make - and a
    check that could not be made is a failure rather than a pass. Crossing one currency into
    another would make a policy check read a market, which is the thing policy-as-data forbids.

    Killing mutation: an unstated notional read as a pass.
    """
    rand = tiers.assess(POLICY, ticket(usd=None, ZAR=18_500_000.0), STANDING)

    assert rand['tier'] == 'board', 'a cap nothing could compare against admitted the ticket'
    # `bound` is the DECLARED bound whichever way the check failed - a reader rendering it as an
    # amount never gets a currency where a number belongs
    assert read(rand, 'auto', 'max_notional') == {
        'tier': 'auto', 'check': 'max_notional', 'value': None, 'bound': CAP, 'passed': False}
    for said in rand['refusals'][:2]:
        assert 'not stated in USD' in said, said
    assert "'auto'" in rand['refusals'][0] and "'senior'" in rand['refusals'][1]

    # a tenor the ticket does not carry is the same answer for the same reason
    undated = tiers.assess(POLICY, dict(ticket(), tenor_years=None), STANDING)
    assert undated['tier'] == 'senior'
    assert 'unknown tenor' in undated['refusals'][0] and '2.0 years' in undated['refusals'][0]


def test_a_market_nobody_declared_fails_the_tier_that_prices_on_it():
    """A tier naming a market is a tier that prices on a NAME, and the name is resolved off the
    `markets` fold rather than off whatever the book is carrying. Nothing declared under it fails
    by name; a name standing on another vector fails naming both; and the declared one passes on
    the hash, which is the whole check.

    Killing mutation: an undeclared market read as a pass.
    """
    unmarked = tiers.assess(POLICY, ticket(), {})
    assert unmarked['tier'] == 'senior'
    assert 'official' in unmarked['refusals'][0] and 'nothing in this record' in \
        unmarked['refusals'][0]

    moved = tiers.assess(POLICY, ticket(), {'official': MOVED})
    assert moved['tier'] == 'senior'
    assert VALUES in moved['refusals'][0] and MOVED in moved['refusals'][0]
    assert read(moved, 'auto', 'market')['value'] == VALUES

    assert tiers.assess(POLICY, ticket(), STANDING)['tier'] == 'auto'


def test_the_evaluator_is_pure_over_plain_data():
    """No log, no clock, no store, no home - asserted on the SIGNATURES and on the module's own
    imports rather than on the docstring, so an evaluator that learned to read the record turns
    this red the day it does.

    The import half is the stronger claim: the module names nothing outside its own package, so
    there is no handle here to reach a home with even if a signature grew one.

    Killing mutation: the evaluator importing `time`.
    """
    for named in (tiers.assess, tiers.check, tiers.standing_approval):
        parameters = set(inspect.signature(named).parameters)
        assert parameters.isdisjoint({'log', 'clock', 'now', 'store', 'home'}), named.__name__

    source = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'derivus_spine', 'tiers.py')
    with open(source, encoding='utf-8') as handle:
        tree = ast.parse(handle.read(), filename=source)
    absolute = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            absolute.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level:
            absolute.add(node.module or '')
    assert absolute == set(), sorted(absolute)

    # and the document it is handed is the one a fold answers: a home that declared none has
    # declared no workflow, and this refuses rather than inventing one
    with pytest.raises(MalformedEvent) as refusal:
        tiers.assess(None, ticket(), STANDING)
    assert 'declared no workflow' in str(refusal.value)

    # a ticket or a market table of the wrong TYPE raises where a field the ticket simply does not
    # state fails its tier - the two are different mistakes with different remedies
    for wrong in (('a ticket', STANDING), (ticket(), None),
                  (dict(ticket(), notional_in='a million'), STANDING)):
        with pytest.raises(MalformedEvent) as refusal:
            tiers.assess(POLICY, *wrong)
        assert 'half of one' in str(refusal.value), wrong

    # and the POLICY is guarded the way the ticket is: a row that is not a named tier refuses by
    # name, since a hand-built document is what this module is handed everywhere but `in_force`
    for broken in ({'tiers': ['auto']}, {'tiers': [None]},
                   {'tiers': [{'max_tenor_years': 5}]}, {'tiers': [dict(name=17)]}):
        with pytest.raises(MalformedEvent) as refusal:
            tiers.assess(broken, ticket(), STANDING)
        assert 'not a named tier' in str(refusal.value), broken

    # and every BOUND it would compare a ticket against, the way `firmness._windows` checks each
    # window rather than the pair - a cap that is a number, a tenor that is not, a market that is
    for broken in ({'name': 'auto', 'max_notional': 5_000_000.0},
                   {'name': 'auto', 'max_notional': {'amount': 1.0}},
                   {'name': 'auto', 'max_tenor_years': 'two years'},
                   {'name': 'auto', 'market': 17}):
        with pytest.raises(MalformedEvent) as refusal:
            tiers.assess({'tiers': [broken]}, ticket(), STANDING)
        assert 'not a bound' in str(refusal.value), broken


# --------------------------------------------------------------------------------------------
# The standing approval: the latest verdict, and who may have filed it.

def test_the_latest_verdict_stands_and_four_eyes_reads_who_filed_it():
    """A verdict is never withdrawn, so what stands is what was filed LAST - by LSN, so the order a
    caller hands the list in cannot change the answer. Four eyes is a rule about subjects on one
    ticket: the booker's own verdicts are not read, and the same approval stands where the tier
    does not ask for it.

    Killing mutation: the booker's verdict read as the latest under four eyes, which lets a booker
    withdraw another seat's rejection by approving after it.
    """
    four_eyes = POLICY['tiers'][1]
    open_tier = {'name': 'senior'}

    unfiled, why = tiers.standing_approval(four_eyes, BOOKER, [])
    assert unfiled is None and 'no verdict is filed' in why and "'senior'" in why

    rejected = [verdict('approval', APPROVER, 8),
                verdict('rejection', APPROVER, 9, 'the terms moved under it')]
    lsn, why = tiers.standing_approval(four_eyes, BOOKER, rejected)
    assert lsn is None and 'rejection' in why and 'LSN 9' in why
    assert 'the terms moved under it' in why

    assert tiers.standing_approval(four_eyes, BOOKER, rejected + [
        verdict('approval', APPROVER, 10)]) == (10, None)
    # handed newest first, the answer does not move: the latest is the latest by LSN
    assert tiers.standing_approval(four_eyes, BOOKER, list(reversed(rejected + [
        verdict('approval', APPROVER, 10)]))) == (10, None)

    own = [verdict('approval', BOOKER, 11)]
    lsn, why = tiers.standing_approval(four_eyes, BOOKER, own)
    assert lsn is None and 'one seat' in why and BOOKER in why
    assert tiers.standing_approval(open_tier, BOOKER, own) == (11, None), \
        'a tier that does not ask for four eyes refused the booker anyway'

    # FOUR EYES DOES NOT READ THE BOOKER AT ALL, whichever end of the list its rows sit at: its
    # approval after another seat's rejection withdraws nothing, and another seat's approval stands
    # whether the booker's own verdict came before it or after it.
    behind = [verdict('rejection', APPROVER, 8, 'the terms moved under it'),
              verdict('approval', BOOKER, 11)]
    for handed in (behind, list(reversed(behind))):
        lsn, why = tiers.standing_approval(four_eyes, BOOKER, handed)
        assert lsn is None and 'rejection' in why and 'LSN 8' in why, why
    ahead = [verdict('approval', BOOKER, 8), verdict('approval', APPROVER, 11)]
    after = [verdict('approval', APPROVER, 8), verdict('rejection', BOOKER, 11, 'withdrawn')]
    for handed, standing in ((ahead, 11), (list(reversed(ahead)), 11), (after, 8)):
        assert tiers.standing_approval(four_eyes, BOOKER, handed) == (standing, None), handed


def test_a_tier_covering_a_node_is_read_for_a_ticket_booking_into_it_and_skipped_otherwise():
    """A TIER'S SCOPE IS A NODE, read before its checks: a ticket booking into `BANK/FX` or under
    it may fall in a tier covering `BANK/FX`, and one booking into `BANK/FXO`, into `BANK` above it
    or naming no portfolio at all falls through to the next tier with the route saying why - a
    tier covering nothing the ticket books into is not a tier it can be signed under. A tier
    covering `*` covers every ticket, as a grant at `*` reaches every node.

    Killing mutation: `*` read as a node's name, which routes nothing through it.
    """
    scoped = {'tiers': [{'name': 'fx', 'scope': 'BANK/FX', 'four_eyes': True}, {'name': 'board'}]}
    for portfolio, expected in (('BANK/FX', 'fx'), ('BANK/FX/Options', 'fx'),
                                ('BANK/FXO', 'board'), ('BANK', 'board'), (None, 'board')):
        verdict = tiers.assess(scoped, dict(ticket(), portfolio=portfolio), STANDING)
        assert verdict['tier'] == expected, (portfolio, verdict)
        assert read(verdict, 'fx', 'scope') == {'tier': 'fx', 'check': 'scope', 'value': portfolio,
                                                'bound': 'BANK/FX', 'passed': expected == 'fx'}
        if expected == 'board':
            assert "the 'fx' tier covers BANK/FX" in verdict['refusals'][0], verdict['refusals']
    everything = policy.parse_tiers({'tiers': [{'name': 'all', 'scope': '*'}]}, 'the firm')
    assert [tiers.assess(everything, dict(ticket(), portfolio=portfolio), STANDING)['tier']
            for portfolio in ('BANK', 'BANK/FX', None)] == ['all', 'all', 'all']


def test_check_raises_tier_refused_carrying_the_sentences_assess_lists():
    """`assess` returns and `check` raises, the two answering the same thing - so a caller that
    wants the route reads it and a caller that wants the refusal catches it, and neither of them
    holds a second copy of what a bound means.

    Killing mutation: the refusal carrying the last sentence of the route alone.
    """
    closed = {'tiers': POLICY['tiers'][:2]}
    over = ticket(usd=100_000_000.0)

    assert tiers.check(POLICY, over, STANDING)['tier'] == 'board'
    listed = tiers.assess(closed, over, STANDING)
    assert listed['tier'] is None and len(listed['refusals']) == 2

    with pytest.raises(TierRefused) as refusal:
        tiers.check(closed, over, STANDING)
    said = str(refusal.value)
    assert 'falls in no tier' in said
    for sentence in listed['refusals']:
        assert sentence in said, sentence
