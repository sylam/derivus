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

"""The folds, gated on the synthetic book and on a second fixture for what it cannot say.

Every gate here drives a real home in a temp directory through the ordinary writer. The first book
is the design's own (`tests/test_spine.py`), imported rather than written again, whose late booking,
backdated amendment, republished print and restated close make as-of and as-at disagree; the second
holds the facts that one cannot - a position closed to zero, one replay tuple attested twice, one
policy declared twice, a print superseded twice, and two verdicts filed in the order they disagree
in. Nothing is patched: a fold that needed a doctored library to pass would be a fold nobody could
run.

The golden per projector is the row shape as a committed file. One value in a golden is MINTED by a
home rather than stated by the record - the checkpoint verifying key generated at genesis - and the
writer's `record_time` is stamped per run; both are asserted against the log itself before they are
replaced, so a golden compares what the fold derived and nothing a home invented.
"""
import hashlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from derivus_spine import (
    CapabilityDenied, MalformedEvent, SpineLog, SpineRefusal, UnknownEventType, WriterBusy,
    canonical_bytes, projections, verify_home)
from derivus_spine.capability import CAPABILITIES_POLICY, canonical_document
from derivus_spine.genesis import VERIFYING_KEY_POLICY
from derivus_spine.policy import FIXINGS_POLICY, TOLERANCE_POLICY, declare
from derivus_spine.projections import (
    PROJECTORS, SEEDS, SUMMARIES, fixings_at, fold, knocked, read_seed, seed_at)
from derivus_spine.verbs import STANDING, apply_lifecycle, complete_run, file_quote
from derivus_spine.vocabulary import ADMIN, EVENT_TYPES

import test_spine_imports as imports
from test_spine import (
    ACTOR, BOOK, INSTRUMENT, MON, TUE, WED, fill, seeded, synthetic_book)

#: Where the synthetic book stands, and the two positions its restatement turns on: the first
#: official close, and the head. The republished print sits at 16, between them.
HEAD = 21
CLOSE = 15
#: The key the book's observations are filed under, and the administrators that print it.
INDEX = 'EURUSD-ECB'
DATE = '2026-08-26'
ECB = 'ECB'
BFIX = 'BFIX'

GOLDENS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures', 'spine_goldens')
#: What the two values no fold derives stand as in a golden.
STAMPED = 'the writer stamped this'
MINTED = 'minted at genesis'


def published(log):
    """The blob the genesis declaration of the checkpoint verifying key names - minted per home."""
    for frame in log.frames():
        if frame['event_type'] != 'policy_declared':
            continue
        body = log.open_body(frame)
        if body.get('policy') == VERIFYING_KEY_POLICY:
            return body['blob']
    return None


def stated(name, rows, log):
    """`rows` with what no fold derives replaced by its own name, each ASSERTED against the log
    first: every activity row's record_time against the frame it was read from, and the verifying
    key's address against the declaration that published it."""
    if name == 'activity':
        stamped = dict((frame['lsn'], frame['record_time']) for frame in log.frames())
        for row in rows:
            assert row['record_time'] == stamped[row['lsn']], row['lsn']
            row['record_time'] = STAMPED
    if name == 'decisions':
        for row in rows['policies']:
            if row['policy'] == VERIFYING_KEY_POLICY:
                assert row['blob'] == published(log), 'the key genesis published'
                row['blob'] = MINTED
    return rows


def golden(name):
    """The committed rows for `name`."""
    with open(os.path.join(GOLDENS, name + '.json'), encoding='utf-8') as handle:
        return json.load(handle)


def closed_book(tmp_path):
    """The second fixture: a clip and the clip that closes it, a rejection filed before the approval
    it argues with, three prints of one key, one policy declared twice, and one replay tuple
    attested twice. Answers `(home, log, marks)`.
    """
    home = seeded(tmp_path, 'closed', clips=())
    log = SpineLog(home)
    marks = {'plan': hashlib.sha256(b'the plan two seats disagreed about').hexdigest()}
    values = b'{"EURUSD":1.0851}'
    marks['values'] = log.store.put(values)
    log.append('fill', fill('EXEC-OPEN'), actor=ACTOR, book=BOOK, effective_time=MON)
    log.append('fill', fill('EXEC-CLOSE', quantity=-1000000.0), actor=ACTOR, book=BOOK,
               effective_time=TUE)
    log.append('rejection', {'plan_hash': marks['plan'], 'reason': 'the terms moved under it'},
               actor='subject-desk-two', book=BOOK, effective_time=TUE)
    log.append('approval', {'plan_hash': marks['plan']}, actor=ACTOR, book=BOOK, effective_time=TUE)
    for value, when in ((1.0851, MON), (1.0857, TUE), (1.0860, WED)):
        log.append('fixing_observed',
                   {'index': INDEX, 'date': DATE, 'source': ECB, 'value': value},
                   actor='subject-administrator', effective_time=when)
    declare(log, ACTOR, TOLERANCE_POLICY, {'tolerances': {'mtm': 1e-9}})
    marks['tolerance'] = declare(log, ACTOR, TOLERANCE_POLICY, {'tolerances': {'mtm': 1e-6}})
    claim = {'plan_hash': marks['plan'], 'values_hash': marks['values'],
             'engine_version': 'derivus-1.0', 'seed': 1}
    marks['attested'] = complete_run(log, ACTOR, STANDING, claim, b'{"Calc": {"first": true}}',
                                     values, b'{"mtm": 1.0}', book=BOOK)
    complete_run(log, ACTOR, STANDING, claim, b'{"Calc": {"second": true}}', values,
                 b'{"mtm": 2.0}', book=BOOK)
    file_quote(log, 'subject-desk-two', 'Q-1', 'ZeroCostCollar', marks['plan'], values,
               {'floor': 17.10}, 4100.0, ticket=marks['plan'], book=BOOK)
    file_quote(log, ACTOR, 'Q-2', 'Straddle', marks['plan'], values, {}, 0.0, book=BOOK)
    # an id that sorts BEFORE Q-1 and is struck after it, so LSN order and alphabetical order part
    file_quote(log, ACTOR, 'A-9', 'Straddle', marks['plan'], values, {}, 0.0, book=BOOK)
    # the same id restated, and then a BACKDATED restatement that does not win by arriving last
    marks['quoted'] = file_quote(log, 'subject-desk-two', 'Q-1', 'ZeroCostCollar', marks['plan'],
                                 values, {'floor': 17.25}, 4200.0, ticket=marks['plan'], book=BOOK)
    file_quote(log, 'subject-desk-two', 'Q-1', 'ZeroCostCollar', marks['plan'], values,
               {'floor': 16.00}, 9999.0, ticket=marks['plan'], book=BOOK, effective_time=MON)
    return home, log, marks


class PositionsOnward(projections.Positions):
    """The positions projector one version on, reading the same bodies."""

    version = projections.Positions.version + 1


def test_every_projector_replays_to_its_committed_golden(tmp_path):
    """Fold equals materialisation: burn the state, refold from genesis, and require the rows to be
    the committed ones in canonical form - so a row reordered, a field renamed or a value derived
    differently is a red gate rather than a silent change of what a reader sees. The strip's table
    of sentences is committed the same way, because it is data the fixture cannot exercise."""
    home, log, marks = synthetic_book(tmp_path)
    assert set(PROJECTORS) == {'activity', 'agreements', 'attestations', 'blotter', 'decisions',
                               'denials', 'entities', 'lifecycle', 'markets', 'positions',
                               'quotes'}

    for name in sorted(PROJECTORS):
        projector = PROJECTORS[name]
        rows = stated(name, projector.rows(fold(log, projector)), log)
        assert canonical_bytes(rows) == canonical_bytes(golden(name)), name

    # the strip carries a sentence for every type the writer can hold, and each is the one the
    # committed table names; a type it does not know renders its own name rather than dropping out
    assert canonical_bytes(SUMMARIES) == canonical_bytes(golden('summaries'))
    assert set(SUMMARIES) == set(EVENT_TYPES)
    assert PROJECTORS['activity'].reads is None
    log.close()


def test_a_seeded_fold_is_the_from_genesis_fold_byte_for_byte(tmp_path):
    """Seed equivalence, for every projector: the state cached at the close and advanced to the head
    is the state folded from genesis, in canonical bytes, as are the rows off it. The fold is PURE
    in its seed - the same seed folds twice to the same answer and is left where it was - and a
    position behind the seed refuses rather than answering the state in front of it."""
    home, log, marks = synthetic_book(tmp_path)
    assert log.head()[0] == HEAD

    for name in sorted(PROJECTORS):
        projector = PROJECTORS[name]
        whole = fold(log, projector, lsn=HEAD)
        seed = seed_at(log, projector, CLOSE)
        seeded_fold = fold(log, projector, lsn=HEAD, seed=seed)
        assert canonical_bytes(seeded_fold) == canonical_bytes(whole), name
        assert canonical_bytes(projector.rows(seeded_fold)) \
            == canonical_bytes(projector.rows(whole)), name
        assert canonical_bytes(fold(log, projector, lsn=HEAD, seed=seed)) \
            == canonical_bytes(whole), name
        assert seed['lsn'] == CLOSE, 'the seed the caller holds is where it was'

    with pytest.raises(SpineRefusal) as behind:
        fold(log, PROJECTORS['activity'], lsn=CLOSE - 1, seed=seed_at(log, PROJECTORS['activity'],
                                                                     CLOSE))
    assert 'walks forward' in str(behind.value)
    with pytest.raises(SpineRefusal) as position:
        seed_at(log, PROJECTORS['positions'], 16)
    assert 'fixing_observed' in str(position.value) and 'official_close_declared' in str(
        position.value)
    log.close()


def test_the_seeded_fold_sees_the_restatement_behind_its_seed(tmp_path):
    """A seed is a starting point and never an answer. At the close the official market is the first
    values vector; folded on to the head it is the restated one, with the close it stands over
    named."""
    home, log, marks = synthetic_book(tmp_path)
    markets = PROJECTORS['markets']

    seed = seed_at(log, markets, CLOSE)
    at_close = markets.rows(seed['state'])['closes'][0]
    assert at_close['values_hash'] == marks['values'] and at_close['supersedes_lsn'] is None

    standing = markets.rows(fold(log, markets, seed=seed))['closes'][0]
    assert standing['values_hash'] == marks['restated']
    assert standing['supersedes_lsn'] == CLOSE and standing['lsn'] == 17
    log.close()


def test_a_republished_print_supersedes_by_as_of_key_and_the_first_is_still_read(tmp_path):
    """Supersession in both directions and under the whole key. The administrator's republication of
    one (index, date, source) wins by `(effective_time, lsn)` and the print it beat stays on the row;
    a print BACKDATED behind the one standing does not win merely by arriving last; and a second
    administrator's print of the same index and date is a SECOND row, because a source that was
    never asked does not supersede the one that was."""
    home, log, marks = synthetic_book(tmp_path)
    log.append('fixing_observed', {'index': INDEX, 'date': DATE, 'source': BFIX, 'value': 1.0900},
               actor='subject-administrator', effective_time=WED)
    log.append('fixing_observed', {'index': INDEX, 'date': DATE, 'source': ECB, 'value': 1.0800},
               actor='subject-administrator', effective_time=TUE)

    assert fixings_at(log, sources={INDEX: [ECB]})[(INDEX, DATE)]['value'] == 1.0857
    assert fixings_at(log, lsn=CLOSE, sources={INDEX: [ECB]})[(INDEX, DATE)]['value'] == 1.0851

    rows = PROJECTORS['lifecycle'].rows(fold(log, PROJECTORS['lifecycle']))['fixings']
    assert [(row['source'], row['value']) for row in rows] == [(BFIX, 1.0900), (ECB, 1.0857)]
    assert rows[1]['supersedes'] == [{'value': 1.0851, 'lsn': 14}, {'value': 1.0800, 'lsn': 23}]
    assert rows[1]['lsn'] == 16 and log.open_body(log.frame_at(14))['value'] == 1.0851
    log.close()


def test_a_knock_is_read_off_the_fold_and_is_never_a_fact(tmp_path):
    """Consequence purity. The record holds the observations; the crossing is derived where it is
    asked for, the log carries no knock, and the writer refuses to file one."""
    home = seeded(tmp_path, clips=())
    log = SpineLog(home)
    for date, value in (('2026-08-24', 1.0851), ('2026-08-25', 1.0920), ('2026-08-26', 1.0880)):
        apply_lifecycle(log, ACTOR, 'fixing_observed',
                        {'index': INDEX, 'date': date, 'source': ECB, 'value': value},
                        book=BOOK, effective_time=WED)

    observed = fixings_at(log, sources={INDEX: [ECB]})
    assert knocked(observed, INDEX, 1.09) == {'knocked': True, 'knocked_on': '2026-08-25'}
    assert knocked(observed, INDEX, 1.20) == {'knocked': False, 'knocked_on': None}
    assert knocked(observed, INDEX, 1.086, rising=False) == {'knocked': True,
                                                             'knocked_on': '2026-08-24'}

    assert set(frame['event_type'] for frame in log.frames()) == {
        'policy_declared', 'checkpoint', 'fixing_observed'}
    with pytest.raises(UnknownEventType):
        apply_lifecycle(log, ACTOR, 'knocked_out', {'instrument': INSTRUMENT, 'choice': 'out'})
    assert log.head()[0] == 7, 'the refusal and the fold both left the head where it was'
    log.close()


def test_facts_sharing_an_effective_time_fold_in_lsn_order(tmp_path):
    """Determinism: the fold is a function of the log, not of a sort's tie-breaking. The two prints
    of the republished key share one truth-time and the later LSN stands; three clips at one instant
    fold into one row in the order they were written."""
    home, log, marks = synthetic_book(tmp_path)
    assert log.frame_at(14)['effective_time'] == log.frame_at(16)['effective_time']
    assert fixings_at(log, sources={INDEX: [ECB]})[(INDEX, DATE)]['lsn'] == 16

    rows = PROJECTORS['activity'].rows(fold(log, PROJECTORS['activity']))
    assert [row['lsn'] for row in rows] == list(range(1, HEAD + 1))
    log.close()

    clips = SpineLog(seeded(tmp_path, 'clips'))
    position = PROJECTORS['positions'].rows(fold(clips, PROJECTORS['positions']))[0]
    assert (position['clips'], position['first_lsn'], position['last_lsn']) == (3, 5, 7)
    assert PROJECTORS['blotter'].rows(fold(clips, PROJECTORS['blotter']))[0]['last_fact']['lsn'] == 7
    clips.close()


def test_a_projector_one_version_on_folds_the_same_bodies_and_a_seed_is_verified_not_trusted(
        tmp_path):
    """Version tolerance, and the seed as evidence rather than as state. A projector whose rows
    moved on still folds the bodies an earlier writer wrote; the seed minted by the version before
    it refuses where it is read AND where it is folded; and a seed minted over another home's close,
    one torn, or one whose state was edited beneath an intact close, refuses by name instead of
    folding a fiction nothing else can detect."""
    home, log, marks = synthetic_book(tmp_path)
    shipped, onward = PROJECTORS['positions'], PositionsOnward()
    filed_name = 'positions-{}-{}.json'.format(shipped.version, CLOSE)

    assert canonical_bytes(onward.rows(fold(log, onward))) == canonical_bytes(
        shipped.rows(fold(log, shipped)))
    assert read_seed(log, shipped, CLOSE) is None

    seed = seed_at(log, shipped, CLOSE)
    assert read_seed(log, shipped, CLOSE) == seed
    with pytest.raises(SpineRefusal) as filed:
        read_seed(log, onward, CLOSE)
    assert filed_name in str(filed.value)
    assert 'version {}'.format(onward.version) in str(filed.value)
    with pytest.raises(SpineRefusal) as handed:
        fold(log, onward, seed=seed)
    assert 'version {}'.format(shipped.version) in str(handed.value)
    assert 'version {}'.format(onward.version) in str(handed.value)
    with pytest.raises(SpineRefusal) as other:
        fold(log, shipped, seed={'projector': 'blotter', 'version': 1, 'lsn': 6, 'state': {}})
    assert "'blotter'" in str(other.value)

    elsewhere, elsewhere_log, _ = synthetic_book(tmp_path / 'elsewhere')
    seed_at(elsewhere_log, shipped, CLOSE)
    elsewhere_log.close()
    filed_at = home / SEEDS / filed_name
    filed_at.write_bytes((elsewhere / SEEDS / filed_name).read_bytes())
    with pytest.raises(SpineRefusal) as foreign:
        read_seed(log, shipped, CLOSE)
    assert 'not that history' in str(foreign.value)

    filed_at.write_bytes(b'{"projector": "positions", "vers')
    with pytest.raises(SpineRefusal) as torn:
        read_seed(log, shipped, CLOSE)
    assert 'is not JSON' in str(torn.value)

    seed_at(log, shipped, CLOSE)
    doctored = json.loads(filed_at.read_text(encoding='utf-8'))
    doctored['state'][sorted(doctored['state'])[0]]['quantity'] = 42.0
    filed_at.write_text(json.dumps(doctored), encoding='utf-8')
    with pytest.raises(SpineRefusal) as edited:
        read_seed(log, shipped, CLOSE)
    assert 'edited beneath its own close' in str(edited.value)
    log.close()


def test_a_fold_never_claims_the_home_and_sees_the_next_append(tmp_path):
    """Reading never claims: a handle folding beside a writer sees that writer's next frame on its
    next fold, while a second WRITER is refused. The attestation the run files lands on the fold
    that reads `run_completed` and nowhere else."""
    home, log, marks = synthetic_book(tmp_path)
    reader = SpineLog(home)
    attestations = PROJECTORS['attestations']
    assert attestations.rows(fold(reader, attestations)) == []

    claim = {'plan_hash': marks['plan'], 'values_hash': marks['restated'],
             'engine_version': 'derivus-1.0', 'seed': 1}
    complete_run(log, ACTOR, STANDING, claim, b'{"Calc": {}}', b'{"EURUSD":1.0857}',
                 b'{"mtm": 1.0}', book=BOOK)

    rows = attestations.rows(fold(reader, attestations))
    assert len(rows) == 1 and rows[0]['lsn'] == HEAD + 1
    assert rows[0]['plan_hash'] == marks['plan'] and rows[0]['lane'] == STANDING
    with pytest.raises(WriterBusy):
        SpineLog(home).append('rehash_declared', {'algorithm': 'sha256'}, actor=ACTOR)
    log.close()


def test_a_fixing_with_no_declared_source_refuses_and_the_order_is_the_authority(tmp_path):
    """The `fixings` policy is the authority an observation is read under. A home declaring none has
    named an authority for nothing and resolves nothing; once one is declared, an index it does not
    name refuses BY NAME where the caller asked for that index and is left alone where it did not.
    The order picks the print, never the administrator who printed last."""
    home, log, marks = synthetic_book(tmp_path)
    log.append('fixing_observed', {'index': INDEX, 'date': DATE, 'source': BFIX, 'value': 1.0900},
               actor='subject-administrator', effective_time=WED)

    assert fixings_at(log) == {}, 'a home declaring no fixings policy resolves nothing'

    declare(log, ACTOR, FIXINGS_POLICY, {'sources': {'FxRate.USD': [ECB]}})
    with pytest.raises(SpineRefusal) as refused:
        fixings_at(log)
    assert INDEX in str(refused.value) and FIXINGS_POLICY in str(refused.value)
    assert BFIX in str(refused.value) and ECB in str(refused.value)
    assert fixings_at(log, indices=['FxRate.USD']) == {}, 'an index nobody asked about is let be'

    declare(log, ACTOR, FIXINGS_POLICY, {'sources': {INDEX: [BFIX, ECB]}})
    stood = fixings_at(log)[(INDEX, DATE)]
    assert (stood['source'], stood['value'], stood['lsn']) == (BFIX, 1.0900, 22)
    assert fixings_at(log, sources={INDEX: [ECB, BFIX]})[(INDEX, DATE)]['source'] == ECB

    with pytest.raises(MalformedEvent):
        declare(log, ACTOR, FIXINGS_POLICY, {'sources': {INDEX: ECB}})
    log.close()


def test_a_refused_append_is_readable_as_a_row(tmp_path):
    """The `denials` fold, which is what makes "nothing outside its seat's verb" a reading rather
    than a body walk. The writer already files a refusal as a fact; the envelope says only that one
    happened, and the strip renders one declared sentence for every denial alike, so WHO was
    refused WHICH verb over WHAT scope is inside the body and nowhere else.

    The committed golden is `[]`, the design's synthetic book declaring no capabilities document
    and so refusing nobody - the hole `quotes` has too - so the rows are driven here: two seats
    turned away, one of them twice, and a second refusal of one fact coalescing onto the LSN it
    already has by the ordinary tag rule.
    """
    home = seeded(tmp_path, 'refused', clips=())
    log = SpineLog(home)
    blob = log.store.put(canonical_document({
        'grants': [{'subject': ACTOR, 'verb': ADMIN, 'book': '*'}],
        'read': [{'subject': ACTOR, 'class': 'firm'}]}))
    log.append('policy_declared', {'policy': CAPABILITIES_POLICY, 'blob': blob}, actor=ACTOR,
               blob_refs=(blob,))

    for attempt in range(2):
        with pytest.raises(CapabilityDenied):
            log.append('fill', fill('EXEC-STRANGER'), actor='subject-nobody', book=BOOK)
    with pytest.raises(CapabilityDenied):
        log.append('market_declared', {'name': 'official', 'values_hash': blob}, actor=ACTOR)

    rows = PROJECTORS['denials'].rows(fold(log, PROJECTORS['denials']))
    assert [(row['subject'], row['verb'], row['book'], row['attempted_type']) for row in rows] == [
        ('subject-nobody', 'book', BOOK, 'fill'), (ACTOR, 'mark', '*', 'market_declared')], \
        'a repeated refusal is one fact and coalesces onto the LSN it already has'
    assert [row['lsn'] for row in rows] == [6, 7]
    log.close()


def test_the_import_surface_still_holds_and_a_subpackage_cannot_hide():
    """The dependency budget, asked of the new module and of every path a module could be added at:
    the gate's own list is every `.py` under the package however deep, so a subpackage cannot
    smuggle an import past it."""
    sources = imports.spine_sources()
    walked = sorted(os.path.join(base, name) for base, _, names in os.walk(imports.SPINE)
                    for name in names if name.endswith('.py'))
    assert sources == walked

    module = os.path.join(imports.SPINE, 'projections.py')
    assert module in sources
    imported = imports.imported_names(module)
    assert imported <= imports.ALLOWED and imported.isdisjoint(imports.FORBIDDEN)
    assert 'duckdb' in imports.FORBIDDEN, 'no reading plane lives in this package'


def test_the_seeds_directory_stands_beside_the_log_and_the_verifier_never_reads_it(tmp_path):
    """A seed is a file beside `log/` and `blobs/` that the verification does not know about: both
    modes answer exactly what they answered before one was minted, and deleting it costs a refold
    and nothing else."""
    home, log, marks = synthetic_book(tmp_path)
    positions = PROJECTORS['positions']
    entitled, chain = verify_home(home), verify_home(home, entitled=False)

    seed = seed_at(log, positions, CLOSE)
    assert (home / SEEDS).is_dir()
    assert verify_home(home) == entitled and verify_home(home, entitled=False) == chain

    filed = json.loads((home / SEEDS / 'positions-{}-{}.json'.format(
        positions.version, CLOSE)).read_text(encoding='utf-8'))
    assert filed == seed and filed['lsn'] == CLOSE
    for path in (home / SEEDS).iterdir():
        path.unlink()
    assert read_seed(log, positions, CLOSE) is None
    assert canonical_bytes(fold(log, positions, lsn=CLOSE)) == canonical_bytes(seed['state'])
    log.close()


def test_the_second_fixture_pins_what_the_synthetic_book_cannot_say(tmp_path):
    """A position closed out is a row and not an absence; the FIRST attestation of a replay tuple
    stands, whole, which is what `verbs.attestation` answers; the LAST declaration of a policy
    stands, which is what `policy.in_force` answers; a print superseded twice keeps both prints it
    beat; verdicts read in the order they were filed; and a quote names the SEAT that struck it,
    read off the envelope, with the ticket beside it where the caller computed one."""
    home, log, marks = closed_book(tmp_path)

    closed = PROJECTORS['positions'].rows(fold(log, PROJECTORS['positions']))
    assert [(row['instrument'], row['quantity'], row['clips']) for row in closed] == [
        (INSTRUMENT, 0.0, 2)], 'a position closed out is a row of the record, not an absence'

    attested = PROJECTORS['attestations'].rows(fold(log, PROJECTORS['attestations']))
    assert canonical_bytes(attested) == canonical_bytes([{
        'plan_hash': marks['plan'], 'values_hash': marks['values'], 'engine_version': 'derivus-1.0',
        'seed': 1, 'lane': STANDING, 'job': marks['attested']['job'],
        'result': marks['attested']['result'], 'lsn': 14}])

    decided = PROJECTORS['decisions'].rows(fold(log, PROJECTORS['decisions']))
    assert [(row['verdict'], row['lsn']) for row in decided['plans'][0]['verdicts']] == [
        ('rejection', 7), ('approval', 8)]
    standing = [row for row in decided['policies'] if row['policy'] == TOLERANCE_POLICY][0]
    assert (standing['blob'], standing['lsn']) == (marks['tolerance']['blob'], 13)

    printed = PROJECTORS['lifecycle'].rows(fold(log, PROJECTORS['lifecycle']))['fixings'][0]
    assert printed['value'] == 1.0860 and printed['lsn'] == 11
    assert printed['supersedes'] == [{'value': 1.0851, 'lsn': 9}, {'value': 1.0857, 'lsn': 10}]

    quoted = PROJECTORS['quotes'].rows(fold(log, PROJECTORS['quotes']))
    assert [row['quote_id'] for row in quoted] == ['Q-2', 'A-9', 'Q-1'], \
        'the rows read in LSN order, which is not their ids\' order'
    assert quoted[-1] == {'quote_id': 'Q-1', 'booker': 'subject-desk-two', 'book': BOOK,
                          'plan_hash': marks['plan'], 'values_hash': marks['values'],
                          'ticket': marks['plan'], 'structure': 'ZeroCostCollar',
                          'solved': {'floor': 17.25}, 'edge': 4200.0, 'effective_time': None,
                          'lsn': marks['quoted']['lsn']}, \
        'the LATER filing under one id stands, and the backdated one did not win by arriving last'
    assert quoted[0]['booker'] == ACTOR and quoted[0]['ticket'] is None, \
        'a quote filed without a ticket carries the absence rather than somebody else\'s hash'
    log.close()
