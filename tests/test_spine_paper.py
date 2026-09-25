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

"""Where a position sits, the paper it sits under, and the seed a reader starts from.

A fill says where its position sits - an agreement and a portfolio - and the positions fold keys by
both, so one instrument under two agreements, or in two portfolios, is two rows; a fill filed
before either existed sits under its netting set and its book. Legal entities and agreements are
declared facts, folded by their ids, a restatement standing by the as-of key and an agreement
citing its terms by address. And a seed filed in a folder the deployment names is where a reader
starts: the newest one at or behind its head, verified, its own projector's version only.

Every gate drives a real home in a temp directory through the ordinary writer; nothing is patched.
"""
import hashlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from derivus_spine import SpineLog, SpineRefusal, canonical_bytes, verify_home
from derivus_spine import cli
from derivus_spine.projections import (
    PROJECTORS, SEEDS, UNSEEDED, close_on, fold, fold_from, latest_seed, read_seed, seed_at)
from derivus_spine.verbs import book, declare_agreement, declare_entity

from test_spine import (ACTOR, BOOK, INSTRUMENT, MON, OTHER, TERMS, TUE, WED, fill, seeded,
                        synthetic_book)

#: The first official close of the synthetic book and the close that restates it, both true at
#: the same instant on the same day.
FIRST_CLOSE, RESTATED_CLOSE, CLOSE_DAY = 15, 17, '2026-08-26'
#: A netting set's terms as the store holds them - bytes the record addresses and never reads.
PAPER = b'{"Netted":"True","Object":"NettingCollateralSet","Reference":"ISDA-A"}'
RESTRUCK = b'{"Netted":"True","Object":"NettingCollateralSet","Reference":"ISDA-A","Settlement_Period":2}'


def keyed(rows):
    """The positions rows as `(instrument, agreement, portfolio, quantity, clips, amended_to)`."""
    return [(row['instrument'], row['agreement'], row['portfolio'], row['quantity'], row['clips'],
             row['amended_to']) for row in rows]


def test_a_position_is_keyed_where_it_sits(tmp_path):
    """One instrument, five clips, four positions - and an amendment carrying all four forward.

    A clip filed before the key existed sits under its netting set and its book. Two agreements are
    two positions however the portfolio reads, credit exposure being per agreement - which is what a
    back-to-back pair is, one instrument long under the client's paper and short under the
    dealer's. Two portfolios are two positions under one agreement. A partial unwind is a clip of
    the same key and nets it. The amendment moves every row under its own key, the old rows standing
    at zero naming where they went.

    Killing mutations: the key dropped back to the instrument alone folds the whole instrument into
    one row of 1.0; the agreement fallback to the netting set removed files the first clip under
    `None`; the portfolio fallback to the book removed files it under an empty key.
    """
    home = seeded(tmp_path, clips=())
    log = SpineLog(home)
    log.append('fill', fill('EXEC-V1', quantity=1.0), actor=ACTOR, book=BOOK, effective_time=MON)
    for reference, quantity, agreement, portfolio in (
            ('EXEC-OPEN', 1.0, 'ISDA-A', 'Rates/EM'),
            ('EXEC-UNWIND', -0.5, 'ISDA-A', 'Rates/EM'),
            ('EXEC-DEALER', -1.0, 'ISDA-B', 'Rates/EM'),
            ('EXEC-G10', 1.0, 'ISDA-A', 'Rates/G10')):
        book(log, ACTOR, TERMS, quantity, 'LEI-5493001KJTIIGC8Y1R12', agreement, reference,
             book=BOOK, effective_time=TUE, price=101.25, agreement=agreement,
             portfolio=portfolio)
    positions = PROJECTORS['positions']
    assert keyed(positions.rows(fold(log, positions))) == [
        (INSTRUMENT, 'CSA-0007', BOOK, 1.0, 1, None),
        (INSTRUMENT, 'ISDA-A', 'Rates/EM', 0.5, 2, None),
        (INSTRUMENT, 'ISDA-A', 'Rates/G10', 1.0, 1, None),
        (INSTRUMENT, 'ISDA-B', 'Rates/EM', -1.0, 1, None)]

    log.append('amendment', {'instrument': INSTRUMENT, 'amended_to': OTHER},
               actor=ACTOR, book=BOOK, effective_time=WED)
    moved = keyed(positions.rows(fold(log, positions)))
    assert moved == [
        (INSTRUMENT, 'CSA-0007', BOOK, 0.0, 0, OTHER),
        (INSTRUMENT, 'ISDA-A', 'Rates/EM', 0.0, 0, OTHER),
        (INSTRUMENT, 'ISDA-A', 'Rates/G10', 0.0, 0, OTHER),
        (INSTRUMENT, 'ISDA-B', 'Rates/EM', 0.0, 0, OTHER),
        (OTHER, 'CSA-0007', BOOK, 1.0, 1, None),
        (OTHER, 'ISDA-A', 'Rates/EM', 0.5, 2, None),
        (OTHER, 'ISDA-A', 'Rates/G10', 1.0, 1, None),
        (OTHER, 'ISDA-B', 'Rates/EM', -1.0, 1, None)], 'a row is never dropped'

    # the price rides the fill it was dealt on, and a body without the three validates as before
    bodies = [log.open_body(frame) for frame in log.frames() if frame['event_type'] == 'fill']
    assert [body.get('price') for body in bodies] == [None, 101.25, 101.25, 101.25, 101.25]
    assert set(bodies[0]) == {'instrument', 'quantity', 'counterparty', 'netting_set',
                              'execution_reference'}
    log.close()


def test_the_paper_folds_by_its_ids_and_a_restatement_names_what_it_stood_over(tmp_path):
    """An entity and an agreement are facts folded under their ids. A second declaration restates,
    standing by the as-of key, so a BACKDATED one lands on the platter and displaces nothing; an
    agreement's restatement names the declaration it stood over, as a close does, and its terms are
    a citation - the store holds them and a whole-home verification resolves them.

    Killing mutations: `_stand` bypassed for entities lets the backdated name win; the terms dropped
    from `BLOB_FIELDS` leaves the verification passing over an address nothing resolves.
    """
    home = seeded(tmp_path, clips=())
    log = SpineLog(home)
    declare_entity(log, ACTOR, 'LEI-PARENT', 'A group', effective_time=MON)
    declare_entity(log, ACTOR, 'LEI-X', 'A counterparty', parent='LEI-PARENT', effective_time=TUE)
    declare_entity(log, ACTOR, 'LEI-X', 'A counterparty, renamed', parent='LEI-PARENT',
                   effective_time=WED)
    declare_entity(log, ACTOR, 'LEI-X', 'a name backdated behind the one in force',
                   effective_time=MON)
    first = declare_agreement(log, ACTOR, 'ISDA-A', 'LEI-X', 'ISDA 2002 with CSA', PAPER,
                              effective_time=TUE)
    second = declare_agreement(log, ACTOR, 'ISDA-A', 'LEI-X', 'ISDA 2002 with CSA', RESTRUCK,
                               effective_time=WED)

    entities = PROJECTORS['entities'].rows(fold(log, PROJECTORS['entities']))
    assert [(row['entity'], row['name'], row['parent']) for row in entities] == [
        ('LEI-PARENT', 'A group', None),
        ('LEI-X', 'A counterparty, renamed', 'LEI-PARENT')]
    agreements = PROJECTORS['agreements'].rows(fold(log, PROJECTORS['agreements']))
    assert [(row['agreement'], row['entity'], row['kind'], row['terms'], row['supersedes_lsn'])
            for row in agreements] == [
        ('ISDA-A', 'LEI-X', 'ISDA 2002 with CSA', hashlib.sha256(RESTRUCK).hexdigest(),
         first['lsn'])]
    assert second['terms'] == hashlib.sha256(RESTRUCK).hexdigest()
    # as at the first declaration, the agreement reads the terms it was first declared under
    at_first = PROJECTORS['agreements'].rows(fold(log, PROJECTORS['agreements'], lsn=first['lsn']))
    assert at_first[0]['terms'] == first['terms'] and at_first[0]['supersedes_lsn'] is None
    assert log.store.get(first['terms']) == PAPER
    log.close()
    assert verify_home(home)['events'] == second['lsn']
    # a citation is a closure the verification holds: the terms gone, the home no longer verifies
    first_terms = first['terms']
    (home / 'blobs' / first_terms[:2] / first_terms[2:4] / first_terms).unlink()
    with pytest.raises(SpineRefusal) as lost:
        verify_home(home)
    assert first_terms in str(lost.value)


def test_a_reader_starts_at_the_newest_seed_a_named_folder_holds(tmp_path):
    """A seed filed where the deployment says - a folder every seat reads, not the home's own - is
    where a reader starts: the newest at or behind its head, verified, and the state it advances to
    is the from-genesis state byte for byte. A position behind every seed folds from genesis. A
    seed of another version is not a candidate, since a shared folder may hold two releases'; a
    close past the log's head is not one a copy behind it can stand at; and a seed of this version
    that does not verify refuses by name rather than being stepped over.
    """
    home, log, marks = synthetic_book(tmp_path)
    shared = tmp_path / 'shared'
    positions = PROJECTORS['positions']
    first = seed_at(log, positions, FIRST_CLOSE, shared)
    second = seed_at(log, positions, RESTATED_CLOSE, shared)
    assert not (home / SEEDS).exists(), 'a named folder is where the seed went'

    assert latest_seed(log, positions, shared) == second
    assert latest_seed(log, positions, shared, lsn=16) == first
    assert latest_seed(log, positions, shared, lsn=FIRST_CLOSE - 1) is None
    for lsn in (None, 16, FIRST_CLOSE - 1):
        assert canonical_bytes(fold_from(log, positions, lsn=lsn, folder=shared)) == \
            canonical_bytes(fold(log, positions, lsn=lsn))

    named = 'positions-{}-{}.json'.format(positions.version, RESTATED_CLOSE)
    (shared / 'positions-99-{}.json'.format(RESTATED_CLOSE)).write_bytes(
        (shared / named).read_bytes())
    (shared / 'positions-{}-999.json'.format(positions.version)).write_bytes(
        (shared / named).read_bytes())
    assert latest_seed(log, positions, shared) == second, 'another version and a later close'

    (shared / named).write_bytes(b'{"projector": "positions", "vers')
    with pytest.raises(SpineRefusal) as torn:
        latest_seed(log, positions, shared)
    assert 'is not JSON' in str(torn.value)
    log.close()


def test_the_seed_verb_files_every_fold_but_the_strip_at_a_named_day(tmp_path, capsys):
    """`DV_Spine seed --at <day> --out <folder>` stands at the last close true on that day - the
    restated one, which stands where two share an instant - and files every projector's seed but the
    strip's, whose state is its history. A day with no close, and an `--at` that is neither a day
    nor a position, refuse by name and write nothing."""
    home, log, marks = synthetic_book(tmp_path)
    assert close_on(log, CLOSE_DAY) == RESTATED_CLOSE and close_on(log) == RESTATED_CLOSE
    assert close_on(log, '2026-08-25') is None
    log.close()
    shared = tmp_path / 'shared'

    assert cli.main(['seed', '--home', str(home), '--at', CLOSE_DAY, '--out', str(shared)]) == 0
    answer = json.loads(capsys.readouterr().out)
    assert answer['close_lsn'] == RESTATED_CLOSE and answer['folder'] == str(shared)
    wanted = sorted(name for name in PROJECTORS if name not in UNSEEDED)
    assert [seed['projector'] for seed in answer['seeds']] == wanted
    reader = SpineLog(home)
    for name in wanted:
        assert read_seed(reader, PROJECTORS[name], RESTATED_CLOSE, shared) is not None, name
    reader.close()

    assert cli.main(['seed', '--home', str(home), '--at', '2026-08-25', '--out',
                     str(tmp_path / 'none')]) == 1
    assert 'no official close on or before 2026-08-25' in capsys.readouterr().err
    assert cli.main(['seed', '--home', str(home), '--at', 'yesterday']) == 1
    assert 'neither an LSN nor a day' in capsys.readouterr().err
    assert not (tmp_path / 'none').exists()
