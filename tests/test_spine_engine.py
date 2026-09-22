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

"""The seam between the engine and the book of record, driven through both at once.

`test_spine_verbs.py` gates the spine half with no engine near it; this is the expensive other
half - real homes in temp directories, a real book file, a real `BaseValuation`, the real
`Market Prices` partition. Nothing is monkeypatched: the injected executor is an ordinary
function, which is the seam's own design, and every fault is data on disk.

THE SWITCH IS `DV_SPINE_HOME` AND THE FIRST GATE IS THAT IT IS OFF: with it unset the edge behaves
to the byte as it did before this increment, and the Context verbs refuse BY NAME rather than
reaching for a default home. That is the regression bar, asserted first.

  * ATTESTATION LANES, all four. Telemetry and curiosity mint nothing, asserted as ABSENCE against
    a head that does not move; a `result_pinned` that fails re-execution is refused by name,
    re-executed through the real engine; one matching a known tuple resolves as a cache hit with
    the executor counted rather than believed.
  * PROVENANCE: the plan hash is RE-DERIVED by recompiling the blob-stored job document at the
    recorded LSN and required to equal the recorded tuple. The compiler-as-a-fold over fixings is
    increment 4's; what is gated here is that the object it will read is stored and recompiles.
  * FIRMNESS, all three: a moved book refuses on the plan dimension, an aged market pin on the
    values dimension, and the two are DISJOINT - a vol tick through the real partition moves
    `values_hash`, leaves `plan_hash` bit-identical, and does not trip the plan dimension.
  * ATOMICITY: a priced ticket references exactly one values hash and one book plan, and both
    staleness policies fire on a deliberately aged fixture.
  * THE DUAL WRITE'S ORDER: the event goes first and the book file follows, and a booking the
    record refuses leaves the file byte-identical.
"""
import hashlib
import itertools
import json
import os
import sys
import threading

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import derivus
from derivus import service, spine, utils
from derivus.config import CustomJsonEncoder
from derivus.schema import deal_at
from derivus_spine import SpineLog, init_home, verify_home
from derivus_spine import policy, projections, verbs

ACTOR = 'subject-desk-one'
BASE = pd.Timestamp('2024-06-28')
RATE, SPOT, AMOUNT = 0.02, 18.5, 1_000_000.0
USDZAR = 1.0 / SPOT
JSON = {'content-type': 'application/json'}
CLIENT_SET = 'CLIENT_A'
CLIENT = TestClient(service.app)

CASHFLOW = {'Object': 'FixedCashflowDeal', 'Reference': 'CF1', 'Currency': 'ZAR',
            'Discount_Rate': 'ZAR', 'Calendars': None, 'Amount': AMOUNT,
            'Payment_Date': BASE + pd.DateOffset(years=2)}
FACTORS = {
    'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Spot': 1.0},
    'FxRate.ZAR': {'Domestic_Currency': None, 'Interest_Rate': 'ZAR', 'Spot': SPOT},
    'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                         'Curve': utils.Curve([], [[0.0, RATE], [5.0, RATE]])},
    'InterestRate.ZAR': {'Currency': 'ZAR', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                         'Curve': utils.Curve([], [[0.0, RATE], [5.0, RATE]])}}
CSA = {'.CreditSupportList': [[0.0, 0.0]]}
COLLAR = {'pair': 'USDZAR', 'expiry': '1Y', 'notional': AMOUNT,
          'notional_currency': 'USD', 'floor': USDZAR * 0.95}


def job(deals=(CASHFLOW,), factors=FACTORS, sections={}, **calculation):
    """A job document as the objects a market data file holds - `test_service`'s own shape."""
    return {'Calc': {
        'Calculation': dict({'Object': 'BaseValuation', 'Base_Date': BASE, 'Currency': 'USD',
                             'MCMC_Simulations': 1, 'Random_Seed': 1}, **calculation),
        'Deals': {'Tag_Titles': '', 'Reference': 'spine-desk',
                  'Deals': {'Children': [{'Instrument': {'.Deal': deal}} for deal in deals]}},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': dict({
            'System Parameters': {'Base_Currency': 'USD', 'Base_Date': BASE},
            'Price Factors': factors}, **sections)}}}


def netting_set(reference, counterparty, deals=()):
    """One `NettingCollateralSet` naming its counterparty where the engine reads it, which is also
    where a fill's counterparty comes from - so the two cannot name different clients."""
    return {'Instrument': {'.Deal': {
        'Object': 'NettingCollateralSet', 'Reference': reference, 'Netted': 'True',
        'Collateralized': 'False', 'Agreement_Currency': 'USD', 'Balance_Currency': 'USD',
        'Funding_Rate': 'USD', 'Liquidation_Period': 0.0, 'Settlement_Period': 0.0,
        'Credit_Support_Amounts': {
            'Counterparty': counterparty, 'Received_Threshold': CSA, 'Posted_Threshold': CSA,
            'Independent_Amount': CSA, 'Minimum_Received': CSA, 'Minimum_Posted': CSA}}},
        'Children': [{'Instrument': {'.Deal': deal}} for deal in deals]}


def dump(document):
    return json.dumps(document, cls=CustomJsonEncoder)


def fx_vol_snapshot(atm_3m=14.0):
    """A USDZAR snapshot through the Bloomberg package's own normalization - canned observations
    standing in for the terminal, the real pipeline downstream. `atm_3m` is all a TICK moves."""
    from derivus_bloomberg import (FXQuoteSecurity, FXVolDefinition, RawBloombergObservation,
                                   normalize_fx_vol)
    raw = {('3M', 'ATM', None): atm_3m, ('3M', 'RR', 0.25): -1.2, ('3M', 'BF', 0.25): 0.35,
           ('1Y', 'ATM', None): 15.0, ('1Y', 'RR', 0.25): -1.6, ('1Y', 'BF', 0.25): 0.45}
    definition = FXVolDefinition(
        pair='USDZAR', surface_name='USD.ZAR', currency='USD',
        expiries={'3M': 0.25, '1Y': 1.0}, pillars=(0.25,),
        securities={coordinate: FXQuoteSecurity('USDZAR {} {} {}'.format(*coordinate))
                    for coordinate in raw})
    observations = [
        RawBloombergObservation(expiry, quote_type, pillar,
                                'USDZAR {} {} {}'.format(expiry, quote_type, pillar),
                                'PX_LAST', value)
        for (expiry, quote_type, pillar), value in raw.items()]
    return normalize_fx_vol(definition, observations, pd.Timestamp('2024-06-28 16:30'))


def fx_vol_quotes(atm_3m=14.0):
    from derivus_bloomberg import to_market_prices_block
    return {'FXVolPrices.USD.ZAR': to_market_prices_block(fx_vol_snapshot(atm_3m))}


# --------------------------------------------------------------------------------------------
# Homes, books, and reading the record back.

@pytest.fixture
def unrecorded(tmp_path, monkeypatch):
    """A desk with NO spine home, made explicit so a developer box carrying the variable cannot make
    this file lie."""
    monkeypatch.delenv('DV_SPINE_HOME', raising=False)
    monkeypatch.delenv('DV_SPINE_ACTOR', raising=False)
    monkeypatch.setenv('DV_HOME', str(tmp_path))
    yield tmp_path


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    """A minted spine home with an actor for the appends. Genesis is four events, so every head
    assertion below counts from there."""
    home = tmp_path / 'spine'
    init_home(home, ACTOR)
    monkeypatch.setenv('DV_SPINE_HOME', str(home))
    monkeypatch.setenv('DV_SPINE_ACTOR', ACTOR)
    monkeypatch.setenv('DV_HOME', str(tmp_path))
    yield home


@pytest.fixture
def desk(tmp_path):
    """A live one-cashflow book, taken down after the gate."""
    path = tmp_path / 'book.json'
    path.write_text(json.dumps(json.loads(dump(job())), indent=2), newline='\n')
    service.BOOK = service.Book(str(path))
    yield path
    service.BOOK = None


@pytest.fixture
def quoting(tmp_path):
    """A desk that can be quoted at: the one-cashflow book with a client's netting set on it and a
    real USDZAR surface ticked in through `/book/market`."""
    document = job(sections={'Bootstrapper Configuration': {'FXVolSurfaceParameters': {}}})
    document['Calc']['Deals']['Deals']['Children'].append(netting_set(CLIENT_SET, 'CPTY_A'))
    path = tmp_path / 'book.json'
    path.write_text(json.dumps(json.loads(dump(document)), indent=2), newline='\n')
    service.BOOK = service.Book(str(path))
    ticked = CLIENT.post('/book/market', content=dump({'quotes': fx_vol_quotes()}),
                         headers=JSON).json()
    assert ticked['written'] is True, ticked
    yield path
    service.BOOK = None


def opened(home):
    """A read-only handle on the record. Reading never claims the home, so a gate may hold one
    while the service writes through its own."""
    return SpineLog(home)


def head(home):
    log = opened(home)
    try:
        return log.head()[0]
    finally:
        log.close()


def facts(home, event_type=None):
    """`(lsn, type, body)` for every event on the record, or every one of a type - read off the
    platter and decrypted, because the assertion everywhere here is what the log actually holds."""
    log = opened(home)
    try:
        return [(frame['lsn'], frame['event_type'], log.open_body(frame))
                for frame in log.frames()
                if event_type is None or frame['event_type'] == event_type]
    finally:
        log.close()


def blob(home, digest):
    log = opened(home)
    try:
        return log.store.get(digest)
    finally:
        log.close()


def declare(home, name, document):
    """Declare one of the two increment-3 policies onto a home, through the ordinary writer."""
    log = opened(home)
    try:
        return policy.declare(log, ACTOR, name, document)
    finally:
        log.close()


def submit(document, **body):
    return CLIENT.post('/execute', content=dump(dict(document, **body)), headers=JSON).json()


def own_job(marker, **calculation):
    """A job whose replay tuple belongs to ONE gate. The result store is content-addressed and lives
    for the length of the process, so two gates posting the same job would be one execution and the
    second would read the first's answer - the feature working, and a way to pass for the wrong
    reason. Every gate moves one number nobody asserts on.
    """
    offset = int(hashlib.sha256(marker.encode('utf-8')).hexdigest()[:4], 16)
    return job(deals=(dict(CASHFLOW, Reference=marker, Amount=AMOUNT + offset),), **calculation)


def drained(submitted):
    """Wait for the one worker and read the summary back."""
    service.EXECUTOR.queue.join()
    return CLIENT.get('/results/{}'.format(submitted['result_id'])).json()


#: How long a gate waits on the one worker before it says so instead of hanging the suite.
WORKER_SECONDS = 30.0


class Barrier:
    """A job that HOLDS the single worker until the gate lets it go. Nothing patched: the executor
    runs whatever object answers `run_job`, which is how `XvaJob`, `SolveJob` and `BloombergJob`
    already ride it. It turns "a submission observed while the same tuple is still queued" from a
    timing window into a fact.
    """

    def __init__(self):
        self.running = threading.Event()
        self.released = threading.Event()

    def run_job(self):
        self.running.set()
        self.released.wait(WORKER_SECONDS)
        return None, {'Results': {}, 'Stats': {}}


def holding(counter=itertools.count()):
    """Back the one worker up behind a barrier and answer it. The queue is drained first, so what
    waits behind the barrier is only what the gate puts there. Each barrier is filed under its own
    name: the result store is content-addressed, so a second under one name would coalesce onto the
    first, never be dequeued, and hang the gate instead of failing it.
    """
    service.EXECUTOR.queue.join()
    barrier = Barrier()
    service.EXECUTOR.submit(
        service.Job('barrier-{}'.format(next(counter)), barrier, {}), service.HEAVY)
    return barrier


def quote_of(structure, params, **extra):
    submitted = CLIENT.post('/book/structure', content=dump(
        dict({'structure': structure, 'params': params}, **extra)), headers=JSON).json()
    service.EXECUTOR.queue.join()
    result = CLIENT.get('/results/{}'.format(submitted['result_id'])).json()
    assert result['status'] == 'done', result.get('error')
    return result['stats']['Quote']


# --------------------------------------------------------------------------------------------
# The switch, off.

def test_the_lane_names_the_engine_spells_are_the_lanes_the_record_knows():
    """Two spellings of one vocabulary, pinned together: the engine names the lanes at call sites
    that must not pay for the spine import, so `derivus.spine` carries its own three strings.
    """
    assert (spine.TELEMETRY, spine.CURIOSITY, spine.STANDING) == (
        verbs.TELEMETRY, verbs.CURIOSITY, verbs.STANDING)
    assert spine.LANES == verbs.LANES
    assert service.DEFAULT_LANE == spine.CURIOSITY


def test_importing_the_engine_lands_neither_the_spine_nor_its_one_dependency():
    """The extra is an EXTRA: a fresh interpreter imports the engine, the seam and the HTTP surface
    and reports what arrived. `pip install derivus` must not grow a `cryptography` dependency for a
    book of record that box does not run, which is why every import in `derivus/spine.py` is inside
    the function that needs it.
    """
    import subprocess

    code = ('import json, sys; import derivus, derivus.spine, derivus.service; '
            'print(json.dumps(sorted({n.split(".")[0] for n in sys.modules})))')
    done = subprocess.run(
        [sys.executable, '-c', code], cwd=os.path.dirname(os.path.dirname(os.path.abspath(
            __file__))), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    assert done.returncode == 0, done.stderr
    landed = set(json.loads(done.stdout))
    assert 'derivus' in landed, 'the engine did not import'
    assert 'derivus_spine' not in landed and 'cryptography' not in landed, sorted(
        landed & {'derivus_spine', 'cryptography'})


def test_with_no_spine_home_the_edge_is_the_edge_it_always_was(unrecorded, desk):
    """THE REGRESSION BAR. No home configured means the record does not exist as far as the engine
    is concerned: a lane is accepted and INERT - including one nobody has heard of - a booking
    books, a quote files with no `pinned` field, and the Context verbs refuse BY NAME rather than
    reaching for a default. That last is the sharpest: `DV_Spine` falls back to `~/.derivus_spine`
    because a person typing a verb means that home, and an engine that fell back would start
    recording on any box where somebody once ran `init`.
    """
    assert spine.home() is None and spine.configured() is False

    for lane in ('telemetry', 'curiosity', 'standing', 'exploration', None):
        document = own_job('UNRECORDED-{}'.format(lane))
        result = drained(submit(document, lane=lane) if lane else submit(document))
        assert result['status'] == 'done', (lane, result.get('error'))
        assert 'attested' not in result, 'a box that records nothing recorded something'

    booked = CLIENT.post('/book/deals', content=dump({'action': 'add', 'deal': dict(
        CASHFLOW, Reference='CF2', Amount=250_000.0)}), headers=JSON).json()
    assert booked['written'] is True and 'recorded' not in booked

    for verb, arguments in (('book', (CASHFLOW, 1.0, 'LEI', 'CSA', 'EXEC-1')),
                            ('amend', (CASHFLOW, dict(CASHFLOW, Amount=2.0))),
                            ('apply_lifecycle', ('election', {'instrument': 'a' * 64,
                                                              'choice': 'exercise'})),
                            ('declare_market', ('official',))):
        with pytest.raises(spine.SpineRefused) as refusal:
            getattr(derivus.Context().load_json((dump(job()), 'posted')), verb)(*arguments)
        assert 'DV_SPINE_HOME' in str(refusal.value), verb


def test_a_configured_home_that_is_not_a_home_refuses_by_name(tmp_path, monkeypatch, desk):
    """Set but not minted is a NAMED refusal and never a quiet fall-back: "configured a home that is
    not there" and "configured none" are different facts, and reading the first as the second would
    silently un-record a box somebody meant to record."""
    monkeypatch.setenv('DV_SPINE_HOME', str(tmp_path / 'never-minted'))
    monkeypatch.setenv('DV_SPINE_ACTOR', ACTOR)
    refused = CLIENT.post('/execute', content=dump(dict(own_job('NO-HOME'),
                                                        lane=spine.STANDING)), headers=JSON)
    service.EXECUTOR.queue.join()
    result = CLIENT.get('/results/{}'.format(refused.json()['result_id'])).json()
    assert result['status'] == 'error'
    assert 'log/' in result['error'] and 'DV_Spine init' in result['error']


def test_a_configured_home_still_refuses_an_append_nobody_signed(recorded):
    """A home configured and an actor not: every event carries the pseudonymous subject reference
    that submitted it, so there is nothing to stamp and the verb refuses by name rather than
    putting a name in the record that nobody chose."""
    os.environ.pop('DV_SPINE_ACTOR')
    context = derivus.Context().load_json((dump(job()), 'posted'))
    with pytest.raises(spine.SpineRefused) as refusal:
        context.declare_market('official')
    assert 'DV_SPINE_ACTOR' in str(refusal.value)
    assert head(recorded) == 4, 'a refused append wrote something'


# --------------------------------------------------------------------------------------------
# The lanes.

def test_a_standing_run_attests_at_birth_and_the_other_lanes_mint_nothing(recorded, desk):
    """A run is recorded IFF its output will be cited by a fact. Telemetry and curiosity mint
    NOTHING and the assertion is ABSENCE - the head does not move - which is the only honest way to
    say so. A standing run appends `run_completed` with the whole replay tuple at birth and reports
    the LSN it landed at, so a caller can cite it without folding for it.
    """
    genesis = head(recorded)
    for silent in (spine.TELEMETRY, spine.CURIOSITY):
        result = drained(submit(own_job('SILENT-' + silent), lane=silent))
        assert result['status'] == 'done', result.get('error')
        assert 'attested' not in result and head(recorded) == genesis, silent

    # the default is curiosity, so a caller who says nothing records nothing
    assert drained(submit(own_job('DEFAULT-LANE')))['status'] == 'done'
    assert head(recorded) == genesis, 'the default lane recorded something'

    standing = drained(submit(own_job('STANDING-BIRTH', Random_Seed=7), lane=spine.STANDING))
    assert standing['status'] == 'done', standing.get('error')
    assert head(recorded) == genesis + 1
    assert standing['attested']['lane'] == spine.STANDING
    assert standing['attested']['lsn'] == genesis + 1

    lsn, _, body = facts(recorded, 'run_completed')[0]
    assert lsn == standing['attested']['lsn']
    assert body['plan_hash'] == standing['plan_hash']
    assert body['values_hash'] == standing['values_hash']
    assert body['engine_version'] == standing['engine_version'] == derivus.__version__
    assert body['seed'] == 7 and body['lane'] == spine.STANDING
    assert verify_home(recorded)['events'] == head(recorded)


def test_a_synthetic_tick_sequence_mints_nothing_and_the_absence_is_asserted(recorded, quoting):
    """A telemetry repaint mints no event, asserted by absence over a synthetic tick sequence: four
    market ticks moving the ATM vol, each installing quotes, re-bootstrapping the surface and
    rewriting the file, with a what-if priced between them. Every one is a READING, superseded
    before anything could cite it. The assertion is the HEAD and not a filtered count, which would
    pass on a log full of the wrong events.
    """
    genesis = head(recorded)
    for atm in (14.1, 14.2, 14.3, 14.4):
        ticked = CLIENT.post('/book/market', content=dump({'quotes': fx_vol_quotes(atm)}),
                             headers=JSON).json()
        assert ticked['written'] is True and ticked['updated'] == ['FXVolPrices.USD.ZAR']
        priced = CLIENT.post('/book/price', json={}).json()
        assert priced['status'] in ('queued', 'running', 'done')
        # drain before the next tick: what this gate asserts is a head that has NOT moved, and a
        # run still in flight at the last tick has not finished minting nothing
        service.EXECUTOR.queue.join()

    assert head(recorded) == genesis, 'a repaint reached the book of record'
    assert facts(recorded, 'run_completed') == []
    assert verify_home(recorded)['events'] == genesis


def test_a_standing_run_whose_numbers_already_exist_still_attests(recorded, desk):
    """Content addressing dedupes NUMBERS; the lane is about STANDING, and this is where the two
    would be conflated. A job priced first as a what-if and then declared standing coalesces onto a
    result the worker will never revisit, so an attestation made only at completion would never be
    made and a fact would cite numbers the record does not hold. The standing submission attests
    from the store instead, and a third submission coalesces on the attestation's own idempotency
    tag rather than writing a second row about one run.
    """
    document = own_job('COALESCED')
    genesis = head(recorded)
    assert drained(submit(document, lane=spine.CURIOSITY))['status'] == 'done'
    assert head(recorded) == genesis, 'the what-if recorded something'

    standing = submit(document, lane=spine.STANDING)
    assert standing['status'] == 'done', 'the numbers were not already there'
    assert standing['attested']['lsn'] == genesis + 1
    assert head(recorded) == genesis + 1
    # and the result the store serves carries it too, rather than only this one answer
    assert drained(standing)['attested'] == standing['attested']

    again = submit(document, lane=spine.STANDING)
    assert again['attested'] == standing['attested'] and head(recorded) == genesis + 1

    # the bytes the record holds ARE the ones the store was serving, which is what makes attesting
    # off a finished result honest rather than convenient
    _, _, body = facts(recorded, 'run_completed')[0]
    assert blob(recorded, body['result']) == spine.result_stored(
        service.EXECUTOR.result(standing['result_id']))
    assert verify_home(recorded)['events'] == head(recorded)


def test_a_standing_run_that_coalesces_onto_one_in_flight_is_attested_when_it_lands(recorded, desk):
    """The other half, which a `done` status cannot reach. The same tuple explored first and
    declared standing a moment later coalesces while the numbers do NOT exist yet, so the standing
    caller is handed `queued` and the job the worker dequeues is the WHAT-IF's, carrying a lane
    that mints nothing - so an attestation read off the dequeued submission would never be made and
    the standing caller would be served numbers with no `run_completed` behind them.

    The standing submission is promoted onto the run inside `submit`, under the lock that publishes
    results, and the worker attests THAT submission. The head is asserted twice: unmoved while the
    run is in flight, moved by exactly one `run_completed` once it publishes. The barrier is what
    makes `queued` observable without a race.
    """
    genesis = head(recorded)
    barrier = holding()
    try:
        assert barrier.running.wait(WORKER_SECONDS), 'the worker never picked the barrier up'
        document = own_job('IN-FLIGHT')
        curious = submit(document, lane=spine.CURIOSITY)
        standing = submit(document, lane=spine.STANDING)

        assert curious['status'] == 'queued' and standing['status'] == 'queued', (curious, standing)
        assert standing['result_id'] == curious['result_id'], 'not the same tuple, not the case'
        assert 'attested' not in standing, 'numbers that did not exist yet were attested'
        assert head(recorded) == genesis, 'the record moved before the run did'
    finally:
        barrier.released.set()

    result = drained(standing)
    assert result['status'] == 'done', result.get('error')
    assert head(recorded) == genesis + 1, 'a standing run was served with no run_completed'
    assert result['attested']['lane'] == spine.STANDING
    assert result['attested']['lsn'] == genesis + 1

    lsn, _, body = facts(recorded, 'run_completed')[0]
    assert lsn == result['attested']['lsn'] and body['lane'] == spine.STANDING
    assert body['plan_hash'] == result['plan_hash'] and body['values_hash'] == result['values_hash']
    assert body['engine_version'] == result['engine_version'] == derivus.__version__
    # the tuple was attested from the run's own bytes, not from whatever the what-if left behind
    assert blob(recorded, body['result']) == spine.result_stored(
        service.EXECUTOR.result(standing['result_id']))
    assert verify_home(recorded)['events'] == head(recorded)


def test_an_attestation_the_record_refuses_fails_the_run_it_was_for(recorded, desk):
    """A standing run whose attestation is refused has NOT acquired standing, so serving its numbers
    would be the unbacked citation the lane rule prevents. The refusal travels as the run's own
    error and the record holds the denial and nothing else."""
    log = opened(recorded)
    try:
        blob_id = log.store.put(json.dumps(
            {'grants': [], 'read': []}, sort_keys=True, separators=(',', ':')).encode('utf-8'))
        log.append('policy_declared', {'policy': 'capabilities', 'blob': blob_id},
                   actor=ACTOR, blob_refs=(blob_id,))
    finally:
        log.close()
    standing = head(recorded)

    result = drained(submit(own_job('UNSCOPED'), lane=spine.STANDING))
    assert result['status'] == 'error'
    assert 'run_completed' in result['error'] and ACTOR in result['error']
    assert [event_type for _, event_type, _ in facts(recorded)][standing:] == \
        ['capability_denied'], 'the record kept an attestation nobody was scoped for'


def test_a_standing_run_off_a_plan_id_refuses_by_name(recorded, desk):
    """A `plan_id` names a PARSE the cache holds, where an attestation carries the job document the
    plan recompiles from. `Context.save_json` is not a complete round trip, so serialising the
    parse back would store a document that is not the one that ran. The refusal names the remedy;
    the curiosity lane over the same plan still runs."""
    prepared = CLIENT.post('/prepare', content=dump(own_job('PLAN-ID')), headers=JSON).json()
    refused = CLIENT.post('/execute', json={'plan_id': prepared['plan_id'],
                                            'lane': spine.STANDING})
    assert refused.status_code == 422
    assert 'plan_id' in refused.json()['detail'] and 'curiosity' in refused.json()['detail']
    assert head(recorded) == 4

    ran = CLIENT.post('/execute', json={'plan_id': prepared['plan_id'],
                                        'lane': spine.CURIOSITY}).json()
    assert 'result_id' in ran and head(recorded) == 4


def test_only_a_standing_job_owes_an_attestation_whatever_evidence_it_carries(recorded, desk):
    """`attests` asks two questions - is this the standing lane, and is there evidence to attest
    from - and both are pinned because only one is load-bearing at today's call sites: every lane
    but standing is handed evidence of None, so the lane test could stop working and every gate
    here would pass, right up to the day another lane wants the job document too.

    So the function is asked over plain tuples with evidence present in EVERY lane, which is the
    only arrangement in which the lane test is the thing being tested.
    """
    evidence = {'job': b'{"Calc":{}}', 'values': b'{}'}
    assert spine.configured() is True, 'this gate is about the recording posture'

    for silent in (spine.TELEMETRY, spine.CURIOSITY):
        assert service.attests(service.Job('r', None, {}, silent, evidence)) is False, silent
    assert service.attests(service.Job('r', None, {}, spine.STANDING, evidence)) is True

    # the other half is the QUOTE's case: a standing run with nothing to attest from files the
    # richer `quote_filed`, and a `run_completed` beside it would be two records of one act
    assert service.attests(service.Job('r', None, {}, spine.STANDING, None)) is False
    assert service.attests(service.Job('r', None, {})) is False


def test_the_stored_job_is_the_job_and_not_the_submission_that_carried_it(recorded, desk):
    """What a standing attestation stores is the `Calc` ENVELOPE and nothing beside it: a posted
    body may also carry `Patch`, `lane` or `plan_id`, none of which is the job. This blob is the
    FIRST LINK of the provenance chain, and the gate above would pass on either because the
    engine's loader tolerates a surplus top-level key.

    The patch is not lost by it: the values vector filed beside the job carries the whole market as
    patched, which is the model of a result as engine(plan, values). Both are asserted.
    """
    document = own_job('SUBMISSION-TRIM')
    moved = SPOT + 1.25
    standing = drained(submit(document, lane=spine.STANDING, Patch={'FxRate.ZAR': {'Spot': moved}}))
    assert standing['status'] == 'done', standing.get('error')

    _, _, body = facts(recorded, 'run_completed')[0]
    stored = json.loads(blob(recorded, body['job']).decode('utf-8'))
    assert sorted(stored) == ['Calc'], 'the submission rode into the record as the job'
    assert stored['Calc'] == json.loads(dump(document))['Calc']

    # the patch is in the VALUES vector, so the tuple the record holds is reproducible from the two
    # blobs it cites and from nothing else
    context = derivus.Context().load_json((json.dumps(stored), 'recompiled'))
    context.patch_market(spine.read_values(blob(recorded, body['values_hash'])))
    assert context.values_hash() == body['values_hash']
    assert context.plan_hash() == body['plan_hash']
    assert context.market_patch()['FxRate.ZAR']['Spot'] == moved, \
        'the values vector does not carry the market the run actually read'


def test_an_unknown_lane_refuses_where_the_record_will_act_on_it(recorded, desk):
    """With a home configured a lane is a decision the record acts on, so an unknown one is refused
    by name rather than read as the default - and the refusal is `verbs.check_lane`'s own, the
    lanes being the record's vocabulary and not the service's."""
    refused = CLIENT.post('/execute',
                          content=dump(dict(own_job('UNKNOWN-LANE'), lane='exploration')),
                          headers=JSON)
    assert refused.status_code == 422
    assert 'telemetry, curiosity, standing' in refused.json()['detail']
    assert head(recorded) == 4


# --------------------------------------------------------------------------------------------
# Provenance: the plan recompiles, and the result reproduces.

def test_the_plan_hash_recompiles_from_the_stored_job_at_the_recorded_lsn(recorded, desk):
    """Recompile the book's plan at its recorded LSN and require the identical plan hash - the
    auditor's move, made mechanical. The record stores the JOB DOCUMENT rather than the plan,
    because a plan is re-derivable and the record never trusts what it can re-derive: the gate
    pulls the cited blob, loads it through the engine's decoder, applies the cited values vector
    and requires BOTH hashes back, then re-executes and requires the result bytes to the byte.

    THE BOUNDARY: this recompiles the document the run was submitted with. The compiler as a FOLD
    over fixings is increment 4's; until it exists, the object it will read is what is gated here.
    """
    standing = drained(submit(own_job('PROVENANCE'), lane=spine.STANDING))
    lsn, _, body = facts(recorded, 'run_completed')[0]
    assert lsn == standing['attested']['lsn']

    stored_job = blob(recorded, body['job'])
    stored_values = blob(recorded, body['values_hash'])
    stored_result = blob(recorded, body['result'])
    # the values citation and the store address are ONE number, which is what lets the engine's
    # hash be a blob id at all
    assert hashlib.sha256(stored_values).hexdigest() == body['values_hash']

    context = derivus.Context().load_json((stored_job.decode('utf-8'), 'recompiled'))
    context.patch_market(spine.read_values(stored_values))
    assert context.plan_hash() == body['plan_hash'], 'the plan did not recompile to its own hash'
    assert context.values_hash() == body['values_hash']

    _, out = context.run_job()
    assert spine.result_of(out) == stored_result, 'the run did not reproduce to the byte'


# --------------------------------------------------------------------------------------------
# Promotion, through the real engine.

def test_a_result_pinned_matching_a_known_tuple_cache_hits_without_re_executing(recorded, desk):
    """A tuple this hub attested itself needs no re-execution, and the gate COUNTS rather than
    believes: the executor handed in records its calls and is never called. The pin still lands and
    names the tolerance policy in force - a promotion under no declared standard is not one this
    record carries."""
    declare(recorded, policy.TOLERANCE_POLICY, {'tolerances': {'mtm': 1e-9}})
    standing = drained(submit(own_job('CACHE-HIT'), lane=spine.STANDING))
    _, _, body = facts(recorded, 'run_completed')[0]
    ran = []

    pinned = spine.pin_result(
        {name: body[name] for name in ('plan_hash', 'values_hash', 'engine_version', 'seed')},
        blob(recorded, body['job']), blob(recorded, body['values_hash']),
        blob(recorded, body['result']),
        execute=lambda job_bytes, values, version: ran.append(1) or (version, b''))

    assert pinned['resolution'] == 'cache hit' and ran == []
    assert standing['attested']['lsn'] < pinned['lsn']
    lsn, _, pin = facts(recorded, 'result_pinned')[0]
    assert lsn == pinned['lsn'] and pin['result'] == body['result']
    assert verify_home(recorded)['events'] == head(recorded)


def test_a_result_pinned_that_will_not_reproduce_is_refused_by_name(recorded, desk):
    """Through the real engine on a tuple this hub never witnessed: a genuine plan and values hash
    carrying a DOCTORED result. `Context.pin_result` injects the engine as the executor, the job
    runs, and the refusal names the class, both numbers and the tolerance policy - and NOTHING is
    appended. The honest claim over the same job then lands, reproducing to the byte.
    """
    declare(recorded, policy.TOLERANCE_POLICY, {'tolerances': {'mtm': 1e-9}})
    unseen = own_job('UNSEEN')
    context = derivus.Context().load_json((dump(unseen), 'unseen'))
    standing = head(recorded)

    with pytest.raises(spine.SpineRefused) as refusal:
        derivus.Context().pin_result(
            spine.canonical(unseen), spine.values_of(context), b'{"mtm":{"nonsense":1}}',
            spine.replay(context))
    said = str(refusal.value)
    assert 'mtm' in said and 'Nothing is pinned' in said, said
    assert head(recorded) == standing, 'a refused pin wrote something'

    _, out = derivus.Context().load_json((dump(unseen), 'unseen')).run_job()
    pinned = derivus.Context().pin_result(
        spine.canonical(unseen), spine.values_of(context), spine.result_of(out),
        spine.replay(context))
    assert pinned['resolution'] == 'reproduced' and head(recorded) == standing + 1
    assert verify_home(recorded)['events'] == head(recorded)


# --------------------------------------------------------------------------------------------
# The Context verbs.

def test_the_context_verbs_book_amend_and_file_the_three_lifecycle_facts(recorded):
    """The five delegators on a real home. Each canonicalises through the ENGINE's own encoder, so
    an instrument's id is the content hash a job document would give it, and hands plain data to
    the spine - which keeps storage out of every module under `derivus/` but this one.
    `declare_market` is the one that uses its context: the values vector it names is THIS
    context's, which is why officialness is a property of the name.
    """
    context = derivus.Context().load_json((dump(job()), 'posted'))
    amended = dict(CASHFLOW, Amount=750_000.0)

    booked = context.book(CASHFLOW, -AMOUNT, 'LEI-5493001KJTIIGC8Y1R12', 'CSA-0007', 'EXEC-1',
                          book='spine-desk')
    assert booked['instrument'] == derivus.content_hash(CASHFLOW)
    assert facts(recorded, 'fill')[0][2] == {
        'instrument': derivus.content_hash(CASHFLOW), 'quantity': -AMOUNT,
        'counterparty': 'LEI-5493001KJTIIGC8Y1R12', 'netting_set': 'CSA-0007',
        'execution_reference': 'EXEC-1'}

    linked = context.amend(CASHFLOW, amended, book='spine-desk')
    assert facts(recorded, 'amendment')[0][2] == {
        'instrument': derivus.content_hash(CASHFLOW), 'amended_to': derivus.content_hash(amended)}
    assert linked['amended_to'] == derivus.content_hash(amended)

    context.apply_lifecycle('election', {'instrument': derivus.content_hash(CASHFLOW),
                                         'choice': 'exercise'}, book='spine-desk')
    context.apply_lifecycle('fixing_observed', {'index': 'USDZAR-SARB', 'date': '2026-08-26',
                                                'source': 'SARB', 'value': 18.62})
    context.apply_lifecycle('determination', {'subject': 'CF1',
                                              'ruling': 'the barrier was touched at 14:02'},
                            book='spine-desk')
    with pytest.raises(spine.SpineRefused):
        context.apply_lifecycle('knocked_out', {'instrument': 'a' * 64, 'choice': 'x'})

    marked = context.declare_market('private/desk-one/screen')
    assert marked['values_hash'] == context.values_hash(), \
        'a market names the values vector this context is carrying'
    assert blob(recorded, marked['values_hash']) == spine.values_of(context)

    assert verify_home(recorded)['events'] == head(recorded) == 4 + 6


def test_a_booking_through_the_book_verb_writes_the_event_before_the_file(recorded, desk):
    """The dual write's ORDER: `/book/deals` appends the `fill` and then rewrites the book file, so
    what is TRUE is on the platter before the desk's copy of it. A booking the record refuses
    leaves the file byte-identical - asserted on the BYTES, not the parse - and the three refusals
    name a quantity, an execution reference and a netting set with a counterparty, none of which
    has a defensible default.
    """
    document = json.loads(desk.read_text())
    document['Calc']['Deals']['Deals']['Children'].append(netting_set(CLIENT_SET, 'CPTY_A'))
    desk.write_text(json.dumps(document, indent=2), newline='\n')
    before = desk.read_bytes()
    deal = dict(CASHFLOW, Reference='CF2', Amount=250_000.0)

    for missing in ({'quantity': 250_000.0}, {'execution_reference': 'EXEC-7'}, {}):
        refused = CLIENT.post('/book/deals', content=dump(dict(
            {'action': 'add', 'deal': deal, 'parent_reference': CLIENT_SET}, **missing)),
            headers=JSON)
        assert refused.status_code == 422, missing
        assert desk.read_bytes() == before, 'a refused booking moved the file'

    # at the root there is no set above it, so there is no counterparty to name
    rootless = CLIENT.post('/book/deals', content=dump({
        'action': 'add', 'deal': deal, 'quantity': 1.0, 'execution_reference': 'EXEC-7'}),
        headers=JSON)
    assert rootless.status_code == 422 and 'NettingCollateralSet' in rootless.json()['detail']
    assert desk.read_bytes() == before

    booked = CLIENT.post('/book/deals', content=dump({
        'action': 'add', 'deal': deal, 'parent_reference': CLIENT_SET, 'quantity': -250_000.0,
        'execution_reference': 'EXEC-7'}), headers=JSON).json()
    assert booked['written'] is True and desk.read_bytes() != before

    lsn, _, fill = facts(recorded, 'fill')[0]
    assert lsn == booked['recorded']['lsn']
    assert fill['quantity'] == -250_000.0 and fill['execution_reference'] == 'EXEC-7'
    assert fill['netting_set'] == CLIENT_SET and fill['counterparty'] == 'CPTY_A'
    # the instrument is the deal as BOOKED, read back off the file the write produced
    node = deal_at(json.loads(desk.read_text()), booked['deal_path'])
    assert fill['instrument'] == derivus.content_hash(service.instrument_of(node))

    # and an amendment links the terms that were to the terms that are
    amended = CLIENT.post('/book/deals', content=dump({
        'action': 'amend', 'deal_path': booked['deal_path'],
        'fields': {'Amount': 260_000.0}}), headers=JSON).json()
    assert amended['written'] is True
    _, _, link = facts(recorded, 'amendment')[0]
    assert link['instrument'] == fill['instrument'] and link['amended_to'] != link['instrument']
    assert verify_home(recorded)['events'] == head(recorded)


# --------------------------------------------------------------------------------------------
# The quote: two hashes pinned, two dimensions checked.

def test_a_quote_pins_the_books_two_hashes_and_files_them_as_one_ticket(recorded, quoting,
                                                                        tmp_path):
    """THE ATOMICITY GATE: a priced ticket references exactly one values hash and one book plan.
    One quote, one `quote_filed`, one of each in its body beside the solved coordinate and the
    edge. The pending file carries the same pair, so the approval reads the pins off the desk's
    copy while the record holds the authoritative one.

    The hashes are the BOOK's, taken before the live spot lands on the quote's copy: an approval
    asks whether the market and the book this trade would LAND against have moved.
    """
    document, _ = service.BOOK.read()
    context = service.load(document)
    quote = quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET,
                     request='the client wants a year of downside at zero cost')

    lsn, _, body = facts(recorded, 'quote_filed')[0]
    assert body['quote_id'] == quote['quote_id'] and body['structure'] == 'ZeroCostCollar'
    assert body['plan_hash'] == context.plan_hash()
    assert body['values_hash'] == context.values_hash()
    assert body['request'].startswith('the client wants')
    assert len(body['solved']) == 1 and body['edge'] == pytest.approx(quote['edge'])
    assert sorted(body) == ['edge', 'plan_hash', 'quote_id', 'request', 'solved', 'structure',
                            'values_hash']

    filed = json.loads((tmp_path / 'tmp' / (quote['quote_id'] + '.json')).read_text())
    assert filed['pinned'] == {'plan_hash': body['plan_hash'],
                               'values_hash': body['values_hash'], 'lsn': lsn}
    assert blob(recorded, body['values_hash']) == spine.values_of(context)
    assert verify_home(recorded)['events'] == head(recorded)


class SpotMovingParams(dict):
    """The quote's own parameters, moving the document's spot the instant `patch_live_spot` reads
    the pair off them - the instant a terminal's crosses would land in production.

    THE GATE'S OWN DATA, not a patch: `params` is caller-supplied and `StructureJob` only reads it,
    so a dict that moves the spot on `params['pair']` reproduces deterministically, with no
    terminal in the room, the one thing this box cannot otherwise produce.
    """

    def __init__(self, params, document, spot):
        super().__init__(params)
        self.document, self.spot, self.moved = document, spot, False

    def __getitem__(self, key):
        if key == 'pair' and not self.moved:
            self.moved = True
            self.document['Calc']['MergeMarketData']['ExplicitMarketData'][
                'Price Factors']['FxRate.ZAR']['Spot'] = self.spot
        return super().__getitem__(key)


def test_a_quote_pins_the_book_before_the_live_spot_lands_on_its_copy(recorded, quoting):
    """The pins are the BOOK's, and the ordering that makes them so is gated rather than trusted.
    `StructureJob.run_job` takes its two hashes at the TOP, before `patch_live_spot` moves this
    copy's spot: pin after and `values_hash` describes a market existing only inside one quote, so
    every approval on a box with a live terminal would refuse on the values dimension for a market
    that never moved, naming a remedy that cannot work.

    Nothing else here can see that - a desk box with no terminal never patches the spot - so the
    missing half is supplied as data. The disjointness half comes free: a SPOT is values-plane data
    like a vol, so moving it moves `values_hash` and leaves `plan_hash` bit-identical.
    """
    document, _ = service.BOOK.read()
    book_context = service.load(document)
    moved_spot = SPOT + 1.5

    quoted_document = service.BOOK.read()[0]
    job = service.StructureJob(quoted_document, 'ZeroCostCollar',
                               SpotMovingParams(COLLAR, quoted_document, moved_spot), CLIENT_SET)
    job.run_job()

    assert job.params.moved is True, 'the spot never moved - this gate proved nothing'
    quoted = service.load(job.document)
    assert quoted.values_hash() != book_context.values_hash(), 'the live spot moved nothing'
    assert quoted.plan_hash() == book_context.plan_hash(), \
        'a spot moved the plan - the values plane and the plan plane are not disjoint'

    lsn, _, body = facts(recorded, 'quote_filed')[0]
    assert body['values_hash'] == book_context.values_hash(), \
        'the quote pinned its own spot-patched market rather than the book it would land against'
    assert body['plan_hash'] == book_context.plan_hash()
    assert blob(recorded, body['values_hash']) == spine.values_of(book_context)

    # the approval reads the same pair off the desk's copy, so it stands against the book
    verdict = spine.package().firmness.assess(
        {'plan_hash': body['plan_hash'], 'values_hash': body['values_hash']},
        {'plan_hash': book_context.plan_hash(), 'values_hash': book_context.values_hash()},
        {'values': 1.0, 'plan': 1.0}, spine.firmness_policy())
    assert verdict['firm'] is True, verdict['refusals']
    assert verify_home(recorded)['events'] == head(recorded)


def test_a_moved_book_refuses_the_approval_on_the_plan_dimension(recorded, quoting):
    """The book moved since the solve, so the marginal charge was priced against a portfolio this
    trade would no longer join. Only the PLAN dimension refuses - the market was not touched - and
    it names the dimension and the remedy: a desk told "stale" learns nothing."""
    quote = quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET)

    document = json.loads(quoting.read_text())
    document['Calc']['Deals']['Deals']['Children'].append(
        json.loads(dump({'Instrument': {'.Deal': dict(CASHFLOW, Reference='LATE',
                                                      Amount=5_000.0)}})))
    quoting.write_text(json.dumps(document, indent=2), newline='\n')

    refused = CLIENT.post('/book/quote', json={'quote_id': quote['quote_id']})
    said = refused.json()['detail']
    assert refused.status_code == 422
    assert 'the plan dimension' in said and 'the book moved under it' in said
    assert 'the values dimension' not in said, 'the two dimensions were conflated'
    assert quote['quote_id'] in said

    assert facts(recorded, 'fill') == [], 'a refused approval booked'


def test_an_aged_market_refuses_the_approval_on_the_values_dimension(recorded, quoting):
    """The market pin is past its window on a market that has NOT moved - a different failure from
    a moved one, with a different remedy. Aged by DECLARATION: a firmness policy with a zero-second
    values window, put in the record as a hashed blob through the ordinary writer, which makes the
    fixture deterministic rather than a sleep.
    """
    declare(recorded, policy.FIRMNESS_POLICY, {'values_seconds': 0, 'plan_seconds': 600})
    quote = quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET)

    refused = CLIENT.post('/book/quote', json={'quote_id': quote['quote_id']})
    said = refused.json()['detail']
    assert refused.status_code == 422
    assert 'the values dimension' in said and 'old against a 0s window' in said
    assert 'check the tick' in said
    assert 'the plan dimension' not in said, 'the book was blamed for the market'
    assert facts(recorded, 'fill') == []


def test_both_staleness_dimensions_fire_on_a_deliberately_aged_fixture(recorded, quoting):
    """Both staleness policies fire on a deliberately aged fixture: two zero-second windows, one
    quote, both dimensions named in one refusal - a quote stale on both has two remedies, and
    reporting one sends a salesperson back into the other."""
    declare(recorded, policy.FIRMNESS_POLICY, {'values_seconds': 0, 'plan_seconds': 0})
    quote = quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET)

    refused = CLIENT.post('/book/quote', json={'quote_id': quote['quote_id']})
    said = refused.json()['detail']
    assert refused.status_code == 422
    assert 'the values dimension' in said and 'the plan dimension' in said
    assert ' AND ' in said


def test_a_vol_tick_moves_the_values_hash_and_never_trips_the_plan_dimension(recorded, quoting):
    """THE DISJOINTNESS GATE, and the reason the `Market Prices` partition was a prerequisite. A
    pure market tick with NO booking in sight - the ATM vol moves, the block value-updates, the
    surface re-bootstraps, the file is rewritten - moves `values_hash` and leaves `plan_hash`
    BIT-IDENTICAL: quote values are the values plane, and pillars, expiries and conventions the
    plan. So the approval refuses on values while the plan dimension stands, asserted directly as
    well as through the verdict.
    """
    quote = quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET)
    _, _, pinned = facts(recorded, 'quote_filed')[0]

    ticked = CLIENT.post('/book/market', content=dump({'quotes': fx_vol_quotes(14.9)}),
                         headers=JSON).json()
    assert ticked['written'] is True and ticked['updated'] == ['FXVolPrices.USD.ZAR']

    standing = service.load(service.BOOK.read()[0])
    assert standing.plan_hash() == pinned['plan_hash'], \
        'a vol tick moved the plan - the values plane and the plan plane are not disjoint'
    assert standing.values_hash() != pinned['values_hash'], 'a vol tick moved nothing'

    verdict = spine.package().firmness.assess(
        pinned, {'plan_hash': standing.plan_hash(), 'values_hash': standing.values_hash()},
        {'values': 1.0, 'plan': 1.0}, spine.firmness_policy())
    assert verdict['plan']['firm'] is True, 'a market tick tripped the book dimension'
    assert verdict['values']['firm'] is False

    refused = CLIENT.post('/book/quote', json={'quote_id': quote['quote_id']})
    assert refused.status_code == 422
    assert 'the market moved under it' in refused.json()['detail']
    assert 'the plan dimension' not in refused.json()['detail']


def test_a_firm_quote_books_the_mirror_and_the_fill_lands_before_the_file(recorded, quoting):
    """The whole approval under a spine home: three windows passed - the desk's `firm_seconds` and
    the record's two dimensions - the `fill` appended, then the file written. The execution
    reference is the QUOTE ID, a quote being an act its id names, and the quantity carries the
    DESK's side: the quote is client paper and the approval books the mirror.
    """
    quote = quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET)
    before = quoting.read_bytes()

    booked = CLIENT.post('/book/quote', json={'quote_id': quote['quote_id']}).json()
    assert booked['written'] is True and quoting.read_bytes() != before
    assert booked['firmness']['firm'] is True

    lsn, _, fill = facts(recorded, 'fill')[0]
    assert lsn == booked['recorded']['lsn']
    assert fill['execution_reference'] == quote['quote_id']
    assert fill['netting_set'] == CLIENT_SET and fill['counterparty'] == 'CPTY_A'
    assert abs(fill['quantity']) == AMOUNT
    bought = [leg['buy_sell'] for leg in quote['legs'] if leg['buy_sell']][0]
    assert (fill['quantity'] < 0) is (bought == 'Buy'), \
        'the desk took the same side as its client'

    node = deal_at(json.loads(quoting.read_text()), booked['deal_path'])
    assert fill['instrument'] == derivus.content_hash(service.instrument_of(node))
    assert verify_home(recorded)['events'] == head(recorded)


def test_the_record_and_the_desk_file_disagree_only_in_the_order_they_were_written(recorded,
                                                                                   quoting):
    """Everything the record holds verifies from genesis, and the desk file is a copy of part of it
    written afterwards. A quote, an approval, a market tick and a standing run in one session, then
    the chain re-derived from its own bytes by the replica-shaped verifier - entitled and
    chain-only both, a book of record only its writer can check not being one.
    """
    quote = quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET)
    assert CLIENT.post('/book/quote', json={'quote_id': quote['quote_id']}).json()['written']
    assert CLIENT.post('/book/market', content=dump({'quotes': fx_vol_quotes(14.6)}),
                       headers=JSON).json()['written']
    assert drained(submit(own_job('POSTURE'), lane=spine.STANDING))['status'] == 'done'

    types = [event_type for _, event_type, _ in facts(recorded)]
    assert types.count('quote_filed') == 1 and types.count('fill') == 1
    assert types.count('run_completed') == 1
    assert 'market_declared' not in types, 'a tick declared a market'

    entitled = verify_home(recorded)
    assert entitled['events'] == head(recorded) and entitled['checkpoints_verified'] == 1
    assert verify_home(recorded, entitled=False)['head_hash'] == entitled['head_hash']


# --------------------------------------------------------------------------------------------
# The book file's pin, and the record read against it.

@pytest.fixture
def booking(tmp_path):
    """A book that is one client's netting set and nothing else - the least book a fill can land
    against, a fill carrying a counterparty and a netting set on the row, and the least the
    reconcile gates can read, every deal in it being one somebody booked."""
    document = job(deals=())
    document['Calc']['Deals']['Deals']['Children'].append(netting_set(CLIENT_SET, 'CPTY_A'))
    path = tmp_path / 'book.json'
    path.write_text(json.dumps(json.loads(dump(document)), indent=2), newline='\n')
    service.BOOK = service.Book(str(path))
    yield path
    service.BOOK = None


def head_hash(home):
    log = opened(home)
    try:
        return log.head()[1]
    finally:
        log.close()


def book_one(reference, amount=250_000.0):
    """Book one cashflow under the client's set and answer the outcome."""
    return CLIENT.post('/book/deals', content=dump(
        {'action': 'add', 'deal': dict(CASHFLOW, Reference=reference, Amount=amount),
         'parent_reference': CLIENT_SET, 'quantity': amount,
         'execution_reference': 'EXEC-' + reference}), headers=JSON).json()


def rewritten(path, edit):
    """The book file edited BY HAND, the way a desk with a text editor edits it - the divergence
    every reconcile gate is about."""
    document = json.loads(path.read_text())
    edit(document['Calc']['Deals']['Deals']['Children'][0])
    path.write_text(json.dumps(document, indent=2), newline='\n')


def test_with_no_home_the_book_file_carries_no_pin_at_all(unrecorded, desk):
    """GATE 1, the pin's half of the regression bar: a booking on a box that records nothing writes
    the file it always wrote. No key, and the word nowhere in the bytes.

    Killing mutation: the `Spine` key written unconditionally in `_land`, which puts an LSN from
    nowhere into every book file on every desk that records nothing.
    """
    assert spine.configured() is False
    booked = CLIENT.post('/book/deals', content=dump({'action': 'add', 'deal': dict(
        CASHFLOW, Reference='NOPIN', Amount=250_000.0)}), headers=JSON).json()
    assert booked['written'] is True

    text = desk.read_text()
    assert service.SPINE_PIN not in text, 'a box that records nothing wrote an LSN'
    assert set(json.loads(text)) == {'Calc'}
    assert 'lsn' not in CLIENT.get('/book').json()
    assert CLIENT.get('/book/status').json()['spine'] is None
    assert service.BOOK.pinned() is None


def test_the_pin_cannot_move_the_plan_hash(recorded, booking):
    """GATE 2. The pin is a SIBLING of `Calc`: `Context.load_json` reads `Calc` alone and
    `plan_hash` hashes `params` and `deals`, so the file carrying an LSN prices as the same plan.

    Killing mutation: the pin written inside `Calc`, which moves the plan hash on every booking and
    takes the disjointness gate with it.
    """
    assert book_one('PINNED')['written'] is True
    document = json.loads(booking.read_text())
    assert set(document) == {'Calc', service.SPINE_PIN}
    assert set(document[service.SPINE_PIN]) == {'lsn', 'head', 'hydrated_at'}

    pinned = derivus.Context().load_json((json.dumps(document), 'pinned'))
    bare = derivus.Context().load_json((json.dumps({'Calc': document['Calc']}), 'bare'))
    assert pinned.plan_hash() == bare.plan_hash(), 'the pin moved the plan'
    assert pinned.values_hash() == bare.values_hash()
    assert service.risk_etag(document) == service.risk_etag({'Calc': document['Calc']})


def test_the_pin_moves_with_the_append(recorded, booking):
    """GATE 3. The file says which LSN it was hydrated at, and after each booking that is the
    record's own head - the event goes first and the file follows it.

    Killing mutation: the pin stamped from a head read BEFORE the append, which leaves every file
    one event behind the record it claims to be a copy of.
    """
    seen = []
    for reference in ('FIRST', 'SECOND'):
        assert book_one(reference)['written'] is True
        document = json.loads(booking.read_text())
        seen.append(document[service.SPINE_PIN]['lsn'])
        assert seen[-1] == head(recorded), reference
        assert document[service.SPINE_PIN]['head'] == head_hash(recorded)

    assert seen == [5, 6] and CLIENT.get('/book').json()['lsn'] == seen[-1]
    status = CLIENT.get('/book/status').json()['spine']
    assert status['lsn'] == seen[-1] and status['events_behind'] == 0
    assert status['positions_behind'] == 0
    assert 'divergence' not in status, 'the status read folded the record'
    assert CLIENT.get('/book/reconcile').json()['in_record_not_in_file'] == []


def test_reconcile_names_a_divergence_by_its_instrument(recorded, booking):
    """GATE 4. A hand edit to the file is visible instead of silent, and it is named by the
    INSTRUMENT ADDRESS rather than by a reference - a renamed deal is different terms, so both
    halves of a rename are divergences.

    Killing mutations: comparing references instead of instrument hashes, which reconciles a renamed
    deal clean and hides the one edit a desk is most likely to make by hand; and skipping the
    subtree of a node marked `Ignore`, which is a node somebody booked whether or not the engine
    prices it.
    """
    assert book_one('RECONCILE')['written'] is True
    clean = CLIENT.get('/book/reconcile').json()
    assert clean['events_behind'] == clean['positions_behind'] == 0
    assert clean['lsn'] == head(recorded)
    assert clean['in_record_not_in_file'] == clean['in_file_not_in_record'] == []
    assert clean['quantity_mismatch'] == []

    booked = deal_at(json.loads(booking.read_text()), '0/0')
    address = derivus.content_hash(service.instrument_of(booked))

    # a node the desk told the engine to ignore is still a node somebody booked
    rewritten(booking, lambda node: node['Children'][0].update({'Ignore': 'True'}))
    ignored = CLIENT.get('/book/reconcile').json()
    assert ignored['in_file_not_in_record'] == ignored['in_record_not_in_file'] == []
    rewritten(booking, lambda node: node['Children'][0].pop('Ignore'))

    rewritten(booking, lambda node: node['Children'].clear())
    lost = CLIENT.get('/book/reconcile').json()
    assert [row['instrument'] for row in lost['in_record_not_in_file']] == [address]
    assert lost['in_record_not_in_file'][0]['netting_set'] == CLIENT_SET
    assert lost['in_file_not_in_record'] == [] and lost['quantity_mismatch'] == []

    rewritten(booking, lambda node: node['Children'].append(
        {'Instrument': {'.Deal': dict(json.loads(dump(CASHFLOW)), Reference='BY_HAND')}}))
    added = CLIENT.get('/book/reconcile').json()
    assert [row['reference'] for row in added['in_file_not_in_record']] == ['BY_HAND']
    assert [row['instrument'] for row in added['in_record_not_in_file']] == [address]

    renamed_node = {'Instrument': {'.Deal': dict(booked['Instrument']['.Deal'],
                                                 Reference='RENAMED')}}
    rewritten(booking, lambda node: node.update({'Children': [renamed_node]}))
    renamed = CLIENT.get('/book/reconcile').json()
    assert [row['instrument'] for row in renamed['in_record_not_in_file']] == [address]
    assert [row['reference'] for row in renamed['in_file_not_in_record']] == ['RENAMED']
    status = CLIENT.get('/book/status').json()['spine']
    assert 'divergence' not in status and status['events_behind'] == 0


def test_a_fill_the_file_never_took_is_what_reconcile_is_for(recorded, booking):
    """THE FOLD IS AT THE HEAD. A fill the record took and the file did not is the one failure this
    verb advertises, and it lives entirely past the pin - `events_behind` and `positions_behind`
    explain how far, and the row itself says which trade.

    Killing mutations: the fold taken at the file's own pin, under which every event after the pin
    is invisible by construction and the verb reports nothing while the record is a trade ahead;
    and `positions_behind` counting every event, which makes a policy declaration read as drift -
    the two numbers exist precisely so it does not.
    """
    assert book_one('KEPT')['written'] is True
    assert CLIENT.get('/book/reconcile').json()['in_record_not_in_file'] == []

    lost = spine.book(json.loads(dump(dict(CASHFLOW, Reference='LOST', Amount=500_000.0))),
                      500_000.0, 'CPTY_A', CLIENT_SET, 'EXEC-LOST')
    answer = CLIENT.get('/book/reconcile').json()
    assert [row['instrument'] for row in answer['in_record_not_in_file']] == [
        lost['body']['instrument'] if 'body' in lost else answer[
            'in_record_not_in_file'][0]['instrument']]
    assert len(answer['in_record_not_in_file']) == 1
    assert answer['in_record_not_in_file'][0]['quantity'] == 500_000.0
    assert answer['events_behind'] == answer['positions_behind'] == 1
    assert answer['in_file_not_in_record'] == [] and answer['quantity_mismatch'] == []

    # a POLICY past the pin is not drift: it moves the events and no position
    declare(recorded, policy.FIXINGS_POLICY, {'sources': {'FxRate.ZAR': ['ECB']}})
    after = CLIENT.get('/book/reconcile').json()
    assert (after['events_behind'], after['positions_behind']) == (2, 1)
    assert len(after['in_record_not_in_file']) == 1, 'a policy moved a position'


# --------------------------------------------------------------------------------------------
# The record's own readings - the strip, the markets, and the verdict a banner reads off them.

#: Every field one line of the strip carries. The envelope and nothing else, which is what lets a
#: replica holding no key render the sequence.
STRIP_FIELDS = {'lsn', 'record_time', 'effective_time', 'actor', 'event_type', 'book', 'summary'}


def strip(**params):
    answer = CLIENT.get('/book/activity', params=params)
    assert answer.status_code == 200, answer.text
    return answer.json()


def merged(held, page):
    """The strip's merge, mirrored from `web/src/spine.ts`: LSN order and no row twice, since a
    page is what came after a cursor and a cursor answered off a fold that moved repeats one."""
    rows = dict((row['lsn'], row) for row in held)
    rows.update((row['lsn'], row) for row in page['rows'])
    return [rows[lsn] for lsn in sorted(rows)]


#: A reconcile answer's three lists where none has been fetched - what the banner knows before it
#: asks, which is nothing about what the two hold.
NO_LISTS = {'in_record_not_in_file': [], 'in_file_not_in_record': [], 'quantity_mismatch': []}


def verdict(pinned, answer):
    """The banner's verdict, mirrored from `web/src/spine.ts`.

    THE LISTS DECIDE AND THE COUNTS NEVER DO: every write to the book file re-pins it at the head,
    the write that is the other half of a divergence included, so a verdict gated on the counts is
    told `clean` about a trade the desk deleted through its own verb. DRIFT IS WHAT THE RECORD
    MOVING FORWARD CANNOT EXPLAIN - a deal the file holds that nobody booked, a clip count the two
    disagree on, or a trade the record holds and the file has LOST under a pin that has seen every
    position there is.
    """
    if pinned is None:
        return 'none'
    lists = answer or NO_LISTS
    lost = len(lists['in_record_not_in_file'])
    if lists['in_file_not_in_record'] or lists['quantity_mismatch'] \
            or (lost and not (pinned['positions_behind'] or 0) > 0):
        return 'drifted'
    return 'behind' if lost else 'clean'


def stored(home, values):
    """`values` in the blob store - where a close's vector and a snapshot live, the writer
    refusing a fact that cites bytes nobody put first."""
    log = opened(home)
    try:
        return log.store.put(values)
    finally:
        log.close()


def filed(home, event_type, body, book=None):
    """One fact through the ordinary writer, the way any other client files one."""
    log = opened(home)
    try:
        return log.append(event_type, body, actor=ACTOR, book=book)['lsn']
    finally:
        log.close()


def test_the_strip_is_every_event_and_a_page_is_what_came_after(recorded, booking):
    """The activity read is the record's own sequence with nothing left out: the fold
    opens no body, so every type is a line and the summary is the declared sentence for it.

    `?since=` answers what came after, and merging that page onto the rows already held reproduces
    the whole strip - the rule `web/src/spine.ts` renders by, mirrored here. With no `?since=`
    `?limit=` keeps the NEWEST rows, which is a strip's first paint.

    Killing mutation: the first page taken from the START instead of the end (`rows[:limit]`),
    which paints a strip with the four genesis events and never what just happened.
    """
    for reference in ('ONE', 'TWO'):
        assert book_one(reference)['written'] is True

    whole = strip()
    assert [row['lsn'] for row in whole['rows']] == list(range(1, head(recorded) + 1))
    assert whole['lsn'] == head(recorded)
    assert [row['event_type'] for row in whole['rows'][-2:]] == ['fill', 'fill']
    assert all(set(row) == STRIP_FIELDS for row in whole['rows'])
    assert all(row['summary'] == projections.SUMMARIES[row['event_type']] for row in whole['rows'])
    assert [row['lsn'] for row in strip(limit=3)['rows']] == [row['lsn'] for row
                                                              in whole['rows'][-3:]]

    assert book_one('THREE')['written'] is True
    page = strip(since=whole['lsn'])
    assert [row['lsn'] for row in page['rows']] == [head(recorded)]
    assert page['lsn'] == head(recorded)
    assert merged(whole['rows'], page) == strip()['rows']
    assert strip(since=head(recorded)) == {'lsn': head(recorded), 'rows': []}

    # a type this hub has no sentence for renders its own NAME rather than dropping out of the
    # sequence, which is what lets a replica of a newer hub still show every LSN
    activity = projections.PROJECTORS['activity']
    state = activity.initial()
    activity.apply(state, dict(page['rows'][0], event_type='a_type_from_a_newer_hub'), None)
    assert activity.rows(state)[0]['summary'] == 'a_type_from_a_newer_hub'


def test_a_page_walks_the_record_forward_one_event_at_a_time(recorded, booking):
    """A PAGE UNDER A CAP WALKS. With `?since=` the page is the OLDEST rows after it and the
    cursor is the last row delivered, so a reader given that cursor reaches every event in turn
    and steps over none; the walk ends at the head with an empty page rather than at a cursor that
    was never true. `?limit=1` is the smallest walk there is, and the whole record is its ten LSNs.

    Killing mutations: the page capped from the END under a `since` (the oldest rows are dropped
    and no later call can reach them, since the cursor is already past them); and the cursor
    answered as the handle's OPEN-TIME head rather than the last row delivered, under which the
    first step of the walk jumps to the end of the record.
    """
    for reference in ('ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX'):
        assert book_one(reference)['written'] is True
    assert head(recorded) == 10

    walked, cursor = [], 0
    for _ in range(11):
        page = strip(since=cursor, limit=1)
        walked.extend(row['lsn'] for row in page['rows'])
        cursor = page['lsn']
    assert walked == list(range(1, 11)), 'a page under a cap lost an event'
    assert strip(since=cursor, limit=1) == {'lsn': 10, 'rows': []}

    # and the empty edges: an unstated `since`, one past the head, and a limit of nothing
    assert strip(since='', limit=2)['rows'] == strip(limit=2)['rows']
    assert strip(since=99) == {'lsn': 10, 'rows': []}
    assert strip(limit=-3) == {'lsn': 10, 'rows': []}
    # an empty page under a cursor hands the cursor back, so a walker with a limit of nothing
    # is not teleported past what it has not read
    assert strip(since=3, limit=0) == {'lsn': 3, 'rows': []}
    assert strip(since=3, limit=-3) == {'lsn': 3, 'rows': []}
    assert CLIENT.get('/book/activity', params={'since': 'yesterday'}).status_code == 422


def test_a_restated_close_names_the_close_it_stands_over(recorded, booking):
    """The markets read is the fold's own answer: the close standing per market carrying
    the LSN of the close it RESTATED - a close is superseded by a new close rather than corrected
    in place - beside the names a values vector was declared under and the snapshots registered.

    Killing mutation: the superseded LSN dropped, after which a restated close is indistinguishable
    from a first one and nothing on the screen says the day was marked twice.
    """
    vector = stored(recorded, b'{"EURUSD":1.0851}')
    restated = stored(recorded, b'{"EURUSD":1.0857}')
    filed(recorded, 'market_declared', {'name': 'official', 'values_hash': vector})
    first = filed(recorded, 'official_close_declared',
                  {'market': 'official', 'values_hash': vector})
    second = filed(recorded, 'official_close_declared',
                   {'market': 'official', 'values_hash': restated})
    snapshot = stored(recorded, b'{"surface":"the vol cube"}')
    filed(recorded, 'snapshot_registered', {'blob': snapshot}, book=CLIENT_SET)

    answer = CLIENT.get('/book/markets').json()
    assert answer['lsn'] == head(recorded)
    assert answer['closes'] == [{'market': 'official', 'values_hash': restated,
                                 'supersedes_lsn': first, 'effective_time': None, 'lsn': second}]
    assert answer['names'] == [{'name': 'official', 'values_hash': vector, 'actor': ACTOR,
                                'effective_time': None, 'lsn': first - 1}]
    assert answer['snapshots'] == [{'blob': snapshot, 'book': CLIENT_SET, 'lsn': head(recorded)}]


def test_the_record_reads_are_a_404_on_a_box_that_records_nothing(unrecorded, desk):
    """A reading refuses nothing and a desk that records nothing is a STATE rather than an
    error: the two reads 404 in the service's own sentence naming the variable, the status block
    is null, and the banner's verdict is `none`.

    Killing mutation: either read answering an empty strip or an empty markets list, which tells a
    desk that keeps no record that its record holds nothing.
    """
    assert spine.configured() is False
    for path in ('/book/activity', '/book/markets'):
        answer = CLIENT.get(path)
        assert answer.status_code == 404, path
        assert spine.SPINE_HOME in answer.json()['detail'], path
    assert CLIENT.get('/book/status').json()['spine'] is None
    assert verdict(None, None) == 'none'


def test_the_banner_reads_clean_then_behind_then_drifted(recorded, booking):
    """What the reconcile banner says, through the answers it says it from: a file written
    under the record is CLEAN; fills the record took and the file never saw are the record being
    AHEAD, by two counts that are not one number; and a deal the file holds that nobody booked is
    DRIFT, which no beat of the poll closes.

    Killing mutation: `positions_behind` counted as every event, after which the fixings policy
    filed past the pin reads as a third trade the file has lost.
    """
    assert book_one('CLEAN')['written'] is True
    pinned = CLIENT.get('/book/status').json()['spine']
    assert (pinned['events_behind'], pinned['positions_behind']) == (0, 0)
    assert verdict(pinned, CLIENT.get('/book/reconcile').json()) == 'clean'

    for reference, amount in (('AHEAD_ONE', 100_000.0), ('AHEAD_TWO', 200_000.0)):
        spine.book(json.loads(dump(dict(CASHFLOW, Reference=reference, Amount=amount))),
                   amount, 'CPTY_A', CLIENT_SET, 'EXEC-' + reference)
    declare(recorded, policy.FIXINGS_POLICY, {'sources': {'FxRate.ZAR': ['ECB']}})
    pinned = CLIENT.get('/book/status').json()['spine']
    assert (pinned['events_behind'], pinned['positions_behind']) == (3, 2)
    answer = CLIENT.get('/book/reconcile').json()
    assert len(answer['in_record_not_in_file']) == 2
    assert answer['in_file_not_in_record'] == answer['quantity_mismatch'] == []
    assert verdict(pinned, answer) == 'behind'

    rewritten(booking, lambda node: node['Children'].append(
        {'Instrument': {'.Deal': dict(json.loads(dump(CASHFLOW)), Reference='BY_HAND')}}))
    drifted = CLIENT.get('/book/reconcile').json()
    assert [row['reference'] for row in drifted['in_file_not_in_record']] == ['BY_HAND']
    assert verdict(CLIENT.get('/book/status').json()['spine'], drifted) == 'drifted'


def test_what_the_file_lost_is_drift_under_a_pin_that_has_seen_every_position(recorded, booking):
    """THE COUNTS NEVER DECIDE, and this is why: a deal deleted through the desk's own verb
    records nothing and re-pins the file AT THE HEAD, so the record holds a live position the file
    has lost while both counts read zero; and a hand edit already shown as drift is re-pinned over
    by the next booking, which moves the counts back to zero with both lists still standing.

    Killing mutation: the verdict gated on the counts - `clean` where they are zero - under which
    a deleted trade is invisible on every screen forever, nothing ever moving those counts again.
    """
    assert book_one('DELETED')['written'] is True
    dropped = CLIENT.post('/book/deals', content=dump(
        {'action': 'delete', 'deal_path': '0/0', 'reference': 'DELETED'}), headers=JSON).json()
    assert dropped['written'] is True

    pinned = CLIENT.get('/book/status').json()['spine']
    assert (pinned['events_behind'], pinned['positions_behind']) == (0, 0)
    answer = CLIENT.get('/book/reconcile').json()
    assert [row['quantity'] for row in answer['in_record_not_in_file']] == [250_000.0]
    assert answer['in_file_not_in_record'] == answer['quantity_mismatch'] == []
    assert verdict(pinned, answer) == 'drifted'

    # the other half: a hand edit, then a booking that re-pins over it - both lists standing and
    # both counts back at zero, which is the answer a desk must not be told is clean
    rewritten(booking, lambda node: node['Children'].append(
        {'Instrument': {'.Deal': dict(json.loads(dump(CASHFLOW)), Reference='BY_HAND')}}))
    assert book_one('AFTER')['written'] is True
    pinned = CLIENT.get('/book/status').json()['spine']
    assert (pinned['events_behind'], pinned['positions_behind']) == (0, 0)
    answer = CLIENT.get('/book/reconcile').json()
    assert [row['reference'] for row in answer['in_file_not_in_record']] == ['BY_HAND']
    assert len(answer['in_record_not_in_file']) == 1
    assert verdict(pinned, answer) == 'drifted'


def test_a_home_with_no_book_at_all_still_reads_the_record(recorded):
    """The strip and the markets open no DOCUMENT, so they answer on a box serving no book - the
    replica posture the strip exists for, where the record is the whole of what there is to read.
    Reconcile is the one that compares against a file, and says so in the book verb's own words.

    Killing mutation: the two reads asking for the live book first, which 404s `No book is being
    served` at a box whose only job is to read the record.
    """
    assert service.BOOK is None
    assert CLIENT.get('/book/activity').json()['lsn'] == head(recorded)
    assert CLIENT.get('/book/markets').json()['lsn'] == head(recorded)
    refused = CLIENT.get('/book/reconcile')
    assert refused.status_code == 404 and 'No book is being served' in refused.json()['detail']


def test_a_shredded_home_answers_the_records_own_words_and_never_a_500(recorded, booking):
    """A home whose class key is gone is CRYPTO-SHREDDED, which is an entitlement fact rather than
    a fault: the strip still reads, opening no body, and every read that opens one answers the
    spine's own sentence as a 422.

    Killing mutation: the handler dropped, after which a reading takes the verb down and a desk
    meets a stack trace where the record's own words belong.
    """
    assert book_one('SHREDDED')['written'] is True
    filed(recorded, 'official_close_declared',
          {'market': 'official', 'values_hash': stored(recorded, b'{"EURUSD":1.0851}')})
    key = recorded / 'keys' / 'class_firm.key'
    shredded = key.with_suffix('.gone')
    key.rename(shredded)
    try:
        assert CLIENT.get('/book/activity').status_code == 200
        for path in ('/book/markets', '/book/reconcile'):
            refused = CLIENT.get(path)
            assert refused.status_code == 422, path
            assert 'class_firm.key' in refused.json()['detail'], path
    finally:
        shredded.rename(key)


# --------------------------------------------------------------------------------------------
# The plan compiler as a fold over fixings supersession.

EQUITY = {
    'EquityPrice.EQ': {'Spot': 100.0, 'Currency': 'USD', 'Interest_Rate': 'USD', 'Issuer': '',
                       'Respect_Default': 'No', 'Jump_Level': 0.0},
    'DividendRate.EQ': {'Currency': 'USD', 'Floor': None,
                        'Curve': utils.Curve([], [[0.0, 0.02], [5.0, 0.02]])},
    'VolatilityGrid.EQ': {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
                          'Surface': utils.Curve([], [[m, t, 0.25] for m in (0.6, 1.0, 1.4)
                                                      for t in (0.02, 2.0)])}}
WATCHED = BASE - pd.DateOffset(days=30)
INDEX = 'EquityPrice.EQ'


def barrier_book(observed=''):
    """A book whose one deal declares its observations in `Barrier_Dates` - one monitoring date
    already past, blank unless the desk typed a close into it."""
    return job(deals=({
        'Object': 'EquityBarrierOption', 'Reference': 'BR', 'Currency': 'USD',
        'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
        'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
        'Strike_Price': 100.0, 'Units': 1.0, 'Cash_Rebate': 0.0,
        'Expiry_Date': BASE + pd.DateOffset(days=365), 'Barrier_Type': 'Up_And_Out',
        'Barrier_Price': 115.0, 'Barrier_Monitoring_Frequency': pd.DateOffset(days=0),
        'Barrier_Dates': [[WATCHED, observed]]},), factors=dict(FACTORS, **EQUITY))


def observe(home, value, effective_time=None, index=INDEX, date=WATCHED):
    """One administrator's print of a day, filed as the fact it is."""
    log = opened(home)
    try:
        return log.append('fixing_observed', {
            'index': index, 'date': date.strftime('%Y-%m-%d'), 'source': 'EXCHANGE',
            'value': value}, actor=ACTOR, effective_time=effective_time)
    finally:
        log.close()


def watched_row(document):
    return document['Calc']['Deals']['Deals']['Children'][0]['Instrument']['.Deal'][
        'Barrier_Dates']


def plan_of(document):
    return derivus.Context().load_json((dump(document), 'compiled')).plan_hash()


def test_the_plan_compiles_from_the_fold_rather_than_from_what_was_typed(recorded, desk):
    """GATE 10. The plan is terms PLUS the observations the record holds: the fold writes the
    watched day's close onto the deal's own `Barrier_Dates` row, and that is a different program
    from the one whose cell is blank - and the same program the close typed by hand compiles to.

    Killing mutation: `compiled_job` returning its argument under a configured home, which leaves
    the plan the desk typed and the record's own observations disagreeing in silence.
    """
    declare(recorded, policy.FIXINGS_POLICY, {'sources': {INDEX: ['EXCHANGE']}})
    observe(recorded, 108.5)

    terms = barrier_book()
    compiled = spine.compiled_job(terms)
    assert watched_row(compiled) == [[WATCHED, 108.5]]
    assert watched_row(terms) == [[WATCHED, '']], 'the submitted document was edited in place'
    assert plan_of(compiled) != plan_of(terms), 'the fold did not move the plan'
    assert plan_of(compiled) == plan_of(barrier_book(108.5)), \
        'the folded plan is not the plan the same close typed by hand compiles to'

    # and the standing lane still recompiles to its own recorded hash out of the stored blob
    standing = drained(submit(own_job('FOLD-PROVENANCE'), lane=spine.STANDING))
    _, _, body = facts(recorded, 'run_completed')[0]
    recompiled = derivus.Context().load_json(
        (blob(recorded, body['job']).decode('utf-8'), 'recompiled'))
    recompiled.patch_market(spine.read_values(blob(recorded, body['values_hash'])))
    assert recompiled.plan_hash() == body['plan_hash'] == standing['plan_hash']


def test_a_republished_fixing_moves_the_plan_and_each_lsn_keeps_its_own(recorded, desk):
    """GATE 11. Two prints of one `(index, date, source)`: the plan compiled at the earlier LSN is
    the earlier print's and the plan compiled now is the restatement's, so an auditor recompiles AT
    an LSN and gets that day's answer rather than today's.

    Killing mutation: `fixings_at` ordering by LSN alone, so a print backdated behind the one in
    force wins merely by arriving last and every plan ever recompiled moves with it.
    """
    declare(recorded, policy.FIXINGS_POLICY, {'sources': {INDEX: ['EXCHANGE']}})
    first = observe(recorded, 108.5, effective_time='2024-06-27T16:00:00.000000Z')['lsn']
    observe(recorded, 111.25, effective_time='2024-06-27T17:30:00.000000Z')

    terms = barrier_book()
    earlier, later = spine.compiled_job(terms, lsn=first), spine.compiled_job(terms)
    assert watched_row(earlier) == [[WATCHED, 108.5]]
    assert watched_row(later) == [[WATCHED, 111.25]]
    assert plan_of(earlier) != plan_of(later)
    assert plan_of(earlier) == plan_of(barrier_book(108.5))
    assert plan_of(later) == plan_of(barrier_book(111.25))


def test_a_plan_named_and_a_plan_run_are_one_plan(recorded, desk):
    """`/prepare` names a parse and `/execute` runs one, so both compile the job against the record
    first: a client that streams a plan once and ticks a delta runs the plan the record makes, not
    the one that was typed.

    Killing mutation: `/prepare` loading the raw document, under which the plan id it hands back
    names the terms-only program and the two routes to one job are two runs.
    """
    declare(recorded, policy.FIXINGS_POLICY, {'sources': {INDEX: ['EXCHANGE']}})
    observe(recorded, 108.5)

    terms = barrier_book()
    named = CLIENT.post('/prepare', content=dump(terms), headers=JSON).json()
    assert named['plan_id'] == plan_of(spine.compiled_job(terms))
    assert named['plan_id'] != plan_of(terms), 'the record holds a close this plan did not read'

    posted = CLIENT.post('/execute', content=dump(terms), headers=JSON).json()
    streamed = CLIENT.post('/execute', content=dump({'plan_id': named['plan_id']}),
                           headers=JSON).json()
    assert posted['result_id'] == streamed['result_id'], 'two routes, two runs'
    service.EXECUTOR.queue.join()


def test_a_print_dated_after_the_base_date_is_left_standing(recorded, desk):
    """A `fixing_observed` carries its date as text, so a forward-dated print is a legal fact. The
    compiler fills a monitoring row only where its day is ON OR BEFORE the book's own.

    Killing mutation: the fold written into every row the record holds a print for, which prices a
    barrier as already observed on a day that has not happened.
    """
    ahead = BASE + pd.DateOffset(days=30)
    declare(recorded, policy.FIXINGS_POLICY, {'sources': {INDEX: ['EXCHANGE']}})
    observe(recorded, 108.5)
    observe(recorded, 133.75, date=ahead)

    both = job(deals=(dict(barrier_book()['Calc']['Deals']['Deals']['Children'][0][
        'Instrument']['.Deal'], Barrier_Dates=[[WATCHED, ''], [ahead, '']]),),
        factors=dict(FACTORS, **EQUITY))
    filled = watched_row(spine.compiled_job(both))
    assert filled[0][1] == 108.5, 'the day behind the base date was not filled'
    assert filled[1][1] == '', 'a print dated after the base date was written onto the row'


def test_a_fixing_whose_authority_nobody_declared_refuses_by_name(recorded, desk):
    """GATE 12. A plan may not read a fixing nobody vouched for: an index THIS PLAN COMPILES
    AGAINST that the `fixings` policy does not name refuses BY NAME, and the refusal reaches a desk
    as a 422 carrying the spine's own sentence rather than a paraphrase of it. An index the log
    holds prints of that no deal here names is left alone, because the compiler asks for exactly
    the indices its deals declare.

    Killing mutation: the compiler asking for every index the log holds prints of rather than its
    own, which refuses a plan over an administrator nobody involved in it ever chose.
    """
    stranger = 'FxRate.ZAR'
    observe(recorded, 108.5)
    observe(recorded, 18.5, index=stranger)

    declare(recorded, policy.FIXINGS_POLICY, {'sources': {stranger: ['ECB']}})
    with pytest.raises(spine.SpineRefused) as refusal:
        spine.compiled_job(barrier_book())
    assert INDEX in str(refusal.value) and 'EXCHANGE' in str(refusal.value)

    refused = CLIENT.post('/execute', content=dump(dict(barrier_book(), lane=spine.CURIOSITY)),
                          headers=JSON)
    assert refused.status_code == 422
    assert str(refusal.value) == refused.json()['detail'], 'the desk read a paraphrase'

    # the deal's own index declared and the stranger's still not: the plan compiles regardless
    declare(recorded, policy.FIXINGS_POLICY, {'sources': {INDEX: ['EXCHANGE']}})
    assert watched_row(spine.compiled_job(barrier_book())) == [[WATCHED, 108.5]]
