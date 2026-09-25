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
  * THE IDENTITY: a tick that re-reads a board at the same prices leaves `values_hash`
    bit-identical, one moved mid moves it, and the stored vector reads back onto another market.
  * THE ACCEPTANCE, in order: the desk's own window, the book having MOVED under the solve, the
    BOARD the quote was struck on being older than a declared window, and the ticket re-derived -
    each refusing with the head unmoved - while a moved MARKET is reported and books. The values
    and plan planes are DISJOINT: a vol tick through the real partition moves `values_hash` and
    leaves `plan_hash` bit-identical.
  * THE TIER STEP inside the closure, the decision verbs, and the acceptance standing whatever the
    workflow then says.
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
from derivus import service, spine, structures, utils
from derivus.config import CustomJsonEncoder
from derivus.schema import deal_at, walk_job_deals
from derivus_spine import SpineLog, init_home, verify_home
from derivus_spine import policy, projections, verbs
from derivus_spine.capability import CAPABILITIES_POLICY, canonical_document

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


#: When the canned snapshot says its quotes were seen, and a later reading of the same board.
SNAPPED = pd.Timestamp('2024-06-28 16:30')
RESNAPPED = pd.Timestamp('2024-06-28 17:45')


def fx_vol_snapshot(atm_3m=14.0, seen=SNAPPED):
    """A USDZAR snapshot through the Bloomberg package's own normalization - canned observations
    standing in for the terminal, the real pipeline downstream. `atm_3m` is all a TICK moves, and
    `seen` is when the board was read, which moves every row's stamp and no number."""
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
    return normalize_fx_vol(definition, observations, seen)


def fx_vol_quotes(atm_3m=14.0, seen=SNAPPED):
    from derivus_bloomberg import to_market_prices_block
    return {'FXVolPrices.USD.ZAR': to_market_prices_block(fx_vol_snapshot(atm_3m, seen))}


def tick(atm_3m=14.0, seen=SNAPPED):
    """One market tick through the real verb, asserted to have landed."""
    ticked = CLIENT.post('/book/market', content=dump({'quotes': fx_vol_quotes(atm_3m, seen)}),
                         headers=JSON).json()
    assert ticked['written'] is True, ticked
    return ticked


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
    tick()
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
    """Declare one of the reserved policies onto a home, through the ordinary writer."""
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
    silently un-record a box somebody meant to record.

    The refusal is the QUEUE's since 5c, so it arrives at the submission in the same words rather
    than as the run's own error after the numbers were computed - a record nobody can open is one
    no job on this box gets past, whatever lane it declared.
    """
    monkeypatch.setenv('DV_SPINE_HOME', str(tmp_path / 'never-minted'))
    monkeypatch.setenv('DV_SPINE_ACTOR', ACTOR)
    for lane in (spine.STANDING, spine.CURIOSITY):
        refused = CLIENT.post('/execute', content=dump(dict(own_job('NO-HOME'), lane=lane)),
                              headers=JSON)
        assert refused.status_code == 422, lane
        said = refused.json()['detail']
        assert 'log/' in said and 'DV_Spine init' in said, lane


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
    error and the record holds the denial and nothing else.

    THE QUEUE ASKS FOR THE SCOPE THE APPEND WILL NEED, so the two checks can only disagree where the
    DOCUMENT MOVED between them: the worker is held behind a barrier, the job is admitted under a
    `book` grant over `*`, that grant is withdrawn while the run waits, and the attestation meets a
    document the submission never saw. Anything else would be the hub paying for a Monte Carlo the
    record was always going to refuse - increment 3's own boundary, which this closes.
    """
    entitle(recorded, [(ACTOR, verb, '*') for verb in ('admin', 'validate', 'book')])
    barrier = holding()
    try:
        assert barrier.running.wait(WORKER_SECONDS), 'the worker never picked the barrier up'
        submitted = submit(own_job('UNSCOPED'), lane=spine.STANDING)
        assert submitted['status'] == 'queued', submitted
        entitle(recorded, [(ACTOR, verb, '*') for verb in ('admin', 'validate')])
        standing = head(recorded)
    finally:
        barrier.released.set()

    result = drained(submitted)
    assert result['status'] == 'error'
    assert 'run_completed' in result['error'] and ACTOR in result['error']
    assert [event_type for _, event_type, _ in facts(recorded)][standing:] == \
        ['capability_denied'], 'the record kept an attestation nobody was scoped for'

    # and the ordinary desk seat - `book` over its own book and nothing firm-level - never reaches
    # the executor at all, because the fact it would file is firm-level
    entitle(recorded, [(ACTOR, verb, '*') for verb in ('admin', 'validate')]
            + [(ACTOR, 'book', 'spine-desk')])
    stored = dict(service.EXECUTOR.results)
    turned = CLIENT.post('/execute', content=dump(dict(own_job('UNADMITTED'),
                                                       lane=spine.STANDING)), headers=JSON)
    assert turned.status_code == 422 and 'book' in turned.json()['detail']
    assert "over '*'" in turned.json()['detail'], 'the queue asked over the job\'s own book'
    assert service.EXECUTOR.results == stored, 'the queue ran a job it had refused'


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

    marked = context.declare_market('private/{}/screen'.format(ACTOR))
    assert marked['values_hash'] == context.values_hash(), \
        'a market names the values vector this context is carrying'
    assert blob(recorded, marked['values_hash']) == spine.values_of(context)

    assert verify_home(recorded)['events'] == head(recorded) == 4 + 6


def test_a_market_resolves_by_name_and_a_private_one_resolves_for_its_owner(recorded):
    """The composition nothing performed before: a NAME, folded to a values hash, fetched from the
    store and read back as the objects `patch_market` takes. The proof is that it moves a book - a
    second context carrying another spot is patched onto the declared market and hashes to it.

    A name nobody declared refuses rather than falling back on whatever market is loaded, which is
    the entire point of binding a process to a market by name. A `private/<subject>/<name>` market
    resolves for the SUBJECT ITS NAME NAMES - which the declaring verb holds to the seat declaring
    it, so the name and the fold cannot disagree about who owns a board - and for nobody else here:
    the surveillance and admin read of one is an entitlement class this record does not yet
    classify, and a per-market ACL is the thing the design forbids.

    A NAME RESOLVES TO ITS LATEST DECLARATION, and an official close is one way of declaring what a
    market stands on, so a close at a second vector moves what the name answers and a declaration
    after it moves it back. Latest by the fold's as-of key, so a close backdated behind the
    declaration in force does not displace it by arriving last.
    """
    context = derivus.Context().load_json((dump(job()), 'posted'))
    context.declare_market('official')
    context.declare_market('private/subject-desk-two/screen', actor='subject-desk-two')
    other = derivus.Context().load_json((dump(job(factors=dict(
        FACTORS, **{'FxRate.ZAR': dict(FACTORS['FxRate.ZAR'], Spot=SPOT + 1.0)}))), 'posted'))

    standing = spine.resolve_market('official')
    assert standing == {'name': 'official', 'values_hash': context.values_hash(),
                        'lsn': standing['lsn'], 'values': standing['values']}
    assert other.values_hash() != context.values_hash()
    other.patch_market(standing['values'])
    assert other.values_hash() == context.values_hash(), \
        'the resolved market did not read back as the market it names'

    with pytest.raises(spine.SpineRefused) as refusal:
        spine.resolve_market('settlement')
    assert "'settlement'" in str(refusal.value) and 'official' in str(refusal.value)

    mine = spine.resolve_market('private/subject-desk-two/screen', actor='subject-desk-two')
    assert mine['values_hash'] == context.values_hash()
    for stranger in (ACTOR, None):
        with pytest.raises(spine.SpineRefused) as refusal:
            spine.resolve_market('private/subject-desk-two/screen', actor=stranger)
        assert 'entitlement class' in str(refusal.value), stranger
        assert 'subject-desk-two' in str(refusal.value)

    # the close is a declaration of the name, and the two orders answer the two vectors
    closed = derivus.Context().load_json((dump(job(factors=dict(
        FACTORS, **{'FxRate.ZAR': dict(FACTORS['FxRate.ZAR'], Spot=SPOT + 2.0)}))), 'posted'))
    restated = derivus.Context().load_json((dump(job(factors=dict(
        FACTORS, **{'FxRate.ZAR': dict(FACTORS['FxRate.ZAR'], Spot=SPOT + 3.0)}))), 'posted'))
    assert len({context.values_hash(), closed.values_hash(), restated.values_hash()}) == 3
    closed.declare_close('official')
    assert spine.resolve_market('official')['values_hash'] == closed.values_hash(), \
        'a close on the name did not move what the name resolves to'
    restated.declare_market('official')
    assert spine.resolve_market('official')['values_hash'] == restated.values_hash(), \
        'a declaration after a close did not move it back'
    # latest by the as-of key, not by LSN: a close BACKDATED behind the name's declaration is the
    # record's last row on that market and does not displace it - the fold's own supersession rule
    restated.declare_market('desk')
    closed.declare_close('desk', effective_time='2020-01-02T16:00:00.000000Z')
    assert spine.resolve_market('desk')['values_hash'] == restated.values_hash(), \
        'a backdated close displaced the declaration in force by arriving last'

    assert verify_home(recorded)['events'] == head(recorded)


def test_a_designated_process_resolves_the_market_the_policy_names_and_never_a_private_one(
        recorded):
    """`process=` is the other half of the binding: the name must be the market the tiers policy
    DESIGNATES for that process, so a settlement export cannot be pointed at a market somebody
    picked. A home that designates nothing refuses too - a rule nobody declared is not one a
    process may assume - and a `private/` name refuses whoever asks, its owner included, because no
    designation is ever private.

    The two rules are checked independently rather than one behind the other: a stranger asking for
    a private market under a process meets the OWNER rule, and its owner asking meets the
    designation rule, so neither is dead behind the other.
    """
    context = derivus.Context().load_json((dump(job()), 'posted'))
    context.declare_market('official')
    context.declare_market('dealer')
    context.declare_market('private/subject-desk-two/screen', actor='subject-desk-two')

    with pytest.raises(spine.SpineRefused) as undesignated:
        spine.resolve_market('official', process='settlement_export')
    assert 'nothing at all' in str(undesignated.value)

    assert spine.tiers_policy() is None
    declare(recorded, policy.TIERS_POLICY, {'tiers': [{'name': 'desk', 'four_eyes': True}],
                                            'designations': {'settlement_export': 'official'}})
    assert spine.tiers_policy()['designations'] == {'settlement_export': 'official'}

    assert spine.resolve_market('official', process='settlement_export')['values_hash'] == \
        context.values_hash()
    for named in ('dealer', 'private/subject-desk-two/screen'):
        with pytest.raises(spine.SpineRefused) as refusal:
            spine.resolve_market(named, process='settlement_export', actor='subject-desk-two')
        said = str(refusal.value)
        assert 'settlement_export' in said and "'official'" in said and named in said, named

    # the owner rule is not dead behind the designation rule: a stranger naming a process meets it
    with pytest.raises(spine.SpineRefused) as refusal:
        spine.resolve_market('private/subject-desk-two/screen', process='settlement_export',
                             actor=ACTOR)
    assert 'entitlement class' in str(refusal.value)
    assert verify_home(recorded)['events'] == head(recorded)


def test_the_seam_files_a_decision_a_close_and_reads_them_back_as_folds(recorded):
    """The three delegators and the three reads, on one home. A verdict is never withdrawn, so
    `verdicts` answers the LIST in the order it was filed and never a boolean - "approved" and
    "nobody has ruled" have different remedies. A quote names the seat that struck it off the
    envelope, since no body carries one.
    """
    context = derivus.Context().load_json((dump(job()), 'posted'))
    plan = context.plan_hash()

    context.reject(plan, 'the booker and the approver are one seat', book='spine-desk')
    signed = context.approve(plan, actor='subject-desk-two', book='spine-desk')
    assert [(row['verdict'], row['actor']) for row in spine.verdicts(plan)] == [
        ('rejection', ACTOR), ('approval', 'subject-desk-two')]
    assert spine.verdicts(plan)[-1]['lsn'] == signed['lsn']
    assert spine.verdicts('f' * 64) == [], 'a plan nobody ruled on answered somebody else\'s list'

    closed = context.declare_close('official')
    assert facts(recorded, 'official_close_declared')[0][2] == {
        'market': 'official', 'values_hash': context.values_hash()}
    assert closed['values_hash'] == context.values_hash()

    spine.file_quote('Q-1', 'ZeroCostCollar', plan, spine.values_of(context), {'floor': 17.25},
                     4200.0, ticket='c' * 64, actor_name='subject-desk-two', book_name='spine-desk')
    quoted = spine.quotes()
    assert len(quoted) == 1 and quoted[0]['booker'] == 'subject-desk-two'
    assert quoted[0]['ticket'] == 'c' * 64 and quoted[0]['plan_hash'] == plan
    assert quoted[0]['values_hash'] == context.values_hash()
    assert verify_home(recorded)['events'] == head(recorded)


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


def test_the_paper_is_declared_and_a_booking_names_the_agreement_it_sits_under(recorded, desk):
    """Legal's half and the desk's, through the service.

    An entity is declared and then an agreement with it, its terms judged by the engine before
    anything appends: a block that is not a netting set, one carrying positions, one stating a
    balance or a holding, one its own declarations refuse and one naming an entity nobody declared
    each refuse by name with nothing recorded. A booking naming the agreement files it on the fill,
    the counterparty being the agreement's entity, beside the portfolio - the book's own where none
    is stated - and the price it was done at; the position is keyed by the agreement and the
    portfolio. A booking naming an agreement nobody declared, or sitting under another set than
    the agreement's own, refuses by name and leaves the file byte-identical.

    Killing mutations: the counterparty read off the file's set rather than the agreement books
    `CPTY_A` where the paper says `LEI-A`; the portfolio default dropped files the second booking
    under the book only through the fold's fallback, and the fill carries no portfolio.
    """
    def declared(body):
        return CLIENT.post('/book/agreements', content=dump(body), headers=JSON)

    terms = netting_set(CLIENT_SET, 'CPTY_A')['Instrument']['.Deal']
    assert CLIENT.post('/book/entities', content=dump(
        {'entity': 'LEI-A', 'name': 'Client A', 'parent': 'LEI-GROUP'}),
        headers=JSON).status_code == 200
    head_before = head(recorded)
    for body, said in (
            (dict(terms, Object='FixedCashflowDeal'), 'NettingCollateralSet block'),
            (dict(terms, Children=[{'Instrument': {'.Deal': CASHFLOW}}]), 'carry Children'),
            (dict(terms, Opening_Balance=1_000_000.0), 'Opening_Balance is 1000000.0'),
            (dict(terms, Collateral_Assets={'Cash_Collateral': [{'Currency': 'USD',
                                                                  'Amount': 500_000.0}]}),
             'Cash_Collateral[0].Amount is 500000.0'),
            (dict(terms, Netted='Sometimes'), 'Netted is')):
        refused = declared({'agreement': CLIENT_SET, 'entity': 'LEI-A', 'kind': 'ISDA 2002',
                            'terms': body})
        assert refused.status_code == 422 and said in refused.json()['detail'], said
    stranger = declared({'agreement': CLIENT_SET, 'entity': 'LEI-Z', 'kind': 'ISDA 2002',
                         'terms': terms})
    assert stranger.status_code == 422 and 'declare the entity first' in stranger.json()['detail']
    assert head(recorded) == head_before, 'a refused declaration recorded something'

    signed = declared({'agreement': CLIENT_SET, 'entity': 'LEI-A', 'kind': 'ISDA 2002 with CSA',
                       'terms': terms}).json()
    agreements = CLIENT.get('/book/agreements').json()['agreements']
    assert [(row['agreement'], row['entity'], row['kind']) for row in agreements] == [
        (CLIENT_SET, 'LEI-A', 'ISDA 2002 with CSA')]
    assert agreements[0]['terms_hash'] == signed['terms']
    assert agreements[0]['terms']['Reference'] == CLIENT_SET
    assert CLIENT.get('/book/entities').json()['entities'][0]['parent'] == 'LEI-GROUP'

    document = json.loads(desk.read_text())
    document['Calc']['Deals']['Deals']['Children'] += [
        netting_set(CLIENT_SET, 'CPTY_A'), netting_set('CLIENT_B', 'CPTY_B')]
    desk.write_text(json.dumps(document, indent=2), newline='\n')
    before = desk.read_bytes()
    deal = dict(CASHFLOW, Reference='CF-PAPER', Amount=250_000.0)
    for parent, agreement, said in (('CLIENT_B', CLIENT_SET, 'materialisation'),
                                    (CLIENT_SET, 'ISDA-NOBODY', 'the record declares')):
        refused = CLIENT.post('/book/deals', content=dump({
            'action': 'add', 'deal': deal, 'parent_reference': parent, 'agreement': agreement,
            'quantity': 1.0, 'execution_reference': 'EXEC-PAPER'}), headers=JSON)
        assert refused.status_code == 422 and said in refused.json()['detail'], said
        assert desk.read_bytes() == before, 'a refused booking moved the file'

    for reference, portfolio in (('EXEC-PAPER', 'Rates/EM'), ('EXEC-BOOK', None)):
        booked = CLIENT.post('/book/deals', content=dump(dict(
            {'action': 'add', 'deal': dict(deal, Reference=reference), 'parent_reference':
             CLIENT_SET, 'agreement': CLIENT_SET, 'quantity': 1.0, 'price': 99.5,
             'execution_reference': reference},
            **({} if portfolio is None else {'portfolio': portfolio}))), headers=JSON).json()
        assert booked['written'] is True, booked
    fills = [body for _, _, body in facts(recorded, 'fill')]
    assert [(fill['agreement'], fill['counterparty'], fill['portfolio'], fill['price'])
            for fill in fills] == [(CLIENT_SET, 'LEI-A', 'Rates/EM', 99.5),
                                   (CLIENT_SET, 'LEI-A', 'spine-desk', 99.5)]
    projections = service.spine.package().projections
    positions = projections.PROJECTORS['positions']
    rows = service.spine.folded(lambda log: positions.rows(projections.fold(log, positions)))
    assert {(row['agreement'], row['portfolio']) for row in rows} == {
        (CLIENT_SET, 'Rates/EM'), (CLIENT_SET, 'spine-desk')}
    assert verify_home(recorded)['events'] == head(recorded)


# --------------------------------------------------------------------------------------------
# A market is its numbers.

def test_a_board_read_again_at_the_same_prices_is_the_same_market(recorded, quoting):
    """THE IDENTITY GATE: a tick delivering the SAME numbers under a later stamp leaves
    `values_hash` bit-identical, and one moved mid moves it.

    Rehashing on every tick because a row's clock moved brings no information - when a board was
    read is data on the row, nothing in pricing reads it, and the record stamps every event with its
    own clock. The FILE moves either way, the stamp being value-plane data a tick writes, so this is
    the hash alone that stops reading it.

    Killing mutation: `values_hash` taken of `market_patch()` whole, which calls one market two and
    reports a quote's market as moved after every beat of a cadence that changed no price.
    """
    struck = service.load(service.BOOK.read()[0])
    before = quoting.read_bytes()

    tick(14.0, RESNAPPED)
    restamped = service.load(service.BOOK.read()[0])
    assert quoting.read_bytes() != before, 'the stamp never reached the file - this proved nothing'
    assert restamped.values_hash() == struck.values_hash(), 'a clock moved the market'
    assert restamped.plan_hash() == struck.plan_hash()

    tick(14.9, RESNAPPED)
    moved = service.load(service.BOOK.read()[0])
    assert moved.values_hash() != struck.values_hash(), 'a moved mid moved nothing'
    assert moved.plan_hash() == struck.plan_hash()


def test_the_stored_values_vector_round_trips_onto_another_market(recorded, quoting):
    """`values_of` is the bytes `values_hash` NAMES, so the vector the record stores and the hash a
    quote pins are one number - and reading it back onto a context carrying another board patches
    the numbers and leaves that context's own stamps where they were.

    Killing mutation: the projection taken in one of the two and not the other, which makes every
    stored vector address something the record does not hold.
    """
    struck = service.load(service.BOOK.read()[0])
    assert derivus.content_hash(json.loads(spine.values_of(struck).decode('utf-8'))) == \
        struck.values_hash()

    tick(14.9, RESNAPPED)
    other = service.load(service.BOOK.read()[0])
    stamps = other.market_patch()['FXVolPrices.USD.ZAR']['Points']
    assert other.values_hash() != struck.values_hash()

    other.patch_market(spine.read_values(spine.values_of(struck)))
    assert other.values_hash() == struck.values_hash(), 'the vector did not read back'
    assert [row.get('Timestamp') for row in other.market_patch()[
        'FXVolPrices.USD.ZAR']['Points']] == [row.get('Timestamp') for row in stamps], \
        'reading a values vector moved the reader\'s own clocks'


# --------------------------------------------------------------------------------------------
# The quote: nothing recorded, everything the acceptance needs on the file.

def pending_of(tmp_path, quote):
    """The pending trade as it stands on disk."""
    return json.loads((tmp_path / 'tmp' / (quote['quote_id'] + '.json')).read_text())


def test_a_quote_records_nothing_and_files_what_the_acceptance_will(recorded, quoting, tmp_path):
    """WE DO NOT CARE ABOUT A QUOTE UNTIL THE CLIENT ACCEPTS IT. A desk quotes fifteen times a day
    and cares about the one that comes back, so the head does not move here - asserted as ABSENCE on
    the head rather than on a filtered count - and the fourteen others die in `DV_HOME/tmp`.

    What the file carries is everything the acceptance will file: the book's two hashes, the values
    vector behind them canonicalising back to the hash it is filed under, the TICKET the acceptance
    re-derives off this same book, the age of the oldest stamped pillar, and who quoted it.

    Killing mutation: `/book/structure` left in the standing lane, which files a `quote_filed` for
    every price a salesperson ever says out loud.
    """
    document, _ = service.BOOK.read()
    context = service.load(document)
    before = head(recorded)
    quote = quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET,
                     request='the client wants a year of downside at zero cost')

    assert head(recorded) == before, 'a quote moved the record'
    filed = pending_of(tmp_path, quote)
    assert filed['quoted_by'] == ACTOR and filed['request'].startswith('the client wants')
    pinned = filed['pinned']
    assert pinned['plan_hash'] == context.plan_hash()
    assert pinned['values_hash'] == context.values_hash()
    assert spine.canonical(pinned['values']) == spine.values_of(context)
    assert derivus.content_hash(pinned['values']) == pinned['values_hash'], \
        'the vector on the file is not the one the hash beside it names'
    assert pinned['ticket'] == service.quote_ticket(document, filed['deal'], CLIENT_SET)
    assert pinned['ticket'] != pinned['plan_hash'], 'the ticket is the book plus this quote'
    # the board was stamped at 16:30 on the base date, so the age is the wall clock since then
    assert pinned['pillar_age'] == pytest.approx(
        service.quote_age(SNAPPED.isoformat(), filed['quoted_at']), rel=1e-6)
    assert quote['pinned'] == {name: value for name, value in pinned.items() if name != 'values'}


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


def test_a_quote_pins_the_book_before_the_live_spot_lands_on_its_copy(recorded, quoting, tmp_path):
    """The pins are the BOOK's, and the ordering that makes them so is gated rather than trusted.
    `StructureJob.run_job` takes its two hashes at the TOP, before `patch_live_spot` moves this
    copy's spot: pin after and `values_hash` describes a market existing only inside one quote, so
    every booking on a box with a live terminal would report a market that never moved as moved.

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
    _, out = job.run_job()

    assert job.params.moved is True, 'the spot never moved - this gate proved nothing'
    quoted = service.load(job.document)
    assert quoted.values_hash() != book_context.values_hash(), 'the live spot moved nothing'
    assert quoted.plan_hash() == book_context.plan_hash(), \
        'a spot moved the plan - the values plane and the plan plane are not disjoint'

    pinned = pending_of(tmp_path, out['Stats']['Quote'])['pinned']
    assert pinned['values_hash'] == book_context.values_hash(), \
        'the quote pinned its own spot-patched market rather than the book it would land against'
    assert pinned['plan_hash'] == book_context.plan_hash()
    assert spine.canonical(pinned['values']) == spine.values_of(book_context)

    # the booking reads the same pair off the desk's copy, so it stands against the book
    verdict = spine.package().firmness.assess(
        pinned, {'plan_hash': book_context.plan_hash(), 'values_hash': book_context.values_hash()},
        1.0, spine.firmness_policy())
    assert verdict['firm'] is True and verdict['market']['moved'] is False


# --------------------------------------------------------------------------------------------
# The acceptance: the record first, then the tier, then the fill, then the file.

def accept(quote, **body):
    return CLIENT.post('/book/quote', json=dict({'quote_id': quote['quote_id']}, **body))


def typed(home, *event_types):
    """`(lsn, type)` for every event of these types, in LSN order - what an ORDERING assertion is
    made of."""
    return [(lsn, event_type) for lsn, event_type, _ in facts(home)
            if event_type in event_types]


def test_the_acceptance_files_the_quote_then_the_fill_and_a_retry_coalesces(recorded, quoting):
    """THE ACCEPTANCE IS WHERE THE RECORD MOVES. With no tiers policy the flow is what it was: the
    `quote_filed` under the ACCEPTOR's seat, then the `fill` citing the quote id as its execution
    reference, at consecutive LSNs, then the file. The quote carries the TICKET the acceptance
    re-derived, which is the plan an approval of it would sign.

    THE TICKET IS HELD AGAINST THE WORLD, not against itself: the plan the quote WOULD leave the
    book at is the plan it DID leave it at, read off the file the booking wrote.

    AND AN ACCEPTANCE THAT BOOKED IS TOLD SO BY NAME. The booking was itself a move of the book, so
    the plan check would send a salesperson to re-quote a trade the desk has already done; the
    pending file says where it landed and the second acceptance answers that instead. The retry that
    coalesces is the one a tier held back, which is the desk tier's own gate.

    Killing mutations: the fill appended before the quote, which files a trade against a quote the
    record does not yet hold; `quote_ticket` splicing at the root, which mints a ticket for a plan
    no booking reaches.
    """
    quote = quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET,
                     request='the client wants a year of downside at zero cost')
    struck = pending_of(quoting.parent, quote)['pinned']
    before = quoting.read_bytes()
    booked = accept(quote).json()

    assert booked['written'] is True and quoting.read_bytes() != before
    filed, fill = typed(recorded, 'quote_filed', 'fill')
    assert filed[1] == 'quote_filed' and fill[1] == 'fill'
    assert fill[0] == filed[0] + 1, 'the quote and its fill are not one act'
    assert booked['accepted'] == {'lsn': filed[0], 'ticket': struck['ticket']}
    assert pending_of(quoting.parent, quote)['accepted'] == booked['accepted']

    _, _, body = facts(recorded, 'quote_filed')[0]
    assert body['quote_id'] == quote['quote_id'] and body['structure'] == 'ZeroCostCollar'
    assert body['ticket'] == struck['ticket'] and body['plan_hash'] == struck['plan_hash']
    assert body['request'].startswith('the client wants')
    assert len(body['solved']) == 1 and body['edge'] == pytest.approx(quote['edge'])
    assert blob(recorded, body['values_hash']) == spine.canonical(struck['values'])

    _, _, fill_body = facts(recorded, 'fill')[0]
    assert fill_body['execution_reference'] == quote['quote_id']
    assert fill_body['netting_set'] == CLIENT_SET and fill_body['counterparty'] == 'CPTY_A'
    assert fill_body['quantity'] == 1.0, 'one unit of the mirror as written'
    landed = json.loads(quoting.read_text())
    node = deal_at(landed, booked['deal_path'])
    assert fill_body['instrument'] == derivus.content_hash(service.instrument_of(node))
    # the side and the size live in the mirror's own terms, which is the instrument the fill cites
    bought = [leg['buy_sell'] for leg in quote['legs'] if leg['buy_sell']][0]
    held = [child['Instrument']['.Deal']['Buy_Sell'] for child in node['Children']
            if child['Instrument']['.Deal'].get('Buy_Sell')][0]
    assert held != bought, 'the desk took the same side as its client'
    assert body['ticket'] == service.load(landed).plan_hash(), \
        'the ticket is not the plan this booking left the book at'

    retried = accept(quote).json()
    assert retried['written'] is False
    assert retried['booked'] == {'lsn': fill[0], 'deal_path': booked['deal_path']}
    assert 'already on the book' in retried['note']
    assert 'the book moved under it' not in json.dumps(retried), \
        'a quote that booked was told to re-quote'
    assert typed(recorded, 'quote_filed', 'fill') == [filed, fill], 'a retry filed a second fact'
    assert verify_home(recorded)['events'] == head(recorded)


class RacingBook(service.Book):
    """A live book that lets a SECOND WRITER in at a named moment - THE GATE'S OWN DATA rather than
    a patch, the way `SpotMovingParams` supplies a terminal, the interleaving a lock exists for
    being a second writer and a window.

    An acceptance reads the file twice: once outside its act, for the desk's own window, and once
    inside it for the document the trade lands in. `before` fires on the first, with the lock free,
    so it can be a write through the book's own verbs - one that beat the acceptance to the file.
    `within` fires on the second, which is held under the lock, so it has to be a write that never
    asks for the lock: the only kind that CAN move the file while the act runs, and the window a
    redo would put the record's appends in.
    """

    def __init__(self, path):
        super().__init__(path)
        self.before = self.within = None
        self.outer = False

    def arm(self, before=None, within=None):
        self.before, self.within = before, within

    def read(self):
        self.outer = True
        document, etag = super().read()
        self.outer = False
        landing, self.before = self.before, None
        if landing is not None:
            landing()
        return document, etag

    def _read(self):
        document, etag = super()._read()
        if not self.outer and self.within is not None:
            landing, self.within = self.within, None
            landing()
        return document, etag


def booked_deal(reference, amount=5_000.0):
    """An ordinary `/book/deals` booking under the client's set - a write that moves the PLAN, and
    one the record holds, so a reconcile after it is clean."""
    return CLIENT.post('/book/deals', content=dump({
        'action': 'add', 'deal': dict(CASHFLOW, Reference=reference, Amount=amount),
        'parent_reference': CLIENT_SET, 'quantity': amount,
        'execution_reference': 'EXEC-' + reference}), headers=JSON).json()


def reconciles_clean():
    """Whether the record and the file agree on every one of `/book/reconcile`'s three lists."""
    answer = CLIENT.get('/book/reconcile').json()
    return answer['in_record_not_in_file'] == answer['in_file_not_in_record'] == answer[
        'quantity_mismatch'] == []


def lockless_deal(path, reference, **fields):
    """A deal written STRAIGHT INTO THE FILE, under nobody's lock and outside the record - the
    second writer `Book` picks up through its own mtime check, and the only one that can move the
    file while an act holds the lock. `fields` author it as the gate needs it: badly, and under the
    reference a booking is using, is what a redo would newly say something about."""
    document = json.loads(path.read_text())
    document['Calc']['Deals']['Deals']['Children'].append(
        {'Instrument': {'.Deal': json.loads(dump(dict(CASHFLOW, Reference=reference, **fields)))}})
    path.write_text(json.dumps(document, indent=2), newline='\n')


def booked_only(path):
    """Leave the file holding nothing but what a gate books through the verbs, so a reconcile below
    reads this gate and not the fixture's own cashflow. The book is re-opened on what is left."""
    document = json.loads(path.read_text())
    children = document['Calc']['Deals']['Deals']['Children']
    children[:] = [node for node in children
                   if node['Instrument']['.Deal'].get('Object') == 'NettingCollateralSet']
    path.write_text(json.dumps(document, indent=2), newline='\n')
    service.BOOK = RacingBook(str(path))


def test_a_write_racing_the_acceptance_leaves_no_half_act_in_either_order(recorded, quoting):
    """AN ACCEPTANCE APPENDS BEFORE IT WRITES, SO IT MAY NOT RUN OPTIMISTICALLY. `Book.mutate`
    re-runs an edit that lost the race on the document as it now stands, which is what it is for -
    but a fact the first run put on the record is not undone by losing, so a redo caused by a
    plan-moving write would leave the record holding a quote and a fill for a booking the file never
    took, the caller told to re-quote, and every later acceptance answering "already booked" at a
    deal path that does not exist. `Book.transact` runs the whole act under the lock instead.

    BOTH ORDERS A LOCK CAN ORDER, each with the other write going through the ordinary booking verb
    so the record holds it too. First the other write lands before the acceptance reads: the plan
    refuses and the head does not move. Then the acceptance lands from INSIDE another edit's body -
    which is where `mutate` runs an edit, outside the lock - so that write meets a file that moved
    and redoes onto the booked one. After each: one `quote_filed` and one `fill` per acceptance, a
    reconcile with nothing in any of its three lists, and no answer naming a path the file lacks.
    The order NO lock can order is the gate below.

    Killing mutation: the acceptance back on `mutate`, whose second pass meets the plan check the
    first passed and refuses, leaving the record holding a quote and a fill for a trade the file
    never took and the pending file saying it booked at a path nothing is at.
    """
    booked_only(quoting)
    first = quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET)
    before = head(recorded)
    service.BOOK.arm(before=lambda: booked_deal('LATE-A'))
    refused = accept(first)

    assert 'LATE-A' in quoting.read_text(), 'the deal never landed - this gate proved nothing'
    assert refused.status_code == 422
    assert 'the book moved under it' in refused.json()['detail']
    assert head(recorded) == before + 1, 'the acceptance appended beside the booking that beat it'
    assert facts(recorded, 'quote_filed') == []
    assert [body['execution_reference'] for _, _, body in facts(recorded, 'fill')] == ['EXEC-LATE-A']

    # the other order: the acceptance completes inside another edit's body, so that edit redoes
    second = quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET)
    answered = []

    def racing(document, etag):
        """A values edit whose body runs OUTSIDE the lock, where the acceptance can land whole. A
        spot is values-plane, so this write moves neither the plan nor a row a reconcile reads."""
        if not answered:
            answered.append(accept(second).json())
        document['Calc']['MergeMarketData']['ExplicitMarketData'][
            'Price Factors']['FxRate.ZAR']['Spot'] = SPOT + 0.25
        return True, {'written': True}

    ticked = service.BOOK.mutate(racing)
    landed = json.loads(quoting.read_text())

    assert answered[0]['written'] is True and ticked['written'] is True
    assert landed['Calc']['MergeMarketData']['ExplicitMarketData'][
        'Price Factors']['FxRate.ZAR']['Spot'] == SPOT + 0.25, 'the racing write never redid'
    assert len(facts(recorded, 'quote_filed')) == 1 and len(facts(recorded, 'fill')) == 2

    booked = pending_of(quoting.parent, second)['booked']
    assert deal_at(landed, booked['deal_path'])['Instrument']['.Deal']['Reference'].startswith(
        'ZeroCostCollar-'), 'the booked path names a deal the file does not hold'
    assert reconciles_clean(), 'the record and the file disagree after the redo'


def test_a_lockless_write_inside_the_act_leaves_no_half_act(recorded, quoting):
    """THE ORDER NO LOCK CAN ORDER: a deal written straight into the file, which `Book` picks up
    through its own mtime check, landing on the read the act itself runs - the window a redo would
    put the record's appends in, and the one a writer that takes the lock cannot reach.

    THE HUB IS THE GATEKEEPER AND THE FILE IS ITS PROJECTION, so what such a write meets is not a
    race to be ordered - it loses to the land, as every write the act read before does, and a copy
    of it that outlived the act would be a divergence `/book/reconcile` names. What is gated is the
    INVARIANT, which holds whichever way the race falls rather than the way it fell here: at most
    one `quote_filed` and one `fill`, a `booked.deal_path` naming a deal the file holds, a REFUSED
    acceptance leaving the head where it found it, and the record and the file reconciling on all
    three lists. Self-contained: one home, one book, one acceptance.

    Killing mutation: the acceptance back on `mutate`, whose second pass meets the plan check the
    first passed - the caller is told to re-quote while the record keeps the quote and the fill the
    first pass filed, the file never takes the trade, and the pending file names a deal path
    nothing is at.
    """
    booked_only(quoting)
    quote = quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET)
    raced = []
    service.BOOK.arm(within=lambda: raced.append(lockless_deal(quoting, 'RAW-C')))
    before = head(recorded)
    took = accept(quote)

    assert raced, 'the write never landed inside the act - this gate proved nothing'
    assert len(facts(recorded, 'quote_filed')) <= 1 and len(facts(recorded, 'fill')) <= 1, \
        'the act ran twice and the record holds a quote and a fill for each pass'
    if took.status_code != 200:
        assert head(recorded) == before, 'a refused acceptance left its quote on the record'
    booked = pending_of(quoting.parent, quote).get('booked')
    if booked is not None:
        assert booked['deal_path'] in dict(walk_job_deals(json.loads(quoting.read_text()))), \
            'the pending file says this quote booked where the file holds nothing'
    assert reconciles_clean(), 'the record and the file disagree after the act'


def test_a_booking_raced_by_a_lockless_write_files_its_fill_exactly_once(recorded, quoting):
    """`/book/deals` IS THE OTHER EDIT THAT APPENDS BEFORE IT WRITES: under a home the fill goes on
    the record inside the closure, so it takes `Book.transact` and runs one pass under the lock.

    The write racing it wears THIS BOOKING'S OWN REFERENCE and is authored badly, which is what
    makes a second pass refuse what the first passed: messages are keyed by reference, so the
    verdict now says something about the deal being booked and `newly_said` returns it.

    Killing mutation: `/book/deals` back on `mutate` - the booking is refused over somebody else's
    authoring while the record already holds its fill, and reconcile reports a trade the file lacks.
    """
    booked_only(quoting)
    raced = []
    service.BOOK.arm(within=lambda: raced.append(
        lockless_deal(quoting, 'RACED-IN', Amount='oops')))
    booked = booked_deal('RACED-IN')

    assert raced, 'the write never landed inside the act - this gate proved nothing'
    assert booked['written'] is True, booked
    assert len(facts(recorded, 'fill')) == 1, 'the record holds a fill per pass of the edit'
    assert 'oops' not in quoting.read_text(), 'the badly authored write outlived the act it raced'
    assert reconciles_clean(), 'the record and the file disagree after the booking'


def test_a_moved_market_between_the_quote_and_the_acceptance_books_and_says_so(recorded, quoting):
    """THE RULING: between a quote and the client's word the market may move materially, and we
    follow the spine as usual. The move is REPORTED on the booking - the values struck on, the ones
    standing, and that they differ - and never refused; the desk's own `firm_seconds` is the promise
    that bounds it. The PLAN is untouched by a tick, which is what makes the two separable.

    Killing mutation: the market comparison left as a refusal, which stops a desk booking anything
    on a market that ticks every thirty seconds.
    """
    quote = quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET)
    struck = pending_of(quoting.parent, quote)['pinned']

    assert tick(14.9)['updated'] == ['FXVolPrices.USD.ZAR']
    standing = service.load(service.BOOK.read()[0])
    assert standing.plan_hash() == struck['plan_hash'], \
        'a vol tick moved the plan - the values plane and the plan plane are not disjoint'
    assert standing.values_hash() != struck['values_hash'], 'a vol tick moved nothing'

    booked = accept(quote).json()
    assert booked['written'] is True
    assert booked['market'] == {'pinned': struck['values_hash'],
                                'current': standing.values_hash(), 'moved': True}
    assert booked['firmness']['firm'] is True and booked['firmness']['plan']['moved'] is False
    assert len(facts(recorded, 'fill')) == 1


def test_a_booking_between_the_quote_and_the_acceptance_refuses_with_nothing_appended(recorded,
                                                                                      quoting):
    """The book moved since the solve, so the marginal charge was priced against a portfolio this
    trade would no longer join. It refuses BEFORE anything appends - the record does not hold a
    quote for a trade that was never booked - and names the remedy: a desk told "stale" learns
    nothing, a desk told "the book moved" re-solves.
    """
    quote = quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET)

    document = json.loads(quoting.read_text())
    document['Calc']['Deals']['Deals']['Children'].append(
        json.loads(dump({'Instrument': {'.Deal': dict(CASHFLOW, Reference='LATE',
                                                      Amount=5_000.0)}})))
    quoting.write_text(json.dumps(document, indent=2), newline='\n')
    before = head(recorded)

    refused = accept(quote)
    said = refused.json()['detail']
    assert refused.status_code == 422
    assert 'the book moved under it' in said and quote['quote_id'] in said
    assert 'oldest stamped pillar' not in said, 'the board was blamed for the book'
    assert head(recorded) == before, 'a refused acceptance appended'


def test_a_declared_pillar_window_refuses_a_quote_struck_on_an_old_board(recorded, quoting):
    """A board already stale when the price was given is a price a client may not hold the desk to,
    and the desk declares how stale is too stale. Aged by DECLARATION - a firmness policy with a
    zero-second window, put in the record as a hashed blob through the ordinary writer - which makes
    the fixture deterministic rather than a sleep.

    AND IT IS THE OLDEST PILLAR, not the freshest: a surface refreshed at one expiry and dead at
    another is as old as its deadest row, so one row re-stamped today does not make the board fresh.
    The second stamp is written onto the book as DATA - a stamp is value-plane, so it moves neither
    the plan the acceptance checks nor the market it reports.

    AND A HOME THAT DECLARED NO WINDOW REFUSES NOTHING, which is the other half of the ruling: a
    deployment that has not said how stale is too stale has not asked for the question.
    """
    quote = quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET)
    assert accept(quote).json()['written'] is True, 'no window refused a stale board'

    # one pillar re-read a moment ago; the rest stand at the snapshot's own 16:30, two years back
    document = json.loads(quoting.read_text())
    rows = document['Calc']['MergeMarketData']['ExplicitMarketData'][
        'Market Prices']['FXVolPrices.USD.ZAR']['instrument']['Points']
    rows[0]['Timestamp'] = {'.Timestamp': pd.Timestamp.utcnow().tz_localize(None).isoformat()}
    quoting.write_text(json.dumps(document, indent=2), newline='\n')

    # a window wide enough for the fresh row and far too narrow for the rest of the board
    declare(recorded, policy.FIRMNESS_POLICY, {'pillar_seconds': 1e6})
    aged = quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET)
    struck = pending_of(quoting.parent, aged)['pinned']
    assert struck['pillar_age'] > 1e6, 'the board is not old enough to test the window'
    before = head(recorded)

    refused = accept(aged)
    said = refused.json()['detail']
    assert refused.status_code == 422
    assert 'oldest stamped pillar' in said and '1000000s window' in said
    assert 'the book moved under it' not in said, 'the book was blamed for the board'
    assert head(recorded) == before, 'a refused acceptance appended'


def test_a_stale_desk_window_refuses_before_the_record_is_asked_anything(recorded, quoting):
    """The desk's own mandate is FIRST and it is a promise to a client rather than a statement about
    provenance, so a quote past `firm_seconds` never reaches the record's own checks - asserted on
    the wording, which is the desk's, and on a head that did not move."""
    document = json.loads(quoting.read_text())
    document['Calc'][structures.QUOTE_POLICY] = {'firm_seconds': 0}
    quoting.write_text(json.dumps(document, indent=2), newline='\n')

    quote = quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET)
    before = head(recorded)
    refused = accept(quote)

    assert refused.status_code == 422
    assert structures.QUOTE_POLICY in refused.json()['detail']
    assert 'the book moved under it' not in refused.json()['detail']
    assert head(recorded) == before


def test_a_board_stamped_only_on_its_surfaces_has_an_age(recorded, quoting, tmp_path):
    """ONE PROJECTION FOR THE IDENTITY AND THE AGE. A book bootstrapped once and handed on carries
    its surfaces' `Quote_Timestamp` and no quote rows at all - six of the fixtures under `tests/`
    are such a book - and it has an age for exactly the reason it has an identity: `quote_stamps`
    reads every clock `without_clocks` drops, the value-bound `Date` on a price factor as well as
    the stamp on a row.

    Killing mutation: `quote_stamps` reading the `Market Prices` rows alone, which answers None for
    such a board and refuses every acceptance on it in a sentence asking for something already done.
    """
    document = json.loads(quoting.read_text())
    market = document['Calc']['MergeMarketData']['ExplicitMarketData']
    for row in market['Market Prices']['FXVolPrices.USD.ZAR']['instrument']['Points']:
        row.pop('Timestamp', None)
    assert market['Price Factors']['FXVol.USD.ZAR']['Quote_Timestamp'], 'the surface is unstamped'
    quoting.write_text(json.dumps(document, indent=2), newline='\n')

    quote = quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET)
    pinned = pending_of(tmp_path, quote)['pinned']
    assert pinned['pillar_age'] is not None, 'a surface-stamped board read as having no age'
    assert pinned['pillar_age'] == pytest.approx(service.quote_age(
        SNAPPED.isoformat(), pending_of(tmp_path, quote)['quoted_at']), rel=1e-9)

    declare(recorded, policy.FIRMNESS_POLICY, {'pillar_seconds': 1e9})
    assert accept(quote_of('ZeroCostCollar', COLLAR,
                           netting_set=CLIENT_SET)).json()['written'] is True


def test_a_board_stamped_only_on_its_rows_has_an_age(recorded, quoting, tmp_path):
    """The other half of the same projection. A board bootstrapped from quote rows that carry their
    own clocks has an age off THOSE, whether or not the surface written from them kept one.

    Killing mutation: `quote_stamps` reading the factor clocks alone, which is the half the
    increment added and would otherwise be the only half held.
    """
    document = json.loads(quoting.read_text())
    market = document['Calc']['MergeMarketData']['ExplicitMarketData']
    market['Price Factors']['FXVol.USD.ZAR']['Quote_Timestamp'] = ''
    assert market['Market Prices']['FXVolPrices.USD.ZAR']['instrument']['Points'][0]['Timestamp']
    quoting.write_text(json.dumps(document, indent=2), newline='\n')

    quote = quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET)
    pinned = pending_of(tmp_path, quote)['pinned']
    assert pinned['pillar_age'] is not None, 'a row-stamped board read as having no age'
    assert pinned['pillar_age'] == pytest.approx(service.quote_age(
        SNAPPED.isoformat(), pending_of(tmp_path, quote)['quoted_at']), rel=1e-9)


def test_the_advanced_fold_answers_the_position_it_is_asked_for(recorded, quoting):
    """`advancing`'s three laws, read off the fold itself rather than through a booking.

    It ADVANCES, so a verdict filed after the pair was taken is in the next answer. It is BOUNDED at
    the position it answers, so a frame appended between two asks is not applied twice - a
    duplicated verdict list that `standing_approval`'s `max` would hide. A pair AHEAD of the
    position asked for refolds from genesis rather than refusing, since a fold walks forward. And a
    history that is not the one the pair was taken on drops it: the key is the GENESIS event hash,
    so a home re-minted where the last one stood is a different record.

    Killing mutations: the fold unbounded while the pair is stamped at the position asked for; a
    pair ahead of that position used as a seed; the key taken off the path rather than the history.
    """
    projections = spine.package().projections
    decisions = projections.PROJECTORS['decisions']
    plan, other = 'a' * 64, 'b' * 64

    def verdicts_at(lsn=None):
        return spine.folded(lambda log: spine.advancing(log, decisions, lsn)['plans'])

    spine.approve(plan, actor_name=ACTOR, book_name='spine-desk')
    first = verdicts_at()
    assert [row['lsn'] for row in first[plan]] == [head(recorded)]

    signed = spine.approve(other, actor_name=DESK_TWO, book_name='spine-desk')
    assert [row['lsn'] for row in verdicts_at()[other]] == [signed['lsn']], 'the pair never advanced'
    assert [row['lsn'] for row in verdicts_at()[plan]] == [first[plan][0]['lsn']], \
        'a frame was applied twice'

    # a position BEHIND the pair is a fold from genesis, and the pair is left where it was
    assert other not in verdicts_at(signed['lsn'] - 1)
    assert [row['lsn'] for row in verdicts_at()[other]] == [signed['lsn']]

    # another history answers its own, and is not this one's
    elsewhere = quoting.parent / 'second-spine'
    init_home(elsewhere, ACTOR)
    log = opened(elsewhere)
    try:
        assert spine.advancing(log, decisions)['plans'] == {}
    finally:
        log.close()
    assert [row['lsn'] for row in verdicts_at()[plan]] == [first[plan][0]['lsn']]


def test_a_home_re_minted_where_the_last_one_stood_is_a_different_record(recorded, quoting,
                                                                        tmp_path):
    """THE PAIR IS KEYED ON THE HISTORY, NOT THE PATH. A home deleted and minted again at the same
    place holds none of the verdicts the last one did, and a fold that answered the old ones would
    route a booking on a record that no longer exists - silently, which is the unsafe direction.

    Killing mutation: the key taken off `log.home`, which answers the predecessor's history.
    """
    import shutil

    decisions = spine.package().projections.PROJECTORS['decisions']
    spine.approve('a' * 64, actor_name=ACTOR, book_name='spine-desk')
    assert spine.folded(lambda log: spine.advancing(log, decisions))['plans'].get('a' * 64)

    shutil.rmtree(str(recorded))
    init_home(recorded, ACTOR)
    spine.approve('c' * 64, actor_name=ACTOR, book_name='spine-desk')

    standing = spine.folded(lambda log: spine.advancing(log, decisions))['plans']
    assert 'a' * 64 not in standing, 'the pair answered a history this home no longer has'
    assert 'c' * 64 in standing


def test_the_board_is_aged_at_the_moment_the_quote_was_struck(recorded, quoting):
    """The board's age is taken AT `quoted_at` and travels on the file, so the number a booking is
    measured by is the one the price was given at - not the one the wall clock says when a client
    finally calls back.

    Killing mutation: `board_age` measured against now, which grows a quote's board age for as long
    as the client takes to answer and refuses a price that was fresh when it was given.
    """
    document = json.loads(quoting.read_text())

    # the snapshot's own 16:30, read half an hour later
    assert service.board_age(document, '2024-06-28T17:00:00.000+00:00') == pytest.approx(1800.0)
    assert service.board_age(document, None) > 1e7, 'the wall clock reads the same as the quote'
    assert service.board_age({}, '2024-06-28T17:00:00.000+00:00') is None


def test_an_edited_pending_file_no_longer_says_what_was_quoted(recorded, quoting, tmp_path):
    """The TICKET is re-derived from the pending deal and this book and compared against the one the
    quote pinned. The plan is unmoved here, so the two can differ only where the FILE has been
    edited - which is exactly what this refuses, before anything appends."""
    quote = quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET)
    path = tmp_path / 'tmp' / (quote['quote_id'] + '.json')
    filed = json.loads(path.read_text())
    filed['deal']['Children'][0]['Instrument']['.Deal']['Strike_Price'] *= 1.01
    path.write_text(json.dumps(filed, indent=2), newline='\n')
    before = head(recorded)

    refused = accept(quote)
    assert refused.status_code == 422
    assert 'no longer says what was quoted' in refused.json()['detail']
    assert head(recorded) == before


# --------------------------------------------------------------------------------------------
# The tier step, inside the acceptance closure.

AUTO_SEAT = 'policy/tiers/auto'
DESK_TWO = 'subject-desk-two'

#: A cap the collar's million dollars sits comfortably under, and one it does not.
BIG, SMALL = 5_000_000.0, 100_000.0


def tiers(*rows, **sections):
    """A tiers document out of the rows a gate cares about."""
    return dict(tiers=list(rows), **sections)


def entitle(home, grants):
    """Declare a capabilities document, which is what turns enforcement ON: until one is in force
    the home is the single-user instrument it was, and after it every append needs a grant."""
    log = opened(home)
    try:
        blob = log.store.put(canonical_document(
            {'grants': [{'subject': s, 'verb': v, 'book': b} for s, v, b in grants], 'read': []}))
        return log.append('policy_declared', {'policy': CAPABILITIES_POLICY, 'blob': blob},
                          actor=ACTOR, blob_refs=(blob,))
    finally:
        log.close()


def test_an_automatic_tier_signs_under_its_seat_between_the_quote_and_the_fill(recorded, quoting):
    """A tier naming a SEAT signs for itself, and the order is the whole point: the acceptance, then
    the approval under the tier's own subject, then the fill. Three facts at consecutive LSNs, so an
    auditor reading the log in order sees a trade authorised before it was booked.

    Killing mutation: the approval appended after the fill, which records every automatic booking as
    having been signed off afterwards.
    """
    declare(recorded, policy.TIERS_POLICY, tiers(
        {'name': 'auto', 'seat': AUTO_SEAT, 'max_notional': {'amount': BIG, 'currency': 'USD'}},
        {'name': 'desk', 'four_eyes': True}))
    quote = quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET)

    booked = accept(quote).json()
    assert booked['written'] is True
    assert booked['tier']['name'] == 'auto' and booked['tier']['seat'] == AUTO_SEAT

    steps = typed(recorded, 'quote_filed', 'approval', 'fill')
    assert [event_type for _, event_type in steps] == ['quote_filed', 'approval', 'fill']
    assert [lsn for lsn, _ in steps] == list(range(steps[0][0], steps[0][0] + 3))
    _, _, signed = facts(recorded, 'approval')[0]
    assert signed == {'plan_hash': booked['accepted']['ticket']}
    assert booked['tier']['approval_lsn'] == steps[1][0]
    assert verify_home(recorded)['events'] == head(recorded)


def test_a_seat_the_document_does_not_scope_leaves_the_acceptance_standing(recorded, quoting):
    """An automatic tier whose seat holds no `approve` grant cannot sign, so nothing books - but the
    ACCEPTANCE STANDS, because the client took the price and that is a fact whatever the workflow
    then says. The denial is a chained fact in the writer's own voice, the file is byte-identical,
    and the answer names the seat and the scope it lacks.

    Killing mutation: the `quote_filed` appended after the tier step, which loses the acceptance of
    every quote a misconfigured workflow holds back.
    """
    declare(recorded, policy.TIERS_POLICY, tiers({'name': 'auto', 'seat': AUTO_SEAT}))
    # `validate` beside the rest: quoting is a curiosity job and 5c's queue asks for that scope
    # before it runs one, and the six verbs imply nothing about each other here or at the writer
    entitle(recorded, [(ACTOR, verb, '*')
                       for verb in ('validate', 'book', 'mark', 'admin', 'approve')])
    quote = quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET)
    before = quoting.read_bytes()

    booked = accept(quote).json()
    assert booked['written'] is False and quoting.read_bytes() == before
    assert booked['accepted']['lsn'] == facts(recorded, 'quote_filed')[0][0]
    assert AUTO_SEAT in booked['waits_on'] and 'approve' in booked['waits_on']
    assert facts(recorded, 'fill') == [] and facts(recorded, 'approval') == []

    _, _, denial = facts(recorded, 'capability_denied')[0]
    assert denial['subject'] == AUTO_SEAT and denial['verb'] == 'approve'
    assert verify_home(recorded)['events'] == head(recorded)


def test_a_desk_tier_waits_for_a_second_seat_and_books_on_the_verdict_that_stands(recorded,
                                                                                  quoting):
    """THE TWO-ACT DESK TIER, through the service. A tier naming no seat wants a human: the first
    acceptance files the quote and answers `waits_on` with the file untouched, the ACCEPTOR's own
    approval does not satisfy four eyes, a rejection filed after it is what the record says LAST -
    the answer naming its LSN and its reason - and an approval by another seat after that is what
    books. Four acceptances of one quote, ONE `quote_filed`: the tuples carry no `effective_time`,
    so the writer's own duplicate rule coalesces them onto the LSN they already have.

    Killing mutations: `standing_approval` reading the first verdict in the list rather than the
    latest by LSN, which books on an approval a rejection has since overtaken; and the four-eyes
    test read against any row of the list rather than the one that stands.
    """
    declare(recorded, policy.TIERS_POLICY, tiers({'name': 'desk', 'four_eyes': True}))
    quote = quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET)
    before = quoting.read_bytes()

    waiting = accept(quote).json()
    ticket = waiting['accepted']['ticket']
    assert waiting['written'] is False and quoting.read_bytes() == before
    assert waiting['tier'] == {'name': 'desk', 'seat': None, 'approval_lsn': None}
    assert 'no verdict is filed' in waiting['waits_on']

    signed = CLIENT.post('/book/quote/approve',
                         json={'quote_id': quote['quote_id'], 'actor': ACTOR}).json()
    assert signed['ticket'] == ticket
    own = accept(quote).json()
    assert own['written'] is False and 'one seat' in own['waits_on']
    assert quoting.read_bytes() == before

    # a rejection filed LAST is what stands, whatever was approved before it
    rejected = CLIENT.post('/book/quote/reject', json={
        'quote_id': quote['quote_id'], 'actor': DESK_TWO, 'reason': 'the client is over limit'})
    stale = accept(quote).json()
    assert stale['written'] is False
    assert str(rejected.json()['recorded']['lsn']) in stale['waits_on']
    assert 'the client is over limit' in stale['waits_on']

    # and an approval after the rejection is what the record says last
    approved = CLIENT.post('/book/quote/approve',
                           json={'quote_id': quote['quote_id'], 'actor': DESK_TWO}).json()
    booked = accept(quote).json()
    assert booked['written'] is True and booked['tier']['name'] == 'desk'
    assert booked['tier']['approval_lsn'] == approved['recorded']['lsn']
    assert len(facts(recorded, 'fill')) == 1
    assert [lsn for lsn, _ in typed(recorded, 'quote_filed')] == [waiting['accepted']['lsn']], \
        'four acceptances of one quote minted more than one fact'
    assert verify_home(recorded)['events'] == head(recorded)


def test_a_currency_the_tick_has_not_valued_is_not_a_currency_the_ticket_states(recorded, quoting):
    """A SPOT BLOCK INSTALLED AND NOT YET TICKED carries its declared zero, which is the documented
    sequence for bringing a new currency up. That currency is one this book cannot VALUE the
    notional in, so the ticket does not state it and a tier capping in it fails by name - rather
    than dividing by it inside a closure that has already filed the quote.

    The acceptance stands, the head goes no further than it, and the book file is byte-identical.

    Killing mutation: the cross stated whatever it comes out as, which is a `ZeroDivisionError` out
    of a booking endpoint - a traceback where every other refusal in this package is a sentence.
    """
    document = json.loads(quoting.read_text())
    document['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors']['FxRate.EUR'] = {
        'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Spot': 0.0}
    quoting.write_text(json.dumps(document, indent=2), newline='\n')
    declare(recorded, policy.TIERS_POLICY, tiers(
        {'name': 'auto', 'seat': AUTO_SEAT, 'max_notional': {'amount': BIG, 'currency': 'EUR'}}))

    quote = quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET)
    pending = pending_of(quoting.parent, quote)
    terms = service.quote_terms(json.loads(quoting.read_text()), pending)
    assert 'EUR' not in terms['notional_in'] and set(terms['notional_in']) == {'USD', 'ZAR'}

    # THE OTHER DIRECTION, with the zero on top: the NOTIONAL's own currency unvalued. Its amount
    # still stands - an amount of a currency needs no rate - and every cross off it reads zero,
    # which is not a notional a cap can be compared against and so is not stated
    unvalued = json.loads(quoting.read_text())
    unvalued['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors'][
        'FxRate.USD']['Spot'] = 0.0
    assert service.quote_terms(unvalued, pending)['notional_in'] == {'USD': AMOUNT}
    before = quoting.read_bytes()

    answer = accept(quote).json()
    assert answer['written'] is False and 'tier' not in answer
    assert any('not stated in EUR' in one for one in answer['refused']), answer['refused']
    assert quoting.read_bytes() == before, 'a refused tier moved the book'
    assert head(recorded) == answer['accepted']['lsn'], 'the acceptance is not the last fact'


def test_size_tenor_and_market_route_a_ticket_through_the_tiers(recorded, quoting):
    """The three checks, each one driving the route through the SERVICE rather than the evaluator.

    Over the cap escalates to the tier that admits it; over every tier answers `refused` with every
    sentence of the route AND the acceptance standing, since the client took the price whatever the
    policy says about it. A cap in a currency this book carries no rate for fails ITS tier by name -
    the safe direction, and the one a conversion inside a policy check would lose - while a cap in a
    currency it DOES price reads the notional CROSSED at the book's own spot, which is the case that
    tells a stated notional from an unconverted one. And the market check passes on the name the
    book's own board is declared under and fails on one nothing declared.

    A TICKET NO TIER ADMITS FILES NO REJECTION: the route's sentences are an answer, and a verdict
    is a seat's decision the policy names no seat for, so the head stands at the acceptance.

    Killing mutation: `quote_expiry` answering nothing, which routes every ticket past a tier that
    bounds the tenor.
    """
    declared = spine.declare_market('official', spine.values_of(
        service.load(service.BOOK.read()[0])))
    # both branches of the expiry, read directly: a vanilla leg names its own day and an accrual
    # leg names the last settlement of its strip
    assert service.quote_expiry({'Object': 'StructuredDeal', 'Children': [
        {'Instrument': {'.Deal': {'Object': 'FXAccumulatorOptionDeal', 'Accumulator_ExpiryDates': [
            [{'.Timestamp': '2026-03-30'}, {'.Timestamp': '2026-04-01'}, 0.0],
            [{'.Timestamp': '2026-06-30'}, {'.Timestamp': '2026-07-02'}, 0.0]]}}}]}) == \
        pd.Timestamp('2026-07-02')
    assert service.quote_expiry({'Object': 'StructuredDeal'}) is None

    for rows, expected, said in (
            ([{'name': 'auto', 'seat': AUTO_SEAT,
               'max_notional': {'amount': SMALL, 'currency': 'USD'}},
              {'name': 'desk', 'four_eyes': True}], 'desk', 'caps max_notional'),
            ([{'name': 'auto', 'seat': AUTO_SEAT, 'max_tenor_years': 0.5},
              {'name': 'desk', 'four_eyes': True}], 'desk', 'caps max_tenor_years'),
            ([{'name': 'auto', 'seat': AUTO_SEAT,
               'max_notional': {'amount': BIG, 'currency': 'JPY'}},
              {'name': 'desk', 'four_eyes': True}], 'desk', 'not stated in JPY'),
            # a million dollars is 54,054 rand on this book's own axis, so a 100,000 rand cap
            # admits it - and refuses it if the notional was stated without being crossed
            ([{'name': 'auto', 'seat': AUTO_SEAT,
               'max_notional': {'amount': SMALL, 'currency': 'ZAR'}},
              {'name': 'desk', 'four_eyes': True}], 'auto', None),
            ([{'name': 'auto', 'seat': AUTO_SEAT, 'market': 'eod'},
              {'name': 'desk', 'four_eyes': True}], 'desk', 'nothing in this record has declared'),
            ([{'name': 'auto', 'seat': AUTO_SEAT, 'market': 'official'}], 'auto', None),
            ([{'name': 'auto', 'seat': AUTO_SEAT,
               'max_notional': {'amount': SMALL, 'currency': 'USD'}}], None, 'caps max_notional')):
        declare(recorded, policy.TIERS_POLICY, tiers(*rows))
        quote = quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET)
        pending = pending_of(quoting.parent, quote)
        answer = accept(quote).json()

        if expected == 'auto':
            assert answer['written'] is True and answer['tier']['name'] == 'auto', rows
        elif expected == 'desk':
            assert answer['written'] is False and answer['tier']['name'] == 'desk', rows
            assert 'no verdict is filed' in answer['waits_on']
        else:
            assert answer['written'] is False and 'tier' not in answer, rows
            assert any(said in one for one in answer['refused']), answer['refused']
            assert facts(recorded, 'rejection') == [], 'a tier filed a rejection nobody signed'
            assert head(recorded) == answer['accepted']['lsn'], \
                'a ticket no tier admits moved the record past its own acceptance'
        assert answer['accepted']['lsn'] > declared['lsn'], 'the acceptance did not stand'
        # the sentence the ticket's route collected, which is what "fails that tier by name" means
        if said is not None:
            route = spine.route_ticket(pending['pinned']['ticket'], service.quote_terms(
                json.loads(quoting.read_text()), pending), ACTOR)
            assert any(said in one for one in route['refusals']), (rows, route['refusals'])


# --------------------------------------------------------------------------------------------
# The decision verbs.

def test_a_decision_on_a_quote_nobody_accepted_refuses_by_name(recorded, quoting, tmp_path):
    """A decision is filed over the plan an ACCEPTED quote minted, so there is nothing to rule on
    before the client has taken the price - and a pending file naming a position that holds
    something else is one copied from another home, which `quote_at` catches by opening the ONE
    frame at that LSN rather than folding every quote the desk has struck.

    THE POSITION IS EVIDENCE ABOUT NOTHING BY ITSELF, so three things are refused and each by its
    own check: a position holding another TYPE, a position past the head, and - the shape a file
    carried between two desks that both quote actually has - a position holding ANOTHER QUOTE.

    Killing mutation: the quote-id check dropped, which rules on whatever quote happens to sit at a
    remembered position.
    """
    quote = quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET)
    refused = CLIENT.post('/book/quote/approve',
                          json={'quote_id': quote['quote_id'], 'actor': ACTOR})
    assert refused.status_code == 422 and 'accept it first' in refused.json()['detail']

    waiting = declare(recorded, policy.TIERS_POLICY, tiers({'name': 'desk', 'four_eyes': True}))
    assert accept(quote).json()['written'] is False, 'the tier let it book and moved the plan'
    second = quote_of('ZeroCostCollar', dict(COLLAR, notional=AMOUNT / 2),
                      netting_set=CLIENT_SET)
    assert accept(second).json()['written'] is False

    path = tmp_path / 'tmp' / (quote['quote_id'] + '.json')
    filed = json.loads(path.read_text())
    elsewhere = json.loads((tmp_path / 'tmp' / (second['quote_id'] + '.json')).read_text())
    read = spine.quote_at(filed['accepted']['lsn'], quote['quote_id'])
    assert read['quote_id'] == quote['quote_id'] and read['ticket'] == filed['accepted']['ticket']

    # the second quote's own position: a real `quote_filed`, and not this quote's
    filed['accepted']['lsn'] = elsewhere['accepted']['lsn']
    path.write_text(json.dumps(filed, indent=2), newline='\n')
    other = CLIENT.post('/book/quote/reject', json={
        'quote_id': quote['quote_id'], 'actor': ACTOR, 'reason': 'nothing doing'})
    assert other.status_code == 422
    assert second['quote_id'] in other.json()['detail'] and 'and not' in other.json()['detail']

    filed['accepted']['lsn'] = waiting['lsn']
    path.write_text(json.dumps(filed, indent=2), newline='\n')
    wrong_type = CLIENT.post('/book/quote/reject', json={
        'quote_id': quote['quote_id'], 'actor': ACTOR, 'reason': 'nothing doing'})
    assert wrong_type.status_code == 422 and 'in this record and not the quote' in \
        wrong_type.json()['detail']

    # and a position past the head is the record saying so, not a traceback
    filed['accepted']['lsn'] = 10_000
    path.write_text(json.dumps(filed, indent=2), newline='\n')
    beyond = CLIENT.post('/book/quote/approve',
                         json={'quote_id': quote['quote_id'], 'actor': ACTOR})
    assert beyond.status_code == 422 and 'no LSN 10000' in beyond.json()['detail']


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


# --------------------------------------------------------------------------------------------
# The mark, and the queue that asks first.

def test_a_market_declared_through_the_service_stands_under_the_name_it_was_given(recorded, desk):
    """`POST /book/markets` is the desk's MARK: the live book's own projected values, filed under a
    name and resolving back to it by that name.

    THE RECORD'S REFUSALS REACH THE CALLER UNEDITED. A seat the capabilities document does not
    scope for `mark` is a 422 in the writer's own words with the denial landed as a fact, and a
    `private/` name whose subject is not the seat declaring it is refused at the verb before
    anything appends - neither rule is restated here, because a second place to get one right is a
    second place to get it wrong.

    THE CLOSE IS HELD TO THE SAME OWNER RULE, and is where a caller-chosen market name first reaches
    the wire: a close inside another subject's `private/` namespace would be a board its owner never
    declared and is the only reader of, since `resolve_market` answers a name across the
    declarations of it and the closes on it alike.

    Killing mutations: the verb hashing the document rather than the projected values, after which
    the mark addresses bytes the store does not hold and nothing resolves; a `mark` check written
    into the endpoint, which passes this gate while disagreeing with the writer about a private
    name; and the owner rule absent from `declare_close`, which plants an official close in a
    namespace the same seat was refused one call earlier.
    """
    standing = service.load(service.BOOK.read()[0])
    declared = CLIENT.post('/book/markets', json={'name': 'official'}).json()
    assert declared == {'recorded': {'lsn': head(recorded)}, 'name': 'official',
                        'values_hash': standing.values_hash()}
    resolved = spine.resolve_market('official')
    assert resolved['values_hash'] == standing.values_hash()
    assert resolved['lsn'] == declared['recorded']['lsn'], 'the mark resolved at another position'

    theirs = 'private/{}/screen'.format(DESK_TWO)
    for named, said in (({}, 'declare_market'), ({'name': ''}, 'declare_market'),
                        ({'name': theirs}, 'one seat\'s own board')):
        refused = CLIENT.post('/book/markets', json=named)
        assert refused.status_code == 422 and said in refused.json()['detail'], named
    for named, said in (({'market': ''}, 'declare_close'), ({'market': theirs}, 'own board')):
        refused = CLIENT.post('/book/close', json=named)
        assert refused.status_code == 422 and said in refused.json()['detail'], named
    assert head(recorded) == declared['recorded']['lsn'], 'a refused declaration moved the record'
    assert CLIENT.get('/book/markets').json()['closes'] == [], \
        'a close landed on another seat\'s board'

    # the seat that may BOOK is not the seat that may MARK, and the refusal is itself a fact
    entitle(recorded, [(ACTOR, verb, '*') for verb in ('book', 'admin')])
    unscoped = CLIENT.post('/book/markets', json={'name': 'dealer'})
    assert unscoped.status_code == 422 and 'mark' in unscoped.json()['detail']
    assert facts(recorded, 'capability_denied')[0][2] == {
        'subject': ACTOR, 'verb': 'mark', 'book': '*', 'attempted_type': 'market_declared'}
    assert verify_home(recorded)['events'] == head(recorded)


#: A seat nobody has ever granted anything, named on the REQUEST the way a stranger reaches a desk.
STRANGER = 'subject-nobody-at-all'


def priced(marker, actor=STRANGER):
    """One what-if through the service under a named seat, and whether the executor's store MOVED
    for it - so a gate says the queue never saw the job rather than that its answer was not served.

    The candidate carries the gate's own name: the store is content-addressed and lives for the
    length of the process, so a shared plan would read as a cache hit rather than as a refusal. The
    SEAT rides the request body, which is how a caller names one and is the claim the section
    rests on - relabelling `DV_SPINE_ACTOR` instead would hold only that a deployment whose own
    seat is unscoped cannot price.
    """
    before = dict(service.EXECUTOR.results)
    answer = CLIENT.post('/book/price', content=dump(dict(
        {'deal': dict(CASHFLOW, Reference=marker)},
        **({} if actor is None else {'actor': actor}))), headers=JSON)
    service.EXECUTOR.queue.join()
    return answer, dict(service.EXECUTOR.results) != before


def test_with_no_document_in_force_the_queue_admits_every_job(recorded, desk):
    """ENFORCEMENT ACTIVATES BY DECLARATION, at the queue exactly as at the writer. A home that has
    declared no capabilities document is the single-user instrument the page describes, so a seat
    nothing ever granted anything prices as it did before admission existed: the ordinary answer,
    the job on the queue, and a record that does not move.

    Killing mutation: admission failing closed on an absent document, which stops every job on
    every increment-1 home there is.
    """
    at = head(recorded)

    answer, ran = priced('no-document')
    assert answer.status_code == 200 and ran, answer.text
    assert drained(answer.json())['status'] == 'done'
    assert head(recorded) == at, 'a what-if moved the record'


def test_under_a_document_the_queue_asks_before_the_executor_does(recorded, desk, monkeypatch):
    """THE QUEUE IS THE HUB'S COMPUTE AND IT ASKS FIRST. `pin_result` has no HTTP verb, so a
    submission is the whole of what an unscoped seat could make this box spend, and the check sits
    in `submit` - where every queued job passes - rather than in each verb.

    THE SEAT ASKED ABOUT IS THE REQUEST'S. The document here is the posture a ticking desk must
    declare - `validate` on the deployment's own seat, or the metronome and the diary stop - and
    the stranger is named on the BODY, so what this holds is the contract's own sentence rather
    than "a deployment whose seat is unscoped cannot price".

    A what-if mints nothing, so it is admitted under `validate`: the refusal is a 422 in the
    record's own words, the `capability_denied` is the ONLY new row, and the executor's store is
    exactly what it was - counted, never believed. Asking again is ONE fact, the denial coalescing
    onto the LSN it already has. The same seat granted `validate` over this book prices, and A
    TUPLE ALREADY IN THE STORE IS ASKED ABOUT AGAIN: content addressing dedupes NUMBERS, so a seat
    whose grant has since been withdrawn is refused at the queue rather than served them. An
    unnamed actor under a document is refused by name, because a job nobody signed for is one the
    record could not attribute.

    Killing mutations: the check taken after `queue.put`, which pays for the run and then refuses
    it; the check taken only where the tuple is new, after which a seat asking for numbers the
    store already holds is served them; the actor read off the environment rather than the request,
    which lets the one grant a ticking desk cannot withhold admit every anonymous what-if; and the
    job's own book dropped, which asks for a scope no desk document grants.
    """
    entitle(recorded, [(ACTOR, verb, '*') for verb in ('admin', 'validate')])
    before = head(recorded)

    refused, ran = priced('under-a-document')
    assert refused.status_code == 422 and not ran, refused.text
    said = refused.json()['detail']
    assert STRANGER in said and 'validate' in said and 'spine-desk' in said, said
    denials = facts(recorded, 'capability_denied')
    assert [row[2] for row in denials] == [{'subject': STRANGER, 'verb': 'validate',
                                            'book': 'spine-desk',
                                            'attempted_type': spine.CURIOSITY}]
    assert head(recorded) == before + 1, 'a refused job filed more than the refusal'

    again, ran = priced('under-a-document')
    assert again.status_code == 422 and not ran
    assert facts(recorded, 'capability_denied') == denials, 'a repeated refusal is a second fact'

    entitle(recorded, [(ACTOR, verb, '*') for verb in ('admin', 'validate')]
            + [(STRANGER, 'validate', 'spine-desk')])
    answer, ran = priced('under-a-document')
    assert answer.status_code == 200 and ran, 'an admitted job never reached the executor'
    assert drained(answer.json())['status'] == 'done'

    # the numbers are now in the store, and the grant is withdrawn: a cache hit is not an admission.
    # The withdrawal document may not be a document already declared here - one policy is one blob,
    # so a repeat coalesces onto the frame it already has and nothing comes into force
    entitle(recorded, [(ACTOR, verb, '*') for verb in ('admin', 'validate', 'mark')])
    withdrawn, ran = priced('under-a-document')
    assert withdrawn.status_code == 422 and not ran, 'the store answered a seat the queue refuses'

    # and the deployment's own seat still prices, which is the grant that keeps the desk ticking
    deployment, ran = priced('deployment', actor=None)
    assert deployment.status_code == 200 and ran, deployment.text
    monkeypatch.delenv(spine.SPINE_ACTOR)
    unnamed, ran = priced('unnamed', actor=None)
    assert unnamed.status_code == 422 and not ran
    assert spine.SPINE_ACTOR in unnamed.json()['detail']
    assert verify_home(recorded)['events'] == head(recorded)


def test_the_scope_the_queue_asks_for_is_the_one_the_jobs_lane_will_mint(recorded):
    """WHAT A JOB IS ADMITTED UNDER IS WHAT ITS LANE WILL FILE. A standing run files a
    `run_completed`, so the queue asks for THAT TYPE's own verb and a seat scoped only to validate
    is turned away before the Monte Carlo rather than after it - where increment 3's boundary left
    it, the hub paying for an execution the writer would then refuse.

    And the seat asked about is THE REQUEST'S where it names one: the same job posted under a seat
    the document scopes for `book` runs, and the attestation is filed under that seat rather than
    under the deployment's own, so what was admitted and what was recorded are one name.

    Killing mutations: one verb asked for every lane, which either stops every what-if on a booking
    desk or lets a validate-only seat mint attestations; and the request's actor read for admission
    and not for the append, which records somebody else's run.
    """
    assert (spine.STANDING_TYPE, spine.SETTLEMENT_EXPORT) == (
        'run_completed', policy.DESIGNATED_PROCESSES[0])
    entitle(recorded, [(ACTOR, verb, '*') for verb in ('admin', 'validate')]
            + [(DESK_TWO, 'book', '*')])
    standing = dict(own_job('admission'), lane=spine.STANDING)

    refused = CLIENT.post('/execute', content=dump(standing), headers=JSON)
    assert refused.status_code == 422 and 'book' in refused.json()['detail'], refused.text
    assert ACTOR in refused.json()['detail'], 'the queue asked about a seat nobody named'
    assert facts(recorded, 'capability_denied')[0][2]['attempted_type'] == spine.STANDING_TYPE
    assert facts(recorded, 'run_completed') == []

    ran = submit(standing, actor=DESK_TWO)
    assert drained(ran)['status'] == 'done', ran
    assert len(facts(recorded, 'run_completed')) == 1, 'the admitted standing run attested nothing'
    log = opened(recorded)
    try:
        assert [frame['actor'] for frame in log.frames()
                if frame['event_type'] == 'run_completed'] == [DESK_TWO]
    finally:
        log.close()
    assert verify_home(recorded)['events'] == head(recorded)


def test_the_poll_paths_own_jobs_are_admitted_under_the_deployments_seat(recorded, quoting):
    """THE POLL PATH NAMES NOBODY. A diary compile and a beat of the cadence are TELEMETRY jobs no
    request signed, so the seat the queue admits them under is `DV_SPINE_ACTOR` - the deployment's
    own, which is the whole of what the metronome has - and a document that does not scope that seat
    stops them BY NAME rather than silently, which is the one failure a ticking desk must be told
    about. Neither is run here: what is asserted is which seat the queue asked about.

    Killing mutation: a job carrying no actor admitted unconditionally, which leaves the whole poll
    path outside the document a desk declared.
    """
    service.BOOK_DIARY_CACHE.clear()
    entitle(recorded, [(ACTOR, verb, '*') for verb in ('admin', 'validate')])
    assert CLIENT.get('/book/diary').status_code == 200, 'the compile was refused its own seat'

    # not cleared: admission is asked before the cache is read, so the warm one refuses too
    entitle(recorded, [(ACTOR, 'admin', '*')])
    for refused in (CLIENT.get('/book/diary').json()['detail'],
                    str(pytest.raises(spine.SpineRefused, service.submit_bloomberg, {}).value)):
        assert ACTOR in refused and 'validate' in refused, refused
    assert facts(recorded, 'capability_denied')[-1][2]['attempted_type'] == spine.TELEMETRY
    assert verify_home(recorded)['events'] == head(recorded)
