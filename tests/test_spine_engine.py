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
import asyncio
import hashlib
import itertools
import json
import math
import os
import shutil
import sys
import threading

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import derivus
from derivus import service, spine, structures, utils
from derivus.config import CustomJsonEncoder
from derivus.schema import deal_at, splice_deal, walk_job_deals
from derivus_spine import SpineLog, canonical_bytes, init_home, verify_home
from derivus_spine import oracle, policy, projections
from derivus_spine.capability import CAPABILITIES_POLICY, canonical_document

import test_spine_identity
from test_service import (FUNDED, SHORT_CALL, VANILLA, answered, file_world, recalc, xva_book,
                          xva_rows)

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
#: The identity gates' own IdP - keys minted in-process, the JWKS spelled from them by hand.
idp = test_spine_identity.idp
COLLAR = {'pair': 'USDZAR', 'expiry': '1Y', 'notional': AMOUNT,
          'notional_currency': 'USD', 'floor': USDZAR * 0.95}


def job(deals=(CASHFLOW,), factors=FACTORS, sections={}, **calculation):
    """A job document as the objects a market data file holds - `test_service`'s own shape."""
    return {'Calc': {
        'Calculation': dict({'Object': 'BaseValuation', 'Base_Date': BASE, 'Currency': 'USD',
                             'MCMC_Simulations': 1, 'Quasi_Seed': 1}, **calculation),
        'Deals': {'Reference': 'spine-desk',
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
    document = job(sections={'Bootstrapper Configuration': {'FXVol': {}}})
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
    first, never be dequeued, and hang the gate instead of failing it. No request signs it, so it
    runs under the deployment's own seat, as the poll paths do.
    """
    service.EXECUTOR.queue.join()
    barrier = Barrier()
    service.EXECUTOR.submit(service.Job('barrier-{}'.format(next(counter)), barrier, {},
                                        actor=spine.hub()), service.HEAVY)
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

def test_importing_the_engine_lands_neither_the_spine_nor_its_one_dependency():
    """The extra is an EXTRA: a fresh interpreter imports the engine, the seam and the HTTP surface
    and reports what arrived. `pip install derivus` must not grow a `cryptography` dependency for a
    book of record that box does not run, which is why every import in `derivus/spine.py` is inside
    the function that needs it.

    Killing mutation: a module-level `import derivus_spine.verbs` in `derivus/spine.py`, which lands
    the record and `cryptography` on every engine install.
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

    Killing mutation: `where` answering nothing rather than refusing, which sends a Context verb
    after a home nobody configured.
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

    Killing mutation: `configured` answering no for a home that was never minted, which
    un-records the box in silence.
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
    putting a name in the record that nobody chose.

    Killing mutation: the actor check dropped, which appends under no name.
    """
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
    the LSN it landed at, so a caller can cite it without folding for it: a base valuation's two
    seeds, `seed` null since it declares no `Random_Seed` and `quasi_seed` the `Quasi_Seed` it
    states.

    Killing mutations: `attests` answering no for every job, which serves a standing run's numbers
    with no `run_completed` behind them; the replay tuple read without its `quasi_seed`, which the
    record refuses as a claim with no `quasi_seed` and the run errors.
    """
    genesis = head(recorded)
    for silent in (spine.TELEMETRY, spine.CURIOSITY):
        result = drained(submit(own_job('SILENT-' + silent), lane=silent))
        assert result['status'] == 'done', result.get('error')
        assert 'attested' not in result and head(recorded) == genesis, silent

    # the default is curiosity, so a caller who says nothing records nothing
    assert drained(submit(own_job('DEFAULT-LANE')))['status'] == 'done'
    assert head(recorded) == genesis, 'the default lane recorded something'

    standing = drained(submit(own_job('STANDING-BIRTH', Quasi_Seed=7), lane=spine.STANDING))
    assert standing['status'] == 'done', standing.get('error')
    assert head(recorded) == genesis + 1
    assert standing['attested']['lane'] == spine.STANDING
    assert standing['attested']['lsn'] == genesis + 1

    lsn, _, body = facts(recorded, 'run_completed')[0]
    assert lsn == standing['attested']['lsn']
    assert body['plan_hash'] == standing['plan_hash']
    assert body['values_hash'] == standing['values_hash']
    assert body['engine_version'] == standing['engine_version'] == derivus.__version__
    assert body['seed'] is None and body['quasi_seed'] == standing['quasi_seed'] == 7
    assert body['lane'] == spine.STANDING
    assert verify_home(recorded)['events'] == head(recorded)


def test_a_synthetic_tick_sequence_mints_nothing_and_the_absence_is_asserted(recorded, quoting):
    """A telemetry repaint mints no event, asserted by absence over a synthetic tick sequence: four
    market ticks moving the ATM vol, each installing quotes, re-bootstrapping the surface and
    rewriting the file, with a what-if priced between them. Every one is a READING, superseded
    before anything could cite it. The assertion is the HEAD and not a filtered count, which would
    pass on a log full of the wrong events.

    Killing mutation: the what-if submitted in the standing lane with its evidence, which files a
    run for every repaint.
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

    Killing mutation: the attestation of a run already done dropped, which serves a standing caller
    stored numbers no fact cites.
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

    Killing mutation: the promotion in `submit` dropped, which publishes the what-if's run with
    nothing attested.
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
    would be the unbacked citation the lane rule prevents. The attestation is the HUB's own act, so
    no document refuses it: a seat's grant withdrawn while the run waits behind a barrier costs it
    nothing. What refuses it is the record itself - a home whose `blobs/` went away while the run
    waited is not a home - and then the refusal travels as the run's own error with nothing filed.

    THE QUEUE ASKS THE REQUEST'S SEAT FOR `mark` OVER THE JOB'S OWN BOOK, so the ordinary desk seat
    - `book` over its book, `validate` everywhere - is turned away before the Monte Carlo is paid
    for, the refusal naming that book.

    Killing mutations: the attestation filed under the requester, which a grant withdrawn behind
    the barrier then refuses; and admission still asking `book` over `*`, which admits the desk seat
    to numbers the hub attests as standing.
    """
    marking = [(ACTOR, verb, '*') for verb in ('admin', 'validate', 'mark')]
    entitle(recorded, marking)
    barrier = holding()
    try:
        assert barrier.running.wait(WORKER_SECONDS), 'the worker never picked the barrier up'
        submitted = submit(own_job('WITHDRAWN'), lane=spine.STANDING, actor=ACTOR)
        assert submitted['status'] == 'queued', submitted
        entitle(recorded, [(ACTOR, 'admin', '*')])
    finally:
        barrier.released.set()
    result = drained(submitted)
    assert result['status'] == 'done', result.get('error')
    assert facts(recorded)[-1][1] == 'run_completed' and result['attested']['lsn'] == head(recorded)

    # not the first document again: one policy is one blob, so a repeat would coalesce onto the
    # frame it already has and leave the withdrawal in force
    entitle(recorded, marking + [(DESK_TWO, 'validate', 'spine-desk')])
    blobs = recorded / 'blobs'
    barrier = holding()
    try:
        assert barrier.running.wait(WORKER_SECONDS), 'the worker never picked the barrier up'
        submitted = submit(own_job('UNHOMED'), lane=spine.STANDING, actor=ACTOR)
        assert submitted['status'] == 'queued', submitted
        standing = head(recorded)
        blobs.rename(recorded / 'blobs-away')
    finally:
        barrier.released.set()
    try:
        result = drained(submitted)
    finally:
        (recorded / 'blobs-away').rename(blobs)
    assert result['status'] == 'error' and 'there is no blobs/' in result['error'], result
    assert head(recorded) == standing, 'the record kept an attestation it could not hold'

    entitle(recorded, [(ACTOR, verb, '*') for verb in ('admin', 'validate')]
            + [(ACTOR, 'book', 'spine-desk')])
    stored = dict(service.EXECUTOR.results)
    turned = CLIENT.post('/execute', content=dump(dict(own_job('UNADMITTED'), lane=spine.STANDING,
                                                       actor=ACTOR)), headers=JSON)
    assert turned.status_code == 422, turned.text
    assert "no mark scope over 'spine-desk'" in turned.json()['detail'], turned.text
    assert service.EXECUTOR.results == stored, 'the queue ran a job it had refused'


def test_a_standing_run_off_a_plan_id_refuses_by_name(recorded, desk):
    """A `plan_id` names a PARSE the cache holds, where an attestation carries the job document the
    plan recompiles from. `Context.save_json` is not a complete round trip, so serialising the
    parse back would store a document that is not the one that ran. The refusal names the remedy;
    the curiosity lane over the same plan still runs.

    Killing mutation: the `plan_id` check dropped, which attests a parse rather than a document.
    """
    prepared = CLIENT.post('/prepare', content=dump(own_job('PLAN-ID')), headers=JSON).json()
    refused = CLIENT.post('/execute', json={'plan_id': prepared['plan_id'],
                                            'lane': spine.STANDING})
    assert refused.status_code == 422
    assert 'plan_id' in refused.json()['detail'] and 'curiosity' in refused.json()['detail']
    assert head(recorded) == 4

    ran = CLIENT.post('/execute', json={'plan_id': prepared['plan_id'],
                                        'lane': spine.CURIOSITY}).json()
    assert 'result_id' in ran and head(recorded) == 4


def test_the_stored_job_is_the_job_and_not_the_submission_that_carried_it(recorded, desk):
    """What a standing attestation stores is the `Calc` ENVELOPE and nothing beside it: a posted
    body may also carry `Patch`, `lane` or `plan_id`, none of which is the job. This blob is the
    FIRST LINK of the provenance chain, and the gate above would pass on either because the
    engine's loader tolerates a surplus top-level key.

    The patch is not lost by it: the values vector filed beside the job carries the whole market as
    patched, which is the model of a result as engine(plan, values). Both are asserted.

    Killing mutation: the posted body stored whole, `Patch` and `lane` beside `Calc`.
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
    lanes being the record's vocabulary and not the service's.

    Killing mutation: `check_lane` passing any lane through, which runs `exploration` as a lane
    nobody knows.
    """
    refused = CLIENT.post('/execute',
                          content=dump(dict(own_job('UNKNOWN-LANE'), lane='exploration')),
                          headers=JSON)
    assert refused.status_code == 422
    assert 'telemetry, curiosity, standing' in refused.json()['detail']
    assert head(recorded) == 4


# --------------------------------------------------------------------------------------------
# Provenance: the plan recompiles, and the result reproduces.

# --------------------------------------------------------------------------------------------
# Promotion, through the real engine.

def test_a_result_pinned_matching_a_known_tuple_cache_hits_without_re_executing(recorded, desk):
    """A tuple this hub attested itself needs no re-execution, and the gate COUNTS rather than
    believes: the executor handed in records its calls and is never called. The pin still lands and
    names the tolerance policy in force - a promotion under no declared standard is not one this
    record carries.

    Killing mutation: the attestation lookup skipped, which re-executes a tuple this hub already
    ran.
    """
    declare(recorded, policy.TOLERANCE_POLICY, {'tolerances': {'mtm': 1e-9}})
    standing = drained(submit(own_job('CACHE-HIT'), lane=spine.STANDING))
    _, _, body = facts(recorded, 'run_completed')[0]
    ran = []

    pinned = spine.pin_result(
        {name: body[name] for name in ('plan_hash', 'values_hash', 'engine_version', 'seed',
                                       'quasi_seed')},
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

    Killing mutation: the departures ignored, which pins a doctored result.
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


def test_a_result_pinned_whose_replay_skipped_a_deal_is_refused_by_name(recorded, desk):
    """A claim carrying the engine's own result for a job beside a cash flow on a curve the market
    lacks is refused naming that deal and the curve, nothing appended. Killing mutation: the
    executor running without `utils.run_whole` - the partly priced result is pinned, reproduced."""
    declare(recorded, policy.TOLERANCE_POLICY, {'tolerances': {'mtm': 1e-9}})
    unseen = job(deals=(dict(CASHFLOW, Reference='PIN-WHOLE'),
                        dict(CASHFLOW, Reference='PIN-GAP', Discount_Rate='ZAR-SWAP')))
    context = derivus.Context().load_json((dump(unseen), 'unseen'))
    _, out = derivus.Context().load_json((dump(unseen), 'unseen')).run_job()
    assert out['Stats']['Deals Skipped'] == 1, out['Stats']
    standing = head(recorded)

    with pytest.raises(utils.UnpriceableSchedule,
                       match=r"PIN-GAP \('Cannot find InterestRate.ZAR-SWAP',\) - Skipped"):
        derivus.Context().pin_result(
            spine.canonical(unseen), spine.values_of(context), spine.result_of(out),
            spine.replay(context))
    assert head(recorded) == standing, 'a refused pin wrote something'


def test_a_run_says_the_files_it_read_and_a_replay_reads_the_bytes_that_stood(recorded,
                                                                             tmp_path):
    """THE FILES A JOB NAMES, DECLARED BY THEIR CONTENT, on the book naming its market data file
    by path. Declared under a name through `POST /book/files`, the file's blob is the sha256
    `Context.files` takes of the book's file, one number, and `GET /book/files` lists it beside
    `MarketDataFile`. A standing run of the book attests the digests its job read beside the
    replay tuple - the declared blob and a null calendar, the run's own `Stats['Files']` - and the
    oracle finds the file declared. The file is then rewritten on disk, the ZAR curve's day count
    moved to ACT_360, a plan field the stored values vector patched over it cannot hide: the claim
    replayed reads the bytes that stood and marks the cashflow at the run's own 15,803,002.92, the
    result to the byte, where re-executing off the rewritten path marks it at 15,766,149.02; the
    listing no longer matches the book's file. The job never changed.

    Killing mutations: the replay reading the path (`reading` handing the context nothing), which
    marks the rewritten file; the attestation without its digests (the evidence's `files` not
    filed), which leaves the replay nothing to read the bytes by.
    """
    def mark(result):
        table = json.loads(result.decode('utf-8'))['mtm']['.DataFrame']
        return {row[0]: row[table['columns'].index('Value')] for row in table['data']}

    book, market = file_world(tmp_path)
    try:
        declared = CLIENT.post('/book/files', content=dump({
            'name': 'risk-neutral', 'kind': 'market', 'path': str(market)}), headers=JSON).json()
        document = json.loads(book.read_text())
        assert declared['blob'] == derivus.Context().files(document)['MarketDataFile'] == \
            hashlib.sha256(market.read_bytes()).hexdigest()
        assert [(row['name'], row['kind'], row['blob'], row['matches'])
                for row in CLIENT.get('/book/files').json()['files']] == [
            ('risk-neutral', 'market', declared['blob'], 'MarketDataFile')]

        ran = drained(submit(document, lane=spine.STANDING))
        assert ran['status'] == 'done', ran.get('error')
        _, _, body = facts(recorded, 'run_completed')[0]
        assert body['files'] == ran['stats']['Files'] == {
            'MarketDataFile': declared['blob'], 'CalendDataFile': None}
        assert oracle.report(recorded)['files_declared']['held'] is True

        data = json.loads(market.read_text())
        data['MarketData']['Price Factors']['InterestRate.ZAR']['Day_Count'] = 'ACT_360'
        market.write_text(json.dumps(data))
        job, values = blob(recorded, body['job']), blob(recorded, body['values_hash'])
        _, stood = spine.executor(files=body['files'])(job, values, body['engine_version'])
        _, moved = spine.executor()(job, values, body['engine_version'])
        assert stood == blob(recorded, body['result'])
        assert mark(stood)['CF1'] != mark(moved)['CF1']
        assert CLIENT.get('/book/files').json()['files'][0]['matches'] is None
        assert json.loads(book.read_text()) == document, 'the job moved'
    finally:
        service.BOOK = None


# --------------------------------------------------------------------------------------------
# The Context verbs.

def test_the_context_verbs_book_amend_and_file_the_three_lifecycle_facts(recorded):
    """The five delegators on a real home. Each canonicalises through the ENGINE's own encoder, so
    an instrument's id is the content hash a job document would give it, and hands plain data to
    the spine - which keeps storage out of every module under `derivus/` but this one.
    `declare_market` is the one that uses its context: the values vector it names is THIS
    context's, which is why officialness is a property of the name.

    Killing mutation: `canonical` spelling the terms with `str` rather than the engine's encoder,
    which gives an instrument an id no job document gives it.
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

    # judged at the deepest node holding the terms, which the seam reads - here the book itself
    linked = context.amend(CASHFLOW, amended, book='spine-desk')
    assert facts(recorded, 'amendment')[0][2] == {
        'instrument': derivus.content_hash(CASHFLOW), 'amended_to': derivus.content_hash(amended),
        'portfolio': 'spine-desk'}
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

    Killing mutation: the latest declaration taken by LSN alone, which lets a backdated close
    displace the declaration in force.
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

    Killing mutation: the designation unchecked, which prices a process on whatever market it names.
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

    Killing mutation: `verdicts` answering the latest verdict alone, which reads a plan rejected
    then approved as approved and nothing more.
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
    folded = projections.PROJECTORS['quotes']
    quoted = spine.folded(lambda log: folded.rows(projections.fold(log, folded)))
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

    Killing mutation: the counterparty taken from anything but the set the deal is booked under.
    """
    document = json.loads(desk.read_text())
    # another client's set ahead of this one, so only the set booked under names CPTY_A
    document['Calc']['Deals']['Deals']['Children'] += [
        netting_set('CLIENT_B', 'CPTY_B'), netting_set(CLIENT_SET, 'CPTY_A')]
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


def test_an_amendment_naming_a_field_its_type_does_not_declare_is_refused_and_files_nothing(
        recorded, desk):
    """AN AMENDMENT NAMES FIELDS OF ITS DEAL'S TYPE, and one that type does not declare is refused
    before it merges. On a structure booked whole, a key the structure does not declare and a key on
    one of its legs each refuse naming the key and the type, the file byte-identical and nothing
    filed; an amount the leg declares restrikes the structure. A booking still takes a key its type
    does not declare, as every deal of a legacy book carries one.

    Killing mutation: the judge removed, which writes both keys and files an amendment for each.
    """
    with_client(desk)
    legs = [{'Instrument': {'.Deal': dict(CASHFLOW, Reference='LEG-{}'.format(leg))}}
            for leg in (1, 2)]
    booked = ticketed('EXEC-S', deal={'Object': 'StructuredDeal', 'Reference': 'STRUCT',
                                      'Currency': 'ZAR', 'Children': legs}).json()
    assert booked['written'] is True, booked
    before, head_before = desk.read_bytes(), head(recorded)
    for path, fields, said in (
            (booked['deal_path'], {'Underlying_Amount': 4_000_000.0}, 'StructuredDeal declares no '
             'Underlying_Amount - it declares Currency, MtM, Net_Cashflows, Object, Product_Class,'
             ' Reference, Sales_Margin, Sales_Margin_Currency'),
            (booked['deal_path'] + '/0', {'Amout': 5.0}, 'FixedCashflowDeal declares no Amout')):
        refused = restruck(path, **fields)
        assert refused['written'] is False, refused
        assert [message[:len(said)] for message in refused['refused']] == [said], refused
        assert desk.read_bytes() == before, 'a refused amendment moved the file'
        assert head(recorded) == head_before, 'a refused amendment filed something'

    amended = restruck(booked['deal_path'] + '/0', Amount=2_000_000.0)
    assert amended['written'] is True and amended['recorded']['event_type'] == 'amendment'
    assert [lsn for lsn, _, _ in facts(recorded, 'amendment')] == [amended['recorded']['lsn']]
    tagged = ticketed('EXEC-T', deal=dict(CASHFLOW, Reference='CF-T', Tags=['legacy'])).json()
    assert tagged['written'] is True, tagged


def test_a_trade_restruck_back_and_restruck_again_stands_where_the_file_has_it(recorded, desk):
    """A booked cash flow restruck to a new amount, back, and to the new amount again is three
    amendments on the record, and the record's position stands on the terms the file holds: the
    reconciliation finds no position the file lacks and the trade nowhere unbooked.

    Killing mutation: the third restrike filed as offered, which the writer answers with the
    first's LSN - two amendments, and a position the record still holds on the terms it left.
    """
    with_client(desk)
    booked = ticketed('EXEC-R').json()
    assert booked['written'] is True, booked
    landed = [restruck(booked['deal_path'], Amount=amount) for amount in (
        300_000.0, 250_000.0, 300_000.0)]
    assert all(answer['written'] is True for answer in landed), landed
    filed = [lsn for lsn, _, _ in facts(recorded, 'amendment')]
    assert filed == [answer['recorded']['lsn'] for answer in landed] and len(set(filed)) == 3, filed
    drift = CLIENT.get('/book/reconcile', params={'actor': ACTOR}).json()
    assert drift['in_record_not_in_file'] == [], drift
    assert 'EXEC-R' not in [row['reference'] for row in drift['in_file_not_in_record']], drift


def test_the_agreements_read_names_what_paper_never_states(recorded):
    """The read the paper screen makes says which fields an agreement's terms never state - a
    balance and a holding being settlement state - so a form offers none of them: the netting
    set's `Opening_Balance`, and the `Amount`, `Units` and `Principal` of a collateral table's
    rows. Each one it names is the one a declaration stating it is refused for, by name.

    Killing mutation: the block left off the read, which leaves a form offering every field.
    """
    terms = netting_set(CLIENT_SET, 'LEI-A')['Instrument']['.Deal']
    assert CLIENT.post('/book/entities', content=dump({'entity': 'LEI-A', 'name': 'Client A'}),
                       headers=JSON).status_code == 200
    read = CLIENT.get('/book/agreements').json()
    assert read == {'agreements': [], 'settlement_state': {
        'fields': ['Opening_Balance'], 'collateral_columns': ['Amount', 'Units', 'Principal']}}
    named = read['settlement_state']
    tables = {'Amount': 'Cash_Collateral', 'Units': 'Equity_Collateral',
              'Principal': 'Bond_Collateral'}
    stated = [(field, dict(terms, **{field: 1.0})) for field in named['fields']] + [
        (column, dict(terms, Collateral_Assets={tables[column]: [{column: 1.0}]}))
        for column in named['collateral_columns']]
    for field, body in stated:
        refused = CLIENT.post('/book/agreements', content=dump({
            'agreement': CLIENT_SET, 'entity': 'LEI-A', 'kind': 'ISDA 2002', 'terms': body}),
            headers=JSON)
        assert refused.status_code == 422 and '{} is 1.0'.format(field) in refused.json()[
            'detail'], (field, refused.text)


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

    terms = netting_set(CLIENT_SET, 'LEI-A')['Instrument']['.Deal']
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

    for reference, portfolio in (('EXEC-PAPER', 'spine-desk/Rates/EM'), ('EXEC-BOOK', None)):
        booked = CLIENT.post('/book/deals', content=dump(dict(
            {'action': 'add', 'deal': dict(deal, Reference=reference), 'parent_reference':
             CLIENT_SET, 'agreement': CLIENT_SET, 'quantity': 1.0, 'price': 99.5,
             'execution_reference': reference},
            **({} if portfolio is None else {'portfolio': portfolio}))), headers=JSON).json()
        assert booked['written'] is True, booked
    fills = [body for _, _, body in facts(recorded, 'fill')]
    assert [(fill['agreement'], fill['counterparty'], fill['portfolio'], fill['price'])
            for fill in fills] == [(CLIENT_SET, 'LEI-A', 'spine-desk/Rates/EM', 99.5),
                                   (CLIENT_SET, 'LEI-A', 'spine-desk', 99.5)]
    projections = service.spine.package().projections
    positions = projections.PROJECTORS['positions']
    rows = service.spine.folded(lambda log: positions.rows(projections.fold(log, positions)))
    assert {(row['agreement'], row['portfolio']) for row in rows} == {
        (CLIENT_SET, 'spine-desk/Rates/EM'), (CLIENT_SET, 'spine-desk')}
    assert verify_home(recorded)['events'] == head(recorded)


def test_a_collateral_row_stated_as_cells_is_refused_by_name_wherever_it_is_posted(recorded, desk):
    """A COLLATERAL ROW HAS ONE SPELLING, BY ITS COLUMNS' NAMES, the one every reader of the table
    reads. A netting set whose eligible cash is a list of cells is refused in one sentence wherever
    it is posted - as an agreement's terms with a 422, as a set booked into the book unwritten, and
    in a job handed to `/validate` - never with a 500, and nothing is written or recorded; the same
    row stated by its columns' names is declared.

    Killing mutations: discovery reading every row of a table as an object, which fails the
    booking on `'list' object has no attribute 'get'`; the row of cells left unnamed, which declares
    the agreement.
    """
    said = ("Collateral_Assets.Cash_Collateral[0] is a list of cells: a collateral row is stated "
            "by its columns' names")
    terms = netting_set(CLIENT_SET, 'LEI-A')['Instrument']['.Deal']
    cells = dict(terms, Collateral_Assets={'Cash_Collateral': [['USD', None, {'.Percent': 10.0}]]})
    assert CLIENT.post('/book/entities', content=dump({'entity': 'LEI-A', 'name': 'Client A'}),
                       headers=JSON).status_code == 200
    book, record = desk.read_bytes(), head(recorded)

    def declared(stated):
        return CLIENT.post('/book/agreements', content=dump({
            'agreement': CLIENT_SET, 'entity': 'LEI-A', 'kind': 'CSA', 'terms': stated}),
            headers=JSON)

    refused = declared(cells)
    assert refused.status_code == 422 and said in refused.json()['detail'], refused.text
    booked = CLIENT.post('/book/deals', content=dump({
        'action': 'add', 'deal': {'Instrument': {'.Deal': dict(cells, Reference='NS-CELLS')}},
        'quantity': 1.0, 'execution_reference': 'EXEC-CELLS'}), headers=JSON)
    assert booked.status_code == 200 and booked.json()['written'] is False, booked.text
    assert said in booked.json()['refused'], booked.text
    validated = CLIENT.post('/validate', content=dump(job(deals=(dict(cells, Reference='NS-CELLS'),))),
                            headers=JSON)
    assert validated.status_code == 200 and validated.json()['deals'] == {'NS-CELLS': [said]},         validated.text
    assert desk.read_bytes() == book and head(recorded) == record, 'a refusal wrote or recorded'
    assert declared(dict(terms, Collateral_Assets={'Cash_Collateral': [
        {'Currency': 'USD', 'Haircut_Posted': {'.Percent': 10.0}}]})).status_code == 200


def test_an_agreement_s_holding_is_refused_whatever_it_states(recorded):
    """A HOLDING IS SETTLEMENT STATE, NEVER PAPER, whatever amount it states: an agreement's cash
    row stating an `Amount` of 0.0 is refused naming it as one stating 500,000 is, while a row
    stating none, or null, is declared.

    Killing mutation: the amount tested by its truth, which declares the row stating 0.0.
    """
    terms = netting_set(CLIENT_SET, 'LEI-A')['Instrument']['.Deal']
    assert CLIENT.post('/book/entities', content=dump({'entity': 'LEI-A', 'name': 'Client A'}),
                       headers=JSON).status_code == 200

    def declared(row):
        return CLIENT.post('/book/agreements', content=dump({
            'agreement': CLIENT_SET, 'entity': 'LEI-A', 'kind': 'CSA', 'terms': dict(
                terms, Collateral_Assets={'Cash_Collateral': [row]})}), headers=JSON)

    refused = declared({'Currency': 'USD', 'Amount': 0.0})
    assert refused.status_code == 422, refused.text
    assert 'Collateral_Assets.Cash_Collateral[0].Amount is 0.0' in refused.json()['detail']
    for row in ({'Currency': 'USD'}, {'Currency': 'USD', 'Amount': None}):
        assert declared(row).status_code == 200, row


def test_paper_restated_and_then_restored_stands_restored(recorded):
    """Through the service, as a first-time user files it: an agreement declared with terms A,
    then B, then A again is three facts with A in force, the third answered with a new LSN and its
    row naming B's as the one it stood over, while A declared once more answers the third's LSN. An
    entity renamed and then named back stands under its first name at the third LSN. The oracle and
    the verification hold over what this leaves.

    Killing mutation: the body no longer carrying `supersedes`, so the third answer is the first
    declaration's LSN and B stays in force.
    """
    def declared(path, body):
        answer = CLIENT.post(path, content=dump(body), headers=JSON)
        assert answer.status_code == 200, answer.text
        return answer.json()['recorded']['lsn']

    named = [declared('/book/entities', {'entity': 'LEI-A', 'name': name})
             for name in ('Client A', 'Client A plc', 'Client A')]
    a = netting_set(CLIENT_SET, 'LEI-A')['Instrument']['.Deal']
    first, second, back, again = [
        declared('/book/agreements', {'agreement': CLIENT_SET, 'entity': 'LEI-A',
                                      'kind': 'ISDA 2002', 'terms': terms})
        for terms in (a, dict(a, Liquidation_Period=10.0), a, a)]
    assert len(set(named)) == 3 and len({first, second, back}) == 3 and again == back
    row, = CLIENT.get('/book/agreements').json()['agreements']
    assert (row['lsn'], row['supersedes_lsn'], row['terms']['Liquidation_Period']) == (
        back, second, 0.0)
    entity, = CLIENT.get('/book/entities').json()['entities']
    assert (entity['lsn'], entity['name']) == (named[2], 'Client A')
    assert oracle.failed(oracle.report(recorded)) == []
    assert verify_home(recorded)['events'] == head(recorded) == back


def test_a_name_another_entity_holds_is_refused_in_the_record_s_words(recorded):
    """Through the service: an entity declared under a name standing under another id answers 422
    with the record's own sentence - the name, the id that holds it, the remedy - and records
    nothing, while the id that holds the name declares it again as the fact already standing.

    Killing mutation: the check dropped from `declare_entity`, which answers 200 and files a
    second id under the name.
    """
    def declared(entity):
        return CLIENT.post('/book/entities', content=dump({'entity': entity, 'name': 'Client A'}),
                           headers=JSON)

    assert declared('LEI-A').status_code == 200
    before = head(recorded)
    refused = declared('LEI-B')
    assert (refused.status_code, refused.json()['detail']) == (
        422, "declare_entity: the name 'Client A' stands under entity 'LEI-A', and a name "
             "identifies one legal entity - restate 'LEI-A' if this is that entity, or give "
             "'LEI-B' a name of its own")
    assert declared('LEI-A').json()['recorded']['lsn'] == head(recorded) == before
    assert [row['entity'] for row in CLIENT.get('/book/entities').json()['entities']] == ['LEI-A']


def test_an_agreement_id_another_entity_holds_is_refused_in_the_record_s_words(recorded):
    """Through the service: an agreement declared under an id standing under another entity, its
    terms that entity's own, answers 422 with the record's own sentence - the id, the entity it
    stands under, the remedy - and records nothing, while the entity that holds the id declares it
    again as the fact already standing.

    Killing mutation: the check dropped from `declare_agreement`, which answers 200 and moves the
    agreement, and every position under it, to the other entity.
    """
    def declared(entity):
        return CLIENT.post('/book/agreements', content=dump({
            'agreement': CLIENT_SET, 'entity': entity, 'kind': 'ISDA 2002',
            'terms': netting_set(CLIENT_SET, entity)['Instrument']['.Deal']}), headers=JSON)

    for entity in ('LEI-A', 'LEI-B'):
        assert CLIENT.post('/book/entities', content=dump({'entity': entity, 'name': entity}),
                           headers=JSON).status_code == 200
    first = declared('LEI-A').json()['recorded']['lsn']
    refused = declared('LEI-B')
    assert refused.status_code == 422 and refused.json()['detail'] == (
        "declare_agreement: the agreement 'CLIENT_A' stands under entity 'LEI-A', and an "
        "agreement's id belongs to one legal entity - declare the agreement with 'LEI-B' under an "
        'id of its own'), refused.text
    assert declared('LEI-A').json()['recorded']['lsn'] == head(recorded) == first
    assert [row['entity'] for row in CLIENT.get('/book/agreements').json()['agreements']] == [
        'LEI-A']


def test_a_declared_tree_is_where_a_book_books_and_a_restrike_is_judged(recorded, desk):
    """THE TREE IS DECLARED, BOOK BY BOOK. A book declaring no node books wherever its paths say,
    whatever another book declared; once any node of it stands, a fill books into the book itself
    or a node declared under it, an undeclared path refusing by name with the file untouched. An
    amendment names the deepest node holding every position in the terms it restrikes - read off
    the positions fold under the book lock, never stated by a caller - which is where it is judged.

    Killing mutations: the declared nodes read across every book, which refuses a book another
    book's tree has nothing to do with; and the restrike judged at the first position holding the
    terms rather than at the node holding them all.
    """
    document = json.loads(desk.read_text())
    document['Calc']['Deals']['Deals']['Children'].append(netting_set(CLIENT_SET, 'CPTY_A'))
    desk.write_text(json.dumps(document, indent=2), newline='\n')

    def booking(execution, portfolio, reference='CF-TREE'):
        return CLIENT.post('/book/deals', content=dump({
            'action': 'add', 'deal': dict(CASHFLOW, Reference=reference, Amount=250_000.0),
            'parent_reference': CLIENT_SET, 'quantity': 1.0, 'execution_reference': execution,
            'portfolio': portfolio}), headers=JSON)

    spine.declare_portfolio('another-book/Rates')
    assert booking('EXEC-FREE', 'spine-desk/Anywhere', 'CF-FREE').json()['written'] is True
    for path in ('spine-desk/FX', 'spine-desk/FX/Options', 'spine-desk/FX/Spot'):
        spine.declare_portfolio(path)
    assert [row['path'] for row in spine.portfolios()] == [
        'another-book/Rates', 'spine-desk/FX', 'spine-desk/FX/Options', 'spine-desk/FX/Spot']
    before = desk.read_bytes()
    refused = booking('EXEC-NOWHERE', 'spine-desk/Rates')
    assert refused.status_code == 422 and 'declare the node first' in refused.json()['detail']
    assert desk.read_bytes() == before, 'a refused booking moved the file'

    held = [booking(execution, portfolio).json() for execution, portfolio in (
        ('EXEC-OPT', 'spine-desk/FX/Options'), ('EXEC-SPOT', 'spine-desk/FX/Spot'))]
    assert all(answer['written'] for answer in held), held
    assert booking('EXEC-ROOT', 'spine-desk', 'CF-ROOT').json()['written'] is True
    amended = CLIENT.post('/book/deals', content=dump({
        'action': 'amend', 'deal_path': held[0]['deal_path'],
        'fields': {'Amount': 260_000.0}}), headers=JSON).json()
    assert amended['written'] is True, amended
    assert facts(recorded, 'amendment')[0][2]['portfolio'] == 'spine-desk/FX'
    assert verify_home(recorded)['events'] == head(recorded)


def test_an_election_is_judged_where_its_position_sits(recorded, desk):
    """AN ELECTION IS JUDGED WHERE ITS POSITION SITS, as an amendment is: at the deepest node
    holding every position in its instrument under the book, read off the positions fold, filed as
    the body's `portfolio` and never stated by a caller. So a seat granted `book` at a desk's node
    exercises the trade it booked there; a seat at a sibling node is refused by name with the
    denial landed, and refused again where its body names its own node; and an instrument the
    record holds no position in is judged at the book, which a node seat does not reach.

    Killing mutations: `election` out of `capability.PORTFOLIO_TYPES`, which judges it at the book
    and refuses the trader the exercise of its own trade; and `spine.apply_lifecycle` filing the
    portfolio a caller states, which lands the sibling seat's election.
    """
    with_client(desk)
    fx, rates = 'spine-desk/FX', 'spine-desk/Rates'
    for path in (fx, rates):
        spine.declare_portfolio(path)
    entitle(recorded, [(ACTOR, verb, '*') for verb in ('admin', 'validate')] + [
        (DESK_FX, 'validate', 'spine-desk'), (DESK_FX, 'book', fx), (DESK_TWO, 'book', rates)])
    booked = ticketed('CF-ELECT', portfolio=fx, actor=DESK_FX).json()
    assert booked['written'] is True, booked
    held = booked['recorded']['instrument']

    def elect(seat, instrument=held, **stated):
        return CLIENT.post('/book/lifecycle', content=dump({
            'event_type': 'election', 'actor': seat,
            'body': dict({'instrument': instrument, 'choice': 'exercise'}, **stated)}),
            headers=JSON)

    for stated in ({}, {'portfolio': rates}):
        refused = elect(DESK_TWO, **stated)
        assert refused.status_code == 422, (stated, refused.text)
        assert "holds no book scope over '{}'".format(fx) in refused.json()['detail'], stated
        assert facts(recorded, 'capability_denied')[-1][2] == {
            'subject': DESK_TWO, 'verb': 'book', 'book': fx, 'attempted_type': 'election'}
    assert facts(recorded, 'election') == []
    assert elect(DESK_FX, portfolio=rates).status_code == 200
    assert [body for _, _, body in facts(recorded, 'election')] == [
        {'instrument': held, 'choice': 'exercise', 'portfolio': fx}]
    unheld = elect(DESK_FX, instrument=spine.instrument_address(
        {'Instrument': {'.Deal': dict(CASHFLOW, Reference='CF-NOBODY')}}))
    assert unheld.status_code == 422, unheld.text
    assert "holds no book scope over 'spine-desk'," in unheld.json()['detail']
    assert verify_home(recorded)['events'] == head(recorded)
    assert oracle.failed(oracle.report(recorded)) == []


def test_the_money_a_settlement_moved_reads_back_as_balances(recorded, desk):
    """The back office's half through the service: a settlement that moved money states the amount,
    the asset, its kind, the settlement system's reference and the value date, and `/book/cash`
    reads the movements back with the balances they sum to - the collateral held under an agreement
    one row, a movement restated under its reference standing in place of the one it corrects, and
    `date` reading only what had settled by that day. A movement the record could not read back -
    held under an agreement nobody declared, on a day that is not a day, or described with no
    amount - refuses by name with nothing recorded.

    Killing mutations: the value date dropped on the way to the verb, which the record refuses as
    money moved on no day; the `date` filter dropped, which reads a return posted after that day
    into its balance.
    """
    def moved(**body):
        return CLIENT.post('/book/transition', content=dump(dict({'status': 'settled'}, **body)),
                           headers=JSON)

    terms = netting_set(CLIENT_SET, 'LEI-A')['Instrument']['.Deal']
    CLIENT.post('/book/entities', content=dump({'entity': 'LEI-A', 'name': 'Client A'}),
                headers=JSON)
    CLIENT.post('/book/agreements', content=dump({'agreement': CLIENT_SET, 'entity': 'LEI-A',
                                                  'kind': 'ISDA 2002 with CSA', 'terms': terms}),
                headers=JSON)
    head_before = head(recorded)
    held = dict(subject=CLIENT_SET, amount=1.0, asset='USD', kind='collateral', reference='C-0')
    for body, said in ((dict(held, subject='ISDA-NOBODY', value_date='2024-06-28'),
                        'declare the agreement first'),
                       (dict(held, value_date='28/06/2024'), 'is not a day'),
                       (dict(held, value_date=None), 'value date'),
                       (dict(held, amount=None, value_date='2024-06-28'), 'no amount')):
        refused = moved(**{field: value for field, value in body.items() if value is not None})
        assert refused.status_code == 422 and said in refused.json()['detail'], (said, refused.text)
    assert head(recorded) == head_before, 'a refused movement recorded something'

    for amount, reference, day in ((5_000_000.0, 'COL-1', '2024-06-27'),
                                   (5_000_000.0, 'COL-2', '2024-06-28'),
                                   (4_000_000.0, 'COL-2', '2024-06-28'),
                                   (-1_000_000.0, 'COL-3', '2024-07-01')):
        filed = moved(**dict(held, amount=amount, reference=reference, value_date=day))
        assert filed.status_code == 200, filed.text
        assert filed.json()['value_date'] == day and filed.json()['amount'] == amount
    cash = CLIENT.get('/book/cash').json()
    assert [(row['reference'], row['amount']) for row in cash['movements']] == [
        ('COL-1', 5_000_000.0), ('COL-2', 4_000_000.0), ('COL-3', -1_000_000.0)]
    assert cash['balances'] == [{'kind': 'collateral', 'subject': CLIENT_SET, 'asset': 'USD',
                                 'amount': 8_000_000.0, 'movements': 3}]
    that_day = CLIENT.get('/book/cash', params={'date': '2024-06-28'}).json()
    assert that_day['date'] == '2024-06-28' and that_day['balances'][0]['amount'] == 9_000_000.0
    assert CLIENT.get('/book/cash', params={'date': 'yesterday'}).status_code == 422
    assert verify_home(recorded)['events'] == head(recorded)


def test_a_compile_copies_the_deal_tree_alone_and_loads_a_market_only_for_collateral_held(
        recorded, desk):
    """A compile under a record writes inside the deal tree and nowhere else, so that is all it
    copies: the compiled job's market IS the document's, and the document hashes after the compile
    as it did before. And the market is loaded only to cross collateral the record holds: over a
    collateralised set nothing is held under, the loader is never asked; once 2,500,000 dollars
    settle under it the loader is asked once and the set is written that balance, 2,500,000.0 in
    its dollar `Balance_Currency` at a spot of one.

    Killing mutations: the market loaded before a holding is known, where the loader is asked with
    nothing held; the tree written where it stands rather than copied, where the document's own
    set carries the balance and its hash moves.
    """
    terms = dict(netting_set(CLIENT_SET, 'LEI-A')['Instrument']['.Deal'], Collateralized='True')
    document = json.loads(desk.read_text())
    document['Calc']['Deals']['Deals']['Children'].append({'Instrument': {'.Deal': terms}})
    desk.write_text(json.dumps(document, indent=2), newline='\n')
    CLIENT.post('/book/entities', content=dump({'entity': 'LEI-A', 'name': 'Client A'}),
                headers=JSON)
    assert CLIENT.post('/book/agreements', content=dump({
        'agreement': CLIENT_SET, 'entity': 'LEI-A', 'kind': 'ISDA 2002 with CSA',
        'terms': terms}), headers=JSON).status_code == 200
    asked = []

    def compiled():
        document = service.BOOK.read()[0]
        before = derivus.content_hash(document)
        job = spine.compiled_job(document, strict=False, runs='CreditMonteCarlo',
                                 market=lambda job: asked.append(job) or service.market_of(job))
        assert derivus.content_hash(document) == before, 'the compile wrote the document it read'
        return document, job

    document, job = compiled()
    assert job is document and not asked, 'a market was loaded with no collateral held'
    assert CLIENT.post('/book/transition', content=dump({
        'status': 'settled', 'subject': CLIENT_SET, 'amount': 2_500_000.0, 'asset': 'USD',
        'kind': 'collateral', 'reference': 'COL-1', 'value_date': '2024-06-28'}),
        headers=JSON).status_code == 200
    document, job = compiled()
    assert len(asked) == 1
    assert [node['Instrument']['.Deal'].get('Opening_Balance')
            for _, node in service.netting_sets(job)] == [2_500_000.0]
    assert job['Calc']['MergeMarketData'] is document['Calc']['MergeMarketData']


def test_a_status_poll_is_answered_anew_once_the_record_moves_and_the_paper_has_its_own_stamp(
        recorded, desk):
    """Under a home a poll's key carries where the record's log stands, `spine.platter`: the home
    and each segment's size, by hand. The status is kept under it and a second poll is answered
    what the first kept. A declaration lands a frame, so the status is asked anew: its
    `spine.paper`, 0 on a record declaring no paper, is an entity's LSN once one is declared and
    an agreement's after it, while a fixings policy declared between them moves the key and not
    the paper.

    Killing mutations: the platter left out of a poll's key, where the key by hand differs and the
    status reads the paper where the first poll found it; the paper stamped with the head, where
    it reads 4 on a record declaring none and the policy moves it.
    """
    def paper():
        return CLIENT.get('/book/status').json()['spine']['paper']

    log, started = recorded / 'log', head(recorded)
    key = service.poll_key(service.BOOK, service.xva_path())
    assert spine.platter() == key[4] == (str(recorded),) + tuple(
        (name, (log / name).stat().st_size) for name in sorted(os.listdir(log)))
    assert paper() == 0 and service.BOOK.polls.get(('status',) + key)['spine']['paper'] == 0
    terms = netting_set(CLIENT_SET, 'LEI-A')['Instrument']['.Deal']
    CLIENT.post('/book/entities', content=dump({'entity': 'LEI-A', 'name': 'Client A'}),
                headers=JSON)
    assert service.poll_key(service.BOOK, service.xva_path()) != key and paper() == started + 1
    declare(recorded, policy.FIXINGS_POLICY, {policy.FIXINGS_SECTION: {}})
    assert paper() == started + 1 and head(recorded) == started + 2
    CLIENT.post('/book/agreements', content=dump({
        'agreement': CLIENT_SET, 'entity': 'LEI-A', 'kind': 'ISDA 2002', 'terms': terms}),
        headers=JSON)
    assert paper() == head(recorded) == started + 3


def test_a_risk_poll_with_nothing_moved_is_a_lookup_and_a_frame_that_moves_no_number_runs_nothing(
        recorded, desk):
    """`/book/risk` is kept at two levels. The first is the poll's key, read before anything is
    compiled or hashed: a second poll is answered what the first kept under it. A frame landing
    moves that key, so the book is compiled again; an entity declared moves nothing a risk run
    reads, so the compile hashes as it did and the answer kept under that hash is served - the
    same object, not a second run - and kept under the new key.

    Killing mutations: the answer never kept under the poll's key, where the lookup finds none;
    the second level dropped, where the declaration is a second run and a new object.
    """
    def key():
        return service.poll_key(service.BOOK)

    risk = CLIENT.get('/book/risk').json()
    ran = service.BOOK.risk_cache.get(risk['etag'])
    assert service.BOOK.polls.get(('risk',) + key()) == risk == CLIENT.get('/book/risk').json()
    standing = key()
    CLIENT.post('/book/entities', content=dump({'entity': 'LEI-A', 'name': 'Client A'}),
                headers=JSON)
    assert key() != standing and service.BOOK.polls.get(('risk',) + key()) is None
    again = CLIENT.get('/book/risk').json()
    assert again == risk and service.BOOK.risk_cache.get(again['etag']) is ran
    assert service.BOOK.polls.get(('risk',) + key()) == risk


def test_a_position_is_read_where_it_sits_and_where_the_file_holds_it(recorded, desk):
    """`/book/positions` is the fold at the head joined to the FILE: every position standing under
    its agreement and portfolio, beside the nodes carrying its instrument under that agreement's
    set. One instrument under two agreements is two positions, each reading its own set's nodes; a
    close-out stands at nothing beside the nodes it sat at; a node the file lost answers no path.
    A portfolio whose top node is not the book, or with an empty segment, refuses by name with
    nothing appended.

    Killing mutations: the join by instrument alone hands the legacy set's position the declared
    agreement's nodes too; the standing filter read on quantity alone drops the closed position.
    """
    CLIENT.post('/book/entities', content=dump({'entity': 'LEI-A', 'name': 'Client A'}),
                headers=JSON)
    assert CLIENT.post('/book/agreements', content=dump({
        'agreement': CLIENT_SET, 'entity': 'LEI-A', 'kind': 'ISDA 2002',
        'terms': netting_set(CLIENT_SET, 'LEI-A')['Instrument']['.Deal']}),
        headers=JSON).status_code == 200
    document = json.loads(desk.read_text())
    document['Calc']['Deals']['Deals']['Children'] += [
        netting_set(CLIENT_SET, 'CPTY_A'), netting_set('CLIENT_B', 'CPTY_B')]
    desk.write_text(json.dumps(document, indent=2), newline='\n')

    def booked(execution, parent, **fields):
        return CLIENT.post('/book/deals', content=dump(dict(
            {'action': 'add', 'deal': dict(CASHFLOW, Reference='CF-HELD', Amount=250_000.0),
             'parent_reference': parent, 'quantity': 1.0, 'execution_reference': execution},
            **fields)), headers=JSON)

    head_before = head(recorded)
    for portfolio, posted in (('Rates/EM', 'spine-desk/Rates/EM'),
                              ('spine-desk//EM', 'spine-desk/EM')):
        refused = booked('EXEC-BAD', CLIENT_SET, agreement=CLIENT_SET, portfolio=portfolio)
        assert refused.status_code == 422 and repr(posted) in refused.json()['detail'], portfolio
    assert head(recorded) == head_before, 'a refused portfolio recorded something'

    for execution, parent, fields in (
            ('EXEC-1', CLIENT_SET, {'agreement': CLIENT_SET, 'portfolio': 'spine-desk/Rates/EM'}),
            ('EXEC-2', CLIENT_SET, {'agreement': CLIENT_SET}),
            ('EXEC-3', 'CLIENT_B', {}),
            ('EXEC-4', CLIENT_SET, {'agreement': CLIENT_SET, 'quantity': -1.0})):
        assert booked(execution, parent, **fields).json()['written'] is True, execution
    standing = CLIENT.get('/book/positions').json()['positions']
    assert sorted((row['agreement'], row['portfolio'], row['counterparty'], row['quantity'],
                   row['clips'], row['deal_paths'], row['reference'], row['object'])
                  for row in standing) == [
        (CLIENT_SET, 'spine-desk', 'LEI-A', 0.0, 2, ['1/0', '1/1', '1/2'], 'CF-HELD',
         'FixedCashflowDeal'),
        (CLIENT_SET, 'spine-desk/Rates/EM', 'LEI-A', 1.0, 1, ['1/0', '1/1', '1/2'], 'CF-HELD',
         'FixedCashflowDeal'),
        ('CLIENT_B', 'spine-desk', 'CPTY_B', 1.0, 1, ['2/0'], 'CF-HELD', 'FixedCashflowDeal')]

    document = json.loads(desk.read_text())
    document['Calc']['Deals']['Deals']['Children'][2]['Children'].clear()
    desk.write_text(json.dumps(document, indent=2), newline='\n')
    lost = {row['agreement']: row for row in CLIENT.get('/book/positions').json()['positions']}
    assert (lost['CLIENT_B']['deal_paths'], lost['CLIENT_B']['reference']) == ([], None)
    assert lost[CLIENT_SET]['deal_paths'] == ['1/0', '1/1', '1/2']


def test_a_position_netted_to_nothing_reads_closed_where_it_sat(recorded, desk):
    """A cashflow booked at one and closed by a fill of minus one at its own node stands at
    1 + (-1) = 0 beside that node, the file still holding the deal and both clips behind it, so a
    screen can say it closed. Terms restruck onto others are not a close: the new terms carry the
    one unit and the old terms stand no more. Two rows, whole.

    Killing mutations: the standing filter read on quantity alone, which drops the closed row; the
    fold read whole, which adds the restruck terms' row at nothing with no node.
    """
    document = json.loads(desk.read_text())
    document['Calc']['Deals']['Deals']['Children'].append(netting_set(CLIENT_SET, 'CPTY_A'))
    desk.write_text(json.dumps(document, indent=2), newline='\n')

    def booked(**body):
        answer = CLIENT.post('/book/deals', content=dump(dict(body, actor=ACTOR)),
                             headers=JSON).json()
        assert answer['written'] is True, answer
        return answer

    shut = booked(action='add', deal=dict(CASHFLOW, Reference='CF-SHUT'),
                  parent_reference=CLIENT_SET, quantity=1.0, execution_reference='EXEC-1')
    booked(action='fill', deal_path=shut['deal_path'], reference='CF-SHUT', quantity=-1.0,
           execution_reference='EXEC-2')
    moved = booked(action='add', deal=dict(CASHFLOW, Reference='CF-MOVED', Amount=250_000.0),
                   parent_reference=CLIENT_SET, quantity=1.0, execution_reference='EXEC-3')
    assert restruck(moved['deal_path'], Amount=300_000.0)['written'] is True

    rows = CLIENT.get('/book/positions').json()['positions']
    assert sorted(((row['reference'], row['quantity'], row['deal_paths']) for row in rows),
                  key=str) == [('CF-MOVED', 1.0, [moved['deal_path']]),
                               ('CF-SHUT', 0.0, [shut['deal_path']])]
    closed = next(row for row in rows if row['reference'] == 'CF-SHUT')
    assert closed['first_lsn'] < closed['last_lsn'] and len(closed['instrument']) == 64
    assert {key: value for key, value in closed.items()
            if key not in ('instrument', 'first_lsn', 'last_lsn', 'tickets')} == {
        'agreement': CLIENT_SET, 'portfolio': 'spine-desk', 'book': 'spine-desk',
        'counterparty': 'CPTY_A', 'quantity': 0.0, 'clips': 2, 'amended_to': None,
        'status': 'unticketed', 'pending': 0.0, 'expired': None, 'settles': None,
        'deal_paths': [shut['deal_path']], 'reference': 'CF-SHUT', 'object': 'FixedCashflowDeal'}
    assert [(clip['quantity'], clip['execution_reference']) for clip in closed['tickets']] == [
        (1.0, 'EXEC-1'), (-1.0, 'EXEC-2')]


def test_the_book_prices_every_position_at_its_net(recorded, desk):
    """THE COMPILE WRITES THE RECORD'S NET, and the desk's risk reads the compiled book. A deal
    booked whole prices as written; the same terms booked again at minus a half - a second node in
    the file - leave the first node held at half and the second ignored, so the client's set reads
    exactly half the cashflow; a second minus a half leaves nothing of it priced; and the paths a
    verb resolves against the file still name the nodes they named. A position held in a block
    stating none of its amounts - a hand edit, a legacy file - refuses the read by name rather than
    pricing a whole unit of it.

    A plan named off the book still runs once the record holds a position; another book's position
    in the same terms is that book's; and a book closed out reads as an empty one rather than
    differentiating nothing.

    Killing mutations: the second node of the same terms priced beside the first, which reads the
    whole cashflow at a net of a half; the risk read off the file rather than the compiled book,
    which prices both nodes as written; and the compile summing every book's positions.
    """
    document = json.loads(desk.read_text())
    document['Calc']['Deals']['Deals']['Children'].append(netting_set(CLIENT_SET, 'CPTY_A'))
    desk.write_text(json.dumps(document, indent=2), newline='\n')
    deal = dict(CASHFLOW, Reference='CF-HELD', Amount=250_000.0)

    def booked(execution, quantity):
        answer = CLIENT.post('/book/deals', content=dump({
            'action': 'add', 'deal': deal, 'parent_reference': CLIENT_SET, 'quantity': quantity,
            'execution_reference': execution}), headers=JSON).json()
        assert answer['written'] is True, answer

    def held():
        rows = CLIENT.get('/book/risk').json()['per_deal']
        return {row['reference']: row['value'] for row in rows}[CLIENT_SET]

    booked('EXEC-WHOLE', 1.0)
    whole = held()
    assert whole != 0.0
    prepared = CLIENT.post('/prepare', content=desk.read_text(), headers=JSON).json()
    assert CLIENT.post('/execute', json={'plan_id': prepared['plan_id']}).status_code == 200
    frame = {'Calc': {'Deals': {'Reference': service.book_name(json.loads(desk.read_text()))}}}
    assert spine.compiled_job(frame) is frame
    spine.book(service.instrument_of(deal_at(json.loads(desk.read_text()), '1/0')), 1.0, 'CPTY_A',
               CLIENT_SET, 'EXEC-ELSEWHERE', book_name='another-desk')
    assert held() == whole
    booked('EXEC-HALF', -0.5)
    assert held() == whole / 2
    compiled = service.priced(json.loads(desk.read_text()))
    assert (deal_at(compiled, '1/0')['Instrument']['.Deal']['Amount'],
            deal_at(compiled, '1/1').get('Ignore')) == (125_000.0, 'True')
    booked('EXEC-REST', -0.5)
    assert held() == 0.0
    closed = service.priced(json.loads(desk.read_text()))
    del closed['Calc']['Deals']['Deals']['Children'][0]
    assert service.consolidated_risk(closed)['per_deal'] == []

    document = json.loads(desk.read_text())
    bare = {key: value for key, value in dict(CASHFLOW, Reference='CF-BARE').items()
            if key != 'Amount'}
    splice_deal(document, json.loads(dump(bare)), CLIENT_SET)
    desk.write_text(json.dumps(document, indent=2), newline='\n')
    node = deal_at(document, '1/3')
    spine.book(service.instrument_of(node), 0.5, 'CPTY_A', CLIENT_SET, 'EXEC-BARE',
               book_name='spine-desk')
    refused = CLIENT.get('/book/risk')
    assert refused.status_code == 422 and "FixedCashflowDeal 'CF-BARE'" in refused.json()['detail']


def test_a_trade_a_set_holds_is_marked_inside_its_set(recorded, desk):
    """A TRADE UNDER A SET HAS A MARK OF ITS OWN, off the run that marks the set. Under CLIENT_A:
    250,000 rand booked whole and then less a half at a second node of the same terms, which the
    compile ignores; 100,000 rand; and a structure of 100,000 and 50,000 rand legs. Each is a
    two-year cashflow worth amount x 18.5 x exp(-0.04) dollars: CF-HELD 125,000 x 17.7746... =
    2,221,825.58, CF-B 1,777,460.46 and PAIR 2,666,190.69, summing to the set's 6,665,476.73,
    which `mtm` counts once beside CF1's 17,774,604.62. The legs are the structure's, not trades.

    Killing mutations: the ignored clip counted in the path map, which reads CF-HELD's path as
    None; and every row below the root read as a trade, which lists the structure's legs.
    """
    document = json.loads(desk.read_text())
    document['Calc']['Deals']['Deals']['Children'].append(netting_set(CLIENT_SET, 'CPTY_A'))
    desk.write_text(json.dumps(document, indent=2), newline='\n')
    legs = [{'Instrument': {'.Deal': dict(CASHFLOW, Reference=reference, Amount=amount)}}
            for reference, amount in (('LEG-A', 100_000.0), ('LEG-B', 50_000.0))]
    for deal, execution, quantity in (
            (dict(CASHFLOW, Reference='CF-HELD', Amount=250_000.0), 'EXEC-WHOLE', 1.0),
            (dict(CASHFLOW, Reference='CF-HELD', Amount=250_000.0), 'EXEC-HALF', -0.5),
            (dict(CASHFLOW, Reference='CF-B', Amount=100_000.0), 'EXEC-B', 1.0),
            ({'Object': 'StructuredDeal', 'Reference': 'PAIR', 'Currency': 'ZAR',
              'Children': legs}, 'EXEC-PAIR', 1.0)):
        assert CLIENT.post('/book/deals', content=dump({
            'action': 'add', 'deal': deal, 'parent_reference': CLIENT_SET, 'quantity': quantity,
            'execution_reference': execution}), headers=JSON).json()['written'] is True

    worth = SPOT * math.exp(-RATE * 2.0)
    risk = CLIENT.get('/book/risk').json()
    assert sorted(risk['in_sets'], key=lambda row: str(row['deal_path'])) == [
        {'reference': 'CF-HELD', 'deal_path': '1/0', 'value': pytest.approx(125_000.0 * worth,
                                                                              rel=1e-12)},
        {'reference': 'CF-B', 'deal_path': '1/2', 'value': pytest.approx(100_000.0 * worth,
                                                                           rel=1e-12)},
        {'reference': 'PAIR', 'deal_path': '1/3', 'value': pytest.approx(150_000.0 * worth,
                                                                           rel=1e-12)}]
    assert risk['per_deal'] == [
        {'reference': CLIENT_SET, 'deal_path': '1', 'value': pytest.approx(375_000.0 * worth,
                                                                             rel=1e-12)},
        {'reference': 'CF1', 'deal_path': '0', 'value': pytest.approx(AMOUNT * worth, rel=1e-12)}]
    assert risk['mtm'] == pytest.approx(1_375_000.0 * worth, rel=1e-12)


def test_a_structure_is_held_whole(recorded, desk):
    """A STRUCTURE BOOKED WHOLE IS ONE INSTRUMENT, its legs inside its terms. A leg booked into it
    refuses by name, since it would move the instrument the position is in; a leg amended amends
    the structure, the position carried onto the terms the edit leaves - the clip left on the old
    terms naming nothing, so the restruck leg's payment carries its own key and settling it makes
    the day's close legal with that clip still in the file; and a position the record holds in a
    leg of a structure it also holds refuses the read, where pricing the leg as the structure's
    would read neither.

    Killing mutations: the refusal dropped, which books the leg into the held structure; the
    amendment filed against the leg rather than the structure, which prices the edited structure
    as written beside its old terms at half; and the stranded legs priced as their structure's.
    """
    document = json.loads(desk.read_text())
    document['Calc']['Deals']['Deals']['Children'].append(netting_set(CLIENT_SET, 'CPTY_A'))
    splice_deal(document, {'Object': 'StructuredDeal', 'Reference': 'SOLO', 'Currency': 'ZAR'},
                CLIENT_SET)
    desk.write_text(json.dumps(document, indent=2), newline='\n')
    legs = [dict(CASHFLOW, Reference='LEG-A', Amount=100_000.0),
            dict(CASHFLOW, Reference='LEG-B', Amount=50_000.0)]

    def structure(reference, deals):
        return {'Object': 'StructuredDeal', 'Reference': reference, 'Currency': 'ZAR',
                'Children': [{'Instrument': {'.Deal': deal}} for deal in deals]}

    def booked(deal, parent, execution, quantity):
        return CLIENT.post('/book/deals', content=dump({
            'action': 'add', 'deal': deal, 'parent_reference': parent, 'quantity': quantity,
            'execution_reference': execution}), headers=JSON)

    assert booked(structure('PAIR', legs), CLIENT_SET, 'EXEC-PAIR', 1.0).json()['written']
    refused = booked(dict(CASHFLOW, Reference='LEG-C'), 'PAIR', 'EXEC-LEG', 1.0)
    assert refused.status_code == 422 and "StructuredDeal 'PAIR'" in refused.json()['detail']
    assert booked(structure('PAIR', legs), CLIENT_SET, 'EXEC-PAIR-HALF', -0.5).json()['written']
    amended = CLIENT.post('/book/deals', content=dump({
        'action': 'amend', 'deal_path': '1/1/0', 'fields': {'Amount': 300_000.0}}), headers=JSON)
    assert amended.json()['written'] is True
    compiled = service.priced(json.loads(desk.read_text()))
    assert (deal_at(compiled, '1/1/0')['Instrument']['.Deal']['Amount'],
            deal_at(compiled, '1/2').get('Ignore')) == (150_000.0, 'True')

    def settled(key):
        assert CLIENT.post('/book/transition', content=dump(
            {'subject': key, 'status': 'settled', 'actor': ACTOR}), headers=JSON).status_code == 200

    day, written = CASHFLOW['Payment_Date'].strftime('%Y-%m-%d'), json.loads(desk.read_text())
    restruck, *others = (derivus.content_hash(service.instrument_of(deal_at(written, path)))
                         for path in ('1/1/0', '1/1/1', '0'))
    for row in CLIENT.get('/book/diary').json()['rows']:
        if row['kind'] == 'payment' and row['instrument'] in others:
            settled(row['key'])
    (waiting,) = CLIENT.get('/book/close/check', params={'date': day}).json()['outstanding']
    assert (waiting['instrument'], waiting['key']) == (restruck, spine.cashflow_key(
        restruck, waiting['leg'], 'payment', day)), 'the restruck leg keys nothing'
    settled(waiting['key'])
    assert CLIENT.get('/book/close/check', params={'date': day}).json()['legal'] is True
    assert deal_at(json.loads(desk.read_text()), '1/2/0')['Instrument']['.Deal']['Amount'] == \
        100_000.0, 'the clip on the old terms is not in the file'

    for execution, leg in zip(('EXEC-X', 'EXEC-Y'), legs):
        assert booked(dict(leg, Reference=leg['Reference'].replace('LEG', 'SOLO')), 'SOLO',
                      execution, 1.0).json()['written']
    solo = [dict(leg, Reference=leg['Reference'].replace('LEG', 'SOLO')) for leg in legs]
    assert booked(structure('SOLO', solo), CLIENT_SET, 'EXEC-SOLO', -0.5).json()['written']
    stranded = CLIENT.get('/book/risk')
    assert stranded.status_code == 422
    assert "StructuredDeal 'SOLO'" in stranded.json()['detail'], stranded.json()


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
    vector behind them canonicalising back to the hash it is filed under, the age of the oldest
    stamped pillar, who quoted it, where it books and the pin of the terms it books.

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
    vector = service.pinned_values(filed)
    assert vector == spine.values_of(context) and 'values' not in pinned, (
        'the vector is on the pending file, or is not the book\'s')
    assert derivus.content_hash(json.loads(vector)) == pinned['values_hash'], \
        'the vector on file is not the one the hash beside it names'
    assert (tmp_path / 'tmp' / (pinned['values_hash'] + service.VALUES_SUFFIX)).read_bytes() == vector
    assert filed['terms'] == service.quoted_terms(filed['deal'], CLIENT_SET, None)
    assert filed['portfolio'] == 'spine-desk', 'a quote naming no node books into its book'
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

    Killing mutation: the pins taken after the live spot lands, which pins a market only the quote
    ever saw.
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

    filed = pending_of(tmp_path, out['Stats']['Quote'])
    pinned = filed['pinned']
    assert pinned['values_hash'] == book_context.values_hash(), \
        'the quote pinned its own spot-patched market rather than the book it would land against'
    assert pinned['plan_hash'] == book_context.plan_hash()
    assert service.pinned_values(filed) == spine.values_of(book_context)

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


def test_a_pending_quote_keeps_its_vector_apart_and_one_filed_with_it_still_books(
        recorded, quoting):
    """A pending quote's file holds the two hashes it pinned and not the values vector, which is
    kept once under its hash beside the pending files, so two quotes struck on one market share one
    vector file and the list of quotes reads no vector. A pending file written with the vector on
    it - the first quote restated that way, its vector file gone with the second's - is accepted
    all the same, and the record files that vector under the hash the quote pinned.

    Killing mutations: the vector written on the pending file again, where the file carries
    `values`; a pending file's own vector never read, where the acceptance finds no vector file.
    """
    quotes = [quote_of('ZeroCostCollar', dict(COLLAR, notional=AMOUNT * size),
                       netting_set=CLIENT_SET) for size in (1, 2)]
    files = sorted(os.listdir(quoting.parent / 'tmp'))
    first = pending_of(quoting.parent, quotes[0])
    kept = quoting.parent / 'tmp' / (first['pinned']['values_hash'] + service.VALUES_SUFFIX)
    assert [name for name in files if name.endswith(service.VALUES_SUFFIX)] == [kept.name]
    assert sorted(first['pinned']) == ['pillar_age', 'plan_hash', 'values_hash']
    assert len(CLIENT.get('/book/quotes').json()['quotes']) == 2

    vector = kept.read_bytes()
    first['pinned']['values'] = json.loads(vector)
    (quoting.parent / 'tmp' / (quotes[0]['quote_id'] + '.json')).write_text(json.dumps(first))
    kept.unlink()
    booked = accept(quotes[0]).json()
    assert booked['written'] is True, booked
    _, _, body = facts(recorded, 'quote_filed')[0]
    assert body['values_hash'] == first['pinned']['values_hash']
    assert blob(recorded, body['values_hash']) == vector


def test_the_acceptance_files_the_quote_then_the_fill_and_a_retry_coalesces(recorded, quoting):
    """THE ACCEPTANCE IS WHERE THE RECORD MOVES. With no tiers policy the flow is what it was: the
    `quote_filed` under the ACCEPTOR's seat, then the `fill` citing the quote id as its execution
    reference, at consecutive LSNs, then the file. The quote and the fill carry ONE ticket - the
    plan the booking left the book at, read off the file it wrote, and the fill as it is filed.

    AND AN ACCEPTANCE THAT BOOKED IS TOLD SO BY NAME. The booking was itself a move of the book, so
    the plan check would send a salesperson to re-quote a trade the desk has already done; the
    pending file says where it landed and the second acceptance answers that instead. The retry that
    coalesces is the one a tier held back, which is the desk tier's own gate.

    Killing mutation: the fill appended before the quote, which files a trade against a quote the
    record does not yet hold.
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
    assert booked['accepted'] == {'lsn': filed[0], 'ticket': booked['ticket']}
    assert pending_of(quoting.parent, quote)['accepted'] == booked['accepted']

    _, _, body = facts(recorded, 'quote_filed')[0]
    assert body['quote_id'] == quote['quote_id'] and body['structure'] == 'ZeroCostCollar'
    assert body['ticket'] == booked['ticket'] and body['plan_hash'] == struck['plan_hash']
    assert body['request'].startswith('the client wants')
    assert len(body['solved']) == 1 and body['edge'] == pytest.approx(quote['edge'])
    assert blob(recorded, body['values_hash']) == service.pinned_values(
        pending_of(quoting.parent, quote))

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
    assert fill_body['ticket'] == spine.ticket(service.load(landed).plan_hash(), {
        field: value for field, value in fill_body.items() if field != 'ticket'}), \
        'the ticket is not the plan it left and the fill as filed'
    assert fill_body['ticket'] == body['ticket'], 'the fill does not carry what was signed'

    retried = accept(quote).json()
    assert retried['written'] is False
    assert retried['booked'] == {'lsn': fill[0], 'deal_path': booked['deal_path']}
    assert 'already on the book' in retried['note']
    assert 'the book moved under it' not in json.dumps(retried), \
        'a quote that booked was told to re-quote'
    assert typed(recorded, 'quote_filed', 'fill') == [filed, fill], 'a retry filed a second fact'
    assert verify_home(recorded)['events'] == head(recorded)


def test_an_accepted_premium_settles_on_the_pair_s_spot_date(recorded, tmp_path):
    """THE ACCEPTANCE DATES WHAT CHANGES HANDS. A straddle quoted at its strike on the rand desk's
    book, each rate on its own calendar, charges the client a premium, and the acceptance books
    the mirror at it: the fill states the pair's spot date for the trade day - Wednesday 18
    November for Monday 16 November, the day the quote's own legs count delivery from - and the
    diary announces the premium due that day, received, in the book's currency.

    Killing mutation: the value date the trade day itself, the pair's spot convention unread.
    """
    from test_structures import DESK_DAY, desk_book

    document = desk_book(True)
    document['Calc']['Deals']['Deals']['Children'].append(
        json.loads(dump(netting_set(CLIENT_SET, 'CPTY_A'))))
    path = tmp_path / 'book.json'
    path.write_text(json.dumps(document, indent=2), newline='\n')
    service.BOOK = service.Book(str(path))
    try:
        quote = quote_of('Straddle', {'pair': 'USDZAR', 'expiry': '6M', 'notional': AMOUNT,
                                      'notional_currency': 'USD', 'strike': 16.8},
                         netting_set=CLIENT_SET)
        assert quote['charged_on'] == 'premium' and quote['net'] > 0.0
        booked = accept(quote).json()
        assert booked['written'] is True, booked
        (_, _, fill), = facts(recorded, 'fill')
        assert (fill['value_date'], fill['price']) == ('2026-11-18', -quote['net'])
        assert DESK_DAY == '2026-11-16'
        (row,) = [row for row in CLIENT.get('/book/diary').json()['rows']
                  if row['kind'] == spine.PREMIUM]
        assert (row['leg'], row['due_date'], row['currency']) == (
            quote['quote_id'], '2026-11-18', 'USD')
        assert row['amount'] == pytest.approx(quote['net'], rel=1e-15)
    finally:
        service.BOOK = None


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
    """Whether the record and the file agree on both of `/book/reconcile`'s position lists."""
    answer = CLIENT.get('/book/reconcile').json()
    return answer['in_record_not_in_file'] == answer['in_file_not_in_record'] == []


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
    The order NO lock can order is the gate below, and it is what holds the acceptance on `transact`:
    in both orders here the acceptance's own pass is never raced, so `mutate` lands it alike.

    Killing mutation: the pins checked against the read above the closure (`check_pins` handed
    `document`, not `book`), which never sees the booking that beat the acceptance and refuses in
    the file's words rather than naming the book that moved.
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


def test_a_quote_books_once_on_a_desk_that_records_nothing(unrecorded, quoting):
    """A QUOTE BOOKS ONCE ON EITHER DESK. With no record the acceptance still writes where the trade
    landed onto the pending file, `lsn` null, and reads it again under the book lock: an acceptance
    landing between this one's first read and its act - a double-click - leaves this one answered
    as booked, and so is a third sent after both. One collar in the file.

    Killing mutations: the no-record acceptance writing no `booked`, which books the collar twice;
    the pending file not read again under the lock, which books the raced one twice.
    """
    quote = quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET)
    service.BOOK = RacingBook(str(quoting))
    raced = []
    service.BOOK.arm(before=lambda: raced.append(accept(quote).json()))
    outer, again = accept(quote).json(), accept(quote).json()

    assert raced and raced[0]['written'] is True, raced
    landed = {'lsn': None, 'deal_path': raced[0]['deal_path']}
    assert (outer['written'], outer.get('booked')) == (False, landed), outer
    assert (again['written'], again.get('booked')) == (False, landed), again
    assert 'already on the book' in again['note']
    assert [node['Instrument']['.Deal']['Reference'] for _, node in walk_job_deals(
        json.loads(quoting.read_text()))].count(quote['deal']['Reference']) == 1, 'booked twice'


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

    Killing mutation: the plan comparison dropped from the firmness verdict, which books a charge
    solved against another portfolio.
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


def test_a_booking_elsewhere_inside_the_firm_window_is_read_as_a_market_move(recorded, quoting):
    """A BOOK MOVE IS A MARKET MOVE INSIDE THE DESK'S FIRM WINDOW. Between one seat's quote and its
    acceptance a second seat books a cashflow on another client's netting set. With a `Quote
    Policy` declared the quote books as priced: `firmness.plan` reads the plan it was priced on, the
    one it lands in and that they differ, `quote_filed` keeps the first and the fill's ticket is
    over the plan the book lands at. Past the window the desk's own refusal stands, and with no
    window declared the moved book refuses as before - neither appending.

    Killing mutations: `held` unread in `firmness.assess`, which refuses inside the window; `held`
    true with no window declared, which books there.
    """
    document = json.loads(quoting.read_text())
    document['Calc']['Deals']['Deals']['Children'].append(
        json.loads(dump(netting_set('CLIENT_B', 'CPTY_B'))))
    quoting.write_text(json.dumps(document, indent=2), newline='\n')

    def raced(window, seat_two):
        """Seat one quotes under `window`, seat two books on CLIENT_B, and seat one accepts:
        `(the plan priced on, the plan standing at the acceptance, the head then, the answer)`."""
        document = json.loads(quoting.read_text())
        document['Calc'].pop(structures.Structure.POLICY.name, None)
        document['Calc'].update({structures.Structure.POLICY.name: window} if window else {})
        quoting.write_text(json.dumps(document, indent=2), newline='\n')
        service.BOOK = service.Book(str(quoting))
        quote = quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET, actor=ACTOR)
        other = CLIENT.post('/book/deals', content=dump({
            'action': 'add', 'deal': dict(CASHFLOW, Reference=seat_two, Amount=5_000.0),
            'parent_reference': 'CLIENT_B', 'quantity': 1.0, 'execution_reference': seat_two,
            'actor': DESK_TWO}), headers=JSON).json()
        assert other['written'] is True, other
        return (pending_of(quoting.parent, quote)['pinned'],
                service.load(service.BOOK.read()[0]).plan_hash(), head(recorded),
                accept(quote, actor=ACTOR))

    priced, standing, _, answer = raced({'firm_seconds': 600}, 'SEAT-TWO-IN')
    booked = answer.json()
    assert booked.get('written') is True, booked
    assert booked['firmness']['plan'] == {'pinned': priced['plan_hash'], 'current': standing,
                                          'moved': True}
    (_, _, filed), = facts(recorded, 'quote_filed')
    _, _, fill = facts(recorded, 'fill')[-1]
    signed = {field: value for field, value in fill.items() if field != 'ticket'}
    assert filed['plan_hash'] == priced['plan_hash'], 'the record lost the plan it was priced on'
    lands = service.load(json.loads(quoting.read_text())).plan_hash()
    assert fill['ticket'] == filed['ticket'] == spine.ticket(lands, signed), \
        'the ticket is not over the plan the quote lands in'

    for window, said in (({'firm_seconds': 0}, 'holds a quote firm for 0s'),
                         (None, 'the book moved under it')):
        _, _, before, answer = raced(window, 'SEAT-TWO-' + said.split()[-1].upper())
        assert answer.status_code == 422 and said in answer.json()['detail'], answer.json()
        assert head(recorded) == before, 'a refused acceptance appended'


def test_an_edited_quote_refuses_whether_or_not_the_book_moved(recorded, quoting):
    """A QUOTE BOOKS ITS OWN TERMS. The pending file pins the mirror, the netting set and the model
    pin it books (`service.quoted_terms`), and an acceptance books nothing whose file no longer
    hashes to them. Inside the desk's firm window seat two books on another client's set, so the
    book has moved and the plan is only a reading - and a strike edited on disk still refuses by
    name; the same edit on a book that did not move refuses too, and so does the file pointed at
    another client's netting set or handed a model pin the quote never carried. None moves the
    record's head or the book file.

    Killing mutations: the pin not compared, which books the edited strike both times; the pin
    leaving out the netting set, which books the collar under CLIENT_B; the pin leaving out the
    models, which merges a pin nobody priced.
    """
    document = json.loads(quoting.read_text())
    document['Calc']['Deals']['Deals']['Children'].append(
        json.loads(dump(netting_set('CLIENT_B', 'CPTY_B'))))
    document['Calc'][structures.Structure.POLICY.name] = {'firm_seconds': 600}
    quoting.write_text(json.dumps(document, indent=2), newline='\n')
    service.BOOK = service.Book(str(quoting))

    def strike(filed):
        filed['deal']['Children'][0]['Instrument']['.Deal']['Strike_Price'] *= 1.01

    def other_set(filed):
        filed['quote']['netting_set'] = 'CLIENT_B'

    def model_pin(filed):
        filed['quote']['valuation_configuration'] = {'FxRate.ZAR': 'GBMAssetPriceModel'}

    for moved, edit in ((True, strike), (False, strike), (False, other_set), (False, model_pin)):
        quote = quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET, actor=ACTOR)
        if moved:
            assert CLIENT.post('/book/deals', content=dump({
                'action': 'add', 'deal': dict(CASHFLOW, Reference='SEAT-TWO', Amount=5_000.0),
                'parent_reference': 'CLIENT_B', 'quantity': 1.0,
                'execution_reference': 'SEAT-TWO', 'actor': DESK_TWO}),
                headers=JSON).json()['written'] is True
        path = quoting.parent / 'tmp' / (quote['quote_id'] + '.json')
        filed = json.loads(path.read_text())
        edit(filed)
        path.write_text(json.dumps(filed, indent=2), newline='\n')
        before, book = head(recorded), quoting.read_bytes()

        refused = accept(quote, actor=ACTOR)
        assert refused.status_code == 422, (moved, edit.__name__, refused.json())
        assert 'no longer says what was quoted' in refused.json()['detail'], (moved, edit.__name__)
        assert (head(recorded), quoting.read_bytes()) == (before, book), (moved, edit.__name__)


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

    Killing mutation: the pillar window unread, which books a quote struck on a board two years old.
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
    the wording, which is the desk's, and on a head that did not move.

    Killing mutation: the desk's firm window unread, which hands a stale quote to the record's
    checks.
    """
    document = json.loads(quoting.read_text())
    document['Calc'][structures.Structure.POLICY.name] = {'firm_seconds': 0}
    quoting.write_text(json.dumps(document, indent=2), newline='\n')

    quote = quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET)
    before = head(recorded)
    refused = accept(quote)

    assert refused.status_code == 422
    assert structures.Structure.POLICY.name in refused.json()['detail']
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
    """`advancing`'s three laws, read off the fold itself rather than through a booking, on the
    pairs the served book keeps.

    It ADVANCES, so a verdict filed after the pair was taken is in the next answer. It is BOUNDED at
    the position it answers, so a frame appended between two asks is not applied twice - a
    duplicated verdict list that `standing_approval`'s `max` would hide. A pair AHEAD of the
    position asked for refolds from genesis rather than refusing, since a fold walks forward, and
    stays where it stands, so the next ask at the head pays for nothing twice. And a history that
    is not the one the pair was taken on drops it: the key is the GENESIS event hash, so a home
    re-minted where the last one stood is a different record.

    Killing mutations: the fold unbounded while the pair is stamped at the position asked for; a
    pair ahead of that position used as a seed; the pair moved back to a position asked behind it;
    the genesis check dropped, which answers another history with this one's verdicts. The key
    taken off the path is the re-minted home's gate.
    """
    projections = spine.package().projections
    decisions = projections.PROJECTORS['decisions']
    plan, other = 'a' * 64, 'b' * 64

    def verdicts_at(lsn=None):
        return spine.folded(
            lambda log: spine.advancing(log, decisions, lsn, service.folds())['plans'])

    spine.approve(plan, actor_name=ACTOR, book_name='spine-desk')
    first = verdicts_at()
    assert [row['lsn'] for row in first[plan]] == [head(recorded)]

    signed = spine.approve(other, actor_name=DESK_TWO, book_name='spine-desk')
    assert [row['lsn'] for row in verdicts_at()[other]] == [signed['lsn']], 'the pair never advanced'
    assert [row['lsn'] for row in verdicts_at()[plan]] == [first[plan][0]['lsn']], \
        'a frame was applied twice'

    # a position BEHIND the pair is a fold from genesis, and the pair is left where it was
    assert other not in verdicts_at(signed['lsn'] - 1)
    assert service.folds()[decisions.name]['lsn'] == signed['lsn'], 'the held pair moved back'
    assert [row['lsn'] for row in verdicts_at()[other]] == [signed['lsn']]

    # another history answers its own, and is not this one's - filed to the pair's own position,
    # so it is the genesis key that drops the pair rather than the pair standing ahead of it
    elsewhere = quoting.parent / 'second-spine'
    init_home(elsewhere, ACTOR)
    for seconds in (1.0, 2.0):
        declare(elsewhere, policy.FIRMNESS_POLICY, {'pillar_seconds': seconds})
    log = opened(elsewhere)
    try:
        assert log.head()[0] == signed['lsn'], 'not the pair\'s position - this proved nothing'
        assert spine.advancing(log, decisions, folds=service.folds())['plans'] == {}
    finally:
        log.close()
    assert [row['lsn'] for row in verdicts_at()[plan]] == [first[plan][0]['lsn']]


def test_a_home_re_minted_where_the_last_one_stood_is_a_different_record(recorded, quoting,
                                                                        tmp_path):
    """THE PAIR IS KEYED ON THE HISTORY, NOT THE PATH. A home deleted and minted again at the same
    place, under the book that kept the pairs, holds none of the verdicts the last one did, and a
    fold that answered the old ones would route a booking on a record that no longer exists -
    silently, which is the unsafe direction.

    Killing mutation: the key taken off `log.home`, which answers the predecessor's history.
    """
    decisions = spine.package().projections.PROJECTORS['decisions']

    def plans():
        return spine.folded(
            lambda log: spine.advancing(log, decisions, folds=service.folds())['plans'])

    spine.approve('a' * 64, actor_name=ACTOR, book_name='spine-desk')
    assert plans().get('a' * 64)

    shutil.rmtree(str(recorded))
    init_home(recorded, ACTOR)
    spine.approve('c' * 64, actor_name=ACTOR, book_name='spine-desk')

    standing = plans()
    assert 'a' * 64 not in standing, 'the pair answered a history this home no longer has'
    assert 'c' * 64 in standing


def test_a_home_copied_from_another_is_its_own_record(minted, tmp_path, monkeypatch):
    """THE PAIRS ARE THE SERVED BOOK'S, NOT THE PROCESS'S. Two homes copied from one minted record
    share its genesis and part at the first fact either files; each is served with the pairs its
    own book keeps, so a fold on one never answers from the other's - which is what lets a gate copy
    a minted home rather than mint its own.

    Killing mutation: the pairs kept at module level under the genesis alone, which answers the
    second copy with the first copy's verdict.
    """
    decisions = spine.package().projections.PROJECTORS['decisions']
    monkeypatch.setenv('DV_SPINE_ACTOR', ACTOR)
    for name, plan in (('first', 'a' * 64), ('second', 'b' * 64)):
        shutil.copytree(str(minted), str(tmp_path / name))
        monkeypatch.setenv('DV_SPINE_HOME', str(tmp_path / name))
        spine.approve(plan, actor_name=ACTOR, book_name='spine-desk')
        own = {}
        plans = spine.folded(lambda log: spine.advancing(log, decisions, folds=own))['plans']
        assert set(plans) == {plan}, 'the {} copy answered another copy\'s history'.format(name)


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
    bare = dict(document, Calc=dict(document['Calc'], MergeMarketData={
        'MarketDataFile': '', 'ExplicitMarketData': {}}))
    assert service.board_age(bare, '2024-06-28T17:00:00.000+00:00') is None


# --------------------------------------------------------------------------------------------
# The tier step, inside the acceptance closure.

DESK_TWO = 'subject-desk-two'

#: A cap the collar's million dollars sits comfortably under, and one it does not.
BIG, SMALL = 5_000_000.0, 100_000.0


def tiers(*rows, **sections):
    """A tiers document out of the rows a gate cares about."""
    return dict(tiers=list(rows), **sections)


def entitle(home, grants, **sections):
    """Declare a capabilities document, which is what turns enforcement ON: until one is in force
    the home is the single-user instrument it was, and after it every append needs a grant.
    `sections` are its others as stated - `verbs`, the acts it moves."""
    log = opened(home)
    try:
        blob = log.store.put(canonical_document(dict(
            {'grants': [{'subject': s, 'verb': v, 'book': b} for s, v, b in grants], 'read': []},
            **sections)))
        return log.append('policy_declared', {'policy': CAPABILITIES_POLICY, 'blob': blob},
                          actor=ACTOR, blob_refs=(blob,))
    finally:
        log.close()


def test_a_home_written_before_the_seats_retired_still_books_exports_marks_and_verifies(recorded,
                                                                                         quoting):
    """THE RECORD'S OWN VERSION RULE, end to end. A home carries the two documents as an earlier
    grammar spelled them, put on the record as data: a capabilities document granting `draft`
    beside every verb it then had, and a tiers policy whose automatic tier names the seat it signed
    under. Read tolerantly - the retired grant dropped, the seated tier read as the automatic one
    it was - the desk accepts a quote through the hub's own approval, strikes its settlement file
    and marks its book at the designated close, reads its P&L, and the oracle holds over it.

    Killing mutations: a stored document read in today's grammar - the capabilities one folding to
    UNREADABLE, which refuses the quote, and the tiers one refused at every read, which refuses the
    acceptance, the export and the marks.
    """
    from derivus_spine import oracle
    from gates.spine_game.red import forged

    log = opened(recorded)
    try:
        drafted = log.store.put(canonical_bytes({'grants': [
            {'subject': ACTOR, 'verb': verb, 'book': '*'} for verb in (
                'admin', 'book', 'draft', 'validate', 'approve', 'mark', 'document',
                'settle')], 'read': []}))
        log.accept(forged(log, 'policy_declared', {'policy': CAPABILITIES_POLICY,
                                                   'blob': drafted}, ACTOR))
        seated = log.store.put(canonical_bytes({
            'designations': {'pnl': 'official', 'settlement_export': 'official'},
            'tiers': [{'name': 'auto', 'seat': ACTOR}]}))
    finally:
        log.close()
    log = opened(recorded)
    try:
        log.append('policy_declared', {'policy': policy.TIERS_POLICY, 'blob': seated},
                   actor=ACTOR, blob_refs=(seated,))
    finally:
        log.close()

    quote = quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET, actor=ACTOR)
    booked = accept(quote, actor=ACTOR).json()
    assert booked['written'] is True and booked['tier']['four_eyes'] is False, booked
    for path, body in (('/book/close', {'actor': ACTOR}), ('/book/marks', {'actor': ACTOR}),
                       ('/book/settlements', {'due_before': '2024-07-31', 'actor': ACTOR})):
        answer = CLIENT.post(path, content=dump(body), headers=JSON)
        assert answer.status_code == 200, (path, answer.text)
        if path == '/book/marks':
            assert drained(answer.json())['status'] == 'done'
    assert CLIENT.get('/book/pnl', params={'actor': ACTOR}).status_code == 200
    assert oracle.failed(oracle.report(recorded)) == []


def test_the_seam_refuses_the_writers_name_whoever_names_it(recorded, monkeypatch):
    """THE RESERVED NAME IS NOBODY'S, whoever says it: an act naming `writer`, and one naming
    nobody where `DV_SPINE_ACTOR` says `writer` and no document is in force, are refused at the
    seam in its own sentence, and nothing is written.

    Killing mutation: the seam passing the name through, which leaves the refusal to the writer and
    a poll path reading the deployment's seat off the environment unguarded.
    """
    before = head(recorded)
    for named, deployment in (('writer', ACTOR), (None, 'writer')):
        monkeypatch.setenv('DV_SPINE_ACTOR', deployment)
        with pytest.raises(spine.SpineRefused) as refused:
            spine.book(CASHFLOW, 1.0, 'CPTY_A', CLIENT_SET, 'EXEC-VOICE', actor_name=named,
                       book_name='spine-desk')
        assert 'name the seat that is acting' in str(refused.value), (named, deployment)
    assert head(recorded) == before


def test_an_automatic_tier_is_the_hubs_own_approval_between_the_quote_and_the_fill(recorded,
                                                                                    quoting):
    """A tier is AUTOMATIC unless it declares `four_eyes`, and its approval is the hub's own act -
    filed in the writer's voice, the workflow an admin declared deciding it rather than a seat - so
    a document scoping NOBODY to approve stops none of it. The order is the whole point: the
    acceptance, the approval, the fill, three facts at consecutive LSNs, so an auditor reading the
    log in order sees a trade authorised before it was booked; the fill carries the ticket that
    approval signed and the tier it fell in.

    Killing mutations: the approval appended after the fill, which records every automatic booking
    as signed off afterwards; the approval filed under the acceptor, which a document scoping it for
    nothing but booking refuses, holding back every trade the workflow was declared to pass; and the
    route's tier left off the fill, which leaves the record unable to say which tickets the hub
    signs.
    """
    declare(recorded, policy.TIERS_POLICY, tiers(
        {'name': 'auto', 'max_notional': {'amount': BIG, 'currency': 'USD'}},
        {'name': 'desk', 'four_eyes': True}))
    # `validate` beside the rest: quoting is a curiosity job the queue asks that scope for, and the
    # verbs imply nothing about each other here or at the writer - nobody holds `approve`
    entitle(recorded, [(ACTOR, verb, '*') for verb in ('validate', 'book', 'mark', 'admin')])
    quote = quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET, actor=ACTOR)

    booked = accept(quote, actor=ACTOR).json()
    assert booked['written'] is True, booked
    assert booked['tier'] == {'name': 'auto', 'four_eyes': False,
                              'approval_lsn': typed(recorded, 'approval')[0][0], 'fell_past': []}

    steps = typed(recorded, 'quote_filed', 'approval', 'fill')
    assert [event_type for _, event_type in steps] == ['quote_filed', 'approval', 'fill']
    assert [lsn for lsn, _ in steps] == list(range(steps[0][0], steps[0][0] + 3))
    _, _, signed = facts(recorded, 'approval')[0]
    assert signed == {'plan_hash': booked['accepted']['ticket']}
    _, _, filled = facts(recorded, 'fill')[0]
    assert (filled['ticket'], filled['tier']) == (booked['accepted']['ticket'], 'auto'), filled
    assert facts(recorded, 'capability_denied') == []
    log = opened(recorded)
    try:
        assert [frame['actor'] for frame in log.frames()
                if frame['event_type'] == 'approval'] == ['writer']
    finally:
        log.close()
    assert verify_home(recorded)['events'] == head(recorded)


def test_a_four_eyes_acceptance_books_pending_and_reads_the_verdict_that_stands(recorded,
                                                                                 quoting):
    """UNDER FOUR EYES THE TRADE LANDS AND WAITS ON ITS STATUS. The acceptance files the quote and
    books the fill carrying its ticket, reading PENDING with `waits_on` saying why - a pending
    trade prices, settles and exports all the same - and the verdicts are filed over that ticket
    where it books: the ACCEPTOR's own approval clears nothing, a rejection filed after it is what
    the record says LAST, and another seat's approval after that is what stands. A second
    acceptance is told the quote is already booked: ONE `quote_filed`, ONE fill.

    Killing mutations: the fill held back under four eyes, which leaves the acceptance booking
    nothing; the status read with the booker's own verdicts, which clears a ticket its own booker
    signed; and the route's tier left off the fill, which leaves a forged approval of it standing
    on the record.
    """
    declare(recorded, policy.TIERS_POLICY, tiers({'name': 'desk', 'four_eyes': True}))
    quote = quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET)
    booked = accept(quote).json()
    ticket = booked['accepted']['ticket']
    assert booked['written'] is True and booked['ticket'] == ticket, booked
    assert booked['tier'] == {'name': 'desk', 'four_eyes': True, 'approval_lsn': None,
                              'fell_past': []}
    assert 'no verdict is filed' in booked['waits_on']
    fill = facts(recorded, 'fill')[0][2]
    assert (fill['ticket'], fill['tier']) == (ticket, 'desk'), fill

    def standing():
        row = next(row for row in CLIENT.get('/book/positions').json()['positions']
                   if row['instrument'] == fill['instrument'])
        return row['status'], row['pending']

    assert standing() == ('pending', 1.0)
    for actor, verb, status in ((ACTOR, 'approve', 'pending'), (DESK_TWO, 'reject', 'rejected'),
                                (DESK_TWO, 'approve', 'approved')):
        decided = CLIENT.post('/book/' + verb, json={
            'quote_id': quote['quote_id'], 'actor': actor, 'reason': 'the client is over limit'})
        assert decided.json()['ticket'] == ticket and decided.json()['status'] == status, verb
        assert standing()[0] == status, (actor, verb)
    assert standing()[1] == 0.0
    retried = accept(quote).json()
    assert retried['written'] is False and 'already on the book' in retried['note']
    assert len(facts(recorded, 'fill')) == len(facts(recorded, 'quote_filed')) == 1
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
        {'name': 'auto', 'max_notional': {'amount': BIG, 'currency': 'EUR'}}))

    quote = quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET)
    pending = pending_of(quoting.parent, quote)
    terms = terms_of(json.loads(quoting.read_text()), pending)
    assert 'EUR' not in terms['notional_in'] and set(terms['notional_in']) == {'USD', 'ZAR'}

    # THE OTHER DIRECTION, with the zero on top: the NOTIONAL's own currency unvalued. Its amount
    # still stands - an amount of a currency needs no rate - and every cross off it reads zero,
    # which is not a notional a cap can be compared against and so is not stated
    unvalued = json.loads(quoting.read_text())
    unvalued['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors'][
        'FxRate.USD']['Spot'] = 0.0
    assert terms_of(unvalued, pending)['notional_in'] == {'USD': AMOUNT}
    before = quoting.read_bytes()

    answer = accept(quote).json()
    assert answer['written'] is False and 'tier' not in answer
    assert any('not stated in EUR' in one for one in answer['refused']), answer['refused']
    assert quoting.read_bytes() == before, 'a refused tier moved the book'
    assert head(recorded) == answer['accepted']['lsn'], 'the acceptance is not the last fact'


def terms_of(document, pending):
    """What the tiers read of a pending quote, as the acceptance hands them over."""
    return service.ticket_terms(document, pending['deal'], pending['pinned']['values_hash'])


def test_size_tenor_and_market_route_a_ticket_through_the_tiers(recorded, quoting):
    """The three checks, each one driving the route through the SERVICE rather than the evaluator.

    Over the cap escalates to the tier that admits it, whose fill lands pending; over every tier
    answers `refused` with every sentence of the route AND the acceptance standing, since the
    client took the price whatever the policy says about it. A cap in a currency this book carries
    no rate for fails ITS tier by name - the safe direction, and the one a conversion inside a
    policy check would lose - while a cap in a currency it DOES price reads the notional CROSSED at
    the book's own spot, which is the case that tells a stated notional from an unconverted one.
    And the market check passes on the name the book's own board is declared under and fails on
    one nothing declared.

    A TICKET NO TIER ADMITS FILES NO REJECTION: the route's sentences are an answer, and a verdict
    is a seat's decision the policy names no seat for, so the head stands at the acceptance.

    Killing mutation: `deal_expiry` answering nothing, which routes every ticket past a tier that
    bounds the tenor.
    """
    declared = spine.declare_market('official', spine.values_of(
        service.load(service.BOOK.read()[0])))
    # both branches of the expiry, read directly: a vanilla leg names its own day, an accrual leg
    # the last settlement of its strip, and a swaption the later of its expiry and its maturity
    assert service.deal_expiry({'Object': 'StructuredDeal', 'Children': [
        {'Instrument': {'.Deal': {'Object': 'FXAccumulatorOptionDeal', 'Accumulator_ExpiryDates': [
            [{'.Timestamp': '2026-03-30'}, {'.Timestamp': '2026-04-01'}, 0.0],
            [{'.Timestamp': '2026-06-30'}, {'.Timestamp': '2026-07-02'}, 0.0]]}}}]}) == \
        pd.Timestamp('2026-07-02')
    assert service.deal_expiry({'Object': 'SwaptionDeal', 'Option_Expiry_Date': {
        '.Timestamp': '2025-06-30'}, 'Swap_Maturity_Date': {'.Timestamp': '2030-06-30'}}) == \
        pd.Timestamp('2030-06-30')
    assert service.deal_expiry({'Object': 'StructuredDeal'}) is None

    for rows, expected, said in (
            ([{'name': 'auto', 'max_notional': {'amount': SMALL, 'currency': 'USD'}},
              {'name': 'desk', 'four_eyes': True}], 'desk', 'caps max_notional'),
            ([{'name': 'auto', 'max_tenor_years': 0.5},
              {'name': 'desk', 'four_eyes': True}], 'desk', 'caps max_tenor_years'),
            ([{'name': 'auto', 'max_notional': {'amount': BIG, 'currency': 'JPY'}},
              {'name': 'desk', 'four_eyes': True}], 'desk', 'not stated in JPY'),
            # a million dollars is 54,054 rand on this book's own axis, so a 100,000 rand cap
            # admits it - and refuses it if the notional was stated without being crossed
            ([{'name': 'auto', 'max_notional': {'amount': SMALL, 'currency': 'ZAR'}},
              {'name': 'desk', 'four_eyes': True}], 'auto', None),
            ([{'name': 'auto', 'market': 'eod'},
              {'name': 'desk', 'four_eyes': True}], 'desk', 'nothing in this record has declared'),
            ([{'name': 'auto', 'market': 'official'}], 'auto', None),
            # a tier covering a node the acceptance does not book into is read past
            ([{'name': 'auto', 'scope': 'spine-desk/FX'},
              {'name': 'desk', 'four_eyes': True}], 'desk', 'covers spine-desk/FX'),
            ([{'name': 'auto', 'max_notional': {'amount': SMALL, 'currency': 'USD'}}], None,
             'caps max_notional')):
        declare(recorded, policy.TIERS_POLICY, tiers(*rows))
        quote = quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET)
        pending = pending_of(quoting.parent, quote)
        answer = accept(quote).json()

        if expected is not None:
            assert answer['written'] is True and answer['tier']['name'] == expected, rows
            assert (expected == 'desk') == ('no verdict is filed' in answer.get('waits_on', ''))
        else:
            assert answer['written'] is False and 'tier' not in answer, rows
            assert any(said in one for one in answer['refused']), answer['refused']
            assert facts(recorded, 'rejection') == [], 'a tier filed a rejection nobody signed'
            assert head(recorded) == answer['accepted']['lsn'], \
                'a ticket no tier admits moved the record past its own acceptance'
        assert answer['accepted']['lsn'] > declared['lsn'], 'the acceptance did not stand'
        # the sentence the ticket's route collected, which is what "fails that tier by name" means
        if said is not None:
            route = spine.route_ticket(answer['ticket'], terms_of(
                json.loads(quoting.read_text()), pending), ACTOR)
            assert any(said in one for one in route['refusals']), (rows, route['refusals'])


def ticketed(execution, amount=250_000.0, **body):
    """One direct booking under the client's set by the desk's own seat, as `/book/deals`
    answers it."""
    return CLIENT.post('/book/deals', content=dump(dict(
        {'action': 'add', 'deal': dict(CASHFLOW, Reference=execution, Amount=amount),
         'parent_reference': CLIENT_SET, 'quantity': 1.0, 'execution_reference': execution,
         'actor': ACTOR}, **body)), headers=JSON)


def with_client(path):
    """The book with the client's netting set on it, as the file holds one."""
    document = json.loads(path.read_text())
    document['Calc']['Deals']['Deals']['Children'].append(netting_set(CLIENT_SET, 'CPTY_A'))
    path.write_text(json.dumps(document, indent=2), newline='\n')


def test_every_booking_is_a_ticket_the_tiers_route(recorded, desk):
    """EVERY BOOKING IS A TICKET. A direct booking's fill carries a ticket over the plan the book
    has WITH it and the fill as filed, its route's tier and sentences aside, and where a tiers policy
    is in force it is routed on the size and the tenor its terms name, as an acceptance is: under
    the cap - 250,000 rand, 4,625,000 dollars on this book's axis - the hub signs in its own voice
    and then the fill lands, approved; over it - 5,000,000 rand - the fill lands PENDING under four
    eyes, the booker's own approval clearing nothing and another seat's clearing it; a tier
    covering the node the booking sits at reads it; and a booking no tier admits - a cap its size
    breaks, a tenor cap it settles past - books nothing.

    Killing mutations: the ticket taken before the splice, which is the book's own plan and every
    booking's; and a tier's scope read at the book rather than where the booking sits, which routes
    a node's booking past the node's own tier.
    """
    with_client(desk)
    declare(recorded, policy.TIERS_POLICY, tiers(
        {'name': 'fx', 'scope': 'spine-desk/FX'},
        {'name': 'auto', 'max_notional': {'amount': BIG, 'currency': 'USD'}},
        {'name': 'desk', 'four_eyes': True}))

    signed = ticketed('EXEC-AUTO').json()
    fill = facts(recorded, 'fill')[-1]
    assert signed['tier'] == {'name': 'auto', 'four_eyes': False, 'approval_lsn': fill[0] - 1,
                              'fell_past': ["the 'fx' tier covers spine-desk/FX and this ticket "
                                            "books into 'spine-desk' - it is read for a ticket in "
                                            "that node or under it"]}
    assert fill[2]['ticket'] == signed['ticket'] == spine.ticket(
        service.load(json.loads(desk.read_text())).plan_hash(), {
            field: value for field, value in fill[2].items()
            if field not in ('ticket', 'tier', 'fell_past')}), \
        'the ticket is not the plan the book has with this booking'
    assert facts(recorded, 'approval')[-1][2] == {'plan_hash': signed['ticket']}

    waiting = ticketed('EXEC-BIG', amount=BIG).json()
    assert waiting['written'] is True and waiting['tier']['name'] == 'desk'
    assert 'no verdict is filed' in waiting['waits_on']
    placed = ticketed('EXEC-FX', amount=BIG, portfolio='spine-desk/FX').json()
    assert placed['tier']['name'] == 'fx', 'the node\'s own tier was read past'

    def status(execution):
        return next(row['status'] for row in CLIENT.get('/book/positions').json()['positions']
                    if row['reference'] == execution)

    assert [status(execution) for execution in ('EXEC-AUTO', 'EXEC-BIG', 'EXEC-FX')] == [
        'approved', 'pending', 'approved']
    for seat, reads in ((ACTOR, 'pending'), (DESK_TWO, 'approved')):
        decided = CLIENT.post('/book/approve', json={'ticket': waiting['ticket'], 'actor': seat})
        assert decided.json()['status'] == reads == status('EXEC-BIG'), seat

    declare(recorded, policy.TIERS_POLICY, tiers(
        {'name': 'capped', 'max_notional': {'amount': BIG, 'currency': 'USD'}},
        {'name': 'short', 'max_tenor_years': 1.0}))
    before, head_before = desk.read_bytes(), head(recorded)
    refused = ticketed('EXEC-NONE', amount=BIG).json()
    assert refused['written'] is False and 'recorded' not in refused, refused
    assert any('caps max_notional' in one for one in refused['refused'])
    assert any('caps max_tenor_years' in one for one in refused['refused']), refused['refused']
    assert desk.read_bytes() == before and head(recorded) == head_before, \
        'a booking no tier admits booked something'
    assert verify_home(recorded)['events'] == head(recorded)


def capped(home, amount):
    """Declare an automatic tier capped at `amount` dollars and a four-eyes one behind it."""
    declare(home, policy.TIERS_POLICY, tiers(
        {'name': 'auto', 'max_notional': {'amount': amount, 'currency': 'USD'}},
        {'name': 'desk', 'four_eyes': True}))


def dollars(path, deal_path, quantity=1.0):
    """The dollars the ticket of what the file at `path` holds at `deal_path` carries, as its
    booking hands it to the tiers."""
    document = json.loads(path.read_text())
    return service.ticket_terms(document, service.instrument_of(deal_at(document, deal_path)), None,
                                quantity)['notional_in'].get('USD')


def booked(deal, execution, quantity=1.0, **body):
    """`deal` booked under the client's set by the desk's own seat, as `/book/deals` answers."""
    return CLIENT.post('/book/deals', content=dump(dict(
        {'action': 'add', 'deal': deal, 'parent_reference': CLIENT_SET, 'quantity': quantity,
         'execution_reference': execution, 'actor': ACTOR}, **body)), headers=JSON).json()


#: A 2Y monthly TARF on a million dollars a fixing, geared twice against the client.
TARF = {'pair': 'USDZAR', 'expiry': '2Y', 'notional': AMOUNT, 'notional_currency': 'USD',
        'buy_currency': 'USD', 'fixing_frequency': '1M', 'target': USDZAR * 0.1, 'leverage': 2.0}


def test_a_strip_is_sized_at_its_amount_at_every_fixing_still_to_come(recorded, quoting):
    """A STRIP DEALS ITS AMOUNT AT EVERY FIXING (CRE52.37(4)). A 2Y monthly TARF quoted on a
    million dollars a fixing deals up to its 2,000,000 `LeverageNotional` at each of 24 fixings, so
    its ticket carries 24 x 2,000,000 = 48,000,000 USD whatever the quote states, and an automatic
    tier capped at 5,000,000 passes it to four eyes. Its leg booked partway through its life, its
    first five fixings printed behind the base date and a notional of one dollar in the request,
    carries the 19 to come, 38,000,000, which an automatic tier capped at 40,000,000 signs.

    Killing mutations: the rows not counted - 2,000,000, the TARF signed by the hub; the leveraged
    amount not read - 24,000,000, still four eyes, so the size is asserted; every row counted
    whatever its date - the seasoned leg 48,000,000 and pending.
    """
    capped(recorded, BIG)
    quote = quote_of('TargetRedemptionForward', TARF, netting_set=CLIENT_SET)
    pending = pending_of(quoting.parent, quote)
    answer = accept(quote).json()
    assert answer['written'] is True and answer['tier']['name'] == 'desk', answer
    assert terms_of(json.loads(quoting.read_text()), pending)['notional_in']['USD'] == 48_000_000.0

    capped(recorded, 40_000_000.0)
    leg = dict(pending['deal']['Children'][0]['Instrument']['.Deal'], Reference='TARF-SEASONED')
    leg['TARF_ExpiryDates'] = [[{'.Timestamp': str((BASE - pd.DateOffset(months=5 - k)).date())}] * 2
                               + [USDZAR] for k in range(5)] + leg['TARF_ExpiryDates'][5:]
    seasoned = booked(leg, 'EXEC-SEASONED', notional=1.0, notional_currency='USD')
    assert seasoned['written'] is True and seasoned['tier']['name'] == 'auto', seasoned
    assert dollars(quoting, seasoned['deal_path']) == 38_000_000.0


#: A year of monthly observations paying 8% a year on a million dollars inside a band about spot.
RANGE = {'pair': 'USDZAR', 'expiry': '1Y', 'notional': AMOUNT, 'notional_currency': 'USD',
         'fixing_frequency': '1M', 'tiers': [[USDZAR * 0.9, USDZAR * 1.1, 8.0]]}


def test_a_range_accrual_is_sized_at_its_payoff_at_every_observation_to_come(recorded, quoting):
    """A range accrual can still pay its `Payoff` at every observation to come. Quoted at 8% a year
    on a million dollars to a last observation 367 days out, it pays 1,000,000 x 8% x 367/365 / 12
    = 6,703.20 a fixing, and its ticket carries 12 x 6,703.20 = 80,438.36 USD, which an automatic
    tier capped at 80,000 passes to four eyes.

    Killing mutation: the rows not counted - 6,703.20, signed by the hub.
    """
    capped(recorded, 80_000.0)
    quote = quote_of('RangeAccrual', RANGE, netting_set=CLIENT_SET)
    pending = pending_of(quoting.parent, quote)
    answer = accept(quote).json()
    assert answer['written'] is True and answer['tier']['name'] == 'desk', answer
    assert terms_of(json.loads(quoting.read_text()), pending)['notional_in']['USD'] == \
        pytest.approx(AMOUNT * 0.08 * 367 / 365, rel=1e-14)


#: A digital call a year out on a share at 50 paying 100,000 dollars, and one on the rand paying
#: 100,000 dollars.
DIGITAL = {'Object': 'EquityBinaryOption', 'Reference': 'DIGITAL', 'Currency': 'USD', 'Discount_Rate': 'USD',
           'Equity': 'EQ', 'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
           'Strike_Price': 50.0, 'Payoff': 1e5, 'Expiry_Date': BASE + pd.DateOffset(years=1)}
FX_DIGITAL = {'Object': 'FXBinaryOption', 'Reference': 'FX_DIGITAL', 'Currency': 'USD', 'Underlying_Currency': 'ZAR',
              'Discount_Rate': 'USD', 'FX_Volatility': 'USD.ZAR', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
              'Strike_Price': USDZAR, 'Payoff': 1e5, 'Expiry_Date': BASE + pd.DateOffset(years=1),
              'Delivery_Date': BASE + pd.DateOffset(years=1)}


def test_a_cash_digital_is_sized_at_its_payout_and_one_paying_shares_at_their_price(recorded, quoting):
    """A DIGITAL PAYING CASH DEALS ITS PAYOUT, in the currency its settlement pays it in; ONE PAYING
    UNITS DEALS THEM AT TODAY'S PRICE (CRE52.36). On a share at 50, a digital call paying 100,000
    dollars carries 100,000 and the same paying a thousand shares 1,000 x 50 = 50,000; a digital
    call on the rand paying 100,000 dollars carries 100,000: an automatic tier capped at 200,000
    signs each.

    Killing mutation: the cash digital read as units - its 100,000 as shares at 50, 5,000,000,
    passed to four eyes.
    """
    document = json.loads(quoting.read_text())
    document['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors'].update(json.loads(dump(
        dict(EQUITY, **{INDEX: dict(EQUITY[INDEX], Spot=50.0)}))))
    quoting.write_text(json.dumps(document, indent=2), newline='\n')
    capped(recorded, 2e5)
    for deal, size in ((DIGITAL, 1e5), (dict(DIGITAL, Reference='SHARES', Payoff=1e3, Payoff_Style='Asset'), 5e4),
                       (FX_DIGITAL, 1e5)):
        dealt = booked(deal, 'EXEC-' + deal['Reference'])
        assert dealt['written'] is True and dealt['tier']['name'] == 'auto', dealt
        assert dollars(quoting, dealt['deal_path']) == size, deal['Reference']


#: Two years from the base date, a year a period: a swap of the share's return on a thousand shares,
#: then two thousand, and a leg paying 2% on a million dollars grown by an index, then two million.
YEARS = [BASE + pd.DateOffset(years=k) for k in range(3)]
SWAPLETS = {'Object': 'EquitySwapletListDeal', 'Reference': 'SHARES', 'Currency': 'USD', 'Equity_Currency': 'USD',
            'Discount_Rate': 'USD', 'Equity': 'EQ', 'Buy_Sell': 'Buy', 'Amount_Type': 'Shares', 'Cashflows': {'Items': [
                {'Start_Date': start, 'End_Date': end, 'Payment_Date': end, 'Amount': amount, 'Start_Multiplier': 1.0,
                 'End_Multiplier': 1.0, 'Dividend_Multiplier': 0.0}
                for start, end, amount in zip(YEARS[:-1], YEARS[1:], (1e3, 2e3))]}}
LINKER = {'Object': 'YieldInflationCashflowListDeal', 'Reference': 'LINKER', 'Currency': 'USD', 'Discount_Rate': 'USD',
          'Buy_Sell': 'Buy', 'Index': 'CPI', 'Index_Reference': {'Months_Lag': 3, 'Reference_Type': 'Interpolated'},
          'Cashflows': {'Items': [{
              'Payment_Date': end, 'Notional': notional, 'Base_Reference_Date': start, 'Base_Reference_Value': 0.0,
              'Final_Reference_Date': end, 'Final_Reference_Value': 0.0, 'Accrual_Start_Date': start,
              'Accrual_End_Date': end, 'Accrual_Day_Count': 'ACT_365', 'Accrual_Year_Fraction': 1.0,
              'Yield': utils.Percent(2.0), 'Rate_Multiplier': 1.0}
              for start, end, notional in zip(YEARS[:-1], YEARS[1:], (1e6, 2e6))]}}


def test_a_swaplet_list_is_sized_at_its_shares_price_or_its_principal_and_a_linker_at_its_notional(
        recorded, quoting):
    """A TOTAL RETURN SWAP DEALS ITS UNITS AT TODAY'S PRICE, OR ITS PRINCIPAL (CRE52.36), AND AN
    INFLATION LEG ITS NOTIONAL, each its largest period's as a table is sized: on a share at 50, a
    swaplet list of a thousand shares then two thousand carries 2,000 x 50 = 100,000 dollars, and
    the same stating 1.5 million dollars then three million as its principal 3,000,000; a leg on a
    million dollars grown by an index, then two million, 2,000,000.

    Killing mutation: a type naming a currency sized in it whatever its amount states (`if
    part.currency` in `service.deal_size`) - the shares 2,000 dollars.
    """
    document = json.loads(quoting.read_text())
    document['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors'].update(json.loads(dump(dict(
        EQUITY, **{INDEX: dict(EQUITY[INDEX], Spot=50.0)}, **{
            'PriceIndex.CPI': {'Index': utils.Curve([], [[float((pd.Timestamp(day) - utils.excel_offset).days), 100.0]
                                                         for day in ('2024-01-01', '2024-05-01')]),
                               'Last_Period_Start': pd.Timestamp('2024-05-01'), 'Publication_Period': 'Monthly',
                               'Next_Publication_Date': BASE + pd.DateOffset(days=12), 'Currency': 'USD'},
            'InflationRate.CPI': {'Price_Index': 'CPI', 'Day_Count': 'ACT_365', 'Currency': 'USD',
                                  'Curve': utils.Curve([], [[0.0, 0.025], [5.0, 0.025]])}}))))
    quoting.write_text(json.dumps(document, indent=2), newline='\n')
    principal = dict(SWAPLETS, Reference='PRINCIPAL', Amount_Type='Principal', Cashflows={'Items': [
        dict(item, Amount=1.5e3 * item['Amount']) for item in SWAPLETS['Cashflows']['Items']]})
    for deal, size in ((SWAPLETS, 1e5), (principal, 3e6), (LINKER, 2e6)):
        dealt = booked(deal, 'EXEC-' + deal['Reference'])
        assert dealt['written'] is True, dealt
        assert dollars(quoting, dealt['deal_path']) == size, deal['Reference']


#: A swap buying a million dollars for rand a month out and 100,000 rand back for dollars a year
#: out - 1,850,000 dollars on this book's axis.
SWAP = {'Object': 'FXSwapDeal', 'Reference': 'SWAP', 'Near_Buy_Far_Sell_Ccy': 'USD',
        'Near_Sell_Far_Buy_Ccy': 'ZAR', 'Near_Buy_Far_Sell_Discount_Rate': 'USD',
        'Near_Sell_Far_Buy_Discount_Rate': 'ZAR', 'Near_Settlement_Date': BASE + pd.DateOffset(
            months=1), 'Far_Settlement_Date': BASE + pd.DateOffset(years=1),
        'Near_Buy_Amount': AMOUNT, 'Near_Sell_Amount': AMOUNT * USDZAR, 'Far_Buy_Amount': 1e5,
        'Far_Sell_Amount': 1e5 * SPOT}

#: A forward buying one dollar for 100,000,000 rand a year out: 1,850,000,000 dollars on this
#: book's axis, all of it on the side it sells.
OFF_MARKET = {'Object': 'FXForwardDeal', 'Reference': 'OFF', 'Buy_Currency': 'USD',
              'Buy_Amount': 1.0, 'Buy_Discount_Rate': 'USD', 'Sell_Currency': 'ZAR',
              'Sell_Amount': 1e8, 'Sell_Discount_Rate': 'ZAR',
              'Settlement_Date': BASE + pd.DateOffset(years=1)}


def test_a_structure_is_its_largest_leg_and_an_fx_trade_its_larger_side(recorded, quoting):
    """A CONTAINER IS ITS LARGEST CHILD, never their sum: a collar's two legs are each on a million
    dollars, so it is a 1,000,000 trade an automatic tier capped at 1,500,000 signs. AN FX TRADE IS
    ITS LARGER SIDE (CRE52.35): a swap of a million dollars near and 100,000 rand far is its far
    side, 1,850,000 dollars, and an off-market forward buying one dollar for 100,000,000 rand is
    its sell side, 1,850,000,000 dollars - each passed by that tier to four eyes.

    Killing mutations: the children summed - the collar 2,000,000 and pending; the swap read by its
    first exposure alone - 1,000,000, signed by the hub; the other side not read - the forward one
    dollar, signed by the hub.
    """
    capped(recorded, 1_500_000.0)
    quote = quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET)
    answer = accept(quote).json()
    assert answer['written'] is True and answer['tier']['name'] == 'auto', answer
    for deal, size in ((SWAP, 1e5 * SPOT), (OFF_MARKET, 1e8 * SPOT)):
        dealt = booked(deal, 'EXEC-' + deal['Reference'])
        assert dealt['written'] is True and dealt['tier']['name'] == 'desk', dealt
        assert dollars(quoting, dealt['deal_path']) == size


#: A deposit of a thousand rand: 18,500 dollars on this book's axis, were its type to declare it.
DEPOSIT = {'Object': 'DepositDeal', 'Reference': 'DEPOSIT', 'Currency': 'ZAR',
           'Discount_Rate': 'ZAR', 'Interest_Rate': 'ZAR', 'Effective_Date': BASE,
           'Maturity_Date': BASE + pd.DateOffset(months=3), 'Amount': 1e3}


def test_a_fill_is_its_quantity_times_the_instrument_and_an_undeclared_type_no_size(recorded,
                                                                                    desk):
    """THE QUANTITY MULTIPLIES, BY MAGNITUDE. A cash flow of 100,000 rand is 1,850,000 dollars on
    this book's axis: filled three times over it is 5,550,000, past an automatic tier capped at
    5,000,000 and pending, and half of one unwound is 925,000, which the hub signs. A TYPE THAT
    DECLARES NOTHING HAS NO SIZE: a deposit carries no notional at all, fails the automatic tier as
    `unstated_notional` and lands pending.

    Killing mutations: the quantity not read - 1,850,000, signed by the hub; its sign kept - the
    unwind no size and pending; the deposit's type declaring its `Amount` an exposure - 18,500
    dollars, signed by the hub.
    """
    with_client(desk)
    capped(recorded, BIG)
    three = ticketed('EXEC-3', amount=1e5, quantity=3.0).json()
    assert three['written'] is True and three['tier']['name'] == 'desk', three
    assert dollars(desk, three['deal_path'], 3.0) == 5_550_000.0
    half = CLIENT.post('/book/deals', content=dump({
        'action': 'fill', 'deal_path': three['deal_path'], 'reference': 'EXEC-3',
        'quantity': -0.5, 'execution_reference': 'EXEC-HALF', 'actor': ACTOR}),
        headers=JSON).json()
    assert half['written'] is True and half['tier']['name'] == 'auto', half
    assert dollars(desk, three['deal_path'], -0.5) == 925_000.0

    deposit = booked(DEPOSIT, 'EXEC-DEPOSIT')
    assert deposit['written'] is True and deposit['tier']['name'] == 'desk', deposit
    document = json.loads(desk.read_text())
    terms = service.ticket_terms(document, service.instrument_of(deal_at(
        document, deposit['deal_path'])), None)
    assert terms['notional_in'] == {}
    assert any('not stated in USD' in one for one in spine.route_ticket(
        deposit['ticket'], terms, ACTOR)['refusals'])


def test_a_restrike_is_sized_at_every_unit_of_the_positions_it_moves(recorded, desk):
    """A RESTRIKE DEALS ITS TERMS AT THE POSITION IT MOVES, as a fill deals its quantity: two clips
    of one cash flow, restruck to 100,000 rand - 1,850,000 dollars a unit on this book's axis -
    deal 3,700,000, past an automatic tier capped at 2,500,000, and land pending under four eyes.

    Killing mutation: the held quantity not read - one unit, 1,850,000, signed by the hub.
    """
    with_client(desk)
    first = ticketed('EXEC-1', amount=1e3).json()
    assert ticketed('EXEC-1', amount=1e3, execution_reference='EXEC-1B').json()['written'] is True
    capped(recorded, 2_500_000.0)
    amended = restruck(first['deal_path'], Amount=1e5)
    assert amended['written'] is True and amended['tier']['name'] == 'desk', amended


#: Two desks' approvers, each at its own node of the book.
FX_SEAT, RATES_SEAT = 'subject-fx-approver', 'subject-rates-approver'


def test_a_verdict_is_filed_where_its_ticket_books(recorded, desk):
    """A VERDICT IS FILED WHERE THE TICKET BOOKS, read off the record and never off the caller: a
    seat granted `approve` at a node signs what books there, and one at another node is refused
    with the denial landed however it names its own node; a ticket nothing carries has nothing to
    rule on.

    And the hub signs an automatic tier's ticket only where its booker may book: a seat that may
    not is refused at the fill with the denial landed, and no approval stands over a ticket
    nothing books.

    Killing mutations: the verdict filed at the portfolio the caller names, which lets the Rates
    approver sign an FX ticket by naming Rates; and the dry authorization skipped, which leaves the
    writer's voice on a ticket its fill was refused.
    """
    with_client(desk)
    declare(recorded, policy.TIERS_POLICY, tiers({'name': 'desk', 'four_eyes': True}))
    for path in ('spine-desk/FX', 'spine-desk/Rates'):
        spine.declare_portfolio(path)
    entitle(recorded, [(ACTOR, verb, '*') for verb in ('admin', 'book', 'validate')] + [
        (FX_SEAT, 'approve', 'spine-desk/FX'), (RATES_SEAT, 'approve', 'spine-desk/Rates')])
    booked = ticketed('EXEC-FX', portfolio='spine-desk/FX').json()
    assert booked['written'] is True and 'waits_on' in booked, booked

    denied = CLIENT.post('/book/approve', json={'ticket': booked['ticket'], 'actor': RATES_SEAT,
                                                'portfolio': 'spine-desk/Rates'})
    assert denied.status_code == 422 and "over 'spine-desk/FX'" in denied.json()['detail']
    assert facts(recorded, 'capability_denied')[-1][2]['book'] == 'spine-desk/FX'
    signed = CLIENT.post('/book/approve', json={'ticket': booked['ticket'], 'actor': FX_SEAT})
    assert signed.status_code == 200, signed.text
    assert (signed.json()['portfolio'], signed.json()['status']) == ('spine-desk/FX', 'approved')
    assert facts(recorded, 'approval')[-1][2]['portfolio'] == 'spine-desk/FX'
    nothing = CLIENT.post('/book/approve', json={'ticket': 'e' * 64, 'actor': FX_SEAT})
    assert nothing.status_code == 422 and 'nothing on the record carries' in nothing.json()[
        'detail']

    declare(recorded, policy.TIERS_POLICY, tiers({'name': 'auto'}))
    signatures = len(facts(recorded, 'approval'))
    stray = ticketed('EXEC-STRAY', portfolio='spine-desk/FX', actor=FX_SEAT)
    assert stray.status_code == 422, stray.text
    assert facts(recorded, 'capability_denied')[-1][2]['attempted_type'] == 'fill'
    assert len(facts(recorded, 'approval')) == signatures, 'the hub signed a ticket nothing books'


def standing_of(reference):
    """`(status, pending)` of the position the file holds under `reference`."""
    return next((row['status'], row['pending'])
                for row in CLIENT.get('/book/positions').json()['positions']
                if row['reference'] == reference)


def restruck(deal_path, **fields):
    """An amendment of the deal at `deal_path` by the desk's own seat, as `/book/deals` answers."""
    return CLIENT.post('/book/deals', content=dump({
        'action': 'amend', 'deal_path': deal_path, 'fields': fields, 'actor': ACTOR}),
        headers=JSON).json()


def confirmed(key, value_date):
    """A confirmation matched on `value_date`, filed against a clip's key by the desk's own seat."""
    assert CLIENT.post('/book/transition', content=dump({
        'subject': key, 'status': 'confirmed', 'value_date': value_date, 'actor': ACTOR}),
        headers=JSON).status_code == 200


def test_a_restrike_is_a_trade_and_the_position_reads_its_ticket(recorded, desk):
    """AN AMENDMENT IS A TICKET TOO. A trade of two clips booked under four eyes, each signed by a
    second seat and confirmed, reads approved; restruck by its booker to a hundred times its
    amount, the amendment carries a ticket of its own over the plan it leaves and is routed as a
    booking is - so the position it moved reads PENDING on it, the worklist naming that ticket once
    for both clips, the approvals over the fills reaching nothing it restruck - and a second seat's
    signature clears it. Every clip it restruck keeps the key it was filled under and is due a
    re-confirmation, which one filed since clears. Under an automatic tier the hub signs the
    restrike before it lands; one no tier admits writes nothing; and the oracle holds.

    Killing mutations: the positions fold carrying the restruck clips' tickets onto the terms they
    became, which reads a restrike nobody signed as approved; a restruck clip keyed on the terms it
    became, which waits on a key nothing was filed under; a confirmation filed before the restrike
    still counted, which leaves the amended trade confirmed as it was; and a row per clip, which
    lists one restrike twice.
    """
    from derivus_spine import oracle

    def unconfirmed():
        return sorted(row['key'] for row in CLIENT.get('/book/worklist').json()['unconfirmed'])

    with_client(desk)
    declare(recorded, policy.TIERS_POLICY, tiers({'name': 'desk', 'four_eyes': True}))
    booked = ticketed('EXEC-1', amount=1_000.0).json()
    keys = []
    for clip, reference in ((booked, 'EXEC-1'), (ticketed(
            'EXEC-1', amount=1_000.0, execution_reference='EXEC-1B').json(), 'EXEC-1B')):
        CLIENT.post('/book/approve', json={'ticket': clip['ticket'], 'actor': DESK_TWO})
        keys.append(spine.fill_key(clip['recorded']['instrument'], reference))
        confirmed(keys[-1], '2024-06-28')
    assert standing_of('EXEC-1') == ('approved', 0.0) and unconfirmed() == []

    amended = restruck(booked['deal_path'], Amount=100_000.0)
    assert amended['written'] is True and amended['tier']['name'] == 'desk', amended
    assert 'no verdict is filed' in amended['waits_on']
    assert facts(recorded, 'amendment')[-1][2]['ticket'] == amended['ticket'] != booked['ticket']
    assert standing_of('EXEC-1') == ('pending', 2.0), 'the fill\'s approval reached the restrike'
    assert [row['key'] for row in CLIENT.get('/book/worklist').json()['pending']] == [
        amended['ticket']]
    assert unconfirmed() == sorted(keys), 'a restruck clip is due no re-confirmation'
    signed = CLIENT.post('/book/approve', json={'ticket': amended['ticket'], 'actor': DESK_TWO})
    assert signed.json()['status'] == 'approved' and standing_of('EXEC-1') == ('approved', 0.0)
    confirmed(keys[0], '2024-06-29')
    assert unconfirmed() == [keys[1]], 'a re-confirmation cleared nothing'

    declare(recorded, policy.TIERS_POLICY, tiers({'name': 'auto'}))
    automatic = restruck(booked['deal_path'], Amount=50_000.0)
    assert automatic['tier'] == {'name': 'auto', 'four_eyes': False,
                                 'approval_lsn': facts(recorded, 'amendment')[-1][0] - 1,
                                 'fell_past': []}
    declare(recorded, policy.TIERS_POLICY, tiers(
        {'name': 'capped', 'max_notional': {'amount': BIG, 'currency': 'USD'}}))
    before, at = desk.read_bytes(), head(recorded)
    refused = restruck(booked['deal_path'], Amount=BIG)
    assert refused['written'] is False and any('caps max_notional' in one
                                               for one in refused['refused'])
    assert desk.read_bytes() == before and head(recorded) == at, 'a restrike no tier admits landed'
    assert oracle.failed(oracle.report(recorded)) == []


#: A seat that may act on nothing in the book - risk reading it.
RISK_SEAT = 'subject-risk'


def papered(recorded):
    """The client's set on the book, its entity and agreement declared, a tier signing what is
    under the cap and a second seat what is not, and the desk's three seats entitled."""
    terms = netting_set(CLIENT_SET, 'CPTY_A')['Instrument']['.Deal']
    for path, body in (('/book/entities', {'entity': 'CPTY_A', 'name': 'Client A Ltd'}),
                       ('/book/agreements', {'agreement': CLIENT_SET, 'entity': 'CPTY_A',
                                             'kind': 'ISDA 2002', 'terms': terms})):
        assert CLIENT.post(path, content=dump(body), headers=JSON).status_code == 200
    declare(recorded, policy.TIERS_POLICY, tiers(
        {'name': 'auto', 'max_notional': {'amount': BIG, 'currency': 'USD'}},
        {'name': 'desk', 'four_eyes': True}))
    entitle(recorded, [(ACTOR, verb, '*') for verb in ('admin', 'book', 'validate', 'approve')]
            + [(DESK_TWO, 'approve', '*'), (RISK_SEAT, 'validate', '*')])


def test_a_strip_row_names_the_trade_who_did_it_and_what_it_opens(recorded, desk):
    """THE PAGE IS TOLD. Each row of the strip names what it is about in the desk's words - the
    trade's reference and client, the agreement, how much was unwound - says who did it, and the
    deal path or agreement it opens: the agreement declared, the hub's automatic signature under
    the tier that gave it beside a second seat's, a booking, an unwind, a restrike and a refusal
    filed under the seat refused.

    Killing mutations: a denial's `by` read off the envelope, which files every refusal under the
    writer; and the hub's own approval told as a seat's, which reads an automatic signature
    exactly like a trader's.
    """
    with_client(desk)
    papered(recorded)
    auto = ticketed('EXEC-AUTO').json()
    big = ticketed('EXEC-BIG', amount=BIG).json()
    assert CLIENT.post('/book/approve', json={'ticket': big['ticket'], 'actor': DESK_TWO}
                       ).status_code == 200
    assert CLIENT.post('/book/deals', content=dump({
        'action': 'fill', 'deal_path': auto['deal_path'], 'reference': 'EXEC-AUTO',
        'quantity': -0.5, 'execution_reference': 'EXEC-HALF', 'actor': ACTOR}),
        headers=JSON).json()['written'] is True
    seated = CLIENT.post('/book/approve', json={'ticket': big['ticket'], 'actor': RISK_SEAT})
    assert seated.status_code == 422 and 'holds no approve scope' in seated.json()['detail']
    assert restruck(big['deal_path'], Amount=300_000.0)['written'] is True

    told = [(row['by'], row['summary'], row['deal_path'], row['agreement'])
            for row in strip()['rows'] if row['event_type'] in (
                'agreement_declared', 'approval', 'fill', 'amendment', 'capability_denied')]
    auto_trade, big_trade = 'EXEC-AUTO for Client A Ltd', 'EXEC-BIG for Client A Ltd'
    assert told == [
        (ACTOR, 'declared agreement {} (ISDA 2002) with Client A Ltd'.format(CLIENT_SET), None,
         CLIENT_SET),
        ('tier auto', 'signed {} automatically: its tier needs no second seat'.format(auto_trade),
         auto['deal_path'], CLIENT_SET),
        (ACTOR, 'booked {} under {}, execution EXEC-AUTO'.format(auto_trade, CLIENT_SET),
         auto['deal_path'], CLIENT_SET),
        (ACTOR, 'booked {} under {}, execution EXEC-BIG'.format(big_trade, CLIENT_SET),
         big['deal_path'], CLIENT_SET),
        (DESK_TWO, 'signed {}'.format(big_trade), big['deal_path'], CLIENT_SET),
        ('tier auto', 'signed {} automatically: its tier needs no second seat'.format(auto_trade),
         auto['deal_path'], CLIENT_SET),
        (ACTOR, 'unwound 0.5 of {} under {}, execution EXEC-HALF'.format(auto_trade, CLIENT_SET),
         auto['deal_path'], CLIENT_SET),
        (RISK_SEAT, 'refused: holds no approve scope over spine-desk, so its approval was not '
                    'filed', None, None),
        (ACTOR, 'amended the terms of {}: Amount {} to 300000.0 - a restrike, ticketed '
                'again'.format(big_trade, BIG), big['deal_path'], CLIENT_SET)], told


def test_a_restrike_row_says_what_it_moved(recorded, desk):
    """A RESTRIKE'S ROW SAYS WHAT CHANGED, off the two sets of terms the record holds: each field
    it moved with what it was and what it became, as the file writes them, and its sentence says
    them where they are few. Moving the amount and the payment date reads both in the sentence;
    moving four fields counts them; the fill beside them moved nothing.

    Killing mutation: the two sets of terms read the wrong way round, which says the restrike
    moved the amount from 300,000 to 250,000.
    """
    with_client(desk)
    booked = ticketed('EXEC-1').json()
    later = {'.Timestamp': '2027-06-28'}
    assert restruck(booked['deal_path'], Amount=300_000.0, Payment_Date=later)['written']
    assert restruck(booked['deal_path'], Amount=400_000.0, Currency='USD', Discount_Rate='USD',
                    Payment_Date={'.Timestamp': '2028-06-28'})['written']

    told = [{key: value for key, value in row.items()
             if key not in ('lsn', 'record_time', 'effective_time')}
            for row in strip()['rows'] if row['event_type'] in ('fill', 'amendment')]
    trade = {'actor': ACTOR, 'book': 'spine-desk', 'by': ACTOR, 'deal_path': booked['deal_path'],
             'agreement': CLIENT_SET, 'entity': None}
    assert told == [
        dict(trade, event_type='fill', changed=None,
             summary='booked EXEC-1 for CPTY_A under CLIENT_A, execution EXEC-1'),
        dict(trade, event_type='amendment', changed=[
            {'field': 'Amount', 'was': 250_000.0, 'became': 300_000.0},
            {'field': 'Payment_Date', 'was': {'.Timestamp': '2026-06-28'}, 'became': later}],
            summary='amended the terms of EXEC-1 for CPTY_A: Amount 250000.0 to 300000.0, '
                    'Payment_Date 2026-06-28 to 2027-06-28 - a restrike, ticketed again'),
        dict(trade, event_type='amendment', changed=[
            {'field': 'Amount', 'was': 300_000.0, 'became': 400_000.0},
            {'field': 'Currency', 'was': 'ZAR', 'became': 'USD'},
            {'field': 'Discount_Rate', 'was': 'ZAR', 'became': 'USD'},
            {'field': 'Payment_Date', 'was': later, 'became': {'.Timestamp': '2028-06-28'}}],
            summary='amended the terms of EXEC-1 for CPTY_A: 4 fields - a restrike, ticketed '
                    'again')], told


def test_one_trade_s_history_is_asked_by_its_path_through_its_restrikes(recorded, desk):
    """ONE TRADE'S HISTORY, ASKED BY ITS PATH. Two trades under four eyes, the first signed, then
    restruck - which keeps its node's path and moves its instrument - and that restrike signed,
    the second signed last. Asked for the first trade's path, the strip answers its four rows -
    LSNs 6, 8, 9 and 10, after genesis's four, the tiers and the second fill at 7 - the fill and
    the signature on the terms the restrike left included, exactly as the whole strip tells them,
    and none of the second's; a page of two from genesis and the next from its
    cursor walk those four in turn, and the newest page's cursor is the head.

    Killing mutation: the page cut before the trade's rows are picked, which reads the first two
    events of the record and finds no row of the trade in them.
    """
    with_client(desk)
    declare(recorded, policy.TIERS_POLICY, tiers({'name': 'desk', 'four_eyes': True}))
    one, two = ticketed('EXEC-ONE').json(), ticketed('EXEC-TWO').json()
    CLIENT.post('/book/approve', json={'ticket': one['ticket'], 'actor': DESK_TWO})
    amended = restruck(one['deal_path'], Amount=300_000.0)
    CLIENT.post('/book/approve', json={'ticket': amended['ticket'], 'actor': DESK_TWO})
    CLIENT.post('/book/approve', json={'ticket': two['ticket'], 'actor': DESK_TWO})
    (held,) = [row for row in CLIENT.get('/book/positions').json()['positions']
               if row['reference'] == 'EXEC-ONE']
    assert held['deal_paths'] == [one['deal_path']]
    assert held['instrument'] != one['recorded']['instrument'], 'the restrike moved nothing'

    trade = [row for row in strip()['rows'] if row['deal_path'] == one['deal_path']]
    assert [(row['lsn'], row['event_type']) for row in trade] == [
        (6, 'fill'), (8, 'approval'), (9, 'amendment'), (10, 'approval')]
    asked = strip(deal_path=one['deal_path'])
    assert (asked['rows'], asked['lsn']) == (trade, head(recorded))
    first = strip(deal_path=one['deal_path'], since=0, limit=2)
    assert (first['rows'], first['lsn']) == (trade[:2], trade[1]['lsn'])
    rest = strip(deal_path=one['deal_path'], since=first['lsn'], limit=2)
    assert (rest['rows'], rest['lsn']) == (trade[2:], trade[3]['lsn'])
    assert strip(deal_path=one['deal_path'], limit=1) == {'lsn': head(recorded),
                                                          'rows': trade[-1:]}


def test_a_client_s_history_is_asked_by_its_agreement_or_its_entity(recorded, desk):
    """A CLIENT'S HISTORY, ASKED BY CLIENT. Two clients, each an entity and an agreement, declared
    A then B - LSNs 5 to 8 after genesis's four - and trades booked A, B, A at 9, 10 and 11. Asked
    by agreement A, the strip answers the rows opening it, exactly as the whole strip tells them:
    its declaration and its two trades, none of B's; a page of two from genesis and the next from
    its cursor walk those three in turn, and the newest page's cursor is the head. Asked by entity
    A, it answers the entity's declaration and its agreement's; asked by both, the one row opening
    each.

    Killing mutation: `agreement` and `entity` not read, which answers every row of the record,
    the other client's among them.
    """
    with_client(desk)
    document = json.loads(desk.read_text())
    document['Calc']['Deals']['Deals']['Children'].append(netting_set('CLIENT_B', 'CPTY_B'))
    desk.write_text(json.dumps(document, indent=2), newline='\n')
    for client, entity, name in ((CLIENT_SET, 'CPTY_A', 'Client A Ltd'),
                                 ('CLIENT_B', 'CPTY_B', 'Client B Ltd')):
        terms = netting_set(client, entity)['Instrument']['.Deal']
        for path, body in (('/book/entities', {'entity': entity, 'name': name}),
                           ('/book/agreements', {'agreement': client, 'entity': entity,
                                                 'kind': 'ISDA 2002', 'terms': terms})):
            assert CLIENT.post(path, content=dump(body), headers=JSON).status_code == 200
    for execution, client in (('EXEC-A1', CLIENT_SET), ('EXEC-B1', 'CLIENT_B'),
                              ('EXEC-A2', CLIENT_SET)):
        assert ticketed(execution, parent_reference=client).json()['written'] is True

    whole = strip()['rows']
    client = [row for row in whole if row['agreement'] == CLIENT_SET]
    assert [(row['lsn'], row['event_type']) for row in client] == [
        (6, 'agreement_declared'), (9, 'fill'), (11, 'fill')]
    asked = strip(agreement=CLIENT_SET)
    assert (asked['rows'], asked['lsn']) == (client, head(recorded))
    first = strip(agreement=CLIENT_SET, since=0, limit=2)
    assert (first['rows'], first['lsn']) == (client[:2], 9)
    assert strip(agreement=CLIENT_SET, since=9, limit=2) == {'lsn': 11, 'rows': client[2:]}
    assert [(row['lsn'], row['event_type']) for row in strip(entity='CPTY_A')['rows']] == [
        (5, 'entity_declared'), (6, 'agreement_declared')]
    assert strip(entity='CPTY_A', agreement=CLIENT_SET)['rows'] == client[:1]


def test_the_worklist_names_each_ticket_and_the_seat_it_waits_on(recorded, desk):
    """ONE ANSWER FOR WHAT WAITS ON A SEAT. A ticket under four eyes waits on the seat that may sign
    it, named as a trade - its reference, path, client and agreement, its booker, its tier, the
    price it was booked at and the tier's own sentence for what it wants, which its booker's own
    signature does not clear - and on no seat that may sign nothing; a read no seat signs, or one
    on a box narrowing no seat's reads, lists it beside `seat` null. Signed by the second seat, it
    leaves the list.

    Killing mutations: `seat` answered as the name the request carried, which tells a box that
    narrows no seat that its lists wait on one; the tier's four eyes dropped from the sentence,
    which reads the booker's own signature as the verdict standing; and the client named by its
    entity id, which a desk reads as a code.
    """
    def worklist(**params):
        return CLIENT.get('/book/worklist', params=params).json()

    with_client(desk)
    assert worklist(actor=DESK_TWO)['seat'] is None, 'a box narrowing no seat named one'
    papered(recorded)
    waiting = ticketed('EXEC-BIG', amount=BIG, price=1250.0).json()
    assert CLIENT.post('/book/approve', json={'ticket': waiting['ticket'], 'actor': ACTOR}
                       ).json()['status'] == 'pending'

    listed = worklist(actor=DESK_TWO)
    assert (listed['seat'], listed['counts']['pending']) == (DESK_TWO, 1)
    row = listed['pending'][0]
    assert {key: row[key] for key in ('what', 'key', 'reference', 'deal_path', 'client',
                                      'agreement', 'booker', 'tier', 'price', 'currency')} == {
        'what': "EXEC-BIG for Client A Ltd awaits a second seat's approval",
        'key': waiting['ticket'], 'reference': 'EXEC-BIG', 'deal_path': waiting['deal_path'],
        'client': 'Client A Ltd', 'agreement': CLIENT_SET, 'booker': ACTOR, 'tier': 'desk',
        'price': 1250.0, 'currency': structures.reporting_currency(json.loads(desk.read_text()))}
    assert "{!r} booked it, which the 'desk' tier refuses".format(ACTOR) in row['waits_on']
    risk = worklist(actor=RISK_SEAT)
    assert risk['seat'] == RISK_SEAT and set(risk['counts'].values()) == {0}, risk['counts']
    nobody = worklist()
    assert nobody['seat'] is None and nobody['pending'][0]['key'] == waiting['ticket']

    assert CLIENT.post('/book/approve', json={'ticket': waiting['ticket'], 'actor': DESK_TWO}
                       ).json()['status'] == 'approved'
    assert worklist(actor=DESK_TWO)['counts']['pending'] == 0


def test_a_booker_reads_why_its_ticket_waits_on_the_position(recorded, desk):
    """A BOOKER SEES WHY ITS TICKET WAITS. A seat that books and cannot approve files 250,000 rand,
    which the hub signs under the automatic tier, and 5,000,000, which waits under four eyes. Its
    worklist lists nothing to sign; its positions read each ticket's tier and, on the pending one,
    the tier's sentence naming why - as the approver reads them, whose worklist row says the same.
    Restruck to 300,000 rand - 5,550,000 dollars, past the cap - the first trade's clip reads the
    restrike's ticket under the four-eyes tier and waits too. A seat acting only at another node
    sees no position. Signed by the second seat, a ticket waits on nothing; refused by it, it
    stands rejected and waits on nothing either, as does a ticket booked before any tier stood.
    Each ticket says what it fell past: 5,000,000 rand is 92,500,000 dollars at 18.5 on this
    book's axis, past the automatic tier's 5,000,000 cap, and the restrike 300,000 x 18.5 =
    5,550,000; the 250,000 fell in the first tier read, past nothing, and the trade booked before
    any tier stood says nothing at all.

    Killing mutations: a restruck clip keeping the tier its fill fell in, which reads the restrike
    as signed automatically while it waits; the tier's sentence asked of every ticket whatever
    its status, which reads the refused restrike as waiting on an approval and the ticket booked
    before the tiers as wanting a second seat under a tier named None; and the route's sentences
    filed only where no tier admits the ticket, which reads the pending one as falling to four
    eyes for no reason.
    """
    booker, elsewhere = 'subject-booker', 'subject-desk-fx'
    with_client(desk)
    early = ticketed('EXEC-EARLY').json()
    declare(recorded, policy.TIERS_POLICY, tiers(
        {'name': 'auto', 'max_notional': {'amount': BIG, 'currency': 'USD'}},
        {'name': 'desk', 'four_eyes': True}))
    entitle(recorded, [(ACTOR, 'admin', '*'), (ACTOR, 'validate', '*'), (DESK_TWO, 'approve', '*'),
                       (booker, 'book', 'spine-desk'), (booker, 'validate', 'spine-desk'),
                       (elsewhere, 'book', 'spine-desk/FX')])
    small = ticketed('EXEC-SMALL', actor=booker).json()
    big = ticketed('EXEC-BIG', amount=BIG, actor=booker).json()

    def tickets(seat):
        return {row['reference']: [{key: value for key, value in entry.items() if key != 'at'}
                                   for entry in row['tickets']]
                for row in CLIENT.get('/book/positions', params={'actor': seat}).json()[
                    'positions']}

    def ticket(booked, execution, tier, status, waits_on, by=None, actor=booker, fell_past=()):
        return [{'quantity': 1.0, 'instrument': booked['recorded']['instrument'],
                 'execution_reference': execution, 'fill': booked['recorded']['lsn'],
                 'ticket': (by or booked)['ticket'], 'tier': tier,
                 'fell_past': None if fell_past is None else list(fell_past),
                 'lsn': (by or booked)['recorded']['lsn'], 'actor': actor, 'status': status,
                 'waits_on': waits_on}]

    def capped(dollars):
        return ["the 'auto' tier caps max_notional at 5000000 USD and this ticket is {} of it - "
                "book it under a tier that admits the size, or split the clip".format(dollars)]

    waits = ("no verdict is filed against this ticket, and the 'desk' tier wants a seat other "
             "than the booker to sign it")
    before = ticket(early, 'EXEC-EARLY', None, 'unticketed', None, actor=ACTOR, fell_past=None)
    assert CLIENT.get('/book/worklist', params={'actor': booker}).json()['pending'] == []
    assert tickets(booker) == tickets(DESK_TWO) == {
        'EXEC-EARLY': before,
        'EXEC-SMALL': ticket(small, 'EXEC-SMALL', 'auto', 'approved', None),
        'EXEC-BIG': ticket(big, 'EXEC-BIG', 'desk', 'pending', waits,
                           fell_past=capped(92_500_000.0))}
    (row,) = CLIENT.get('/book/worklist', params={'actor': DESK_TWO}).json()['pending']
    assert (row['key'], row['tier'], row['fell_past'], row['waits_on']) == (
        big['ticket'], 'desk', capped(92_500_000.0), waits)
    assert tickets(elsewhere) == {}

    amended = CLIENT.post('/book/deals', content=dump({
        'action': 'amend', 'deal_path': small['deal_path'], 'fields': {'Amount': 300_000.0},
        'actor': booker}), headers=JSON).json()
    assert amended['tier']['name'] == 'desk', amended
    assert CLIENT.post('/book/approve', json={'ticket': big['ticket'], 'actor': DESK_TWO}
                       ).json()['status'] == 'approved'
    assert tickets(booker) == {
        'EXEC-EARLY': before,
        'EXEC-SMALL': ticket(small, 'EXEC-SMALL', 'desk', 'pending', waits, by=amended,
                             fell_past=capped(5_550_000.0)),
        'EXEC-BIG': ticket(big, 'EXEC-BIG', 'desk', 'approved', None,
                           fell_past=capped(92_500_000.0))}
    assert CLIENT.post('/book/reject', json={'ticket': amended['ticket'], 'actor': DESK_TWO,
                                             'reason': 'over the line'}
                       ).json()['status'] == 'rejected'
    assert tickets(booker)['EXEC-SMALL'] == ticket(small, 'EXEC-SMALL', 'desk', 'rejected', None,
                                                   by=amended, fell_past=capped(5_550_000.0))


def test_a_booking_s_own_answer_says_what_its_ticket_fell_past(recorded, quoting):
    """A BOOKER READS WHY ON THE ANSWER. Under an automatic tier capped at 100,000 dollars and a
    four-eyes tier past it, a direct booking of 250,000 rand - 4,625,000 dollars at 18.5 on this
    book's axis - its restrike to 300,000 - 5,550,000 - and the acceptance of a collar on a million
    dollars each answer their `tier` with the four-eyes tier's name and, as `fell_past`, the
    automatic tier's cap sentence at that size, exactly what the position's ticket reads back.

    Killing mutation: the answer's tier left without the route's sentences, which sends a booker to
    read the position back to learn why its ticket waits.
    """
    declare(recorded, policy.TIERS_POLICY, tiers(
        {'name': 'auto', 'max_notional': {'amount': SMALL, 'currency': 'USD'}},
        {'name': 'desk', 'four_eyes': True}))

    def capped(dollars):
        return ["the 'auto' tier caps max_notional at 100000 USD and this ticket is {} of it - "
                "book it under a tier that admits the size, or split the clip".format(dollars)]

    def tickets():
        return {entry['ticket']: entry['fell_past']
                for row in CLIENT.get('/book/positions').json()['positions']
                for entry in row['tickets']}

    booked = ticketed('EXEC-1').json()
    for act, dollars in ((lambda: booked, 4_625_000.0),
                         (lambda: restruck(booked['deal_path'], Amount=300_000.0), 5_550_000.0),
                         (lambda: accept(quote_of('ZeroCostCollar', COLLAR,
                                                  netting_set=CLIENT_SET)).json(), 1_000_000.0)):
        answer = act()
        assert answer['tier'] == {'name': 'desk', 'four_eyes': True, 'approval_lsn': None,
                                  'fell_past': capped(dollars)}, answer
        assert tickets()[answer['ticket']] == capped(dollars)


def test_a_deal_booked_again_as_it_was_is_the_trade_the_record_holds(recorded, desk):
    """A TICKET IS THE TRADE: the plan it leaves and its own fields. A delete records nothing, so a
    deal deleted and booked again as it was - its reference, quantity and price - is the trade the
    record never lost: answered `booked` at the fill it has, the file taking it back and nothing
    filed, signed as it was. Booked again under that reference at another quantity it is refused
    by name; under a reference of its own at that quantity, on the plan the first left, it is a new
    ticket, PENDING under four eyes; and a position reads the worst of its clips, so a clip signed
    after it hides nothing.

    Killing mutations: the trade left out of the ticket, which hands a second execution on the same
    plan the approval over the first; a trade the file lost answered without putting it back, which
    leaves the file behind the record for good; and the position read off its latest ticket alone,
    which reads it approved once a later clip is signed.
    """
    from derivus_spine import oracle

    def deleted(booked):
        assert CLIENT.post('/book/deals', content=dump({
            'action': 'delete', 'deal_path': booked['deal_path']}), headers=JSON).json()['written']

    with_client(desk)
    declare(recorded, policy.TIERS_POLICY, tiers({'name': 'desk', 'four_eyes': True}))
    first = ticketed('EXEC-1', amount=1_000.0).json()
    CLIENT.post('/book/approve', json={'ticket': first['ticket'], 'actor': DESK_TWO})
    deleted(first)
    again = ticketed('EXEC-1', amount=1_000.0).json()
    assert again['written'] is True and again['booked'] == {
        'lsn': first['recorded']['lsn'], 'deal_path': first['deal_path']}, again
    assert len(facts(recorded, 'fill')) == 1 and standing_of('EXEC-1') == ('approved', 0.0)
    assert CLIENT.get('/book/reconcile').json()['in_record_not_in_file'] == []
    assert 'its own reference' in ticketed('EXEC-1', amount=1_000.0, quantity=100.0).json()[
        'detail']
    deleted(again)
    more = ticketed('EXEC-1', amount=1_000.0, quantity=100.0, execution_reference='EXEC-1B').json()
    assert more['ticket'] != first['ticket'] and 'no verdict is filed' in more['waits_on']
    assert standing_of('EXEC-1') == ('pending', 100.0), 'one clip\'s approval cleared a second'
    third = ticketed('EXEC-1', amount=1_000.0, quantity=2.0, execution_reference='EXEC-1C').json()
    CLIENT.post('/book/approve', json={'ticket': third['ticket'], 'actor': DESK_TWO})
    assert standing_of('EXEC-1') == ('pending', 100.0), 'a signed clip hid one nobody signed'
    assert oracle.failed(oracle.report(recorded)) == []


def test_a_trade_sent_again_is_answered_where_it_stands(recorded, desk):
    """A TRADE IS TAKEN ONCE. A client that never saw its answer sends the same booking again after
    it landed: the fill standing under its key - its instrument and execution reference - with the
    fields it states is answered `booked` by name, where it landed, and nothing is spliced or
    filed, even once the trade is restruck; sent again at another quantity it is refused by name, a
    second execution wanting its own reference. A restrike sent again after it landed is answered
    the same way.

    Killing mutations: the check dropped, which books the trade sent again as a second clip; the
    fields ignored, which answers a changed quantity as already booked; a clip's fields read at the
    restrike governing it, which refuses the booking sent again once it is restruck; and a restrike
    that landed filed again, which refuses it as terms that did not move.
    """
    with_client(desk)
    first = ticketed('EXEC-1', amount=1_000.0).json()
    before, at = desk.read_bytes(), head(recorded)
    landed = {'lsn': first['recorded']['lsn'], 'deal_path': first['deal_path']}
    again = ticketed('EXEC-1', amount=1_000.0).json()
    assert again['written'] is False and again['booked'] == landed, again
    assert again['note'].startswith("execution 'EXEC-1' is on the record at LSN")
    assert desk.read_bytes() == before and head(recorded) == at, 'a trade sent again moved the book'
    other = ticketed('EXEC-1', amount=1_000.0, quantity=2.0).json()['detail']
    assert 'quantity 2.0 where it filed 1 ' in other and 'its own reference' in other, other
    assert desk.read_bytes() == before and head(recorded) == at

    amended = restruck(first['deal_path'], Amount=5_000.0)
    before, at = desk.read_bytes(), head(recorded)
    assert ticketed('EXEC-1', amount=1_000.0).json()['booked'] == landed
    sent = restruck(first['deal_path'], Amount=5_000.0)
    assert sent['written'] is False and sent['booked'] == {
        'lsn': amended['recorded']['lsn'], 'deal_path': first['deal_path']}, sent
    assert desk.read_bytes() == before and head(recorded) == at, 'a restrike sent again moved it'


def test_a_retry_lands_every_event_on_the_lsn_it_already_has(recorded, desk):
    """THE RECORD TAKES A TRADE BEFORE THE FILE DOES, so a retry is how a lost file write is healed,
    and a retry files nothing. A booking under an automatic tier whose file write was lost - the
    book file put back as it was - sent again is answered at the fill the record holds, the file
    taking it back: one clip, one node, one approval. A restrike retried the same way is answered
    at the restrike that landed, the file taking back the terms it became.

    Killing mutations: a trade the file lost answered without putting it back, which leaves the
    file behind the record for good; a restrike that landed filed again rather than answered; and
    the tier the route stamped read as a field the booking sent again restates, which refuses it.
    """
    from derivus_spine import oracle

    with_client(desk)
    declare(recorded, policy.TIERS_POLICY, tiers({'name': 'auto'}))
    lost = desk.read_bytes()
    booked = ticketed('EXEC-R').json()
    desk.write_bytes(lost)
    again = ticketed('EXEC-R').json()
    assert again['written'] is True and again['booked'] == {
        'lsn': booked['recorded']['lsn'], 'deal_path': booked['deal_path']}, again
    lost = desk.read_bytes()
    amended = restruck(booked['deal_path'], Amount=500_000.0)
    desk.write_bytes(lost)
    again = restruck(booked['deal_path'], Amount=500_000.0)
    assert again['written'] is True and again['booked'] == {
        'lsn': amended['recorded']['lsn'], 'deal_path': booked['deal_path']}, again
    assert [len(facts(recorded, kind)) for kind in ('fill', 'amendment', 'approval')] == [1, 1, 2]
    reconciled = CLIENT.get('/book/reconcile').json()
    assert reconciled['in_record_not_in_file'] == []
    assert oracle.failed(oracle.report(recorded)) == []


def test_a_retried_acceptance_lands_every_event_on_the_lsn_it_already_has(recorded, quoting):
    """THE RECORD'S OWN IDEMPOTENCY HOLDS FOR AN ACCEPTANCE, its ticket being the plan it leaves the
    book at and the fill's own fields. Refused by a cap no tier admits it under, accepted again,
    and accepted once more after the desk lifts the cap, one quote files ONE `quote_filed`; and
    where the files the act wrote are lost after the record took it - the book and the pending
    file put back as they were - the retry lands the hub's approval and the fill on the LSNs they
    already have and the book takes the trade once.

    Killing mutation: the head put back in the ticket, which files the quote again at every retry.
    """
    declare(recorded, policy.TIERS_POLICY, tiers(
        {'name': 'capped', 'max_notional': {'amount': 1.0, 'currency': 'USD'}}))
    quote = quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET)
    pending = quoting.parent / 'tmp' / (quote['quote_id'] + '.json')
    refused = accept(quote).json()
    assert refused['written'] is False and refused['refused'], refused
    assert accept(quote).json()['accepted'] == refused['accepted'], 'a retry filed the quote again'
    declare(recorded, policy.TIERS_POLICY, tiers({'name': 'auto'}))
    lost = quoting.read_bytes(), pending.read_bytes()
    booked = accept(quote).json()
    assert booked['written'] is True and booked['accepted'] == refused['accepted'], booked
    acts = typed(recorded, 'quote_filed', 'approval', 'fill')
    assert [kind for _, kind in acts] == ['quote_filed', 'approval', 'fill']
    landed = json.loads(quoting.read_text())

    quoting.write_bytes(lost[0])
    pending.write_bytes(lost[1])
    retried = accept(quote).json()
    assert retried['written'] is True and retried['recorded']['lsn'] == acts[-1][0], retried
    assert typed(recorded, 'quote_filed', 'approval', 'fill') == acts, 'a retry filed a second fact'
    held = booked['deal_path'].rpartition('/')[0]
    assert len(deal_at(json.loads(quoting.read_text()), held)['Children']) == len(
        deal_at(landed, held)['Children']), 'the book took the trade twice'
    assert pending_of(quoting.parent, quote)['booked']['lsn'] == acts[-1][0]
    assert verify_home(recorded)['events'] == head(recorded)


def test_every_refusal_a_booking_meets_comes_before_the_hub_signs(recorded, quoting):
    """AN UNTAMPERED HUB NEVER FAILS ITS OWN ORACLE. Under an automatic tier the hub signs a ticket
    in its own voice and books it next, so everything that could refuse the booking is asked
    FIRST: a node nobody declared, a quantity or a price that is not a number, an execution
    reference that is not a name, a seat holding no `book` there - on a direct booking and on an
    acceptance alike, and an acceptance naming a `portfolio` its quote did not - each refused by
    name with no approval standing, while a clean booking signs and lands.

    Killing mutations: the dry authorization skipped, which signs a ticket the append then refuses -
    an approval the oracle names as the writer's voice forged; and a stale acceptance `portfolio`
    ignored, which books where the quote said while the caller named somewhere else.
    """
    from derivus_spine import oracle

    declare(recorded, policy.TIERS_POLICY, tiers({'name': 'auto'}))
    spine.declare_portfolio('spine-desk/FX')
    entitle(recorded, [(ACTOR, verb, '*') for verb in ('admin', 'book', 'validate')])
    for case in ({'portfolio': 'spine-desk/Fx'}, {'quantity': '1'}, {'quantity': True},
                 {'execution_reference': 12345}, {'price': '1.5'},
                 {'actor': 'subject-stranger'}):
        answer = ticketed('EXEC-BAD', **case)
        assert answer.status_code == 422, (case, answer.text)
    quote = quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET, portfolio='spine-desk/Fx',
                     actor=ACTOR)
    assert 'a node declared under it' in accept(quote, actor=ACTOR).json()['detail']
    stale = accept(quote, actor=ACTOR, portfolio='spine-desk/FX').json()['detail']
    assert "the 'spine-desk/FX' this one names is refused" in stale, stale
    assert facts(recorded, 'approval') == facts(recorded, 'fill') == facts(
        recorded, 'quote_filed') == []
    assert ticketed('EXEC-GOOD').json()['written'] is True
    assert len(facts(recorded, 'approval')) == len(facts(recorded, 'fill')) == 1
    assert oracle.failed(oracle.report(recorded)) == []


#: A deliverable FX forward: its tenor is the day it settles, and it names no expiry or maturity.
FORWARD = {'Object': 'FXForwardDeal', 'Reference': 'FWD', 'Buy_Currency': 'USD',
           'Buy_Amount': AMOUNT, 'Buy_Discount_Rate': 'USD', 'Sell_Currency': 'ZAR',
           'Sell_Amount': AMOUNT * SPOT, 'Sell_Discount_Rate': 'ZAR'}


def test_a_trade_is_as_long_as_the_last_day_its_declarations_name(recorded, desk):
    """A TENOR IS READ OFF DECLARATIONS: the latest day a type declares it settles on, expires or
    matures on (`schema.tenor_fields`). A one-year forward books under a five-year tenor cap and
    one settling in six years is refused by it by name - the forward names no expiry or maturity,
    so its settlement IS its tenor.

    Killing mutation: the tenor read off the fields named for an expiry or a maturity alone, which
    dates a forward by nothing and refuses every forward under every tenor cap.
    """
    with_client(desk)
    declare(recorded, policy.TIERS_POLICY, tiers({'name': 'short', 'max_tenor_years': 5.0}))

    def forward(years):
        return CLIENT.post('/book/deals', content=dump({
            'action': 'add', 'deal': dict(FORWARD, Reference='FWD-{}Y'.format(years),
                                          Settlement_Date=BASE + pd.DateOffset(years=years)),
            'parent_reference': CLIENT_SET, 'quantity': 1.0,
            'execution_reference': 'EXEC-{}Y'.format(years), 'actor': ACTOR}), headers=JSON).json()

    assert forward(1)['written'] is True
    refused = forward(6)
    assert refused['written'] is False and any('caps max_tenor_years' in one
                                               for one in refused['refused']), refused


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
    refused = CLIENT.post('/book/approve', json={'quote_id': quote['quote_id'], 'actor': ACTOR})
    assert refused.status_code == 422 and 'accept it first' in refused.json()['detail']

    waiting = declare(recorded, policy.TIERS_POLICY, tiers({'name': 'desk', 'four_eyes': True}))
    assert accept(quote).json()['written'] is True, 'the tier held the booking back'
    second = quote_of('ZeroCostCollar', dict(COLLAR, notional=AMOUNT / 2),
                      netting_set=CLIENT_SET)
    assert accept(second).json()['written'] is True

    path = tmp_path / 'tmp' / (quote['quote_id'] + '.json')
    filed = json.loads(path.read_text())
    elsewhere = json.loads((tmp_path / 'tmp' / (second['quote_id'] + '.json')).read_text())
    read = spine.quote_at(filed['accepted']['lsn'], quote['quote_id'])
    assert read['quote_id'] == quote['quote_id'] and read['ticket'] == filed['accepted']['ticket']

    # the second quote's own position: a real `quote_filed`, and not this quote's
    filed['accepted']['lsn'] = elsewhere['accepted']['lsn']
    path.write_text(json.dumps(filed, indent=2), newline='\n')
    other = CLIENT.post('/book/reject', json={
        'quote_id': quote['quote_id'], 'actor': ACTOR, 'reason': 'nothing doing'})
    assert other.status_code == 422
    assert second['quote_id'] in other.json()['detail'] and 'and not' in other.json()['detail']

    filed['accepted']['lsn'] = waiting['lsn']
    path.write_text(json.dumps(filed, indent=2), newline='\n')
    wrong_type = CLIENT.post('/book/reject', json={
        'quote_id': quote['quote_id'], 'actor': ACTOR, 'reason': 'nothing doing'})
    assert wrong_type.status_code == 422 and 'in this record and not the quote' in \
        wrong_type.json()['detail']

    # and a position past the head is the record saying so, not a traceback
    filed['accepted']['lsn'] = 10_000
    path.write_text(json.dumps(filed, indent=2), newline='\n')
    beyond = CLIENT.post('/book/approve',
                         json={'quote_id': quote['quote_id'], 'actor': ACTOR})
    assert beyond.status_code == 422 and 'no LSN 10000' in beyond.json()['detail']


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
    assert service.pin_of(service.BOOK.read()[0]) is None


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

    Killing mutations: the instrument address taken over the terms less their `Reference`, which
    reconciles a renamed deal clean and hides the one edit a desk is most likely to make by hand;
    and skipping the subtree of a node marked `Ignore`, which is a node somebody booked whether or
    not the engine prices it.
    """
    assert book_one('RECONCILE')['written'] is True
    clean = CLIENT.get('/book/reconcile').json()
    assert clean['events_behind'] == clean['positions_behind'] == 0
    assert clean['lsn'] == head(recorded)
    assert clean['in_record_not_in_file'] == clean['in_file_not_in_record'] == []

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
    assert lost['in_file_not_in_record'] == []

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
    assert answer['in_file_not_in_record'] == []

    # a POLICY past the pin is not drift: it moves the events and no position
    declare(recorded, policy.FIXINGS_POLICY, {'sources': {'FxRate.ZAR': ['ECB']}})
    after = CLIENT.get('/book/reconcile').json()
    assert (after['events_behind'], after['positions_behind']) == (2, 1)
    assert len(after['in_record_not_in_file']) == 1, 'a policy moved a position'


def test_a_partial_unwind_leaves_the_file_and_the_record_agreeing(recorded, booking):
    """THEY MEET WHERE THE BOOK PRICES. A trade unwound by half against the node that books it is
    one node in the file and two fills on the record, the node priced at the net the record holds,
    so the two agree - and closed out they still do. What they disagree about is a node of those
    terms under a client's set the record holds them nowhere under, which prices as written, and a
    trade the file lost, named by the reference on the record's own copy of its terms and by its
    client.

    Killing mutations: the file's nodes counted against the record's fills, which raises a
    disagreement on every partial unwind; the reconcile keyed on the instrument alone, which reads
    the copy under another client's set as the trade booked; and a lost trade named off the file
    alone, which names it nothing.
    """
    def filled(quantity, execution, **body):
        answer = CLIENT.post('/book/deals', content=dump(dict(
            body, quantity=quantity, execution_reference=execution)), headers=JSON).json()
        assert answer['written'] is True, answer
        return answer

    def disagreements():
        return {name: rows for name, rows in CLIENT.get('/book/reconcile').json().items()
                if isinstance(rows, list) and rows}

    booked = filled(1.0, 'EXEC-UNWOUND', action='add', parent_reference=CLIENT_SET,
                    deal=dict(CASHFLOW, Reference='UNWOUND'))
    path = booked['deal_path']
    filled(-0.5, 'EXEC-HALF', action='fill', deal_path=path, reference='UNWOUND')
    held = next(row for row in CLIENT.get('/book/positions').json()['positions']
                if row['reference'] == 'UNWOUND')
    assert (held['quantity'], held['clips'], held['deal_paths']) == (0.5, 2, [path])
    assert disagreements() == {}, 'a partial unwind reads as the file and the record disagreeing'
    filled(-0.5, 'EXEC-CLOSE', action='fill', deal_path=path, reference='UNWOUND')
    assert disagreements() == {}, 'a trade closed out reads as a disagreement'

    copied = deal_at(json.loads(booking.read_text()), path)
    document = json.loads(booking.read_text())
    document['Calc']['Deals']['Deals']['Children'].append(
        netting_set('CLIENT_B', 'CPTY_B', [copied['Instrument']['.Deal']]))
    booking.write_text(json.dumps(document, indent=2), newline='\n')
    assert disagreements() == {'in_file_not_in_record': [{
        'deal_path': '1/0', 'reference': 'UNWOUND', 'instrument': booked['recorded']['instrument'],
        'netting_set': 'CLIENT_B', 'client': 'CPTY_B'}]}, 'the copy reconciled as the booked trade'

    kept = filled(1.0, 'EXEC-KEPT', action='add', parent_reference=CLIENT_SET,
                  deal=dict(CASHFLOW, Reference='KEPT'))
    rewritten(booking, lambda node: node['Children'].pop())
    lost = disagreements()['in_record_not_in_file']
    assert [(row['reference'], row['client'], row['netting_set'], row['quantity'])
            for row in lost] == [('KEPT', 'CPTY_A', CLIENT_SET, 1.0)], lost
    assert lost[0]['instrument'] == kept['recorded']['instrument']


# --------------------------------------------------------------------------------------------
# The record's own readings - the strip, the markets, and the verdict a banner reads off them.

#: Every field one line of the strip carries: the envelope, the sentence the service tells it in,
#: who did it and what it opens.
STRIP_FIELDS = {'lsn', 'record_time', 'effective_time', 'actor', 'event_type', 'book', 'summary',
                'by', 'deal_path', 'agreement', 'entity', 'changed'}


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


#: A reconcile answer's two position lists where none has been fetched - what the banner knows
#: before it asks, which is nothing about what the two hold.
NO_LISTS = {'in_record_not_in_file': [], 'in_file_not_in_record': []}


def verdict(pinned, answer):
    """The banner's verdict, mirrored from `web/src/spine.ts`.

    THE LISTS DECIDE AND THE COUNTS NEVER DO: every write to the book file re-pins it at the head,
    the write that is the other half of a divergence included, so a verdict gated on the counts is
    told `clean` about a trade the desk deleted through its own verb. DRIFT IS WHAT THE RECORD
    MOVING FORWARD CANNOT EXPLAIN - a deal the file holds that nobody booked, or a trade the record
    holds and the file has LOST under a pin that has seen every position there is.
    """
    if pinned is None:
        return 'none'
    lists = answer or NO_LISTS
    lost = len(lists['in_record_not_in_file'])
    if lists['in_file_not_in_record'] \
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
    """The activity read is the record's own sequence with nothing left out: the fold opens no
    body, so every type is a line, its summary the declared sentence for it where the service
    tells it no other - a booking it names by its reference, client and agreement.

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
    assert all(row['summary'] == projections.SUMMARIES[row['event_type']]
               for row in whole['rows'][:-2])
    assert [row['summary'] for row in whole['rows'][-2:]] == [
        'booked 250000 of {} for CPTY_A under {}, execution EXEC-{}'.format(
            reference, CLIENT_SET, reference) for reference in ('ONE', 'TWO')]
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
    """The markets read is the fold's own answer: the close standing per market, the day it is
    for and the LSN of that day's close it RESTATED - a close is superseded by a new close rather
    than corrected in place - beside the names a values vector was declared under and the
    snapshots registered.

    Killing mutation: the superseded LSN dropped, after which a restated close is indistinguishable
    from a first one and nothing on the screen says the day was marked twice.
    """
    vector = stored(recorded, b'{"EURUSD":1.0851}')
    restated = stored(recorded, b'{"EURUSD":1.0857}')
    filed(recorded, 'market_declared', {'name': 'official', 'values_hash': vector})
    first = filed(recorded, 'official_close_declared',
                  {'market': 'official', 'values_hash': vector, 'date': '2026-08-26'})
    second = filed(recorded, 'official_close_declared',
                   {'market': 'official', 'values_hash': restated, 'date': '2026-08-26'})
    snapshot = stored(recorded, b'{"surface":"the vol cube"}')
    log = opened(recorded)
    try:
        log.own('snapshot_registered', {'blob': snapshot}, book=CLIENT_SET)
    finally:
        log.close()

    answer = CLIENT.get('/book/markets').json()
    assert answer['lsn'] == head(recorded)
    assert answer['closes'] == [{'market': 'official', 'values_hash': restated, 'lsn': second,
                                 'date': '2026-08-26', 'supersedes_lsn': first,
                                 'effective_time': None}]
    assert answer['names'] == [{'name': 'official', 'values_hash': vector, 'actor': ACTOR,
                                'effective_time': None, 'lsn': first - 1}]
    assert answer['snapshots'] == [{'blob': snapshot, 'book': CLIENT_SET, 'lsn': head(recorded)}]


def test_the_record_reads_are_a_404_on_a_box_that_records_nothing(unrecorded, desk):
    """A reading refuses nothing and a desk that records nothing is a STATE rather than an
    error: the three reads 404 in the service's own sentence naming the variable, the status block
    is null, and the banner's verdict is `none`.

    Killing mutation: any read answering an empty strip, markets list or positions list, which tells
    a desk that keeps no record that its record holds nothing.
    """
    assert spine.configured() is False
    for path in ('/book/activity', '/book/markets', '/book/positions'):
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
    assert answer['in_file_not_in_record'] == []
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
    assert answer['in_file_not_in_record'] == []
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
    """The strip and the markets need no DOCUMENT, so they answer on a box serving no book - the
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
    a fault: the strip still reads, each row in the fold's own sentence where its body will not
    open, and every other read that opens one answers the spine's own sentence as a 422.

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
    'EquityPriceVol.EQ': {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
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
    """GATE 11. Prints of one `(index, date, source)`: the plan compiled at the earlier LSN is
    the earlier print's and the plan compiled now is the restatement's, so an auditor recompiles AT
    an LSN and gets that day's answer rather than today's - a third print, backdated behind the
    restatement and filed last, standing under it.

    Killing mutation: `fixings_at` ordering by LSN alone, so a print backdated behind the one in
    force wins merely by arriving last (109.0 where 111.25 stands) and every plan ever recompiled
    moves with it.
    """
    declare(recorded, policy.FIXINGS_POLICY, {'sources': {INDEX: ['EXCHANGE']}})
    first = observe(recorded, 108.5, effective_time='2024-06-27T16:00:00.000000Z')['lsn']
    observe(recorded, 111.25, effective_time='2024-06-27T17:30:00.000000Z')
    observe(recorded, 109.0, effective_time='2024-06-27T17:00:00.000000Z')

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


def test_a_print_filed_through_the_service_answers_the_row_a_close_waits_on(recorded, desk):
    """`POST /book/lifecycle` files a print, an election or a ruling through the writer. The
    barrier option's expiry fixing keeps its day's close waiting until the print filed through the
    verb answers it; a consequence is refused by name; and a seat holding `mark` over the book
    alone - a print being the firm's - is refused in the writer's own words, the denial landed and
    nothing printed.

    Killing mutation: the verb filing under the deployment's own seat rather than the one named,
    which prints what a seat the document never scoped for it asked for.
    """
    declare(recorded, policy.FIXINGS_POLICY, {'sources': {INDEX: ['EXCHANGE']}})
    desk.write_text(json.dumps(json.loads(dump(barrier_book(108.5))), indent=2), newline='\n')
    day = (BASE + pd.DateOffset(days=365)).strftime('%Y-%m-%d')

    def filed(event_type, actor=ACTOR, **body):
        return CLIENT.post('/book/lifecycle', content=dump(
            {'event_type': event_type, 'body': body, 'actor': actor}), headers=JSON)

    def waiting():
        return [(row['index'], row['source']) for row in CLIENT.get(
            '/book/close/check', params={'date': day}).json()['outstanding']
            if row['kind'] == 'fixing' and row['due_date'] == day]

    assert waiting() == [(INDEX, None)]
    entitle(recorded, [(ACTOR, verb, '*') for verb in ('admin', 'mark', 'validate')] + [
        (DESK_TWO, 'mark', 'spine-desk')])
    printed = filed('fixing_observed', index=INDEX, date=day, source='EXCHANGE', value=112.0)
    assert printed.json() == {'recorded': {'lsn': head(recorded)}, 'event_type': 'fixing_observed'}
    assert waiting() == []

    knocked = filed('knock', instrument='f' * 64)
    assert knocked.status_code == 422 and 'apply_lifecycle does not file' in knocked.text
    refused = filed('fixing_observed', DESK_TWO, index=INDEX, date=day, source='EXCHANGE',
                    value=113.0)
    assert refused.status_code == 422 and 'holds no mark scope' in refused.json()['detail']
    assert facts(recorded, 'capability_denied')[-1][2] == {
        'subject': DESK_TWO, 'verb': 'mark', 'book': '*', 'attempted_type': 'fixing_observed'}
    assert [body['value'] for _, _, body in facts(recorded, 'fixing_observed')] == [112.0]


def test_a_price_that_is_not_positive_is_no_print(recorded, desk):
    """`POST /book/lifecycle` refuses by name a print of zero or below of an index whose factor
    family declares its level positive - `FxRate.EUR`, `EquityPrice.EQ`, `PriceIndex.CPI` - with
    nothing landed on the log, and files the same of an `InterestRate`, whose level may be either,
    and a zero `CommodityPrice`, which declares itself not positive.

    Killing mutations: the check removed, which lands a zero the fill reads as no level under an FX
    divisor and as an unfixed cell under any other index; any of the four declarations deleted.
    """
    def filed(index, value):
        return CLIENT.post('/book/lifecycle', content=dump({'event_type': 'fixing_observed', 'body': {
            'index': index, 'date': '2024-06-28', 'source': 'EXCHANGE', 'value': value},
            'actor': ACTOR}), headers=JSON)

    before = head(recorded)
    for index, value in itertools.product(('FxRate.EUR', INDEX, 'PriceIndex.CPI'), (0.0, -1.0)):
        refused = filed(index, value)
        assert refused.status_code == 422 and index in refused.json()['detail'], refused.text
    assert head(recorded) == before
    landed = [('InterestRate.USD', 0.0), ('InterestRate.USD', -0.5), ('CommodityPrice.GOLD', 0.0)]
    assert [filed(*print_).status_code for print_ in landed] == [200, 200, 200]
    assert [(body['index'], body['value']) for _, _, body in facts(recorded, 'fixing_observed')] == landed


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
    the mark names bytes that are not the values the book carries; a `mark` check written into the
    endpoint in its own words, which refuses the seat with no denial on the record; and the owner
    rule absent from `declare_close`, which plants an official close in a namespace the same seat
    was refused one call earlier.
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
    unscoped = CLIENT.post('/book/markets', json={'name': 'dealer', 'actor': ACTOR})
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


def test_under_a_document_the_queue_asks_before_the_executor_does(recorded, desk):
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
    whose grant has since been withdrawn is refused at the queue rather than served them. A request
    naming no seat under a document is refused by name, the deployment's own seat - which does
    price - being the poll paths' alone: a job nobody signed for is one the record could not
    attribute.

    Killing mutations: the check taken after `queue.put`, which pays for the run and then refuses
    it; the check taken only where the tuple is new, after which a seat asking for numbers the
    store already holds is served them; an unnamed request falling back on the environment's seat,
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

    # the deployment's own seat prices when a request names it, which is the grant that keeps the
    # desk ticking - and a request naming nobody is not taken for it
    deployment, ran = priced('deployment', actor=ACTOR)
    assert deployment.status_code == 200 and ran, deployment.text
    unnamed, ran = priced('unnamed', actor=None)
    assert unnamed.status_code == 422 and not ran
    assert 'no actor for this act' in unnamed.json()['detail']
    assert spine.SPINE_ACTOR in unnamed.json()['detail']
    assert verify_home(recorded)['events'] == head(recorded)


def test_the_scope_the_queue_asks_for_is_the_seats_own_and_the_hub_attests(recorded):
    """WHAT A STANDING RUN IS ADMITTED UNDER IS `mark` OVER ITS BOOK, asked of the REQUEST's seat:
    numbers a fact is about to cite are a mark of that book, and a seat scoped only to validate is
    turned away before the Monte Carlo rather than after it. A `mark`-only seat is admitted, and
    the attestation is the hub's - filed in the writer's own voice, the seat having asked and the
    hub having run it - so a seat's grant over a book never reaches a fact only the hub may file.
    Under a document a request naming no seat is refused by name, the deployment's own being the
    poll paths' alone.

    Killing mutations: one verb asked for every lane, which either stops every what-if on a booking
    desk or lets a validate-only seat mint attestations; the attestation filed under the requester,
    which a seat could then file for numbers the hub never ran; and an unnamed request falling back
    on the deployment's seat, which admits every anonymous caller as the hub.
    """
    assert (spine.STANDING_TYPE, spine.SETTLEMENT_EXPORT) == (
        'run_completed', policy.DESIGNATED_PROCESSES[0])
    entitle(recorded, [(ACTOR, verb, '*') for verb in ('admin', 'validate')]
            + [(DESK_TWO, 'mark', 'spine-desk')])
    standing = dict(own_job('admission'), lane=spine.STANDING)

    refused = CLIENT.post('/execute', content=dump(dict(standing, actor=ACTOR)), headers=JSON)
    assert refused.status_code == 422, refused.text
    assert "{!r} holds no mark scope over 'spine-desk'".format(ACTOR) in refused.json()['detail']
    assert facts(recorded, 'capability_denied')[0][2]['attempted_type'] == spine.STANDING_TYPE
    anonymous = CLIENT.post('/execute', content=dump(standing), headers=JSON)
    assert anonymous.status_code == 422 and 'no actor for this act' in anonymous.json()['detail']
    assert facts(recorded, 'run_completed') == []

    ran = submit(standing, actor=DESK_TWO)
    assert drained(ran)['status'] == 'done', ran
    assert len(facts(recorded, 'run_completed')) == 1, 'the admitted standing run attested nothing'
    log = opened(recorded)
    try:
        assert [frame['actor'] for frame in log.frames()
                if frame['event_type'] == 'run_completed'] == ['writer']
    finally:
        log.close()
    assert verify_home(recorded)['events'] == head(recorded)


def test_the_poll_paths_own_jobs_are_admitted_under_the_deployments_seat(recorded, quoting):
    """THE POLL PATH NAMES NOBODY. A diary compile and a beat of the cadence are TELEMETRY jobs no
    request signed, so the seat the queue admits them under is `DV_SPINE_ACTOR` - the deployment's
    own, which is the whole of what the metronome has - and a document that does not scope that seat
    stops them BY NAME rather than silently, which is the one failure a ticking desk must be told
    about. Neither is run here: what is asserted is which seat the queue asked about.

    Killing mutation: the poll path queueing its jobs under no seat and a job carrying none
    admitted unconditionally, which leaves the whole poll path outside the document a desk
    declared.
    """
    service.BOOK.diaries.entries.clear()
    entitle(recorded, [(ACTOR, verb, '*') for verb in ('admin', 'validate')])
    assert CLIENT.get('/book/diary').status_code == 200, 'the compile was refused its own seat'

    # not cleared: admission is asked before the cache is read, so the warm one refuses too
    entitle(recorded, [(ACTOR, 'admin', '*')])
    for refused in (CLIENT.get('/book/diary').json()['detail'],
                    str(pytest.raises(spine.SpineRefused, service.submit_bloomberg, {}).value)):
        assert ACTOR in refused and 'validate' in refused, refused
    assert facts(recorded, 'capability_denied')[-1][2]['attempted_type'] == spine.TELEMETRY
    assert verify_home(recorded)['events'] == head(recorded)


def test_the_consolidated_risk_and_the_explain_ask_the_queue_s_question(recorded, desk):
    """THE TWO READS THAT VALUE ON THE REQUEST THREAD ASK WHAT THE QUEUE ASKS. `/book/risk` values
    the book on a miss and `/book/pnl?explain=` three times, neither through the queue, so each
    asks the request's seat a what-if's question - `validate` over the book - before it values:
    the refusal is a 422 in the record's own words, both reads land on ONE `capability_denied`,
    and the cold cache stays empty. A RISK ANSWER ALREADY CACHED IS ASKED ABOUT AGAIN: the desk's
    own read warms it and the stranger is still refused. The seat granted `validate` over the book
    reads both - the risk off the cache, the explain valuing the position the marks held - and a
    request naming no seat under a document is refused by name. A P&L read to now marks the book
    as it stands on this thread too, so it is asked the same question; one to a marked end is not.

    Killing mutations: the admission asked after the cache is read, which serves the stranger the
    answer the desk's read warmed; the explain's admission dropped, which lets an explain to a
    marked end past it for a seat holding nothing; and the read to now admitted, which marks the
    book for one.
    """
    with_client(desk)
    declare(recorded, policy.TIERS_POLICY, tiers({'name': 'auto'},
                                                 designations={'pnl': 'official'}))
    assert ticketed('EXEC-RISK').json()['written'] is True
    for path in ('/book/close', '/book/marks'):
        answer = CLIENT.post(path, content=dump({'actor': ACTOR}), headers=JSON)
        assert answer.status_code == 200, answer.text
    assert drained(answer.json())['status'] == 'done'
    entitle(recorded, [(ACTOR, verb, '*') for verb in ('admin', 'validate')])
    before = head(recorded)

    def risk(**params):
        return CLIENT.get('/book/risk', params=params)

    def explained(**params):
        return CLIENT.get('/book/pnl', params=dict(explain='true', **params))

    cold = risk(actor=STRANGER)
    assert cold.status_code == 422 and not service.BOOK.risk_cache.entries, cold.text
    said = cold.json()['detail']
    assert STRANGER in said and 'validate' in said and 'spine-desk' in said, said
    denials = facts(recorded, 'capability_denied')
    assert [row[2] for row in denials] == [{'subject': STRANGER, 'verb': 'validate',
                                            'book': 'spine-desk',
                                            'attempted_type': spine.CURIOSITY}]
    assert head(recorded) == before + 1, 'a refused read filed more than the refusal'

    warm = risk(actor=ACTOR)
    assert warm.status_code == 200, warm.text
    assert service.BOOK.polls.get(('risk',) + service.poll_key(service.BOOK)) == warm.json(), (
        'the poll kept nothing a stranger could be served')
    for again in (risk(actor=STRANGER), explained(actor=STRANGER, end='2024-06-28'),
                  CLIENT.get('/book/pnl', params=dict(actor=STRANGER))):
        assert again.status_code == 422 and STRANGER in again.json()['detail'], again.text
    assert facts(recorded, 'capability_denied') == denials, 'a repeated refusal is a second fact'
    for unnamed in (risk(), explained(), CLIENT.get('/book/pnl')):
        assert unnamed.status_code == 422 and 'no actor for this act' in unnamed.json()['detail']

    entitle(recorded, [(ACTOR, verb, '*') for verb in ('admin', 'validate')]
            + [(STRANGER, 'validate', 'spine-desk')])
    assert risk(actor=STRANGER).json() == warm.json()
    told = explained(actor=STRANGER)
    assert told.status_code == 200, told.text
    assert told.json()['rows'] and told.json()['explain']['carry'] is not None, told.json()
    assert verify_home(recorded)['events'] == head(recorded)


def test_the_tick_and_the_verification_queue_under_the_seat_that_asked(recorded, quoting):
    """A POSTED TICK AND A VERIFICATION ARE THE CALLER'S; the metronome's beat alone is the
    deployment's. Under a document granting the hub's seat `validate`, a tick posted for a seat
    holding `mark` alone - the market's own question passed - is refused naming it, the denial's
    attempted type the TELEMETRY lane, and so is a verification queued for it, whose job names no
    book and so asks over `*`; a tick or a
    verification naming no seat is refused by name rather than run as the hub; and the executor's
    store is exactly what it was. The beat, which names no seat, is still admitted under
    `DV_SPINE_ACTOR` and runs, refusing the unprovisioned home before it opens a session. The
    verification is queued through `submit_verify`, since its verb refuses a box with no terminal
    before it asks.

    Killing mutations: the helper ignoring its `actor`, which queues the stranger's verification
    under the hub; and the verb passing the hub's seat, which admits the stranger's tick.
    """
    entitle(recorded, [(ACTOR, verb, '*') for verb in ('admin', 'validate', 'mark')]
            + [(STRANGER, 'mark', '*')])
    service.EXECUTOR.queue.join()
    before = dict(service.EXECUTOR.results)

    posted = CLIENT.post('/book/bloomberg', content=dump({'actor': STRANGER}), headers=JSON)
    assert posted.status_code == 422, posted.text
    for said in (posted.json()['detail'], str(pytest.raises(
            spine.SpineRefused, service.submit_verify, {}, actor=STRANGER).value)):
        assert STRANGER in said and 'validate' in said, said
    assert [row[2] for row in facts(recorded, 'capability_denied')] == [
        {'subject': STRANGER, 'verb': 'validate', 'book': 'spine-desk',
         'attempted_type': spine.TELEMETRY}]
    unnamed = CLIENT.post('/book/bloomberg', content=dump({}), headers=JSON)
    assert unnamed.status_code == 422 and 'no actor for this act' in unnamed.json()['detail']
    assert 'no actor for this act' in str(
        pytest.raises(spine.SpineRefused, service.submit_verify, {}).value)
    assert dict(service.EXECUTOR.results) == before, 'a refused trip reached the queue'

    metronome = service.Metronome(60.0)
    metronome.beat()
    service.EXECUTOR.queue.join()
    beat = CLIENT.get('/results/{}'.format(metronome.pending)).json()
    assert beat['status'] == 'done', beat
    assert 'a routine tick never provisions' in beat['stats']['Bloomberg']['refused'][0]
    assert verify_home(recorded)['events'] == head(recorded)


# --------------------------------------------------------------------------------------------
# The paper's set, reads by node, and the seat a token proves.

def agreed(agreement, entity):
    """An entity and an agreement with it on the record through the service, its terms the
    netting set `netting_set` spells for that entity."""
    assert CLIENT.post('/book/entities', content=dump({'entity': entity, 'name': entity}),
                       headers=JSON).status_code == 200
    answer = CLIENT.post('/book/agreements', content=dump(
        {'agreement': agreement, 'entity': entity, 'kind': 'ISDA 2002',
         'terms': netting_set(agreement, entity)['Instrument']['.Deal']}), headers=JSON)
    assert answer.status_code == 200, answer.text


def surviving(path, *counterparties):
    """The book with a survival curve per counterparty, which a netting set naming one prices on."""
    document = json.loads(path.read_text())
    factors = document['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors']
    for counterparty in counterparties:
        factors['SurvivalProb.' + counterparty] = json.loads(dump({
            'Recovery_Rate': 0.4, 'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.2]])}))
    path.write_text(json.dumps(document, indent=2), newline='\n')


def test_the_first_booking_under_a_declared_agreement_brings_its_set(recorded, desk):
    """THE SET IS THE AGREEMENT'S MATERIALISATION, so the first booking naming a declared agreement
    whose set the file does not carry brings it - spliced at the root out of the terms legal
    declared, the booking sitting under it - and every later one books under the set that is
    there. What the set itself wants that the book lacks - its counterparty's survival curve - is
    newly said, so it refuses by name with the file untouched. `/book/reconcile` holds the file's
    set to the declared paper: a field moved is named, and a balance - settlement state - is not.

    Killing mutations: every booking splicing the set again, which gives the second booking a
    second set; and the balance read as paper, which names every set a balance was written into.
    """
    surviving(desk, 'CPTY_A')
    agreed('ISDA-A', 'CPTY_A')
    agreed('ISDA-Z', 'CPTY_Z')
    before = desk.read_bytes()
    wants = ticketed('EXEC-Z', agreement='ISDA-Z', parent_reference=None).json()
    assert wants['written'] is False and wants['refused'] == [
        'no market data for SurvivalProb.CPTY_Z'], wants
    assert desk.read_bytes() == before, 'a refused booking brought its set anyway'

    for execution in ('EXEC-1', 'EXEC-2'):
        booked = ticketed(execution, agreement='ISDA-A', parent_reference=None).json()
        assert booked['written'] is True, booked
    held = [path for path, node in service.netting_sets(json.loads(desk.read_text()))
            if node['Instrument']['.Deal']['Reference'] == 'ISDA-A']
    assert len(held) == 1, 'a second booking brought a second set'
    assert [(fill['agreement'], fill['counterparty']) for _, _, fill in facts(recorded, 'fill')] \
        == [('ISDA-A', 'CPTY_A')] * 2
    assert CLIENT.get('/book/reconcile').json()['terms_mismatch'] == []

    document = json.loads(desk.read_text())
    written = deal_at(document, held[0])['Instrument']['.Deal']
    written.update(Opening_Balance=500_000.0)
    desk.write_text(json.dumps(document, indent=2), newline='\n')
    assert CLIENT.get('/book/reconcile').json()['terms_mismatch'] == [], \
        'settlement state read as paper'
    written.update(Liquidation_Period=10.0)
    desk.write_text(json.dumps(document, indent=2), newline='\n')
    assert CLIENT.get('/book/reconcile').json()['terms_mismatch'] == [{
        'agreement': 'ISDA-A', 'deal_path': held[0], 'fields': ['Liquidation_Period']}]


def test_a_corrected_agreement_is_the_paper_every_run_prices_from_the_event_that_recorded_it(
        recorded, desk):
    """THE PAPER IS THE RECORD'S. The first booking under a declared agreement brings its set into
    the file; legal then corrects the received threshold from out of reach to nothing, and the
    set's CVA falls with it - every run compiles the paper standing at its own position in the
    record over the file's copy, the set's children, balance and holdings' amounts kept, a book
    holding nothing under it included - while the file keeps the paper it was spliced with, which
    `/book/reconcile` names. The book compiled at the position before the correction is the first
    paper's: its xVA job is the first run's and values to the first number.

    Killing mutations: the paper not written over the file's set, under which the second CVA is the
    first; the compile returning the file where only the paper moved, which hands the book holding
    nothing the first paper.
    """
    desk.write_text(json.dumps(json.loads(dump(job(factors=dict(FACTORS, **EQUITY), sections={
        'Price Models': {'GBMAssetPriceModel.EQ': {'Vol': 0.25, 'Drift': 0.0}},
        'Model Configuration': {'.ModelParams': {
            'modeldefaults': {'EquityPrice': 'GBMAssetPriceModel'}, 'modelfilters': {}}}}))),
        indent=2), newline='\n')
    surviving(desk, 'LEI-A')
    assert CLIENT.post('/book/entities', content=dump({'entity': 'LEI-A', 'name': 'Client A'}),
                       headers=JSON).status_code == 200

    def declared(threshold):
        terms = dict(netting_set('ISDA-A', 'LEI-A')['Instrument']['.Deal'], Collateralized='True',
                     Collateral_Assets={'Cash_Collateral': [{'Currency': 'USD'}]})
        terms['Credit_Support_Amounts'] = dict(terms['Credit_Support_Amounts'], Received_Threshold={
            '.CreditSupportList': [[1, threshold]]})
        answer = CLIENT.post('/book/agreements', content=dump({
            'agreement': 'ISDA-A', 'entity': 'LEI-A', 'kind': 'ISDA 2002 with CSA',
            'terms': terms}), headers=JSON)
        assert answer.status_code == 200, answer.text

    def xva():
        CLIENT.post('/book/xva', content=dump({'netting_sets': ['ISDA-A']}), headers=JSON)
        service.EXECUTOR.queue.join()
        row = {row['reference']: row for row in CLIENT.get('/book/xva').json()['sets']}['ISDA-A']
        assert row['status'] == 'done', row
        return row

    declared(1e9)
    booked = ticketed('EXEC-A', agreement='ISDA-A', parent_reference=None,
                      deal=dict(VANILLA, Reference='OPT-A')).json()
    assert booked['written'] is True, booked
    first, before = xva(), head(recorded)
    declared(0.0)
    second = xva()
    assert 0.0 < second['cva'] < first['cva'], (first['cva'], second['cva'])
    assert CLIENT.get('/book/reconcile').json()['terms_mismatch'] == [{
        'agreement': 'ISDA-A', 'deal_path': '1', 'fields': ['Credit_Support_Amounts']}]

    document = json.loads(desk.read_text())
    pinned = spine.compiled_job(document, lsn=before, strict=False, runs='CreditMonteCarlo')
    (_, node), = service.netting_sets(pinned)
    run = service.xva_document(pinned, node, 'LEI-A')
    assert derivus.content_hash(run) == first['plan_hash'], 'the pinned book is not the first one'
    assert float(service.load(run).run_job()[1]['Results']['cva']) == first['cva']

    written = deal_at(document, '1')['Instrument']['.Deal']
    written['Opening_Balance'] = 5.0
    written['Collateral_Assets']['Cash_Collateral'][0]['Amount'] = 3.0
    unheld = dict(document, Calc=dict(document['Calc'], Deals=dict(
        document['Calc']['Deals'], Reference='another-desk')))
    for compiled in (service.priced(document), spine.compiled_job(unheld)):
        compiled = deal_at(compiled, '1')
        terms = compiled['Instrument']['.Deal']
        assert (terms['Credit_Support_Amounts']['Received_Threshold'], terms['Opening_Balance'],
                terms['Collateral_Assets']['Cash_Collateral'][0]['Amount'],
                len(compiled['Children'])) == ({'.CreditSupportList': [[1, 0.0]]}, 5.0, 3.0, 1)


def test_the_set_s_counterparty_is_the_agreement_s_entity(recorded, desk):
    """THE COUNTERPARTY IS STATED ONCE, as the agreement's entity. Terms naming another are refused
    by name with nothing appended; terms naming none are declared naming the entity, as
    `/book/agreements` reads them back, and so is the set the first booking brings - a later
    booking under it naming no agreement is taken, the entity on its fill as on the first's.

    Killing mutations: the entity left out of the terms, which refuses the booking naming no
    agreement on a set that names no counterparty; the refusal dropped, which declares terms
    naming another counterparty with the entity written over it.
    """
    surviving(desk, 'LEI-A')
    assert CLIENT.post('/book/entities', content=dump({'entity': 'LEI-A', 'name': 'Client A'}),
                       headers=JSON).status_code == 200

    def declared(counterparty):
        return CLIENT.post('/book/agreements', content=dump({
            'agreement': CLIENT_SET, 'entity': 'LEI-A', 'kind': 'ISDA 2002',
            'terms': netting_set(CLIENT_SET, counterparty)['Instrument']['.Deal']}), headers=JSON)

    head_before = head(recorded)
    refused = declared('CPTY_A')
    assert refused.status_code == 422 and "name counterparty 'CPTY_A' and the agreement is with " \
        "'LEI-A'" in refused.json()['detail'], refused.text
    assert head(recorded) == head_before, 'a refused declaration recorded something'

    assert declared('').status_code == 200
    assert [row['terms']['Credit_Support_Amounts']['Counterparty']
            for row in CLIENT.get('/book/agreements').json()['agreements']] == ['LEI-A']
    for execution, named in (('EXEC-PAPER', {'agreement': CLIENT_SET}), ('EXEC-SET', {})):
        booked = ticketed(execution, **named).json()
        assert booked.get('written') is True, booked
    assert [fill['counterparty'] for _, _, fill in facts(recorded, 'fill')] == ['LEI-A'] * 2


def test_a_quote_for_a_declared_agreement_brings_its_set_where_the_client_accepts(recorded,
                                                                                  quoting):
    """THE QUOTE PATH MATERIALISES AS THE BOOKING DOES: a quote named for a declared agreement
    whose set the file lacks prices on its own copy with the set spliced in - its pins and ticket
    the book as the acceptance will leave it - and moves nothing, and the acceptance splices the
    same set, under which the mirror books.

    Killing mutation: the quote verb left to refuse a set the file lacks, which makes a direct
    booking the only way to open a declared agreement.
    """
    surviving(quoting, 'CPTY_B')
    agreed('ISDA-B', 'CPTY_B')
    before = quoting.read_bytes()
    quote = quote_of('ZeroCostCollar', COLLAR, netting_set='ISDA-B')
    assert quoting.read_bytes() == before, 'a quote moved the book'
    booked = accept(quote).json()
    assert booked['written'] is True, booked
    held = [path for path, node in service.netting_sets(json.loads(quoting.read_text()))
            if node['Instrument']['.Deal']['Reference'] == 'ISDA-B']
    assert len(held) == 1 and booked['deal_path'].startswith(held[0] + '/')


#: A seat reading one node of the book.
DESK_FX = 'subject-desk-fx'


def test_a_seat_reads_the_rows_under_the_nodes_it_acts_at(recorded, desk):
    """READS BY NODE. A seat granted any verb at a node - `validate`, or `approve` alone - reads
    the positions, the diary and the reconcile rows under it and nothing beside it; one granted
    `validate` over `*` reads the whole book, and so does a read no seat signs on a box checking no
    token - the deployment's own view. Money and the strip are the book's, so a node seat reads
    neither. This is PRESENTATION while one class key opens every body, which the page says.

    Killing mutation: the positions filtered by the book their envelope names rather than the node
    they sit at, which shows a node seat nothing at all.
    """
    with_client(desk)
    for path in ('spine-desk/FX', 'spine-desk/Rates'):
        spine.declare_portfolio(path)
    for execution, node in (('CF-FX', 'spine-desk/FX'), ('CF-RATES', 'spine-desk/Rates')):
        assert ticketed(execution, portfolio=node).json()['written'] is True

    def read(path, seat=None):
        answer = CLIENT.get(path, params={} if seat is None else {'actor': seat})
        assert answer.status_code == 200, answer.text
        return answer.json()

    def references(seat=None):
        return sorted(row['reference'] for row in read('/book/positions', seat)['positions'])

    assert references() == references(DESK_FX) == ['CF-FX', 'CF-RATES'], 'no document narrows'
    entitle(recorded, [(ACTOR, verb, '*') for verb in ('admin', 'book', 'validate', 'settle')] + [
        (DESK_FX, 'validate', 'spine-desk/FX'), (RATES_SEAT, 'approve', 'spine-desk/Rates')])
    fx = next(row['instrument'] for row in read('/book/positions')['positions']
              if row['reference'] == 'CF-FX')
    assert CLIENT.post('/book/transition', content=dump({
        'subject': fx, 'status': 'settled', 'amount': -100.0, 'asset': 'ZAR', 'kind': 'fee',
        'reference': 'FEE-FX', 'value_date': '2024-06-28', 'actor': ACTOR}),
        headers=JSON).status_code == 200
    assert references(DESK_FX) == ['CF-FX'] and references(ACTOR) == ['CF-FX', 'CF-RATES']
    assert references(RATES_SEAT) == ['CF-RATES'], 'a seat reads nothing it may approve'
    assert references() == ['CF-FX', 'CF-RATES'], 'an unnamed read is the deployment\'s view'
    assert {row['instrument'] for row in read('/book/diary', DESK_FX)['rows']} == {fx}
    assert len({row['instrument'] for row in read('/book/diary', ACTOR)['rows']}) == 3
    assert read('/book/cash', DESK_FX)['movements'] == []
    assert [row['reference'] for row in read('/book/cash', ACTOR)['movements']] == ['FEE-FX']

    document = json.loads(desk.read_text())
    document['Calc']['Deals']['Deals']['Children'][-1]['Children'].clear()
    desk.write_text(json.dumps(document, indent=2), newline='\n')
    lost = {seat: [row['instrument'] for row in read('/book/reconcile', seat)[
        'in_record_not_in_file']] for seat in (DESK_FX, ACTOR)}
    assert len(lost[ACTOR]) == 2 and lost[DESK_FX] == [fx]
    assert read('/book/activity', DESK_FX)['rows'] == [] and read('/book/activity', ACTOR)['rows']
    assert read('/book/activity', DESK_FX)['lsn'] == head(recorded), 'the cursor stood still'


def test_a_seat_reads_what_it_may_do_and_where(recorded, desk):
    """WHAT A SEAT MAY DO, read before it acts. With no capabilities document every seat holds every
    verb everywhere, null throughout, and a request naming none acts as the deployment's seat.
    Under one, a seat granted `validate` over the book and `book` at its FX node reads exactly
    those two and nowhere for the other seven, a second seat `approve` everywhere - and the acts
    agree: the FX seat's booking at its node lands, and at the book's root is refused. A request
    naming no seat is refused as each of its acts would be.

    `demands` IS THE MAP IN FORCE, act -> the verb it demands: the vocabulary's until a document
    moves an act, then the document's. Where one moves the approval to `mark` the read says so,
    and the acts agree: the seat holding `approve` is refused the signature naming `mark`, and the
    seat holding `mark` where the ticket books signs it.

    Killing mutations: the verbs read for the seat the deployment runs as rather than the one
    asking, which hands the FX seat the deployment's `admin` and `validate` everywhere; and
    `demands` read off the vocabulary rather than the document in force, which tells a screen
    `approve` signs where the record asks `mark`.
    """
    with_client(desk)
    verbs = ('validate', 'book', 'approve', 'mark', 'document', 'settle', 'confirm', 'collateral',
             'admin')
    asked = {'fill': 'book', 'amendment': 'book', 'election': 'book', 'approval': 'approve',
             'rejection': 'approve', 'official_close_declared': 'mark', 'result_pinned': 'mark',
             'entity_declared': 'document', 'agreement_declared': 'document',
             'status_transition': 'settle', 'confirmed': 'confirm', 'collateral': 'collateral',
             'margin': 'collateral', 'portfolio_declared': 'admin'}

    def seat(actor=None):
        return CLIENT.get('/book/seat', params={} if actor is None else {'actor': actor})

    def read(actor=None):
        answer = seat(actor).json()
        assert sorted(answer) == ['demands', 'seat', 'verbs'], answer
        return answer['seat'], answer['verbs'], {act: answer['demands'][act] for act in asked}

    assert read(DESK_FX) == (DESK_FX, dict.fromkeys(verbs), asked)
    assert read() == (ACTOR, dict.fromkeys(verbs), asked)
    grants = [(ACTOR, 'admin', '*'), (ACTOR, 'validate', '*'), (DESK_TWO, 'approve', '*'),
              (DESK_FX, 'validate', 'spine-desk'), (DESK_FX, 'book', 'spine-desk/FX'),
              (RISK_SEAT, 'mark', 'spine-desk/FX')]
    entitle(recorded, grants)
    assert read(DESK_FX) == (DESK_FX, dict(
        dict.fromkeys(verbs, []), validate=['spine-desk'], book=['spine-desk/FX']), asked)
    assert read(DESK_TWO) == (DESK_TWO, dict(dict.fromkeys(verbs, []), approve=None), asked)
    unnamed = seat()
    assert (unnamed.status_code, unnamed.json()['detail']) == (
        422, spine.UNNAMED.format(spine.SPINE_ACTOR))

    booked = ticketed('CF-FX', portfolio='spine-desk/FX', actor=DESK_FX).json()
    assert booked['written'] is True, booked
    root = ticketed('CF-ROOT', actor=DESK_FX)
    assert root.status_code == 422 and 'holds no book scope' in root.json()['detail'], root.text

    entitle(recorded, grants, verbs={'approval': 'mark'})
    assert read(DESK_TWO) == (DESK_TWO, dict(dict.fromkeys(verbs, []), approve=None),
                              dict(asked, approval='mark'))

    def sign(actor):
        return CLIENT.post('/book/approve', json={'ticket': booked['ticket'], 'actor': actor})

    refused = sign(DESK_TWO)
    assert refused.status_code == 422 and 'holds no mark scope' in refused.json()['detail']
    assert sign(RISK_SEAT).status_code == 200


def test_a_read_naming_its_seat_needs_no_deployment_seat_where_nothing_is_enforced(recorded, desk,
                                                                                   monkeypatch):
    """A READ NEVER RUNS UNDER ITS REQUESTER, and on a home declaring no capabilities document it
    needs no seat at all: the worklist, the diary and the positions answer a seat named on the read
    with `DV_SPINE_ACTOR` unset, the diary compile being the hub's own cache. Under a document the
    compile runs under the deployment's seat, which every poll path already requires.

    Killing mutation: the read's compile admitted under the seat the read names, which files that
    seat's denial of a read under a document.
    """
    with_client(desk)
    assert ticketed('CF-READ').json()['written'] is True
    monkeypatch.delenv('DV_SPINE_ACTOR')
    for path in ('/book/worklist', '/book/diary', '/book/positions'):
        answer = CLIENT.get(path, params={'actor': DESK_FX})
        assert answer.status_code == 200, (path, answer.text)
    monkeypatch.setenv('DV_SPINE_ACTOR', ACTOR)
    entitle(recorded, [(ACTOR, verb, '*') for verb in ('admin', 'validate')] + [
        (DESK_FX, 'approve', 'spine-desk')])
    for path in ('/book/worklist', '/book/diary', '/book/positions'):
        answer = CLIENT.get(path, params={'actor': DESK_FX})
        assert answer.status_code == 200, (path, answer.text)
    assert facts(recorded, 'capability_denied') == [], 'a read filed a denial'


def asked(path, authorization=None):
    """A request as the service's token dependency meets it, for the path a gate names."""
    from starlette.requests import Request

    return Request({'type': 'http', 'method': 'GET', 'scheme': 'http', 'root_path': '',
                    'server': ('testserver', 80), 'path': path, 'query_string': b'',
                    'headers': [] if authorization is None else [
                        (b'authorization', authorization.encode('ascii'))]})


def test_a_bearer_token_names_the_seat_and_nothing_else_does(recorded, desk, tmp_path, idp,
                                                            monkeypatch):
    """THE SEAT FROM A TOKEN. Where the deployment names a key set every request carries an ID
    token its identity provider signed, verified here against that set: none, or a bad one, is a
    401 in the verifier's words; the token's subject is who a fact is filed under whatever the body
    leaves out, and a body naming another seat is refused. The doorbell carries a position and
    nothing else and asks for no token. A READ RUNS UNDER NO REQUESTER: the diary its compile needs
    is the hub's own, admitted under `DV_SPINE_ACTOR`, so a seat holding no `validate` reads what it
    acts on and files no denial. A key set that does not read is a named refusal, never a 500. With
    no key set named nothing is checked: the body's actor stands.

    Killing mutations: the dependency made synchronous, which sets the seat in a threadpool copy of
    the request's context the endpoint never sees - and the fill lands under `DV_SPINE_ACTOR`; and
    the read's compile run under its bearer, which refuses a seat that may only approve and files
    the denial of a read.
    """
    import time

    with_client(desk)
    assert ticketed('EXEC-OFF', actor='subject-said').json()['written'] is True
    keys = tmp_path / 'jwks.json'
    keys.write_text(json.dumps(idp.jwks))
    for name, value in (('DV_SPINE_JWKS', str(keys)),
                        ('DV_SPINE_ISSUER', test_spine_identity.ISSUER),
                        ('DV_SPINE_AUDIENCE', test_spine_identity.AUDIENCE)):
        monkeypatch.setenv(name, value)
    now = time.time()
    token = test_spine_identity.rs256(idp.rsa_key, test_spine_identity.claims(
        exp=now + 300.0, iat=now - 10.0, nbf=now - 10.0))
    signed = dict(JSON, authorization='Bearer ' + token)

    bare = CLIENT.get('/book')
    assert bare.status_code == 401 and 'Authorization: Bearer' in bare.json()['detail']
    head_, payload, signature = token.split('.')
    forged = CLIENT.get('/book', headers={'authorization': 'Bearer ' + '.'.join(
        [head_, payload, test_spine_identity.flip(signature)])})
    assert forged.status_code == 401 and 'signature' in forged.json()['detail']

    booked = CLIENT.post('/book/deals', headers=signed, content=dump({
        'action': 'add', 'deal': dict(CASHFLOW, Reference='EXEC-ON', Amount=250_000.0),
        'parent_reference': CLIENT_SET, 'quantity': 1.0, 'execution_reference': 'EXEC-ON'}))
    assert booked.json()['written'] is True, booked.text
    log = opened(recorded)
    try:
        actors = [frame['actor'] for frame in log.frames() if frame['event_type'] == 'fill']
    finally:
        log.close()
    assert actors == ['subject-said', test_spine_identity.SUBJECT], \
        'the fill is not filed under the token\'s subject'
    impostor = CLIENT.post('/book/deals', headers=signed, content=dump({
        'action': 'add', 'deal': dict(CASHFLOW, Reference='EXEC-NOT', Amount=250_000.0),
        'parent_reference': CLIENT_SET, 'quantity': 1.0, 'execution_reference': 'EXEC-NOT',
        'actor': ACTOR}))
    assert impostor.status_code == 422 and 'bearer token' in impostor.json()['detail']
    asyncio.run(service.bearer(asked(service.DOORBELL)))
    with pytest.raises(service.HTTPException):
        asyncio.run(service.bearer(asked('/book')))

    entitle(recorded, [(ACTOR, verb, '*') for verb in ('admin', 'book', 'validate')] + [
        (test_spine_identity.SUBJECT, 'approve', '*')])
    denials = len(facts(recorded, 'capability_denied'))
    for path in ('/book/diary', '/book/worklist', '/book/positions'):
        answer = CLIENT.get(path, headers=signed)
        assert answer.status_code == 200, (path, answer.text)
    assert len(facts(recorded, 'capability_denied')) == denials, 'a read filed a denial'
    assert spine.hub() == ACTOR, 'a thread no request started lost the deployment\'s seat'
    monkeypatch.setenv('DV_SPINE_JWKS', str(tmp_path / 'missing.json'))
    unread = CLIENT.get('/book', headers=signed)
    assert unread.status_code == 401 and 'does not read as a key set' in unread.json()['detail']


def test_a_read_advances_its_folds_across_bookings_and_homes(recorded, desk, tmp_path,
                                                             monkeypatch):
    """A READ ADVANCES THE SERVED BOOK'S FOLDS, AND THEY ARE ONE HOME'S. Each booking is in the
    next `GET /book/positions`; and a second home served in the same process, its head past where
    the first's pairs stand, answers its own positions - none - and never the first's.

    Killing mutations, both in `spine.advancing`: a held pair answered whatever position is asked,
    which leaves a booking out of the read after it; and the pair's genesis never compared with
    the log's, which seeds the second home's fold with the first home's positions.
    """
    with_client(desk)
    for count, execution in enumerate(('FOLD-1', 'FOLD-2'), 1):
        assert ticketed(execution).json()['written'] is True
        assert len(CLIENT.get('/book/positions').json()['positions']) == count

    other = tmp_path / 'other_home'
    init_home(other, ACTOR)
    monkeypatch.setenv('DV_SPINE_HOME', str(other))
    for node in range(head(recorded)):
        spine.declare_portfolio('other-desk/n{}'.format(node))
    assert head(other) > head(recorded), 'the second home never passed the first\'s pairs'
    assert CLIENT.get('/book/positions').json()['positions'] == []


def test_a_print_no_deal_observes_is_nothing_to_the_plan(recorded, desk):
    """A PLAN ANSWERS FOR THE INDICES ITS DEALS OBSERVE AND NO OTHERS. Under a `fixings` policy
    ordering another index, a print of one it does not order, on a book holding a position and no
    deal that declares an observation, is nothing to the compile: `/prepare` names the plan it
    named before the print.

    Killing mutation: the plan's compile asking for every print the record holds where its deals
    name no index (`fixings(lsn, indices=None)`), which refuses the plan over the index nobody
    ordered.
    """
    with_client(desk)
    declare(recorded, policy.FIXINGS_POLICY, {'sources': {'FxRate.ZAR': ['ECB']}})
    assert ticketed('CF-PLAIN').json()['written'] is True

    def plan():
        document = CLIENT.get('/book').json()['document']
        answer = CLIENT.post('/prepare', content=dump(document), headers=JSON)
        assert answer.status_code == 200, answer.text
        return answer.json()['plan_id']

    before = plan()
    printed = CLIENT.post('/book/lifecycle', content=dump({
        'event_type': 'fixing_observed', 'actor': ACTOR, 'body': {
            'index': 'FxRate.UNDECLARED', 'date': '2024-06-20', 'source': 'WM', 'value': 1.25}}),
        headers=JSON)
    assert printed.status_code == 200, printed.text
    assert plan() == before


#: Seats holding one verb each, for the writes the service asks a verb of itself.
VIEWER, MARKER, SETTLER = 'subject-validates', 'subject-marks', 'subject-settles'


def test_a_delete_asks_book_at_the_node_holding_the_position(recorded, desk):
    """A DELETE ASKS `book` WHERE THE POSITION SITS. It files nothing, so the service asks itself,
    at the deepest node holding the position: a seat holding `validate` alone is refused naming
    `book`, and so is one holding `book` at a sibling node, each with the file untouched and the
    denial landed; the seat holding `book` at the position's node deletes it.

    Killing mutations: the gate removed, which deletes for the seat holding `validate` alone; and
    the node holding the position never read, the delete asked anywhere under the book, which
    deletes for the seat at the sibling node.
    """
    with_client(desk)
    fx, rates = 'spine-desk/FX', 'spine-desk/Rates'
    for path in (fx, rates):
        spine.declare_portfolio(path)
    booked = ticketed('DEL-1', portfolio=fx).json()
    assert booked['written'] is True, booked
    entitle(recorded, [(ACTOR, 'admin', '*'), (ACTOR, 'validate', '*'), (VIEWER, 'validate', '*'),
                       (DESK_TWO, 'book', rates), (DESK_FX, 'book', fx)])

    def delete(seat):
        return CLIENT.post('/book/deals', json={
            'action': 'delete', 'deal_path': booked['deal_path'], 'actor': seat})

    before = desk.read_bytes()
    refused = {seat: delete(seat) for seat in (VIEWER, DESK_TWO)}
    assert {seat: answer.status_code for seat, answer in refused.items()} == {
        VIEWER: 422, DESK_TWO: 422}, {seat: answer.text for seat, answer in refused.items()}
    assert desk.read_bytes() == before, 'a refused delete moved the file'
    for seat, answer in refused.items():
        assert "holds no book scope over '{}'".format(fx) in answer.json()['detail'], seat
    assert [body for _, _, body in facts(recorded, 'capability_denied')] == [
        {'subject': seat, 'verb': 'book', 'book': fx, 'attempted_type': 'delete'}
        for seat in refused]
    deleted = delete(DESK_FX)
    assert deleted.status_code == 200 and deleted.json()['deleted'] == 'DEL-1', deleted.text


def test_a_path_save_asks_mark_over_the_book(recorded, desk):
    """`POST /paths` ASKS `mark` OVER THE BOOK SERVED. Where this box reads the files a job names
    decides the market the book prices on, and the save files nothing, so the service asks
    itself: a seat holding `validate` alone is refused naming `mark` with the maps as they were,
    and the seat holding `mark` over the book saves them.

    Killing mutation: the gate removed, which saves the maps for the seat holding `validate` alone.
    """
    entitle(recorded, [(ACTOR, 'admin', '*'), (VIEWER, 'validate', '*'),
                       (MARKER, 'mark', 'spine-desk')])
    maps = {'path_transform': {'/elsewhere': str(desk.parent)}, 'file_transform': {}}
    refused = CLIENT.post('/paths', json=dict(maps, actor=VIEWER))
    assert refused.status_code == 422, refused.text
    assert "holds no mark scope over 'spine-desk'" in refused.json()['detail']
    assert CLIENT.get('/paths').json()['path_transform'] == {}
    saved = CLIENT.post('/paths', json=dict(maps, actor=MARKER))
    assert saved.status_code == 200, saved.text
    assert CLIENT.get('/paths').json()['path_transform'] == maps['path_transform']


def test_a_close_asks_mark_over_the_firm_before_its_verdict_and_never_validate(recorded, desk):
    """`POST /book/close` ASKS `mark` OVER THE FIRM, FIRST. A seat holding `validate` alone is
    refused in the service's own sentence before anything is compiled - on a day the close is not
    legal too, so it is never told what the day waits on - and the seat holding `mark` over `*` and
    no `validate` declares the close, the diary it stands behind being the hub's. The verb asked
    is the one the declaration demands under the map in force: where a document moves the act to
    `approve`, the seat holding `approve` declares and the one holding `mark` is refused first.

    Killing mutations: the gate removed, which answers the seat holding `validate` alone with what
    the illegal day waits on; the diary compiled under the requester's seat, which refuses the
    marking seat for want of `validate`; and the gate asking `mark` whatever the map says, which
    refuses the seat the record would take.
    """
    with_client(desk)
    assert ticketed('CLOSE-1').json()['written'] is True
    assert CLIENT.post('/book/markets', json={'name': 'official', 'actor': ACTOR}).status_code == 200
    entitle(recorded, [(ACTOR, 'admin', '*'), (ACTOR, 'validate', '*'), (VIEWER, 'validate', '*'),
                       (MARKER, 'mark', '*')])

    def close(seat, day):
        return CLIENT.post('/book/close', json={'date': day, 'actor': seat})

    assert close(MARKER, '2026-12-31').status_code == 422, 'the cashflow of 2026 is unsettled'
    for day in ('2024-06-28', '2026-12-31'):
        refused = close(VIEWER, day)
        assert refused.status_code == 422, (day, refused.text)
        assert "holds no mark scope over '*', so no close is declared" in refused.json()['detail']
    assert facts(recorded, 'official_close_declared') == []
    declared = close(MARKER, '2024-06-28')
    assert declared.status_code == 200, declared.text
    assert [body['date'] for _, _, body in facts(recorded, 'official_close_declared')] == [
        '2024-06-28']

    entitle(recorded, [(ACTOR, 'admin', '*'), (ACTOR, 'validate', '*'), (VIEWER, 'approve', '*'),
                       (MARKER, 'mark', '*')], verbs={'official_close_declared': 'approve'})
    moved = close(MARKER, '2024-06-28')
    assert moved.status_code == 422, moved.text
    assert "holds no approve scope over '*', so no close is declared" in moved.json()['detail']
    assert close(VIEWER, '2024-06-28').status_code == 200


def test_a_settlement_file_asks_settle_over_the_book_and_never_validate(recorded, desk):
    """`POST /book/settlements` ASKS `settle` OVER THE BOOK. The file is struck and nothing is
    filed, so the service asks itself: a seat holding `validate` alone is refused naming `settle`,
    and the seat holding `settle` over the book and no `validate` is answered the file, the diary
    it reads being the hub's.

    Killing mutations: the gate removed, which strikes the file for the seat holding `validate`
    alone; and the diary compiled under the requester's seat, which refuses the settling seat for
    want of `validate`.
    """
    with_client(desk)
    assert CLIENT.post('/book/markets', json={'name': 'official', 'actor': ACTOR}).status_code == 200
    declare(recorded, policy.TIERS_POLICY, tiers(
        {'name': 'all', 'four_eyes': False}, designations={spine.SETTLEMENT_EXPORT: 'official'}))
    entitle(recorded, [(ACTOR, 'admin', '*'), (ACTOR, 'validate', '*'), (VIEWER, 'validate', '*'),
                       (SETTLER, 'settle', 'spine-desk')])

    def export(seat):
        return CLIENT.post('/book/settlements', json={'due_before': '2024-06-28', 'actor': seat})

    refused = export(VIEWER)
    assert refused.status_code == 422, refused.text
    assert "holds no settle scope over 'spine-desk'" in refused.json()['detail']
    assert facts(recorded, 'capability_denied')[-1][2] == {
        'subject': VIEWER, 'verb': 'settle', 'book': 'spine-desk',
        'attempted_type': 'settlement_export'}
    struck = export(SETTLER)
    assert struck.status_code == 200 and struck.json()['market']['name'] == 'official', struck.text


#: `DV_Service --tick`'s own start, in an interpreter of its own so its thread ends with it: the
#: cadence beats until two beats have failed, each warning it logs printed as it is said.
CADENCE = """
import logging, sys, time
from derivus import service
logging.basicConfig(level=logging.WARNING, stream=sys.stdout, format='%(message)s')
service.BOOK = service.Book(sys.argv[1])
ticker = service.Metronome(0.2).start()
ends = time.monotonic() + 60.0
while ticker.failures < 2 and time.monotonic() < ends:
    time.sleep(0.02)
"""


def test_a_tick_the_hub_may_not_mark_is_refused_and_the_cadence_lives(recorded, desk):
    """THE SCHEDULED TICK ASKS `mark` OF THE HUB'S OWN SEAT. Under a capabilities document granting
    `DV_SPINE_ACTOR` no `mark` over the book served, a beat submits nothing, the cadence logs one
    warning a beat naming the grant to declare, and its thread outlives the refusal to beat again;
    the beats' denials are one fact.

    Killing mutation: the gate removed from `Metronome.beat`, which goes on to submit the tick -
    refused on this book for declaring no bootstrapper, in a warning naming no grant.
    """
    import subprocess

    entitle(recorded, [(ACTOR, 'admin', '*'), (ACTOR, 'validate', '*')])
    ran = subprocess.run(
        [sys.executable, '-c', CADENCE, str(desk)], stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, universal_newlines=True, timeout=300,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    said = [line for line in ran.stdout.splitlines() if line.startswith('bloomberg tick')]
    denial = facts(recorded, 'capability_denied')
    assert [body for _, _, body in denial] == [{
        'subject': ACTOR, 'verb': 'mark', 'book': 'spine-desk',
        'attempted_type': 'bloomberg_tick'}], ran.stdout
    assert said[:2] == [
        "bloomberg tick failed ({} in a row): actor '{}' holds no mark scope over 'spine-desk', "
        "so no bloomberg tick is written. Declare a document granting ('{}', mark, 'spine-desk') "
        "through `DV_Spine grant --file`. The refusal is itself recorded at LSN {} - the book is "
        "untouched".format(beat, ACTOR, ACTOR, denial[0][0]) for beat in (1, 2)], ran.stdout


# --------------------------------------------------------------------------------------------
# The official calculation: declared under admin, judged by the engine, read by the credit run.

#: What a hub's credit run is declared to run at in these gates: two batches, its FVA's cost funded
#: at `FUND` and its benefit at the dollar curve the exposure deflates on.
OFFICIAL = {'Simulation_Batches': 2, 'Funding_Valuation_Adjustment': {
    'Funding_Cost_Interest_Curve': 'FUND', 'Funding_Benefit_Interest_Curve': 'USD'}}


def official(blocks, **body):
    """The calculation policy declared through the service's verb: its response."""
    return CLIENT.post('/book/calculations', content=dump(dict({'calculations': blocks}, **body)),
                       headers=JSON)


def test_the_official_calculation_is_judged_by_the_engine_before_the_record_takes_it(recorded,
                                                                                    desk):
    """`POST /book/calculations` declares the `calculation` policy and `GET` reads what stands with
    its address and position, nulls on a record declaring none. The engine judges each block
    against its own type's declarations before anything appends: a field `CreditMonteCarlo` does
    not declare, a value its field does not admit and `Object`, the type being the key and no
    field of it, are each a 422 in the engine's words, and a type no official run reads is one in
    the record's. None moves the head. No market is read: the desk's book carries no `FUND` curve
    and the block naming it lands. Under a capabilities document a seat holding no `admin` is
    refused by the record, the denial landed.

    Killing mutation: the judge dropped from `spine.declare_calculations`, which lands the
    misspelt field as policy.
    """
    started = head(recorded)
    assert CLIENT.get('/book/calculations').json() == {
        'calculations': None, 'blob': None, 'lsn': None}
    for blocks, said in (
            ({'CreditMonteCarlo': {'Simulation_Batchs': 2}},
             'CreditMonteCarlo declares no Simulation_Batchs - it declares '),
            ({'CreditMonteCarlo': {'Credit_Valuation_Adjustment': {'Calculate': 'Maybe'}}},
             "Calculate is 'Maybe', not one of Yes, No"),
            ({'CreditMonteCarlo': dict(OFFICIAL, Object='CreditMonteCarlo')},
             'CreditMonteCarlo declares no Object - it declares '),
            ({'BaseValuation': {'Greeks': 'First'}},
             "states a block for 'BaseValuation', which no official run reads")):
        refused = official(blocks)
        assert refused.status_code == 422 and said in refused.json()['detail'], refused.text
    assert head(recorded) == started, 'a refused declaration recorded something'

    declared = official({'CreditMonteCarlo': OFFICIAL})
    assert declared.status_code == 200, declared.text
    blob = declared.json()['blob']
    assert declared.json() == {'recorded': {'lsn': started + 1}, 'blob': blob}
    assert CLIENT.get('/book/calculations').json() == {
        'calculations': {'CreditMonteCarlo': OFFICIAL}, 'blob': blob, 'lsn': started + 1}
    assert facts(recorded, 'policy_declared')[-1] == (
        started + 1, 'policy_declared', {'policy': policy.CALCULATION_POLICY, 'blob': blob})

    entitle(recorded, [(ACTOR, 'admin', '*')])
    stranger = official({}, actor='subject-nobody')
    assert stranger.status_code == 422 and 'subject-nobody' in stranger.json()['detail']
    assert facts(recorded, 'capability_denied')[-1][2] == {
        'subject': 'subject-nobody', 'verb': 'admin', 'book': '*',
        'attempted_type': 'policy_declared'}
    assert CLIENT.get('/book/calculations').json()['lsn'] == started + 1


@pytest.fixture
def credit_hub(recorded, tmp_path):
    """A recording desk with a credit run to price: NS_A holds a bought call and NS_B one beside a
    sold put, worth either sign, so its FVA's benefit leg reads a real exposure. The book's own
    calculation funds both legs at `FUND` and states no batching."""
    either = [dict(VANILLA, Reference='OPT_B'),
              dict(VANILLA, Reference='PUT_B', Option_Type='Put', Buy_Sell='Sell')]
    yield xva_book(tmp_path, [netting_set('NS_A', 'CPTY_A', [dict(VANILLA, Reference='OPT_A')]),
                              netting_set('NS_B', 'CPTY_B', either)])
    service.BOOK = None


def stating(path, **calculation):
    """The book file at `path` stating `calculation` over its own block, a None dropping its key."""
    document = json.loads(path.read_text())
    block = dict(document['Calc']['Calculation'], **calculation)
    document['Calc']['Calculation'] = {key: value for key, value in block.items()
                                       if value is not None}
    path.write_text(json.dumps(document, indent=2), newline='\n')


def credit_runs(netting_sets=None):
    """A recalc drained: `{reference: (plan_hash, result_id, cva, fva)}` off the rows it landed."""
    asked = [entry['reference'] for entry in recalc(netting_sets)['queued']]
    rows = xva_rows()
    assert {rows[reference]['status'] for reference in asked} == {'done'}, rows
    return {reference: tuple(rows[reference][key] for key in (
        'plan_hash', 'result_id', 'cva', 'fva')) for reference in asked}


def drew(run):
    """`(Batch_Size, Simulation_Batches)` the run a row names reported drawing."""
    stats = CLIENT.get('/results/{}'.format(run[1])).json()['stats']
    return stats['Batch_Size'], stats['Simulation_Batches']


def test_a_credit_run_under_the_policy_is_the_run_of_the_same_block_written_in_the_book_file(
        credit_hub):
    """IDENTITY. The book file states two batches and an FVA costed at `FUND` with its benefit at
    the dollar curve, and no policy stands: the recalc is one plan per set. The file is put back to
    its own block - both legs at `FUND`, no batching - which is another plan and another FVA on
    NS_B, whose benefit leg is real. The policy then declares what the file stated, over that
    conflicting file: each set's plan, run, CVA and FVA are the first recalc's to the bit, the job
    compiled under the policy being the same document, so the same run by its address.

    Killing mutations: the policy not read (`official_calculation` answering the document's own
    block), where the third recalc is the second's; the file's block laid over the policy's, where
    NS_B funds its benefit at the file's `FUND`.
    """
    stating(credit_hub, **OFFICIAL)
    written = credit_runs()
    stating(credit_hub, Simulation_Batches=None, Funding_Valuation_Adjustment=FUNDED)
    own = credit_runs()
    assert official({'CreditMonteCarlo': OFFICIAL}).status_code == 200
    declared = credit_runs()

    assert declared == written and sorted(declared) == ['NS_A', 'NS_B']
    assert all(cva > 0.0 != fva for _, _, cva, fva in declared.values()), declared
    assert all(own[name][0] != declared[name][0] for name in own), 'the file change moved no plan'
    assert own['NS_B'][3] != declared['NS_B'][3], 'the benefit leg reads no exposure'
    assert drew(declared['NS_A']) == (service.XVA_BATCH_SIZE, 2)


def test_the_policy_stands_over_the_book_file_which_stands_where_the_policy_says_nothing(
        credit_hub):
    """PRECEDENCE, key by key. The file states a batch of 512 paths and the policy two batches and
    no size: the run draws two batches of 512, the file's size standing where the policy states
    none. The policy restated with a size of 256, both state it and the run draws 256, the
    policy's. One job each way: two plans, each reporting what it drew.

    Killing mutation: the file's block laid over the policy's, where the second run draws 512.
    """
    stating(credit_hub, Batch_Size=512)
    assert official({'CreditMonteCarlo': {'Simulation_Batches': 2}}).status_code == 200
    silent = credit_runs(['NS_A'])['NS_A']
    assert official({'CreditMonteCarlo': {'Simulation_Batches': 2, 'Batch_Size': 256}}
                    ).status_code == 200
    both = credit_runs(['NS_A'])['NS_A']

    assert (drew(silent), drew(both)) == ((512, 2), (256, 2))
    assert silent[0] != both[0] and silent[2] != both[2]


#: A one-factor Hull-White model for the dollar curve: with it the curve is simulated, so a credit
#: run stating `Deflate_Stochastically: 'Yes'` has paths to deflate on.
HULL_WHITE = {'Alpha': 0.05, 'Lambda': 0.0, 'Quanto_FX_Correlation': 0.0,
              'Quanto_FX_Volatility': {'.Curve': {'meta': [], 'data': [[0.0, 0.0], [10.0, 0.0]]}},
              'Sigma': {'.Curve': {'meta': [], 'data': [[0.0, 0.01], [10.0, 0.01]]}}}


def test_a_field_an_adjustment_block_states_stands_over_the_recalc_s_floor(credit_hub):
    """A recalc lays its floor under each adjustment block, and the floor says
    `Deflate_Stochastically: 'No'`. The dollar curve here carries a Hull-White model, so the field
    can be stated `'Yes'`. Stated in the CVA block of the book file's own calculation, with no
    policy, and then by the policy over a file stating no CVA block, it reaches the run both ways:
    one plan, whose job carries `'Yes'` over the floor's `'No'` beside the floor's other fields,
    and a CVA that is not the floor run's.

    Killing mutation: the floor laid over the block the calculation states (`xva_document`), where
    the job carries the floor's `'No'` and both runs are the floor's.
    """
    document = json.loads(credit_hub.read_text())
    market = document['Calc']['MergeMarketData']['ExplicitMarketData']
    market['Price Models']['HullWhite1FactorInterestRateModel.USD'] = HULL_WHITE
    market['Model Configuration']['.ModelParams']['modelfilters'] = {
        'InterestRate': [[['ID', 'USD'], 'HullWhite1FactorInterestRateModel']]}
    credit_hub.write_text(json.dumps(document, indent=2), newline='\n')
    stated = {'Credit_Valuation_Adjustment': {'Deflate_Stochastically': 'Yes'}}

    floor = credit_runs(['NS_A'])['NS_A']
    stating(credit_hub, **stated)
    own = credit_runs(['NS_A'])['NS_A']
    stating(credit_hub, Credit_Valuation_Adjustment=None)
    assert official({'CreditMonteCarlo': stated}).status_code == 200
    declared = credit_runs(['NS_A'])['NS_A']

    assert declared == own and own[2] != floor[2] and floor[2] > 0.0, (floor, own, declared)
    compiled = service.priced(service.BOOK.read()[0], 'CreditMonteCarlo')
    job = service.xva_document(compiled, dict(service.netting_sets(compiled))['0'], 'CPTY_A')
    assert job['Calc']['Calculation']['Credit_Valuation_Adjustment'] == dict(
        service.XVA_CVA_BLOCK, Deflate_Stochastically='Yes', Calculate='Yes', Counterparty='CPTY_A')
    assert derivus.content_hash(job) == declared[0]


def test_a_declaration_moves_no_plan_but_a_credit_run_s(recorded, quoting):
    """A job compiled for anything but a credit run reads no policy. Across a declaration the
    book's risk keeps its etag, a what-if of the book its result id and the book's own plan its
    name; a quote struck after the declaration pins the plan the one struck before it pinned, and
    that earlier quote books. At the seam, the book taken as a credit run's document: its diary and
    its marks job compile to themselves, and its credit run to a job carrying the policy's block
    over its own, the market shared and the document left as read.

    Killing mutations: the credit block written into every compile (`official_calculation` reading
    it for whatever the job runs as), where the risk's etag moves; the policy read for a job that
    values nothing (`compiled_job` asking it of a diary), where the diary's document is rewritten.
    """
    document = json.loads(quoting.read_text())

    def standing():
        priced = CLIENT.post('/book/price', content=dump({}), headers=JSON).json()
        planned = CLIENT.post('/prepare', content=dump({'Calc': document['Calc']}), headers=JSON)
        return (CLIENT.get('/book/risk').json()['etag'], priced['result_id'],
                planned.json()['plan_id'])

    def pinned(quote):
        return CLIENT.get('/book/quote/{}'.format(quote['quote_id'])).json()['pinned']['plan_hash']

    before, struck = standing(), quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET)
    assert official({'CreditMonteCarlo': OFFICIAL}).status_code == 200
    assert standing() == before
    assert pinned(quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET)) == pinned(struck)

    credit = dict(document, Calc=dict(document['Calc'], Calculation=dict(
        document['Calc']['Calculation'], Object='CreditMonteCarlo')))
    marks, read = spine.PnL.marks_job(credit, {}), derivus.content_hash(credit)
    assert spine.compiled_job(credit, strict=False, runs=False) is credit, 'a diary read the policy'
    assert spine.compiled_job(marks, strict=False) is marks, 'a marks job read the policy'
    compiled = spine.compiled_job(credit, strict=False)
    assert compiled['Calc']['Calculation'] == dict(credit['Calc']['Calculation'], **OFFICIAL)
    assert compiled['Calc']['MergeMarketData'] is credit['Calc']['MergeMarketData']
    assert derivus.content_hash(credit) == read, 'the compile wrote the document it read'

    assert CLIENT.post('/book/quote', json={'quote_id': struck['quote_id']}).json()['written']


def test_a_restated_policy_is_a_new_plan_and_a_job_compiled_before_it_is_the_earlier_run(
        recorded, credit_hub):
    """A recalc asked again over a book and a policy that stand hands back the id its row names.
    The policy restated at three batches, the recalc is asked a run the row does not name - how a
    row reads for a book that moved, a new plan under a new id - and once it lands the row names
    that run and its plan. The book compiled at the position before the restatement is the first
    run's job: the policy a job is written is the one standing where it is compiled.

    Killing mutation: `official_calculation` reading the head whatever position it is asked at,
    where the job compiled before the restatement hashes to the second plan.
    """
    def asked():
        return CLIENT.post('/book/xva', content=dump({'netting_sets': ['NS_A']}),
                           headers=JSON).json()['queued'][0]['result_id']

    assert official({'CreditMonteCarlo': OFFICIAL}).status_code == 200
    first, before = credit_runs(['NS_A'])['NS_A'], head(recorded)
    assert asked() == first[1]
    assert official({'CreditMonteCarlo': dict(OFFICIAL, Simulation_Batches=3)}).status_code == 200
    moved = asked()
    assert moved != first[1], 'the restatement left the credit run where it was'
    second = credit_runs(['NS_A'])['NS_A']
    assert (second[1], drew(second)) == (moved, (service.XVA_BATCH_SIZE, 3))
    assert second[0] != first[0]

    pinned = spine.compiled_job(service.BOOK.read()[0], lsn=before, strict=False,
                                runs='CreditMonteCarlo')
    job = service.xva_document(pinned, dict(service.netting_sets(pinned))['0'], 'CPTY_A')
    assert derivus.content_hash(job) == first[0], 'the book compiled before is not the first run'


def test_the_increment_and_the_credit_charge_start_from_the_policy(recorded, credit_hub):
    """Every credit run starts from the policy. Under one stating two batches and the FVA's two
    curves, the increment of the call NS_A holds runs, as its 'with' leg, the set as it prices:
    the recalc's CVA and FVA to the bit, on two batches of paths. And the credit charge a ticket
    asks of a candidate, a seven-month call, is priced as the set will be once it books: booked,
    the set's recalc reads the charge's 'with' CVA and FVA to the bit.

    Killing mutation: the increment's book compiled as the calculation the file names rather than
    as a credit run (`book_price`), where its legs run at the file's own block, one batch.
    """
    assert official({'CreditMonteCarlo': OFFICIAL}).status_code == 200
    standing = credit_runs(['NS_A'])['NS_A']
    held = answered(deal_path='0/0', reference='OPT_A')
    assert (held['cva']['with'], held['fva']['with']) == standing[2:]
    assert held['paths']['count'] == 2 * service.XVA_BATCH_SIZE

    charge = answered(deal=SHORT_CALL, parent_reference='NS_A')
    booked = CLIENT.post('/book/deals', content=dump({
        'action': 'add', 'deal': SHORT_CALL, 'parent_reference': 'NS_A', 'quantity': 1.0,
        'execution_reference': 'EXEC-C'}), headers=JSON).json()
    assert booked['written'] is True, booked
    assert credit_runs(['NS_A'])['NS_A'][2:] == (charge['cva']['with'], charge['fva']['with'])
    assert charge['cva']['increment'] > 0.0


def test_a_seat_s_own_credit_calculation_goes_over_the_policy(recorded, credit_hub):
    """A seat's own dials go over the policy as they go over the book's. Under a policy stating two
    batches, a named credit calculation stating none runs at the policy's two, and one stating
    three runs at its own three. Neither is official: the policy stands as declared.

    Killing mutation: the compiled calculation laid over the seat's block (`with_overrides`), where
    the calculation stating three runs at the policy's two.
    """
    assert official({'CreditMonteCarlo': OFFICIAL}).status_code == 200
    standing = CLIENT.get('/book/calculations').json()

    def ran(name, **stated):
        saved = CLIENT.post('/calculations', content=dump({'name': name, 'calculation': dict(
            stated, Object='CreditMonteCarlo')}), headers=JSON).json()
        assert saved['written'] is True, saved
        submitted = CLIENT.post('/calculations/run', content=dump(
            {'name': name, 'deal_path': '0'}), headers=JSON).json()
        service.EXECUTOR.queue.join()
        result = CLIENT.get('/results/{}'.format(submitted['result_id'])).json()
        assert result['status'] == 'done', result.get('error')
        return result['stats']['Simulation_Batches']

    assert (ran('as the policy'), ran('my own', Simulation_Batches=3)) == (2, 3)
    assert CLIENT.get('/book/calculations').json() == standing


def test_the_xva_view_names_the_curves_a_credit_run_would_fund_at_and_keeps_the_answer(
        recorded, credit_hub):
    """`GET /book/xva` answers, beside its rows, the curves a recalc asked now funds its FVA at,
    read as the run reads its calculation. With no policy they are the file's, `FUND` both. A
    policy stating both, they are the policy's. One stating the cost alone leaves the benefit null:
    the container it states replaces the file's whole. One stating no FVA block, the file's stand
    again. The answer is kept on the book under the poll's key, so a poll with nothing moved is a
    lookup, and a declaration, a frame, moves the key.

    Killing mutations: the view reading the book file's own block, which answers `FUND` under
    every policy; the answer worked out on every poll, where the book keeps none under the key.
    """
    def funding():
        return CLIENT.get('/book/xva').json()['funding']

    def kept():
        return service.BOOK.polls.get(('funding',) + service.poll_key(service.BOOK))

    assert kept() is None and funding() == {'cost': 'FUND', 'benefit': 'FUND'}
    first = kept()
    assert first == funding() and kept() is first, 'a poll with nothing moved worked it out again'

    assert official({'CreditMonteCarlo': OFFICIAL}).status_code == 200
    assert kept() is None, 'the declaration left the poll key where it was'
    assert funding() == {'cost': 'FUND', 'benefit': 'USD'} == kept()
    cost = {'Funding_Valuation_Adjustment': {'Funding_Cost_Interest_Curve': 'USD'}}
    assert official({'CreditMonteCarlo': cost}).status_code == 200
    assert funding() == {'cost': 'USD', 'benefit': None}
    assert official({'CreditMonteCarlo': {'Simulation_Batches': 2}}).status_code == 200
    assert funding() == {'cost': 'FUND', 'benefit': 'FUND'}


def test_a_block_declared_past_the_engine_s_judge_refuses_the_credit_run_by_name(
        recorded, credit_hub, tmp_path, capsys):
    """The operator's `DV_Spine declare` takes a calculation policy on its shape alone, so a block
    misspelling a field lands. It is judged where it is read: the recalc, the increment and the
    view's funding answer each refuse naming the policy and the field, nothing is queued and no
    row is filed, while a what-if of the book, a base valuation, reads no policy and runs. The
    block declared again through the service, the recalc runs.

    Killing mutation: the judge dropped from `official_calculation`, where the recalc is queued
    and runs at one batch, the field it was meant to state ignored.
    """
    from derivus_spine import cli

    misspelt = tmp_path / 'calculation.json'
    misspelt.write_text(json.dumps({'calculations': {'CreditMonteCarlo': {
        'Simulation_Batchs': 2}}}), encoding='utf-8')
    assert cli.main(['declare', 'calculation', str(misspelt), '--home', str(recorded),
                     '--actor', ACTOR]) == 0, capsys.readouterr().err
    said = ('the calculation policy in force states a CreditMonteCarlo block this engine refuses '
            '- CreditMonteCarlo declares no Simulation_Batchs - it declares ')

    for refused in (CLIENT.post('/book/xva', content=dump({}), headers=JSON),
                    CLIENT.post('/book/price', content=dump(
                        {'deal_path': '0/0', 'xva_increment': True}), headers=JSON),
                    CLIENT.get('/book/xva')):
        assert refused.status_code == 422 and refused.json()['detail'].startswith(said), (
            refused.text)
    assert not (tmp_path / service.XVA_FILE).exists(), 'a refused recalc filed a row'
    assert CLIENT.post('/book/price', content=dump({}), headers=JSON).status_code == 200

    assert official({'CreditMonteCarlo': {'Simulation_Batches': 2}}).status_code == 200
    assert drew(credit_runs(['NS_A'])['NS_A']) == (service.XVA_BATCH_SIZE, 2)
