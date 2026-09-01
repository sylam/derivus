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

"""The seam between the engine and the book of record, driven through both of them at once.

`test_spine_verbs.py` gates the spine half with no engine anywhere near it. This file is the other
half and it is deliberately the expensive one: real homes in temp directories, a real book file, a
real `BaseValuation` through actual `derivus`, and the real `Market Prices` partition. Nothing is
monkeypatched - the injected executor is an ordinary function, which is the seam's own design, and
every fault is data on disk.

THE SWITCH IS `DV_SPINE_HOME` AND THE FIRST GATE IS THAT IT IS OFF. With it unset the edge behaves
to the byte as it did before this increment existed: a lane is accepted and inert, no pending file
grows a field, no verb writes anything, and the Context verbs refuse by name rather than reaching
for a default home. That claim is the regression bar for the whole increment, so it is asserted
first and asserted on the same book the recording gates use.

What the rest of the file holds, in the brief's own words:

  * ATTESTATION LANES, all four. A synthetic tick sequence mints nothing, ABSENCE asserted against
    a head that does not move. A curiosity run mints nothing. A `result_pinned` that fails
    re-execution is refused by name - re-executed through the real engine, against the real
    document the record stored. One matching a known tuple resolves as a cache hit against the
    original attestation, and the executor is never called, which is counted rather than believed.
  * PROVENANCE, in the minimal honest form the increment can carry: the plan hash is RE-DERIVED by
    recompiling the blob-stored job document at the recorded LSN and required to equal the recorded
    tuple - the auditor's own move, made mechanical. The compiler-as-a-fold over fixings
    supersession is increment 4's; what is gated here is that the object the fold will read is
    stored, addressed and recompiles.
  * FIRMNESS, all three. A book that moved refuses on the plan dimension; a market pin past its
    window refuses on the values dimension; and the two are DISJOINT - a pure vol tick through the
    real partition moves `values_hash`, leaves `plan_hash` bit-identical, and does not trip the
    plan dimension. That last one is the gate the `Market Prices` partition prerequisite existed
    for, asserted on a fixture that ticks a vol with no booking in sight.
  * ATOMICITY: a priced ticket references exactly one values hash and one book plan, and both
    staleness policies fire on a deliberately aged fixture.
  * THE DUAL WRITE'S ORDER: under a spine home the event goes first and the book file follows, and
    a booking the record refuses leaves the file byte-identical.
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
from derivus.config import CustomJsonEncoder, deal_at
from derivus_spine import SpineLog, init_home, verify_home
from derivus_spine import policy, verbs

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
    """A job document as the objects a market data file holds - the same shape `test_service`
    authors, so what reaches the endpoint here is what reaches it there."""
    return {'Calc': {
        'Calculation': dict({'Object': 'BaseValuation', 'Base_Date': BASE, 'Currency': 'USD',
                             'MCMC_Simulations': 1, 'Random_Seed': 1}, **calculation),
        'Deals': {'Tag_Titles': '', 'Reference': 'spine-desk',
                  'Deals': {'Children': [{'Instrument': {'.Deal': deal}} for deal in deals]}},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': dict({
            'System Parameters': {'Base_Currency': 'USD', 'Base_Date': BASE},
            'Price Factors': factors}, **sections)}}}


def netting_set(reference, counterparty, deals=()):
    """One `NettingCollateralSet` naming its counterparty where the engine reads it - which is also
    where a fill's counterparty comes from, so the two cannot name different clients."""
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
    standing in for the terminal, everything downstream of them the real pipeline. `atm_3m` is what
    a TICK moves, and it is the whole of what moves."""
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
    """A desk with NO spine home - the posture every existing gate in this suite runs under, made
    explicit so that a developer box carrying the variable cannot make this file lie."""
    monkeypatch.delenv('DV_SPINE_HOME', raising=False)
    monkeypatch.delenv('DV_SPINE_ACTOR', raising=False)
    monkeypatch.setenv('DV_HOME', str(tmp_path))
    yield tmp_path


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    """A minted spine home, configured, with an actor for the appends. Genesis is four events, so
    every head assertion below counts from there."""
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
    """`(lsn, type, body)` for every event on the record, or every one of a type. Read off the
    platter and decrypted, because that is the assertion everywhere here: what is claimed is what
    the log actually holds."""
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
    """A job whose replay tuple belongs to ONE gate.

    The result store is content-addressed and lives for the length of the process, so two gates
    posting the same job would be one execution and the second would read the first's answer. That
    is the feature working, and it is also a way to write a gate that passes for the wrong reason -
    so every gate here moves one number nobody asserts on, and owns its own tuple.
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
    """A job that HOLDS the single worker until the gate lets it go.

    Nothing is patched to get this: the executor runs whatever object it is handed that answers
    `run_job`, which is how `XvaJob`, `SolveJob` and `BloombergJob` already ride it, and this is one
    more of those - submitted through the public `submit`, dequeued by the real worker, released by
    the gate. It is what turns "a submission observed while the same tuple is still queued" from a
    timing window into a fact, and it is why the in-flight gate below is deterministic rather than
    flaky.
    """

    def __init__(self):
        self.running = threading.Event()
        self.released = threading.Event()

    def run_job(self):
        self.running.set()
        self.released.wait(WORKER_SECONDS)
        return None, {'Results': {}, 'Stats': {}}


def holding(counter=itertools.count()):
    """Back the one worker up behind a barrier, and answer it. The queue is drained first, so what
    is waiting behind the barrier afterwards is only what the gate puts there.

    Each barrier is filed under a name of its own, because the result store is content-addressed
    and lives for the length of the process: a second barrier under one name would coalesce onto
    the first, never be dequeued, and hang the gate that waited for it instead of failing it.
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
    """Two spellings of one vocabulary, pinned together.

    The engine names the lanes at call sites that must not pay for the spine import to do it, so
    `derivus.spine` carries its own three strings - `capability.py`'s reason for spelling the
    genesis policy names itself, met from the other side. A pin is what keeps two spellings from
    drifting into two vocabularies, and this is it.
    """
    assert (spine.TELEMETRY, spine.CURIOSITY, spine.STANDING) == (
        verbs.TELEMETRY, verbs.CURIOSITY, verbs.STANDING)
    assert spine.LANES == verbs.LANES
    assert service.DEFAULT_LANE == spine.CURIOSITY


def test_importing_the_engine_lands_neither_the_spine_nor_its_one_dependency():
    """The extra is an EXTRA, proved the way the spine's own import gate proves its budget: a fresh
    interpreter imports the engine, the seam and the HTTP surface, and reports what arrived.

    `pip install derivus` must not grow a `cryptography` dependency for a book of record that box
    does not run - the `fastapi` precedent one module over, and the reason every import in
    `derivus/spine.py` is inside the function that needs it rather than at the top of the file.
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
    """THE REGRESSION BAR, asserted rather than assumed.

    Enforcement by declaration was increment 2's precedent and this is its shape here: no home
    configured means the record does not exist as far as the engine is concerned. A lane is
    accepted and INERT - including one nobody has ever heard of, because there is nothing here for
    it to be wrong about - a booking books, a quote files the pending trade with no `pinned` field,
    and the Context verbs refuse by NAME rather than reaching for a default home. That last one is
    the sharpest: `DV_Spine` falls back to `~/.derivus_spine` because a person typing a verb means
    that home, and an engine that fell back would start recording on any box where somebody once
    ran `init`.
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
    """Set but not minted is a NAMED refusal, never a quiet fall-back: "the deployment configured a
    home that is not there" and "the deployment configured none" are different facts, and reading
    the first as the second would silently un-record a box somebody meant to record. The refusal is
    the spine's own sentence about what a home is made of, carried through unedited."""
    monkeypatch.setenv('DV_SPINE_HOME', str(tmp_path / 'never-minted'))
    monkeypatch.setenv('DV_SPINE_ACTOR', ACTOR)
    refused = CLIENT.post('/execute', content=dump(dict(own_job('NO-HOME'),
                                                        lane=spine.STANDING)), headers=JSON)
    service.EXECUTOR.queue.join()
    result = CLIENT.get('/results/{}'.format(refused.json()['result_id'])).json()
    assert result['status'] == 'error'
    assert 'log/' in result['error'] and 'DV_Spine init' in result['error']


def test_a_configured_home_still_refuses_an_append_nobody_signed(recorded):
    """A home is configured and an actor is not: every event carries the pseudonymous subject
    reference that submitted it, so there is nothing to stamp and the verb refuses by name.
    Inventing one would put a name in the record that nobody chose."""
    os.environ.pop('DV_SPINE_ACTOR')
    context = derivus.Context().load_json((dump(job()), 'posted'))
    with pytest.raises(spine.SpineRefused) as refusal:
        context.declare_market('official')
    assert 'DV_SPINE_ACTOR' in str(refusal.value)
    assert head(recorded) == 4, 'a refused append wrote something'


# --------------------------------------------------------------------------------------------
# The lanes.

def test_a_standing_run_attests_at_birth_and_the_other_lanes_mint_nothing(recorded, desk):
    """The brief's rule in one gate: *a run is recorded IFF its output will be cited by a fact.*

    Telemetry and curiosity mint NOTHING and the assertion is ABSENCE - the head does not move
    across either, which is the only honest way to say "nothing was recorded". A standing run
    appends `run_completed` with the whole replay tuple at birth, and the result reports the LSN it
    landed at so a caller can cite it without folding for it.
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
    """The brief's telemetry gate: *a telemetry repaint mints no event, asserted by absence over a
    synthetic tick sequence.*

    The sequence is real work on a real book - four market ticks moving the ATM vol, each one
    installing quotes, re-bootstrapping the surface and rewriting the file, with a what-if priced
    between them. Every one of those is a READING: superseded by the next before anything could
    cite it. So the head does not move once, and the assertion is the head, not a filtered count -
    a count would pass on a log full of the wrong events.
    """
    genesis = head(recorded)
    for atm in (14.1, 14.2, 14.3, 14.4):
        ticked = CLIENT.post('/book/market', content=dump({'quotes': fx_vol_quotes(atm)}),
                             headers=JSON).json()
        assert ticked['written'] is True and ticked['updated'] == ['FXVolPrices.USD.ZAR']
        priced = CLIENT.post('/book/price', json={}).json()
        assert priced['status'] in ('queued', 'running', 'done')
        # drain before the next tick: market_edit captures the ROOT logger around bootstrap (the
        # roadmap's own Known-defects row), so a pricing thread's ERROR landing in that window
        # refuses a tick that was fine - the race is the engine's, not this gate's claim
        service.EXECUTOR.queue.join()

    assert head(recorded) == genesis, 'a repaint reached the book of record'
    assert facts(recorded, 'run_completed') == []
    assert verify_home(recorded)['events'] == genesis


def test_a_standing_run_whose_numbers_already_exist_still_attests(recorded, desk):
    """Content addressing dedupes NUMBERS; the lane is about STANDING. The two must not be
    conflated, and this gate is where they would be.

    The same job priced first as a what-if and then declared standing coalesces onto a result the
    worker will never revisit - so an attestation made only at completion would never be made at
    all, and a fact would go on to cite numbers the record does not hold. So the standing
    submission attests from the store the run was filed in, the result gains its `attested` block
    where `/results` will serve it, and a third submission coalesces on the attestation's own
    idempotency tag rather than writing a second row about one run.
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

    # the bytes the record holds ARE the ones the store was serving, which is the property that
    # makes attesting off a finished result honest rather than convenient
    _, _, body = facts(recorded, 'run_completed')[0]
    assert blob(recorded, body['result']) == spine.result_stored(
        service.EXECUTOR.result(standing['result_id']))
    assert verify_home(recorded)['events'] == head(recorded)


def test_a_standing_run_that_coalesces_onto_one_in_flight_is_attested_when_it_lands(recorded, desk):
    """The other half of the gate above, and the half a `done` status cannot reach.

    The same tuple explored first and declared standing a moment later coalesces exactly as it does
    when the numbers already exist - except that here the numbers do NOT exist yet, so the status
    the standing caller is handed is `queued` and there is nothing on this thread to attest. The
    job the worker will dequeue is the WHAT-IF's, carrying a lane that mints nothing, so an
    attestation read off the dequeued submission alone would never be made: the standing caller
    would be served numbers with no `run_completed` behind them and no refusal either, which is the
    unbacked citation the lane rule exists to prevent, arriving silently.

    So the standing submission is promoted onto the run inside `submit`, under the lock that
    publishes results, and the worker attests THAT submission when the numbers land. The gate
    asserts the head twice: unmoved while the run is in flight, and moved by exactly one
    `run_completed` in the standing lane once it publishes. `running` is the same arm of the same
    branch as `queued` - one dict lookup apart - and the barrier is what makes `queued` observable
    without a race.
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
    """A standing run whose attestation is refused has NOT acquired standing, so serving its
    numbers as though it had would be exactly the unbacked citation the lane rule exists to
    prevent. The refusal travels as the run's own error, in the spine's own wording, and the record
    holds the denial and nothing else."""
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
    """A `plan_id` names a PARSE the cache holds, and an attestation carries the job document the
    plan recompiles from. `Context.save_json` is explicitly not a complete round trip, so
    serialising the parse back would store a document that is not the one that ran - and a
    provenance chain whose first link is a document nobody can recompile is worse than none. The
    refusal names the remedy; the curiosity lane over the same plan still runs."""
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
    """`attests` is the ONE place that decides whether a finished run owes the record anything, and
    it asks two questions: is this the standing lane, and is there evidence to attest from.

    Both halves are pinned here because only one of them is load-bearing at today's call sites.
    Every lane but standing is handed evidence of None, so the lane test could stop working
    tomorrow and every gate in this file would go on passing - right up to the day some other lane
    wants the job document too, which is increment 4's hydrating projection asking for exactly this
    blob for a run nobody will cite. The failure that day is the silent kind: telemetry minting.

    So the function is asked on its own terms, over plain tuples, with evidence present in every
    lane - which is the only arrangement in which the lane test is the thing being tested.
    """
    evidence = {'job': b'{"Calc":{}}', 'values': b'{}'}
    assert spine.configured() is True, 'this gate is about the recording posture'

    for silent in (spine.TELEMETRY, spine.CURIOSITY):
        assert service.attests(service.Job('r', None, {}, silent, evidence)) is False, silent
    assert service.attests(service.Job('r', None, {}, spine.STANDING, evidence)) is True

    # the other half, and it is the QUOTE's case: a standing run with nothing to attest from files
    # the richer `quote_filed` instead, and a `run_completed` beside it would be two records of one
    # act. A job carrying no lane at all is every submission any existing caller makes
    assert service.attests(service.Job('r', None, {}, spine.STANDING, None)) is False
    assert service.attests(service.Job('r', None, {})) is False


def test_the_stored_job_is_the_job_and_not_the_submission_that_carried_it(recorded, desk):
    """What a standing attestation stores is the `Calc` ENVELOPE and nothing beside it.

    A posted body may also carry `Patch`, `lane` or `plan_id`, and not one of those is the job -
    they are how the submission ARRIVED. The trim matters because this blob is the FIRST LINK of
    the provenance chain: increment 4's auditor recompiles from exactly these bytes, and a link
    that carried the request rather than the document would be a fold over something that never
    ran. The gate above it recompiles the plan and would pass on either, because the engine's
    loader tolerates a surplus top-level key - so the trim needs saying on its own.

    The patch is NOT lost by it, which is the half that makes the trim safe: the values vector
    filed beside the job carries the whole market as patched, which is the brief's own model of a
    result as engine(plan, values) rather than engine(document). So this asserts both - the job is
    the job, and the market the run actually read is recoverable from the record beside it.
    """
    document = own_job('SUBMISSION-TRIM')
    moved = SPOT + 1.25
    standing = drained(submit(document, lane=spine.STANDING, Patch={'FxRate.ZAR': {'Spot': moved}}))
    assert standing['status'] == 'done', standing.get('error')

    _, _, body = facts(recorded, 'run_completed')[0]
    stored = json.loads(blob(recorded, body['job']).decode('utf-8'))
    assert sorted(stored) == ['Calc'], 'the submission rode into the record as the job'
    assert stored['Calc'] == json.loads(dump(document))['Calc']

    # and the patch is in the VALUES vector, where the model says it lives - so the tuple the
    # record holds is reproducible from the two blobs it cites and from nothing else
    context = derivus.Context().load_json((json.dumps(stored), 'recompiled'))
    context.patch_market(spine.read_values(blob(recorded, body['values_hash'])))
    assert context.values_hash() == body['values_hash']
    assert context.plan_hash() == body['plan_hash']
    assert context.market_patch()['FxRate.ZAR']['Spot'] == moved, \
        'the values vector does not carry the market the run actually read'


def test_an_unknown_lane_refuses_where_the_record_will_act_on_it(recorded, desk):
    """With a home configured a lane is a decision the record acts on, so an unknown one is refused
    by name rather than read as the default. The refusal is `derivus_spine.verbs.check_lane`'s own -
    the lanes are the record's vocabulary, not the service's."""
    refused = CLIENT.post('/execute',
                          content=dump(dict(own_job('UNKNOWN-LANE'), lane='exploration')),
                          headers=JSON)
    assert refused.status_code == 422
    assert 'telemetry, curiosity, standing' in refused.json()['detail']
    assert head(recorded) == 4


# --------------------------------------------------------------------------------------------
# Provenance: the plan recompiles, and the result reproduces.

def test_the_plan_hash_recompiles_from_the_stored_job_at_the_recorded_lsn(recorded, desk):
    """The brief's plan-recompilation gate, in the minimal honest form this increment can carry:
    *recompile the synthetic book's plan at its recorded LSN and require the identical plan hash.*

    The auditor's move, made mechanical. The record stores the JOB DOCUMENT rather than the plan,
    because a plan is re-derivable and the record never trusts what it can re-derive: so the gate
    pulls the blob the attestation cites, loads it through the engine's own decoder, applies the
    values vector the same attestation cites, and requires BOTH hashes to come back the ones the
    tuple recorded. It then re-executes and requires the result bytes to reproduce to the byte -
    which is the tolerance policy's fast path, exercised where it should land.

    THE BOUNDARY, stated: this recompiles the document the run was submitted with. The compiler as
    a FOLD - terms plus the fixings each schedule declares, queried from the log with supersession
    resolved - is increment 4's, and until it exists the object the fold will read is what is being
    gated here: stored, addressed, and recompiling to its own hash.
    """
    standing = drained(submit(own_job('PROVENANCE'), lane=spine.STANDING))
    lsn, _, body = facts(recorded, 'run_completed')[0]
    assert lsn == standing['attested']['lsn']

    stored_job = blob(recorded, body['job'])
    stored_values = blob(recorded, body['values_hash'])
    stored_result = blob(recorded, body['result'])
    # the values citation and the store address are ONE number, which is what lets the engine's own
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
    """A tuple this hub attested itself needs no re-execution, and the gate counts rather than
    believes: the executor handed in is a real function that records its calls, and it is never
    called. The pin still lands and still names the tolerance policy in force, because a promotion
    made under no declared standard is not one this record carries."""
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
    """The brief's own sentence, through the real engine on a tuple this hub never witnessed.

    The claim is a genuine one - the plan and values hashes of a job the record has never seen -
    carrying a DOCTORED result. `Context.pin_result` injects the engine as the executor, the job
    runs for real, and what comes back is not what was claimed: the refusal names the class, both
    numbers and the tolerance policy the comparison was held to, and NOTHING is appended.
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

    # and the honest claim over the same job lands, re-executed and reproducing to the byte
    _, out = derivus.Context().load_json((dump(unseen), 'unseen')).run_job()
    pinned = derivus.Context().pin_result(
        spine.canonical(unseen), spine.values_of(context), spine.result_of(out),
        spine.replay(context))
    assert pinned['resolution'] == 'reproduced' and head(recorded) == standing + 1
    assert verify_home(recorded)['events'] == head(recorded)


# --------------------------------------------------------------------------------------------
# The Context verbs.

def test_the_context_verbs_book_amend_and_file_the_three_lifecycle_facts(recorded):
    """The five delegators on a real home. Each one canonicalises through the ENGINE's own encoder,
    so an instrument's id is the content hash a job document would give it, and hands plain data to
    the spine - which is what keeps storage out of every module under `derivus/` but this one.

    `declare_market` is the one that uses its context: the values vector it points a name at is
    THIS context's own, which is why officialness can be a property of the name while every vector
    lives identically in the store.
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
    """The dual write's ORDER, and the two fields a fill will not do without.

    Under a spine home `/book/deals` appends the `fill` and then rewrites the book file, which is
    the durability law applied to the pair: what is TRUE is on the platter before the desk's copy
    of it is. So a booking the record refuses leaves the file byte-identical - asserted on the
    bytes, not on the parse - and the three refusals name what they need: a quantity, an execution
    reference, and a netting set with a counterparty on it, because a fill's body carries all
    three and none of them has a defensible default.
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
    """THE ATOMICITY GATE: *a priced ticket references exactly one values hash and one book plan.*

    One quote, one `quote_filed`, and in its body exactly one of each - beside the coordinate that
    was solved and the edge the desk took. The pending file carries the same pair, so the approval
    reads the pins off the desk's own copy and the record holds the authoritative one.

    The hashes are the BOOK's, taken before the live spot lands on the quote's copy: what an
    approval asks is whether the market and the book this trade would LAND against have moved, and
    a booking lands against the book's market. Which spot the legs were struck on is a different
    question and `spot` already answers it.
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
    """The quote's own parameters, which move the document's spot the instant `patch_live_spot`
    reads the pair off them - which is the instant a terminal's crosses would land in production.

    THE GATE'S OWN DATA, not a patch of anything: `params` is a caller-supplied dict that
    `StructureJob` only ever reads, and `patch_live_spot`'s first act is `params['pair']`. So a dict
    that moves the spot on that read reproduces, deterministically and with no terminal in the room,
    the one thing this box cannot otherwise produce - a live spot landing on the quote's copy of the
    market. Every line of library code runs exactly as written.
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
    copy's spot, and the reason is the approval: what `/book/quote` asks is whether the market and
    the book this trade would LAND against have moved, and a booking lands against the book's own
    market. Pin after the spot patch and `values_hash` describes a market that exists only inside
    one quote - so every approval on a box with a live terminal would refuse on the values
    dimension, for a market that never moved, with a refusal naming a remedy that cannot work.

    NOTHING ELSE IN THIS FILE CAN SEE THAT. A desk box with no terminal never patches the spot, so
    the two orderings answer identically and the claim rides on a comment. This gate supplies the
    missing half as data - a params object that moves the spot exactly where the terminal's crosses
    would - and then asserts the pin is the book's regardless.

    The disjointness half comes free and is worth having twice: a SPOT is values-plane data like a
    vol, so moving it moves `values_hash` and leaves `plan_hash` bit-identical.
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

    # and the approval reads the same pair off the desk's copy, so it stands against the book
    verdict = spine.package().firmness.assess(
        {'plan_hash': body['plan_hash'], 'values_hash': body['values_hash']},
        {'plan_hash': book_context.plan_hash(), 'values_hash': book_context.values_hash()},
        {'values': 1.0, 'plan': 1.0}, spine.firmness_policy())
    assert verdict['firm'] is True, verdict['refusals']
    assert verify_home(recorded)['events'] == head(recorded)


def test_a_moved_book_refuses_the_approval_on_the_plan_dimension(recorded, quoting):
    """The book moved since the solve, so the marginal charge was priced against a portfolio this
    trade would no longer join. Only the PLAN dimension refuses - the market has not been touched -
    and the refusal names the dimension and its own remedy, which is what makes it actionable: a
    desk told "stale" learns nothing; a desk told "the book moved" re-solves."""
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
    """The market pin is past its window, on a market that has not moved at all - a different
    failure from a moved one, with a different remedy.

    The fixture is aged DELIBERATELY and by declaration: a firmness policy with a zero-second
    values window, put in the record as a hashed policy blob through the ordinary writer. That is
    how the brief says staleness windows travel - policy is data - and it is what makes an aged
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
    """The other half of the brief's atomicity gate: *both staleness policies fire on deliberately
    aged fixtures.* Two zero-second windows, one quote, and both dimensions named in one refusal -
    because a quote stale on both has two remedies and reporting one sends a salesperson back into
    the other."""
    declare(recorded, policy.FIRMNESS_POLICY, {'values_seconds': 0, 'plan_seconds': 0})
    quote = quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET)

    refused = CLIENT.post('/book/quote', json={'quote_id': quote['quote_id']})
    said = refused.json()['detail']
    assert refused.status_code == 422
    assert 'the values dimension' in said and 'the plan dimension' in said
    assert ' AND ' in said


def test_a_vol_tick_moves_the_values_hash_and_never_trips_the_plan_dimension(recorded, quoting):
    """THE DISJOINTNESS GATE, and the whole reason the `Market Prices` partition was a prerequisite.

    A pure market tick with NO booking in sight: the ATM vol moves, the block value-updates, the
    surface re-bootstraps and the book file is rewritten. Through the real partition that moves
    `values_hash` and leaves `plan_hash` BIT-IDENTICAL - quote values are the values plane, and
    pillars, expiries, conventions and everything a solve reads are the plan.

    So the approval refuses on the values dimension and the plan dimension stands, and the gate
    asserts the plan hash's bit-identity directly as well as through the verdict. Conflating them
    would produce exactly the failure the brief names - an aged quote approvable at a dead market's
    solve - wearing a single "staleness" number nobody could act on.
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
    the record's two dimensions - the `fill` appended, and then the file written.

    The execution reference is the QUOTE ID, because a quote is an act and its id names that act,
    which is the ticket id the brief's fill rule asks for. The quantity carries the DESK's side:
    the quote is client paper and the approval books the mirror, so a client who bought leaves the
    desk short and the quantity is negative.
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
    """The whole increment's posture in one assertion: everything the record holds verifies from
    genesis, and the desk file is a copy of part of it written afterwards.

    A quote, an approval, a market tick and a standing run in one session, then the chain
    re-derived from its own bytes by the replica-shaped verifier - entitled and chain-only both -
    because a book of record that only its writer can check is not one.
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
