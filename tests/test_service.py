"""One vocabulary, two bindings — and the gates are what says the second one owns no logic.

`derivus.service` is `Context` over HTTP. Every endpoint builds a Context from the posted job,
calls one of its verbs and serialises the answer, so the decisive gate is PARITY: the same job
submitted over HTTP has to produce the same numbers as loading it in process and calling `run_job`.
Anything the wrapper did of its own would show up there as a difference. `/schema`, `/schema/job`,
`/validate` and `/describe` are the same claim for the read verbs — and the skeleton makes it
falsifiable, because a document that does not LOAD cannot answer any of them.

The part that is genuinely new is the dispatcher, and it makes three promises worth holding to.
Ordering: pricing goes through ONE worker, so a base valuation jumps a simulation among the jobs
still WAITING and a running job is never preempted. Identity: a `result_id` is the hash of the
replay tuple, so submitting the same job twice is one execution — dedupe and retry-idempotency are
the same feature, and it has to hold while the first is still running, not just after it finishes.
Survival: a job that fails in the engine is a result like any other, and the next job still runs.

`plan_id` adds a fourth: how a job ARRIVED cannot change what it reports. A plan-id execute and a
full-document execute of the same job name the same result, and a patched execute off a plan leaves
the plan as it was — which is asserted by running an unpatched one after it and demanding the
original id back.

`/book/bloomberg` is the one verb whose dependency this machine may not have, so its gates drive
the seams instead of the terminal: `discover.provision`, `security_map.stale` and `fetch_fx_vol`
are monkeypatched, and the lazy imports inside the job are what makes that reach it. No blpapi,
no socket, no map file - what is under test is the verb's own wiring: the scope it derives from
the map, the refusal a late quote earns, the single atomic write it installs through, and the
progress a poller reads while it runs.

`--tick`'s metronome rides that same seam, so its two gates need neither a terminal nor a patch:
the skip-if-in-flight decision is read off the executor's REAL store while a held job occupies the
worker, and the routine refusal is a beat against a `DV_HOME` holding no map - which the job
answers before it opens a session, in a named refusal the book's bytes are untouched by.

`/book/hn` is the calibration verb, and its three gates are the emitter, the round trip and the
refusal. The emitter is checked against the canned surface the market gates already build - ten
points, the surface's own expiries with the substitutions NAMED, vega-normalised weights, wings
straddling the spot - and then each emitted strike is put back through the family's own moneyness
dispatch and required to return the vol the block carries, which is the whole orientation argument
asserted rather than claimed. The round trip runs the verb and demands the model reprice its own
ten quotes off the parameters the BOOK FILE ends up carrying; it fits a short-dated smile of its
own, for two reasons its fixture states. The refusal is on the REQUEST thread, because a
minutes-long job whose answer is a typo is not an answer.

`/book/structure` and `/book/quote` are one verb split across the desk's own approval: a quote is
given, filed under its id in `DV_HOME/tmp`, and only then booked. What the gates hold is that the
two halves are the same trade - the collar comes back netting to zero, and the BOOK marks the deal
it wrote at zero when the file is priced afterwards. `DV_HOME` is the declared surface for where
those files land, so the gates set it and read the directory it names; the booking half goes
through the same validate-before-write seam `/book/deals` uses, which is asserted by refusing an
authored pending trade in the identical wording an amendment is refused in.

The ordering and dedupe gates are deterministic without sleeping on a clock. A first job blocks
inside `run_job` and announces it through an `Event`, so by the time the others are submitted the
worker is provably busy and they are all in the queue — releasing it makes the queue the only thing
that can decide what runs next. `Queue.join` is the barrier everywhere else: it returns when every
enqueued job has been stored.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import threading
import zipfile

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import derivus
from derivus import service, structures, utils
from derivus.config import CustomJsonEncoder, deal_at

BASE = pd.Timestamp('2024-06-28')
RATE = 0.02
SPOT = 18.5
AMOUNT = 1_000_000.0
EQ_SPOT = 100.0
VOL = 0.25
JSON = {'content-type': 'application/json'}

CLIENT = TestClient(service.app)

#: A two-year ZAR cashflow reported in USD, so one number rides a curve's rate column and a spot at
#: once, and the closed form is `amount x spot x exp(-rate x 2)`.
CASHFLOW = {'Object': 'FixedCashflowDeal', 'Reference': 'CF1', 'Currency': 'ZAR',
            'Discount_Rate': 'ZAR', 'Calendars': None, 'Amount': AMOUNT,
            'Payment_Date': BASE + pd.DateOffset(years=2)}

#: The same binary `test_validate_verb` breaks: `Cash_Payoff` IS its notional, so the declaration
#: makes it required and leaving it out is an authoring message rather than a missing factor.
BINARY = {'Object': 'EquityBinaryOption', 'Reference': 'BIN1', 'Currency': 'USD',
          'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
          'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
          'Strike_Price': EQ_SPOT, 'Expiry_Date': BASE + pd.DateOffset(years=1),
          'Settlement_Date': BASE + pd.DateOffset(years=1)}

FACTORS = {
    'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Spot': 1.0},
    'FxRate.ZAR': {'Domestic_Currency': None, 'Interest_Rate': 'ZAR', 'Spot': SPOT},
    'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                         'Curve': utils.Curve([], [[0.0, RATE], [5.0, RATE]])},
    'InterestRate.ZAR': {'Currency': 'ZAR', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                         'Curve': utils.Curve([], [[0.0, RATE], [5.0, RATE]])}}

EQUITY = {
    'EquityPrice.EQ': {'Spot': EQ_SPOT, 'Currency': 'USD', 'Interest_Rate': 'USD', 'Issuer': '',
                       'Respect_Default': 'No', 'Jump_Level': 0.0},
    'DividendRate.EQ': {'Currency': 'USD', 'Curve': utils.Curve([], [[0.0, 0.0], [5.0, 0.0]])},
    'VolatilityGrid.EQ': {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
                          'Surface': utils.Curve([], [[m, t, VOL] for m in (0.8, 1.0, 1.2)
                                                      for t in (0.02, 2.0)])}}


def job(deals=(CASHFLOW,), factors=FACTORS, sections={}, **calculation):
    """A job document, authored as the objects a market data file holds. Dumping it through
    `CustomJsonEncoder` is what a client posts, so the `.Curve` and `.Timestamp` tokens the endpoint
    receives are exactly the ones a file carries — and the decoder that reads them is the same one.
    `sections` adds further market-data sections (a `Bootstrapper Configuration`, `Market Prices`).
    """
    return {'Calc': {
        'Calculation': dict({'Object': 'BaseValuation', 'Base_Date': BASE, 'Currency': 'USD',
                             'MCMC_Simulations': 1, 'Random_Seed': 1}, **calculation),
        'Deals': {'Tag_Titles': '', 'Reference': 'service',
                  'Deals': {'Children': [{'Instrument': {'.Deal': deal}} for deal in deals]}},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': dict({
            'System Parameters': {'Base_Currency': 'USD', 'Base_Date': BASE},
            'Price Factors': factors}, **sections)}}}


def dump(document):
    return json.dumps(document, cls=CustomJsonEncoder)


def in_process(document):
    """The other binding: the same text, the same decoder, no HTTP."""
    return derivus.Context().load_json((dump(document), 'posted'))


def submit(document):
    return CLIENT.post('/execute', content=dump(document), headers=JSON).json()


def run(document):
    """Submit, wait for the one worker to drain the queue, and read the summary back — which is all
    `/results/{id}` carries now, so the id it was filed under travels with it."""
    submitted = submit(document)
    service.EXECUTOR.queue.join()
    return submitted['result_id'], CLIENT.get('/results/{}'.format(submitted['result_id'])).json()


def fetch(result_id, table, **paging):
    return CLIENT.get('/results/{}/{}'.format(result_id, table), params=paging).json()


def mtm(result_id):
    """`{Reference: Value}` out of the `mtm` table, fetched the way a client fetches one."""
    table = fetch(result_id, 'mtm')
    reference, value = table['columns'].index('Reference'), table['columns'].index('Value')
    return {row[reference]: row[value] for row in table['data']}


class Held:
    """A Context as far as the executor is concerned — it calls `run_job()` and reads nothing else.

    That one verb is the whole seam between the queue and the engine, so the ordering and dedupe
    promises can be observed by handing the executor one of these, and a `Results` tree no
    calculation in the suite produces can be put through the store and the two result endpoints.
    Nothing in the package is patched, and `hold` is what makes the worker's occupancy an event
    rather than a race.
    """

    def __init__(self, name, ran, hold=None, results={}):
        self.name, self.ran, self.hold, self.results = name, ran, hold, results
        self.started = threading.Event()

    def run_job(self):
        self.started.set()
        if self.hold is not None:
            self.hold.wait(timeout=30)
        self.ran.append(self.name)
        return None, {'Results': self.results}


def test_a_job_priced_over_http_is_the_job_priced_in_process():
    """The decisive gate. Identical results, table for table and cell for cell — which is what
    "the wrapper owns no logic" means when it is asserted rather than asserted about.

    The summary claims a shape and the drill-down serves the cells, so both are held against the
    same in-process run: a client that trusted `rows` and paged to it would otherwise be reading a
    number the shape never described. The closed form is here so the comparison cannot pass by both
    sides being empty: a serialised table equal to another serialised table says nothing if neither
    carries a price.
    """
    document = job()
    result_id, result = run(document)
    _, out = in_process(document).run_job()
    expected = json.loads(json.dumps(out['Results'], cls=CustomJsonEncoder))

    assert result['status'] == 'done'
    assert set(result['tables']) == set(expected)
    for name, table in expected.items():
        page, frame = fetch(result_id, name), table['.DataFrame']
        assert result['tables'][name] == {'rows': len(frame['data']), 'columns': frame['columns']}
        assert (page['index'], page['data']) == (frame['index'], frame['data'])
    assert mtm(result_id)['CF1'] == pytest.approx(AMOUNT * SPOT * np.exp(-RATE * 2.0), rel=1e-9)


def test_the_result_carries_the_replay_tuple():
    """A reported number is worth what it can be reproduced from, so the four coordinates travel
    with it — and the two hashes are the loaded job's own, not something the service re-derived."""
    context = in_process(job())
    _, result = run(job())

    assert result['plan_hash'] == context.plan_hash()
    assert result['values_hash'] == context.values_hash()
    assert result['engine_version'] == derivus.__version__
    assert result['seed'] == 1


def test_the_schema_endpoint_is_the_declarations_plus_the_version():
    """What makes a front end thin: it renders panels, tables and enums from `schema.mapping`
    rather than restating them, so the endpoint is that store and the version that emitted it."""
    published = CLIENT.get('/schema').json()

    assert published.pop('engine_version') == derivus.__version__
    assert published == json.loads(json.dumps(derivus.schema.mapping, cls=CustomJsonEncoder))
    # not vacuous: this is the declaration a client reads to know which fields it may patch
    assert published['Factor']['types']['FxRate']['Spot']['bind'] == 'value'


def test_the_schema_publishes_which_deals_take_children():
    """`containers` is `Deal.accepts_children` emitted into the store, so a client - a browser
    SPA, an MCP tool booking under a netting set - answers "may this take children" without
    importing the engine to ask. Held to the accessor over EVERY declared type, both directions,
    and non-vacuously in both."""
    published = CLIENT.get('/schema').json()['Instrument']
    accessor = sorted(t for t in published['types'] if derivus.instruments.accepts_children(t))

    assert published['containers'] == accessor
    assert 'NettingCollateralSet' in published['containers']
    assert 'FixedCashflowDeal' not in published['containers']
    # a container the create menu does not offer would be bookable over MCP and uncreatable in
    # every UI - the same drift test_no_class_is_hidden_from_the_create_menu makes of types
    menued = {t for members in published['groups'].values() for t in members}
    assert set(published['containers']) <= menued


def test_a_done_result_carries_the_run_stats():
    """`Stats` is the run's own account of itself - timings, deals loaded, calibration provenance -
    and dropping it at the store made it unreachable from every HTTP client. It rides the summary
    as a flat dict, never through `tables_of` (which would flatten `Calibrations` into a fake
    table path), and a calculation that reports none reads as `{}` rather than a KeyError."""
    _, result = run(job())

    assert result['stats']['Deals loaded'] == 1
    assert 'stats' not in result['tables'] and 'Stats' not in result['tables']

    service.EXECUTOR.submit(service.Job('statless', Held('statless', [], results={}), {}),
                            service.HEAVY)
    service.EXECUTOR.queue.join()
    assert CLIENT.get('/results/statless').json()['stats'] == {}


def test_the_ui_is_mounted_only_when_it_is_built(tmp_path):
    """The UI is a CLIENT of the service, optional to the core library, so the mount is a flag over
    a directory rather than an import-time assumption - an empty directory refuses. The 404 on
    `/ui/portfolio` is pinned deliberately: `StaticFiles(html=True)` has no SPA fallback, which is
    the fact the front end's no-router decision rests on - a router would ship deep links that 404
    on reload, and this gate is where that lands first."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient as Client

    assert service.mount_ui(FastAPI(), str(tmp_path)) is False

    (tmp_path / 'index.html').write_text('<!doctype html><title>derivus</title>UI-MARKER')
    mounted = FastAPI()
    assert service.mount_ui(mounted, str(tmp_path)) is True
    ui = Client(mounted)
    assert 'UI-MARKER' in ui.get('/ui/').text
    assert ui.get('/ui', follow_redirects=False).status_code in (301, 307)
    assert ui.get('/ui/portfolio').status_code == 404


#: What a client books into the live book: the same cashflow shape, its own reference and size.
BOOKED = dict(CASHFLOW, Reference='CF2', Amount=250_000.0)


@pytest.fixture
def book(tmp_path):
    """A live book over a temp copy of the one-cashflow job, taken down after the gate. Written at
    indent 2, which is what the formatting gate holds the rewrite to."""
    path = tmp_path / 'book.json'
    path.write_text(json.dumps(json.loads(dump(job())), indent=2), newline='\n')
    service.BOOK = service.Book(str(path))
    yield path
    service.BOOK = None


def test_a_missing_book_file_starts_blank_and_takes_its_first_booking(tmp_path):
    """A fresh desk has no job file, so `--book` at an empty path creates the blank book: no
    deals, dated today, the skeleton's USD market data still aboard - which is what lets the very
    first booking validate instead of being refused for market data a bare file would lack."""
    import datetime
    path = tmp_path / 'desk.json'
    service.BOOK = service.open_book(str(path))
    try:
        live = CLIENT.get('/book').json()
        assert live['document']['Calc']['Deals']['Deals']['Children'] == []
        assert live['document']['Calc']['Calculation']['Base_Date'] == {
            '.Timestamp': datetime.date.today().strftime('%Y-%m-%d')}
        assert CLIENT.post('/validate', json=live['document']).json() == {
            'deals': {}, 'factors': []}

        first = CLIENT.post('/book/deals', content=dump({'action': 'add', 'deal': {
            'Object': 'FixedCashflowDeal', 'Reference': 'FIRST', 'Currency': 'USD',
            'Discount_Rate': 'USD', 'Calendars': None, 'Amount': 1000.0,
            'Payment_Date': BASE + pd.DateOffset(years=1)}}), headers=JSON).json()
        assert first['written'] is True and first['deal_path'] == '0'
        assert json.loads(path.read_text())['Calc']['Deals']['Deals']['Children'][0][
            'Instrument']['.Deal']['Reference'] == 'FIRST'
        # reopening an existing book must never overwrite it
        service.BOOK = service.open_book(str(path))
        assert len(CLIENT.get('/book').json()['document']['Calc']['Deals']['Deals'][
            'Children']) == 1
    finally:
        service.BOOK = None


def test_without_a_book_the_book_verbs_are_a_404():
    """A service started without `--book` has no book, and a miss is a refusal naming the fix -
    never a book invented in memory that no file backs."""
    assert CLIENT.get('/book').status_code == 404
    assert '--book' in CLIENT.get('/book').json()['detail']


def test_a_booking_lands_in_the_file_and_every_client_sees_it(book):
    """The file is the source of truth: the booked deal is in the answer, in the file on disk and
    in the next GET - with a moved etag, which is the one question a polling client asks."""
    before = CLIENT.get('/book').json()
    outcome = CLIENT.post('/book/deals', content=dump({'action': 'add', 'deal': BOOKED}),
                          headers=JSON).json()
    after = CLIENT.get('/book').json()
    on_disk = json.loads(book.read_text())

    assert outcome['written'] is True and outcome['deal_path'] == '1'
    assert before['etag'] != after['etag'] == outcome['etag']
    assert after['document'] == on_disk
    assert on_disk['Calc']['Deals']['Deals']['Children'][1][
        'Instrument']['.Deal']['Reference'] == 'CF2'
    # and the written file is still a job the other binding loads and validates clean
    assert in_process(on_disk).validate() == {'deals': {}, 'factors': []}


def test_a_rejected_booking_touches_nothing(book):
    """Validate-before-write, refused on both counts at once: an authoring message against the
    booked deal, and market data the book does not carry. File bytes and etag stand still, and the
    refusal is an ANSWER carrying the messages - the caller reads them to fix the deal."""
    before = book.read_bytes()
    etag = CLIENT.get('/book').json()['etag']
    outcome = CLIENT.post('/book/deals', content=dump({'action': 'add', 'deal': BINARY}),
                          headers=JSON).json()

    assert outcome['written'] is False
    assert 'Cash_Payoff is required' in outcome['refused']
    assert any('EquityPrice.EQ' in message for message in outcome['refused'])
    assert book.read_bytes() == before
    assert CLIENT.get('/book').json()['etag'] == etag


def test_a_booking_naming_market_data_the_book_lacks_is_refused(book):
    """A deal with clean authoring but a curve the book has no block for would load and then be
    silently DROPPED by discovery - the wrong kind of quiet for a booking verb - so the delta of
    missing factors refuses it by name. The book's own pre-existing gaps do not block: only what
    this booking adds."""
    outcome = CLIENT.post('/book/deals', content=dump(
        {'action': 'add',
         'deal': dict(CASHFLOW, Reference='CF9', Currency='GBP', Discount_Rate='GBP')}),
        headers=JSON).json()

    assert outcome['written'] is False
    assert 'no market data for InterestRate.GBP' in outcome['refused']


def test_booking_then_deleting_restores_the_file_bytes(book):
    """The rewrite keeps the file's own indent, so book-then-delete is a no-op to the byte - what
    makes the book diffable and a booking reviewable as the diff of the deal and nothing else."""
    before = book.read_bytes()
    booked = CLIENT.post('/book/deals', content=dump({'action': 'add', 'deal': BOOKED}),
                         headers=JSON).json()
    deleted = CLIENT.post(
        '/book/deals', json={'action': 'delete', 'deal_path': booked['deal_path']}).json()

    assert booked['written'] and deleted['written'] and deleted['deleted'] == 'CF2'
    assert book.read_bytes() == before


def test_a_parent_must_exist_be_unique_and_take_children(book):
    """Appending under the wrong node is a mis-booked trade, not an error message - so a parent
    that is a leaf refuses naming its type, an unknown one refuses naming it, and neither writes.
    `containers` (C2) is what makes the leaf refusal expressible without importing the engine."""
    under_leaf = CLIENT.post('/book/deals', content=dump(
        {'action': 'add', 'deal': BOOKED, 'parent_reference': 'CF1'}), headers=JSON)
    unknown = CLIENT.post('/book/deals', content=dump(
        {'action': 'add', 'deal': BOOKED, 'parent_reference': 'GHOST'}), headers=JSON)

    assert under_leaf.status_code == 422 and 'FixedCashflowDeal' in under_leaf.json()['detail']
    assert unknown.status_code == 422 and 'GHOST' in unknown.json()['detail']
    assert len(CLIENT.get('/book').json()['document']['Calc']['Deals']['Deals']['Children']) == 1


def test_a_booking_nests_under_a_container(book):
    """A container books like any deal and then holds its children: the nested node lands in the
    parent's `Children`, and its `deal_path` is the positional identity every client shares."""
    net = {'Object': 'StructuredDeal', 'Reference': 'STR1', 'Currency': 'ZAR'}
    first = CLIENT.post('/book/deals', content=dump({'action': 'add', 'deal': net}),
                        headers=JSON).json()
    second = CLIENT.post('/book/deals', content=dump(
        {'action': 'add', 'deal': BOOKED, 'parent_reference': 'STR1'}), headers=JSON).json()
    node = json.loads(book.read_text())['Calc']['Deals']['Deals']['Children'][1]

    assert first['written'] and first['deal_path'] == '1'
    assert second['written'] and second['deal_path'] == '1/0'
    assert node['Instrument']['.Deal']['Reference'] == 'STR1'
    assert node['Children'][0]['Instrument']['.Deal']['Reference'] == 'CF2'


def test_an_amendment_lands_in_the_file(book):
    """The edit flow the UI rides: merge one field into a deal at its path, validated first,
    written atomically - and the etag moves so every polling client repaints."""
    etag = CLIENT.get('/book').json()['etag']
    outcome = CLIENT.post('/book/deals', json={
        'action': 'amend', 'deal_path': '0', 'fields': {'Amount': 750_000.0}}).json()
    on_disk = json.loads(book.read_text())

    assert outcome['written'] is True and outcome['deal_path'] == '0'
    assert outcome['etag'] != etag
    assert on_disk['Calc']['Deals']['Deals']['Children'][0][
        'Instrument']['.Deal']['Amount'] == 750_000.0
    assert in_process(on_disk).validate() == {'deals': {}, 'factors': []}


def test_a_bad_amendment_touches_nothing(book):
    """The same validate-delta rule as a booking, wired to the amend branch: pointing CF1 at a
    curve the book lacks refuses by name and leaves the file's bytes standing."""
    before = book.read_bytes()
    outcome = CLIENT.post('/book/deals', json={
        'action': 'amend', 'deal_path': '0', 'fields': {'Discount_Rate': 'GBP'}}).json()

    assert outcome['written'] is False
    assert 'no market data for InterestRate.GBP' in outcome['refused']
    assert book.read_bytes() == before


def test_amending_back_is_byte_identical(book):
    """An edit undone leaves no trace, not even a reformat - the same discipline as
    book-then-delete, so an amendment is reviewable as the diff of the field and nothing else."""
    before = book.read_bytes()
    original = json.loads(book.read_text())['Calc']['Deals']['Deals']['Children'][0][
        'Instrument']['.Deal']['Amount']
    CLIENT.post('/book/deals', json={
        'action': 'amend', 'deal_path': '0', 'fields': {'Amount': 1.0}})
    CLIENT.post('/book/deals', json={
        'action': 'amend', 'deal_path': '0', 'fields': {'Amount': original}})
    assert book.read_bytes() == before


def test_an_amendment_needs_a_real_path(book):
    """An unknown path is a 422 naming it, and a NEGATIVE path refuses rather than silently
    resolving from the end - a wrong path must never quietly amend a different deal."""
    unknown = CLIENT.post('/book/deals', json={
        'action': 'amend', 'deal_path': '7', 'fields': {'Amount': 1.0}})
    negative = CLIENT.post('/book/deals', json={
        'action': 'amend', 'deal_path': '-1', 'fields': {'Amount': 1.0}})
    assert unknown.status_code == 422 and '7' in unknown.json()['detail']
    assert negative.status_code == 422
    assert json.loads(book.read_text())['Calc']['Deals']['Deals']['Children'][0][
        'Instrument']['.Deal']['Amount'] == AMOUNT


def test_a_what_if_prices_the_candidate_and_writes_nothing(book):
    """The par-solve verb: the book plus a candidate priced off an in-memory copy, the file never
    moving. The candidate's own value comes back through the ordinary result surface, which is
    what lets a caller solve an amount against a target and only then book it."""
    before = book.read_bytes()
    submitted = CLIENT.post('/book/price', content=dump({'deal': BOOKED}), headers=JSON).json()
    service.EXECUTOR.queue.join()
    result = CLIENT.get('/results/{}'.format(submitted['result_id'])).json()

    assert result['status'] == 'done'
    assert mtm(submitted['result_id'])['CF2'] == pytest.approx(
        BOOKED['Amount'] * SPOT * np.exp(-RATE * 2.0), rel=1e-3)
    assert book.read_bytes() == before


def fx_vol_snapshot():
    """A USDZAR snapshot built through the Bloomberg package's own normalization - canned
    observations standing in for the terminal, everything downstream of them the real pipeline.
    The block a quote source posts and the snapshot a fetch returns are both this one object, so
    the `/book/market` gates and the `/book/bloomberg` gates tick the same numbers."""
    from derivus_bloomberg import (FXQuoteSecurity, FXVolDefinition, RawBloombergObservation,
                                   normalize_fx_vol)
    raw = {('3M', 'ATM', None): 14.0, ('3M', 'RR', 0.25): -1.2, ('3M', 'BF', 0.25): 0.35,
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


def fx_vol_quotes():
    """That snapshot as the `Market Prices` block a quote source posts to `/book/market`."""
    from derivus_bloomberg import to_market_prices_block
    return {'FXVolPrices.USD.ZAR': to_market_prices_block(fx_vol_snapshot())}


FX_OPTION = {'Object': 'FXOptionDeal', 'Reference': 'OPT1', 'Currency': 'USD',
             'Underlying_Currency': 'ZAR', 'Underlying_Amount': 1_000_000.0,
             'Strike_Price': SPOT, 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
             'Option_Style': 'European', 'Expiry_Date': BASE + pd.DateOffset(years=1),
             'FX_Volatility': 'USD.ZAR', 'Discount_Rate': 'USD'}


def test_a_bloomberg_snapshot_reaches_a_solved_strike(tmp_path):
    """THE practical loop, end to end through one service: canned Bloomberg observations run
    through `derivus_bloomberg`'s own normalization, `/book/market` installs the quote block and
    bootstraps it into the `FXVol` surface the book file then carries, and `/book/solve` finds
    the strike at which an FX option on that surface marks at the target premium. The before/after
    validate pins the surface as load-bearing: the option is unpriceable until the tick lands."""
    path = tmp_path / 'book.json'
    path.write_text(json.dumps(json.loads(dump(job(
        sections={'Bootstrapper Configuration': {'FXVolSurfaceParameters': {}}}))), indent=2), newline='\n')
    service.BOOK = service.Book(str(path))
    try:
        with_option = json.loads(dump(job(
            deals=(CASHFLOW, FX_OPTION),
            sections={'Bootstrapper Configuration': {'FXVolSurfaceParameters': {}}})))
        assert 'FXVol.USD.ZAR' in CLIENT.post('/validate', json=with_option).json()['factors']

        ticked = CLIENT.post('/book/market', content=dump({'quotes': fx_vol_quotes()}),
                             headers=JSON).json()
        on_disk = json.loads(path.read_text())['Calc']['MergeMarketData']['ExplicitMarketData']

        assert ticked['written'] is True
        assert ticked['installed'] == ['FXVolPrices.USD.ZAR']
        assert 'FXVol.USD.ZAR' in ticked['new_factors']
        assert on_disk['Price Factors']['FXVol.USD.ZAR']['Surface_Type'] == 'Malz'
        assert 'FXVolPrices.USD.ZAR' in on_disk['Market Prices']

        target = 500_000.0
        submitted = CLIENT.post('/book/solve', content=dump({
            'deal': FX_OPTION, 'field': 'Strike_Price', 'target': target,
            'bounds': [12.0, 30.0]}), headers=JSON).json()
        service.EXECUTOR.queue.join()
        result = CLIENT.get('/results/{}'.format(submitted['result_id'])).json()
        solved = result['stats']['Solved']

        assert result['status'] == 'done'
        assert 12.0 < solved['value'] < 30.0 and abs(solved['residual']) <= 0.01
        assert mtm(submitted['result_id'])['OPT1'] == pytest.approx(target, abs=0.01)
    finally:
        service.BOOK = None


def test_gamma_travels_the_served_path(tmp_path):
    """Trading options needs second order, and base valuation HAS it - `Greeks: 'All'` is the
    gated cross-gamma block (`test_base_valuation_gamma` owns its oracles). This pins the SERVED
    route a desk actually uses: a what-if with `calculation_overrides` returns `Greeks_Second`,
    its cells are the in-process run's to the bit (the house parity discipline), and the spot
    diagonal is a live positive gamma with the vanna cross carrying real weight beside it."""
    path = tmp_path / 'book.json'
    path.write_text(json.dumps(json.loads(dump(job(
        sections={'Bootstrapper Configuration': {'FXVolSurfaceParameters': {}}}))), indent=2), newline='\n')
    service.BOOK = service.Book(str(path))
    try:
        CLIENT.post('/book/market', content=dump({'quotes': fx_vol_quotes()}), headers=JSON)
        submitted = CLIENT.post('/book/price', content=dump({
            'deal': FX_OPTION, 'calculation_overrides': {'Greeks': 'All'}}), headers=JSON).json()
        service.EXECUTOR.queue.join()
        summary = CLIENT.get('/results/{}'.format(submitted['result_id'])).json()
        served = fetch(submitted['result_id'], 'Greeks_Second')

        # the other binding, over the identical document the what-if built
        document = json.loads(path.read_text())
        document['Calc']['Deals']['Deals']['Children'].append(
            {'Instrument': {'.Deal': json.loads(dump(FX_OPTION))}})
        document['Calc']['Calculation']['Greeks'] = 'All'
        _, out = in_process(document).run_job()
        frame = out['Results']['Greeks_Second']

        assert summary['status'] == 'done' and 'Greeks_Second' in summary['tables']
        assert (served['rows'], len(served['columns'])) == frame.shape
        for served_row, true_row in zip(served['data'], frame.values):
            assert served_row == pytest.approx(list(true_row), rel=1e-12)
        spot = [i for i, label in enumerate(served['index']) if 'FxRate.ZAR' in label][0]
        spot_col = [i for i, label in enumerate(served['columns']) if 'FxRate.ZAR' in label][0]
        assert served['data'][spot][spot_col] > 0, 'the spot diagonal is a real gamma'
        vanna = [abs(cell) for i, row in enumerate(served['data']) for j, cell in enumerate(row)
                 if i != j]
        assert max(vanna) > 0, 'the off-diagonal block is empty - crosses were dropped'
    finally:
        service.BOOK = None


def test_a_quote_update_may_move_only_the_numbers(book):
    """The structure guard, generalized to every family: a re-post moving only
    `Quoted_Market_Value`/`Timestamp` updates; one moving a pillar refuses by name with the file
    untouched - a moved node is a new plan, never a tick.

    The stamp MOVES here, because it is the one member of `schema.MARKET_QUOTE_VALUES` nothing else
    in the repo re-posts through the guard: this docstring claimed it while the body sent the mid
    alone, and a guard misspelling `Timestamp` passed `test_market_prices_partition`, `test_service`
    and `test_mcp` together at 118 passed."""
    quotes = fx_vol_quotes()
    doc = json.loads(book.read_text())
    doc['Calc']['MergeMarketData']['ExplicitMarketData'][
        'Bootstrapper Configuration'] = {'FXVolSurfaceParameters': {}}
    book.write_text(json.dumps(doc, indent=2), newline='\n')

    first = CLIENT.post('/book/market', content=dump({'quotes': quotes}), headers=JSON).json()
    assert first['installed'] == ['FXVolPrices.USD.ZAR']

    ticked = json.loads(dump(fx_vol_quotes()))
    later = json.loads(dump({'stamp': pd.Timestamp('2024-06-28 17:45')}))['stamp']
    for point in ticked['FXVolPrices.USD.ZAR']['instrument']['Points']:
        point['Timestamp'] = later
        if point['Quote_Type'] == 'ATM':
            point['Quoted_Market_Value'] += 0.01
    assert all(point['Timestamp'] != later for point in
               json.loads(dump(fx_vol_quotes()))['FXVolPrices.USD.ZAR']['instrument']['Points']), (
        'the stamp was posted back at the one it already carried')
    before = book.read_bytes()
    second = CLIENT.post('/book/market', content=dump({'quotes': ticked}), headers=JSON).json()
    assert second['updated'] == ['FXVolPrices.USD.ZAR'] and second['written'] is True

    moved = json.loads(dump(fx_vol_quotes()))
    for point in moved['FXVolPrices.USD.ZAR']['instrument']['Points']:
        point['Pillar'] = 0.1
    after_update = book.read_bytes()
    refused = CLIENT.post('/book/market', content=dump({'quotes': moved}), headers=JSON)
    assert refused.status_code == 422 and 'structure differs' in refused.json()['detail']
    assert book.read_bytes() == after_update != before


def test_a_two_way_ticks_beside_the_mid_and_a_moved_pillar_still_refuses(book):
    """The same guard, now that a point may carry a two-way: `Quoted_Bid`/`Quoted_Ask` are on the
    VALUE side of the line the mid is on.

    A spread widens between one print and the next, and a pillar can start or stop being quoted
    two-sided without becoming a different node - so a re-post moving bid, ask and the mid together
    is a tick, and the file takes it. What is structure has not changed: the same post with a moved
    `Pillar` still refuses in the identical wording, and the book is untouched by it.

    The bootstrap runs on every one of these posts, which is the other half of the claim - the
    surface it writes is built from `Quoted_Market_Value` alone, so a block carrying the sides
    ticks a book exactly as a mid-only block does.
    """
    quotes = json.loads(dump(fx_vol_quotes()))
    for point in quotes['FXVolPrices.USD.ZAR']['instrument']['Points']:
        point['Quoted_Bid'] = point['Quoted_Market_Value'] - 0.002
        point['Quoted_Ask'] = point['Quoted_Market_Value'] + 0.002
    doc = json.loads(book.read_text())
    doc['Calc']['MergeMarketData']['ExplicitMarketData'][
        'Bootstrapper Configuration'] = {'FXVolSurfaceParameters': {}}
    book.write_text(json.dumps(doc, indent=2), newline='\n')

    installed = CLIENT.post('/book/market', content=dump({'quotes': quotes}), headers=JSON).json()
    assert installed['installed'] == ['FXVolPrices.USD.ZAR'] and installed['written'] is True

    ticked = json.loads(json.dumps(quotes))
    for point in ticked['FXVolPrices.USD.ZAR']['instrument']['Points']:
        if point['Quote_Type'] == 'ATM':
            point['Quoted_Market_Value'] += 0.01
            point['Quoted_Bid'] += 0.008
            point['Quoted_Ask'] += 0.012
    second = CLIENT.post('/book/market', content=dump({'quotes': ticked}), headers=JSON).json()
    on_disk = json.loads(book.read_text())['Calc']['MergeMarketData']['ExplicitMarketData']
    atm = [point for point in on_disk['Market Prices'][
        'FXVolPrices.USD.ZAR']['instrument']['Points'] if point['Quote_Type'] == 'ATM']

    assert second['updated'] == ['FXVolPrices.USD.ZAR'] and second['written'] is True
    assert all(point['Quoted_Ask'] - point['Quoted_Bid'] == pytest.approx(0.008) for point in atm), (
        'the widened two-way did not reach the file')

    moved = json.loads(json.dumps(ticked))
    for point in moved['FXVolPrices.USD.ZAR']['instrument']['Points']:
        point['Pillar'] = 0.1
    before = book.read_bytes()
    refused = CLIENT.post('/book/market', content=dump({'quotes': moved}), headers=JSON)

    assert refused.status_code == 422 and 'structure differs' in refused.json()['detail']
    assert book.read_bytes() == before


def test_a_market_values_patch_reaches_the_file_and_a_structural_one_is_refused(book):
    """The `patch_market`-shaped half: a spot tick lands in the file through the engine's own
    values seam, and a structural key is refused by the engine's own raise - the service adds no
    judgment of its own."""
    ticked = CLIENT.post('/book/market', json={
        'patch': {'FxRate.ZAR': {'Spot': 19.25}}}).json()
    on_disk = json.loads(book.read_text())

    assert ticked['written'] is True and ticked['patched'] == ['FxRate.ZAR']
    assert on_disk['Calc']['MergeMarketData']['ExplicitMarketData'][
        'Price Factors']['FxRate.ZAR']['Spot'] == 19.25

    before = book.read_bytes()
    structural = CLIENT.post('/book/market', json={
        'patch': {'FxRate.ZAR': {'Interest_Rate': 'GBP'}}})
    assert structural.status_code == 422
    assert book.read_bytes() == before


def test_a_bootstrap_that_complains_writes_nothing(book):
    """A quote block the bootstrap cannot turn into a factor - here a misnamed one no family
    selects - refuses the WHOLE write with the bootstrap's own messages: a book must never carry
    a market its own bootstrap complained about."""
    doc = json.loads(book.read_text())
    doc['Calc']['MergeMarketData']['ExplicitMarketData'][
        'Bootstrapper Configuration'] = {'FXVolSurfaceParameters': {}}
    book.write_text(json.dumps(doc, indent=2), newline='\n')
    before = book.read_bytes()

    ghost = {'GhostPrices.NOWHERE': {'instrument': {'Points': []}}}
    outcome = CLIENT.post('/book/market', content=dump({'quotes': ghost}), headers=JSON).json()

    assert outcome['written'] is False
    assert any('wrote no' in message for message in outcome['refused'])
    assert book.read_bytes() == before


def built_surface(path, quotes=None):
    """A live book carrying a BUILT `FXVol.USD.ZAR`, through the two real entry points: the file is
    a job declaring the surface bootstrapper, and the surface arrives by POSTing a quote block to
    `/book/market` - the canned USDZAR one unless the caller hands in its own. What the
    Heston-Nandi gates start from is therefore a surface the engine built, never one written by
    hand."""
    path.write_text(json.dumps(json.loads(dump(job(sections={
        'Bootstrapper Configuration': {'FXVolSurfaceParameters': {}}}))), indent=2), newline='\n')
    service.BOOK = service.Book(str(path))
    ticked = CLIENT.post('/book/market', content=dump(
        {'quotes': quotes if quotes is not None else fx_vol_quotes()}), headers=JSON).json()
    assert ticked['written'] is True and 'FXVol.USD.ZAR' in ticked['new_factors']
    return path


def desk_smile():
    """A USDZAR smile quoted at 1M, 2M, 3M and 6M, as the `FXVolPrices` block a quote source posts.

    FOUR PILLARS BECAUSE THE LADDER NEEDS SIX CONTRACTS. Ten rungs are not ten quotes: every rung
    the surface does not carry snaps onto one it does, and the canned two-pillar surface collapses
    the whole ladder onto FOUR distinct contracts, which do not identify five parameters. Four
    pillars are the fewest that give the ATM term structure three distinct expiries AND put the
    two wing pairs on two different ones - eight contracts, measured - which is why 6M is here and
    9M and 1Y are not.

    THE RISK REVERSAL IS NEGATIVE, in pair terms, exactly as the canned USDZAR snapshot's is
    (-1.2 at 3M there, -1.2 here) - which is the sign that pair actually trades at. Read on the
    `FxRate.ZAR` axis the model is fitted on, that is a smile whose vol RISES with strike, and it
    is the shape a strictly positive `Gamma_Star` could not represent: the fit used to answer it
    with the leverage channel switched off and a flat smile it called converged. The signed
    leverage share is what admits it, and this fixture is what proves the admission.

    The expiries stop at 6M for the clock rather than the maths: the fit's cost is linear in the
    step count of the longest expiry (252 GARCH steps a year, Fourier-inverted per L-BFGS-B
    evaluation), and on the 3M/1Y canned surface it was measured still running past 21 minutes.
    """
    return {'FXVolPrices.USD.ZAR': {'instrument': {
        'Currency': 'USD', 'Delta_Type': 'Forward', 'Premium_Adjusted': 'Yes',
        'ATM_Convention': 'Delta_Neutral_Straddle', 'Grid_Tolerance': 1e-4,
        'Quote_Sensitivity': 'No',
        'Points': [{'Use': 'Yes', 'Expiry': expiry, 'Pillar': pillar, 'Quote_Type': quote_type,
                    'Quoted_Market_Value': value,
                    'Timestamp': pd.Timestamp('2024-06-28 16:30')}
                   for expiry, atm, rr, bf in ((1.0 / 12.0, 0.140, -0.010, 0.0030),
                                               (2.0 / 12.0, 0.142, -0.011, 0.0032),
                                               (0.25, 0.145, -0.012, 0.0035),
                                               (0.5, 0.150, -0.014, 0.0040))
                   for pillar, quote_type, value in ((0.0, 'ATM', atm), (0.25, 'RR', rr),
                                                     (0.25, 'BF', bf))]}}}


def hn_block(path):
    """The emitter, run the way the verb runs it: off a Context over the book file on disk."""
    from derivus.bootstrappers import HestonNandiModelParameters

    params = in_process(json.loads(path.read_text())).current_cfg.params
    return HestonNandiModelParameters.fx_surface_block(
        'USD.ZAR', params['Price Factors'], params['System Parameters'],
        params['Price Factor Interpolation'])


def test_the_hn_ladder_is_ten_vega_weighted_points_on_the_surfaces_own_strikes(tmp_path):
    """The desk's ladder, off the built surface: ATM at 1M/2M/3M/6M/9M/1Y plus the 25 delta wings
    at 3M and 6M, ten points, nothing past a year.

    What is asserted is what makes them the SURFACE'S points rather than a moneyness grid laid over
    it. The expiries are the surface's own - this one carries 1M/2M/3M/6M, so the 9M and 1Y rungs
    move to the nearest quoted one AT OR UNDER a year and the block SAYS SO in `Quote_Source`,
    which is the whole difference between a substitution and a silent interpolation. Ten rungs are
    therefore EIGHT distinct contracts here, and the count is what the fit is actually given. The
    weights are Black vega, normalised, so they sum to one and the back ATM outweighs the front
    (vega grows with root time) - the property that stops an unweighted fit from serving the back
    end and abandoning the front. And the wings straddle the spot: the strike solved for the pillar
    delta on one side is above it, the other below.
    """
    from derivus.bootstrappers import HestonNandiModelParameters

    name, block = hn_block(built_surface(tmp_path / 'book.json',
                                         json.loads(dump(desk_smile()))))
    try:
        instrument = block['instrument']
        points = instrument['European_Options']
        expiries = sorted({point['Expiry_Date'] for point in points})

        assert name == 'HestonNandiModelPrices.ZAR'
        assert len(points) == 10
        # the surface carries 1/12, 2/12, 0.25 and 0.5 in years, and nothing else
        assert [str(x.date()) for x in expiries] == [
            '2024-07-28', '2024-08-28', '2024-09-27', '2024-12-27']
        assert len({(point['Expiry_Date'], point['Strike']) for point in points}) == 8, (
            'the ladder collapsed further than the fixture says it does')
        assert 'moved to the nearest quoted' in instrument['Quote_Source']
        for moved in ('ATM 0.75 -> 0.5', 'ATM 1 -> 0.5'):
            assert moved in instrument['Quote_Source'], moved
        # the surface's own as-of travels onto the block, to the minute - staleness is data
        assert '2024-06-28' in str(instrument['Quote_Timestamp'])
        assert '16:30' in str(instrument['Quote_Timestamp'])

        # the six ATM rungs are emitted in ladder order, then the two wing pairs
        atm, wings = points[:6], points[6:]
        assert sum(point['Weight'] for point in points) == pytest.approx(1.0)
        assert {point['Expiry_Date'] for point in atm} == set(expiries)
        assert atm[0]['Weight'] < atm[-1]['Weight'], 'the front ATM outweighs the back one'
        assert atm[0]['Expiry_Date'] == expiries[0] and atm[-1]['Expiry_Date'] == expiries[-1]

        assert len({point['Strike'] for point in wings}) == 4, 'the wings sit on two expiries'
        below = [point for point in wings if point['Option_Type'] == 'Put']
        above = [point for point in wings if point['Option_Type'] == 'Call']
        assert len(below) == 2 and len(above) == 2
        assert all(point['Strike'] < SPOT for point in below)
        assert all(point['Strike'] > SPOT for point in above)
        # the wings carry the smile, not the ATM vol repeated - and this pair's RISES with strike
        # in the underlying's own units, which is the sign a one-signed Gamma_Star could not fit
        assert below[0]['Quoted_Market_Value'] < above[0]['Quoted_Market_Value']

        # the block resolves: every reference the fit hard-reads types off the book's own factors
        from derivus import riskfactors

        params = in_process(json.loads((tmp_path / 'book.json').read_text())).current_cfg.params
        factors, interp = params['Price Factors'], params['Price Factor Interpolation']
        assert [HestonNandiModelParameters.resolve(instrument, field, factors)
                for field in ('Underlying', 'Volatility', 'Discount_Rate', 'Yield')] == [
            utils.Factor('FxRate', ('ZAR',)), utils.Factor('FXVol', ('USD', 'ZAR')),
            utils.Factor('InterestRate', ('USD',)), utils.Factor('InterestRate', ('ZAR',))]
        # the surface's axis is log(F/K) on the PAIR, so the lookup is against the forward and
        # inverted - the same pair of switches an FXOptionDeal on this surface sets
        assert (instrument['Use_Forward'], instrument['Invert_Moneyness']) == ('Yes', 'Yes')

        # THE CONVENTION CHAIN, ASSERTED: put each emitted strike back through the family's OWN
        # moneyness dispatch, off the switches the block declares, and the surface has to answer
        # the vol the block carries. That is what makes each quote the surface's own point rather
        # than a number laid beside it, and it is what would fail first if the orientation, the
        # forward or the inversion were wrong.
        #
        # THE VOL IS READ AT THE PILLAR, the strike hangs off the DATE. `Expiry_Date` is the
        # pillar rounded to whole days and the fit reads its own accrual back off it, so the
        # forward is that accrual's; the surface is read at the pillar the quote came from, which
        # under a curve that is not ACT_365 is a different number and under ACT_360 puts the 1Y
        # rung past the last expiry the surface carries
        base = params['System Parameters']['Base_Date']
        surface = riskfactors.construct_factor(
            utils.Factor('FXVol', ('USD', 'ZAR')), factors, interp)
        curve = lambda name: riskfactors.construct_factor(
            utils.Factor('InterestRate', (name,)), factors, interp)
        discount, carry = curve('USD'), curve('ZAR')
        spot = float(riskfactors.construct_factor(
            utils.Factor('FxRate', ('ZAR',)), factors, interp).current_value()[0])
        for point in points:
            days = (point['Expiry_Date'] - base).days
            t = discount.get_day_count_accrual(base, days)
            pillar, _ = HestonNandiModelParameters.fx_surface_expiry(
                surface, days / HestonNandiModelParameters.fx_days_per_year,
                max(HestonNandiModelParameters.fx_atm_expiries))
            forward = spot * np.exp(
                (float(discount.current_value(t)) - float(carry.current_value(t))) * t)
            moneyness = HestonNandiModelParameters.moneyness(
                point['Strike'], spot, forward, surface, True, True)
            assert float(surface.current_value([[moneyness, pillar]])[0]) == pytest.approx(
                point['Quoted_Market_Value'], rel=1e-9)
    finally:
        service.BOOK = None


def hand_authored_hn_block(vols):
    """A `HestonNandiModelPrices.ZAR` block with nine quotes at ONE WEEK, TWO and THREE.

    The emitter's own ladder reaches six months and its fit is a wall-clock quarter of an hour;
    what the shift gate needs is the fit's ARITHMETIC, not its ladder, so the expiries here are
    the shortest that still make three step counts (5, 10 and 15 GARCH steps) and the whole gate
    runs in seconds. Everything else is the block the emitter writes - the same references, the
    same switches, the same columns.
    """
    return {'instrument': {
        'Underlying': 'ZAR', 'Underlying_Type': 'FxRate',
        'Volatility': 'USD.ZAR', 'Volatility_Type': 'FXVol',
        'Discount_Rate': 'USD', 'Discount_Rate_Type': 'InterestRate',
        'Yield': 'ZAR', 'Yield_Type': 'InterestRate',
        'Quote_Type': 'Implied_Volatility', 'Use_Forward': 'Yes', 'Invert_Moneyness': 'Yes',
        'Steps_Per_Year': 252.0, 'Quadrature_Panels': 64,
        'European_Options': [
            {'Expiry_Date': BASE + pd.DateOffset(days=days), 'Strike': SPOT * ratio,
             'Option_Type': 'Call' if ratio >= 1.0 else 'Put', 'Units': 1.0, 'Weight': 1.0 / 9.0,
             'Quoted_Market_Value': vol}
            for days in (7, 14, 21)
            for ratio, vol in zip((0.95, 1.0, 1.05), vols)]}}


def fitted_five(path, block, delta):
    """The five parameters a book fits `block` to, at `Volatility_Delta` `delta` - through the
    market seam, off the file, exactly as a tick calibrates."""
    document = json.loads(path.read_text())
    market = document['Calc']['MergeMarketData']['ExplicitMarketData']
    market['System Parameters']['Volatility_Delta'] = delta
    market['Bootstrapper Configuration'] = {'HestonNandiModelParameters': {}}
    market.get('Market Prices', {}).pop('HestonNandiModelPrices.ZAR', None)
    path.write_text(json.dumps(document, indent=2), newline='\n')
    service.BOOK = service.Book(str(path))
    written = CLIENT.post('/book/market', content=dump(
        {'quotes': {'HestonNandiModelPrices.ZAR': block}}), headers=JSON).json()
    assert written['written'] is True, written
    factor = json.loads(path.read_text())['Calc']['MergeMarketData']['ExplicitMarketData'][
        'Price Factors']['HestonNandiModelParameters.ZAR']
    return {key: factor[key] for key in utils.HN_PARAM_NAMES}


def test_a_volatility_delta_moves_the_fitted_world_once(tmp_path):
    """A scenario shift is the FIT's business, and it must reach the fitted world exactly once.

    It used to reach it twice. `fx_surface_block` folded `Volatility_Delta` into every vol it
    emitted, and `bootstrap` - which reads `Quoted_Market_Value` when the block carries one - added
    `vol_surface.delta` to it again. A 1-vol-point scenario therefore calibrated a 2-vol-point
    world, and nothing said so: both numbers are plausible and neither is reported.

    THE TWO HALVES, asserted separately. The emitter is delta-BLIND: a block authored at 0.01 is
    the block authored at 0.0, bit for bit, because a quote block is a QUOTE and the surface it
    reads is the mid one the book carries either way. And the fit applies the shift ONCE, which is
    stated as an identity between two real calibrations: fitting the unshifted quotes under a 0.01
    scenario must land on the same five parameters as fitting the HAND-BUMPED quotes under no
    scenario at all. The doubled world is what the third fit rules out - the shift has to MOVE the
    answer, or the identity would hold for a shift that did nothing.
    """
    path = built_surface(tmp_path / 'book.json', json.loads(dump(desk_smile())))
    try:
        unshifted = hn_block(path)[1]['instrument']['European_Options']
        document = json.loads(path.read_text())
        document['Calc']['MergeMarketData']['ExplicitMarketData'][
            'System Parameters']['Volatility_Delta'] = 0.01
        path.write_text(json.dumps(document, indent=2), newline='\n')
        service.BOOK = service.Book(str(path))
        shifted = hn_block(path)[1]['instrument']['European_Options']

        assert [point['Quoted_Market_Value'] for point in shifted] == [
            point['Quoted_Market_Value'] for point in unshifted], (
            'the emitted block moved with a scenario shift - the block is a quote')
        assert [point['Strike'] for point in shifted] == [
            point['Strike'] for point in unshifted], 'the strikes moved with the shift'

        vols = (0.14, 0.145, 0.15)
        scenario = fitted_five(path, hand_authored_hn_block(vols), 0.01)
        by_hand = fitted_five(path, hand_authored_hn_block(
            tuple(vol + 0.01 for vol in vols)), 0.0)
        unmoved = fitted_five(path, hand_authored_hn_block(vols), 0.0)

        # MEASURED at 7.7e-9 relative, which is the two worlds' quoted vols differing by one ulp
        # (`q + delta` inside the fit against `q + 0.01` written out) amplified by a line search,
        # not a second application of anything: a doubled shift moves these by percent
        assert scenario == pytest.approx(by_hand, rel=1e-6), (
            'a 1 vol point scenario did not fit the world 1 vol point away')
        assert scenario != pytest.approx(unmoved, rel=1e-3), (
            'the shift moved nothing, so the identity above is vacuous')
    finally:
        service.BOOK = None


def test_a_collapsed_ladder_refuses_and_nothing_past_a_year_is_ever_snapped_to(tmp_path):
    """The two things an unconditional argmin does not do, and both are silent when they happen.

    A COLLAPSED LADDER. The canned surface carries 3M and 1Y and nothing else, so all six ATM
    rungs land on two contracts and both wing pairs land on one expiry - FOUR distinct contracts,
    handed to a five-parameter fit as though they were ten quotes. The repeated ones are weight,
    not observation: what identifies `H0`, `Beta` and `Omega` is the ATM TERM STRUCTURE, and a
    collapse is precisely the destruction of it. So the emitter refuses, naming the surface's own
    pillars, the ladder it was asked for, the count it collapsed to and what to do about it. The
    `expiry.size < 2` guard below it cannot see this - the canned surface IS a grid.

    NOTHING PAST A YEAR, ENFORCED. The rule is the desk's and it was a comment: snapping is an
    argmin over every pillar the surface carries, so a surface quoting 2Y answers the 1Y rung with
    2Y and a fit of sub-year products borrows a parameter from a smile nobody quotes. Here the same
    four-pillar surface carries a 2Y pillar as well, and the ladder must not touch it: every
    emitted expiry is a date inside the year, the 9M and 1Y rungs both land on 6M, and the block
    says so.
    """
    from derivus.bootstrappers import HestonNandiModelParameters

    with pytest.raises(ValueError) as refusal:
        hn_block(built_surface(tmp_path / 'canned.json'))
    service.BOOK = None
    assert '4 distinct contracts' in str(refusal.value)
    assert 'FXVol.USD.ZAR carries pillars 0.25/1' in str(refusal.value)
    assert 'term structure' in str(refusal.value), 'a refusal that does not say what was lost'
    assert 'more expiries' in str(refusal.value), 'a refusal without a remedy'

    long_dated = json.loads(dump(desk_smile()))
    points = long_dated['FXVolPrices.USD.ZAR']['instrument']['Points']
    for pillar, quote_type, value in ((0.0, 'ATM', 0.170), (0.25, 'RR', -0.020),
                                      (0.25, 'BF', 0.0055)):
        points.append(dict(points[0], Expiry=2.0, Pillar=pillar, Quote_Type=quote_type,
                           Quoted_Market_Value=value))

    name, block = hn_block(built_surface(tmp_path / 'book.json', long_dated))
    try:
        instrument = block['instrument']
        emitted = sorted({point['Expiry_Date'] for point in instrument['European_Options']})
        year = BASE + pd.DateOffset(days=int(HestonNandiModelParameters.fx_days_per_year))

        assert name == 'HestonNandiModelPrices.ZAR'
        assert emitted[-1] <= year, 'the ladder snapped onto a pillar past its own cap'
        assert str(emitted[-1].date()) == '2024-12-27', 'the back rungs left the 6M pillar'
        assert 'ATM 1 -> 0.5' in instrument['Quote_Source']
        assert '-> 2' not in instrument['Quote_Source'], 'a 2Y pillar reached the ladder'
    finally:
        service.BOOK = None

    # and where NOTHING is admissible, every rung is DROPPED rather than snapped up onto the
    # nearest thing past the cap - the refusal says which rung did what
    past_the_cap = json.loads(dump(desk_smile()))
    for point in past_the_cap['FXVolPrices.USD.ZAR']['instrument']['Points']:
        point['Expiry'] += 2.0
    with pytest.raises(ValueError) as dropped:
        hn_block(built_surface(tmp_path / 'past.json', past_the_cap))
    service.BOOK = None
    assert '0 distinct contracts' in str(dropped.value)
    assert 'ATM 1 DROPPED - no pillar at or under 1' in str(dropped.value)
    assert '0.25d 0.5 DROPPED' in str(dropped.value)


def test_the_hn_verb_lands_a_fitted_factor_that_reprices_its_own_quotes(tmp_path):
    """The round trip: `/book/hn` authors the block, installs it through the market seam,
    bootstraps, and the five parameters land in the book FILE - after which the model has to
    reprice the ten quotes it was fitted to.

    The yardstick is the family's OWN objective, recomputed here off the written parameters: each
    quote's Black target premium against the Heston-Nandi price of the same contract, weighted by
    the weight the block carries. A calibration that lands finite numbers and misses the smile is
    the failure mode this exists to catch.

    IT RUNS ON THE NEGATIVE RISK REVERSAL, which is the sign USDZAR actually trades at and the one
    the family could not fit at all until the leverage share carried the sign. Read on the
    `FxRate.ZAR` axis the model is fitted on, a negative pair-terms RR is a smile whose vol RISES
    with strike, and a strictly positive `Gamma_Star` could only answer it by switching the
    leverage channel off and reporting a flat smile as a converged calibration. So the gate holds
    the shape rather than the numbers: `Gamma_Star` NEGATIVE, and the optimum INTERIOR - no
    parameter pinned on a box bound, which is what a fit that cannot represent its own data does.

    WHAT IT MEASURED, so a later reading can be compared rather than guessed at, on `desk_smile`'s
    four pillars: 288 s on a quiet box and 549 s with the suite running beside it, to the same
    five numbers BIT FOR BIT - the fit is a deterministic L-BFGS-B and only the clock moves.
    `Omega` 2.757e-6, `Alpha` 7.784e-8, `Beta` 1.079e-3, `Gamma_Star`
    -3529.45, `H0` 7.027e-5 - persistence 0.9708, signed leverage share -0.9989, initial vol
    13.31% and long-run vol 15.64% against a 3M ATM quote of 14.50%. Every one of the five is
    strictly inside its box. Worst point 4.73% (the 3M 25 delta put), weighted residual 6.21e-5.

    AND THE BOUNDS BELOW ARE THAT READING AGAINST A MEASURED MUTANT rather than a wish. The mutant
    is the fit with the LEVERAGE CHANNEL removed and nothing else moved - `Alpha` 0, `Beta` = psi,
    `Omega` = omega + alpha, which holds both the persistence and the stationary per-step variance
    exactly where the fit put them, so the only thing missing is the smile. It reads worst point
    13.13% and weighted residual 2.83e-4: 4.6x the residual, on the same ten quotes. That is the
    number the flat-smile failure would land on, and it is what these bounds sit between.
    """
    import torch

    from derivus import riskfactors
    from derivus.bootstrappers import HestonNandiModelParameters

    path = built_surface(tmp_path / 'book.json', json.loads(dump(desk_smile())))
    try:
        submitted = CLIENT.post('/book/hn', json={'pair': 'USD.ZAR'}).json()
        assert submitted['factor'] == 'HestonNandiModelParameters.ZAR'
        service.EXECUTOR.queue.join()
        result = CLIENT.get('/results/{}'.format(submitted['result_id'])).json()
        outcome = result['stats']['HestonNandi']

        assert result['status'] == 'done', result
        assert outcome['written'] is True and outcome['quotes'] == 10
        assert outcome['block'] == 'HestonNandiModelPrices.ZAR'

        # the projection is the book itself - no second file, and the FILE is what a client reads
        market = json.loads(path.read_text())['Calc']['MergeMarketData']['ExplicitMarketData']
        written = market['Price Factors']['HestonNandiModelParameters.ZAR']
        assert 'HestonNandiModelPrices.ZAR' in market['Market Prices']
        # and the tick does not refit, STRUCTURALLY: the family was borrowed for this run and
        # handed back, so no later bootstrap re-enters a minutes-long least squares
        assert list(market['Bootstrapper Configuration']) == ['FXVolSurfaceParameters']
        assert [key for key in utils.HN_PARAM_NAMES if key not in written] == []
        assert all(np.isfinite(written[key]) for key in utils.HN_PARAM_NAMES)
        assert 0.0 < utils.hn_persistence(*(written[key] for key in (
            'Alpha', 'Beta', 'Gamma_Star'))) < 1.0, 'a non-stationary fit'
        # THE DEGENERATE FIT IS THE ONE TO CATCH, and it is a specific pair of numbers: the
        # leverage channel switched off (`Alpha` at zero, so the smile is flat) with `Gamma_Star`
        # pinned at its bound, which is what an inadmissible skew sign used to produce and what the
        # family still reports as convergence. A model with no skew in it prices a TARF as a
        # lognormal
        assert written['Alpha'] > 0.0, 'the leverage channel is off - the fitted smile is flat'
        # THE SIGN, which is the whole point of the signed leverage share: this surface's smile
        # RISES with strike on the axis the model is fitted on, and only a negative Gamma_Star
        # says so. A positive one here is the old box answering a shape it cannot represent
        assert written['Gamma_Star'] < 0.0, 'a rising smile fitted with equity-leverage skew'
        # AND THE OPTIMUM IS INTERIOR: no parameter sits on a box bound, which is what separates a
        # fit from a surrender. `Gamma_Star`'s box is +-[1, 5000] in magnitude
        assert 1.0 < abs(written['Gamma_Star']) < 4999.0, 'Gamma_Star is pinned at its bound'
        assert 1e-12 < written['Omega'] < 1e-3, 'Omega is pinned at a bound'
        assert 0.0 < written['Beta'], 'Beta is pinned at zero - the leverage share ran to one'
        assert 1e-10 < written['H0'] < 1e-2, 'H0 is pinned at a bound'
        assert {key: written[key] for key in utils.HN_PARAM_NAMES} == pytest.approx(
            {key: outcome['parameters'][key] for key in utils.HN_PARAM_NAMES})

        # reprice the ten, in the objective the fit minimised
        params = in_process(json.loads(path.read_text())).current_cfg.params
        factors, interp = params['Price Factors'], params['Price Factor Interpolation']
        base = params['System Parameters']['Base_Date']
        curve = lambda name: riskfactors.construct_factor(
            utils.Factor('InterestRate', (name,)), factors, interp)
        discount, carry = curve('USD'), curve('ZAR')
        spot = float(riskfactors.construct_factor(
            utils.Factor('FxRate', ('ZAR',)), factors, interp).current_value()[0])
        omega, alpha, beta, gamma, h0 = (written[key] for key in utils.HN_PARAM_NAMES)
        tensor = lambda x: torch.tensor(float(x), dtype=torch.float64)

        worst, weighted = 0.0, 0.0
        for point in params['Market Prices']['HestonNandiModelPrices.ZAR'][
                'instrument']['European_Options']:
            t = discount.get_day_count_accrual(base, (point['Expiry_Date'] - base).days)
            rate, yld = float(discount.current_value(t)), float(carry.current_value(t))
            forward, steps = spot * np.exp((rate - yld) * t), max(int(round(t * 252.0)), 1)
            sign = 1.0 if point['Option_Type'] == 'Call' else -1.0
            target = utils.black_european_option_price(
                forward, point['Strike'], rate, point['Quoted_Market_Value'], t, 1.0, sign)
            model = float(HestonNandiModelParameters.price(
                tensor(spot), tensor(point['Strike']), tensor(sign > 0), tensor(1.0),
                *[tensor(x) for x in (omega, alpha, beta, gamma)],
                tensor((rate - yld) * t / steps), steps, tensor(h0), 64,
                tensor(np.exp(-yld * t))))
            worst = max(worst, abs(model / target - 1.0))
            weighted += point['Weight'] * (target - model) ** 2
            print('HN quote K={:.4f} T={:.4f} {} vol={:.5f} target={:.6f} model={:.6f} '
                  'rel={:+.4%}'.format(point['Strike'], t, point['Option_Type'],
                                       point['Quoted_Market_Value'], target, model,
                                       model / target - 1.0))
        print('HN fit: {:.2f}s params {} worst |rel| {:.4%} weighted residual {:.3e}'.format(
            outcome['seconds'], {k: written[k] for k in utils.HN_PARAM_NAMES}, worst, weighted))

        # MEASURED: worst point 4.73% (the 3M 25 delta put), ATM 1M -0.04%, ATM 3M +1.04%,
        # weighted residual 6.21e-5, against a no-leverage mutant at 13.13% and 2.83e-4 (see the
        # docstring). A one-factor GARCH does not fit a smile exactly and is not asked to - what
        # these hold is that it fits it AT ALL, which a flat smile does not
        assert worst < 0.08, 'the model does not reprice its own quotes'
        assert weighted < 1.2e-4
    finally:
        service.BOOK = None


def test_a_pair_with_no_built_surface_refuses_at_the_verb(tmp_path):
    """A calibration against a surface the book does not carry is refused ON THE REQUEST THREAD,
    by name and with the remedy - never queued as a two-minute job whose answer is that the pair
    was a typo. Nothing is written and nothing is queued."""
    path = built_surface(tmp_path / 'book.json')
    try:
        before = path.read_bytes()
        refused = CLIENT.post('/book/hn', json={'pair': 'EUR.USD'})
        unnamed = CLIENT.post('/book/hn', json={})

        assert refused.status_code == 422
        assert 'FXVol.EUR.USD' in refused.json()['detail']
        assert 'FXVolPrices' in refused.json()['detail'], 'the refusal must name the remedy'
        assert unnamed.status_code == 422 and 'pair' in unnamed.json()['detail']
        assert path.read_bytes() == before
    finally:
        service.BOOK = None


@pytest.fixture
def desk(tmp_path):
    """A live book that declares the `FXVolSurfaceParameters` bootstrapper - what a market tick
    needs to turn quotes into the price factors a pricer reads."""
    path = tmp_path / 'book.json'
    path.write_text(json.dumps(json.loads(dump(job(sections={
        'Bootstrapper Configuration': {'FXVolSurfaceParameters': {}}}))), indent=2), newline='\n')
    service.BOOK = service.Book(str(path))
    yield path
    service.BOOK = None


def canned_map():
    """The verified security map `discover.provision` hands back - one USDZAR block carrying its
    evidence, spelled the way discovery spells the broker grid."""
    def entry(security):
        return {'security': security, 'name': security, 'last_update': '2024-06-28',
                'verified': '2024-06-28'}

    quotes = {label: {'ATM': entry('USDZARV{} BGN Curncy'.format(label)),
                      'RR_0.25': entry('USDZAR25R{} BGN Curncy'.format(label)),
                      'BF_0.25': entry('USDZAR25B{} BGN Curncy'.format(label))}
              for label in ('3M', '1Y')}
    return {'schema': 'derivus-bloomberg-map/1', 'generated': '2024-06-28', 'rejected': {},
            'blocks': {'fx_vol': {'USDZAR': {'expiries': {'3M': 0.25, '1Y': 1.0},
                                             'quotes': quotes}}}}


class FakeTerminal:
    """`BloombergSession` as the verb uses it - a context manager and nothing else, since the
    provision, the freshness check and the fetch are all seams the gates drive. No request is
    built, no socket is opened and blpapi is never reached."""

    def __init__(self, **options):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *error):
        return False


def bloomberg_seams(monkeypatch, provision=None, stale=None):
    """Every seam between the verb and the terminal, replaced. The lazy imports inside
    `BloombergJob.run_job` are what lets a patch reach the job at all: each name is bound off the
    package when the WORKER runs, which is after this. Returns the definitions the fetch was
    handed, so a gate can hold the scope it asked for against what the map built."""
    import derivus_bloomberg
    from derivus_bloomberg import discover, security_map, session

    asked = []

    def fetch_fx_vol(source, definition):
        asked.append(definition)
        return fx_vol_snapshot()

    monkeypatch.setattr(session, 'BloombergSession', FakeTerminal)
    monkeypatch.setattr(derivus_bloomberg, 'fetch_fx_vol', fetch_fx_vol)
    monkeypatch.setattr(security_map, 'stale', stale or (lambda source, securities: {}))
    monkeypatch.setattr(discover, 'provision', provision or (
        lambda source, as_of, on_batch=None: (canned_map(), False)))
    return asked


def ticked(request={}):
    """POST the verb, wait for the one worker to drain, and read the outcome off the result the
    way a poller does - the book write rides the run's own Stats, as a solve's coordinates do."""
    submitted = CLIENT.post('/book/bloomberg', json=request).json()
    service.EXECUTOR.queue.join()
    result = CLIENT.get('/results/{}'.format(submitted['result_id'])).json()
    return result, result.get('stats', {}).get('Bloomberg', {})


def test_the_bloomberg_verb_provisions_fetches_and_ticks_the_book(desk, monkeypatch):
    """THE verb, end to end on a machine with no terminal: the map is provisioned, its scope
    (every fx_vol pair, at the expiries it verified, at the default pillar) is what the fetch is
    asked for, and what comes back is installed and bootstrapped in one atomic write - so the
    book file carries the `FXVol` surface a pricer reads, not just the quotes it came from."""
    asked = bloomberg_seams(monkeypatch)
    result, outcome = ticked()
    on_disk = json.loads(desk.read_text())['Calc']['MergeMarketData']['ExplicitMarketData']

    assert result['status'] == 'done'
    assert outcome['written'] is True
    assert outcome['installed'] == ['FXVolPrices.USD.ZAR']
    assert 'FXVol.USD.ZAR' in outcome['new_factors']
    assert on_disk['Price Factors']['FXVol.USD.ZAR']['Surface_Type'] == 'Malz'
    assert 'FXVolPrices.USD.ZAR' in on_disk['Market Prices']
    # the scope came off the map, which is what "defaults to the desk's own" has to mean
    assert [definition.pair for definition in asked] == ['USDZAR']
    assert sorted(asked[0].expiries) == ['1Y', '3M'] and asked[0].pillars == (0.25,)


def test_a_stale_quote_refuses_the_tick_by_name(desk, monkeypatch):
    """A retired series keeps answering with a plausible price, so the update date is the only
    thing that says so: one late quote refuses the WHOLE tick by name and the book's bytes stand
    still - no half-installed surface, and nothing fetched reaches the file."""
    before = desk.read_bytes()
    asked = bloomberg_seams(monkeypatch, stale=lambda source, securities: {
        'USDZARV3M BGN Curncy': '2015-01-02'})
    result, outcome = ticked()

    assert result['status'] == 'done'
    assert outcome['written'] is False
    assert any('USDZARV3M BGN Curncy' in message and '2015-01-02' in message
               for message in outcome['refused'])
    assert asked == [], 'a late quote must refuse BEFORE anything is fetched'
    assert desk.read_bytes() == before


def test_progress_is_readable_while_the_fetch_runs(desk, monkeypatch):
    """A terminal round trip is minutes of work behind one `result_id`, so the job publishes where
    it has got to and `/results/{id}` merges it while the job waits or runs. Held on an Event, so
    the worker is provably mid-provision when the poll happens - and the entry is gone once the
    result carries its outcome, which is what keeps a done answer from ever showing a stale bar."""
    started, release = threading.Event(), threading.Event()

    def provision(source, as_of, on_batch=None):
        on_batch(1, 3)
        started.set()
        release.wait(timeout=30)
        return canned_map(), True

    bloomberg_seams(monkeypatch, provision=provision)
    submitted = CLIENT.post('/book/bloomberg', json={}).json()
    assert started.wait(timeout=30)
    running = CLIENT.get('/results/{}'.format(submitted['result_id'])).json()
    release.set()
    service.EXECUTOR.queue.join()
    done = CLIENT.get('/results/{}'.format(submitted['result_id'])).json()

    assert submitted['status'] == 'queued'
    assert running['status'] == 'running'
    assert running['progress'] == {'done': 1, 'total': 3, 'note': 'verifying securities'}
    assert done['status'] == 'done' and 'progress' not in done
    assert done['stats']['Bloomberg']['written'] is True
    assert done['stats']['Bloomberg']['provisioned'] is True


def test_the_bloomberg_verb_needs_a_book_and_a_bootstrapper(book):
    """Both refusals are the ones the market verbs already make, in the same words: no book is a
    404 naming the flag that opens one, and a book that declares no bootstrapper is a 422 saying
    nothing can turn quotes into price factors. Neither reaches the terminal or the queue."""
    bare = CLIENT.post('/book/bloomberg', json={})
    service.BOOK = None
    missing = CLIENT.post('/book/bloomberg', json={})

    assert bare.status_code == 422
    assert 'Bootstrapper Configuration' in bare.json()['detail']
    assert missing.status_code == 404 and '--book' in missing.json()['detail']


def test_the_metronome_skips_the_beat_its_last_tick_is_still_in_flight():
    """A terminal round trip can outlast an interval, and the result id's clock stamp means two
    ticks will never coalesce onto one result - so nothing but the metronome stops a slow terminal
    accumulating a queue of them.

    The decision is read off the executor's REAL store, which is why it can be gated without a
    terminal and without patching anything: a job that holds the worker is queued, then running,
    then done, and `pending_status` is exactly that word each time. `beat()` returning while the
    hold is on is the claim - it left `pending` where it was instead of submitting a second tick.
    Non-vacuous by construction: there is no book open here, so a beat that did NOT skip would
    reach `live_book()` and raise the 404 rather than passing quietly.
    """
    metronome = service.Metronome(60.0)
    assert metronome.pending_status() is None, 'nothing submitted yet is nothing to wait on'

    hold = threading.Event()
    held = Held('metronome-tick', [], hold=hold)
    service.EXECUTOR.submit(service.Job('metronome-tick', held, {}), service.HEAVY)
    assert held.started.wait(timeout=30)
    metronome.pending = 'metronome-tick'

    assert metronome.pending_status() == 'running'
    assert metronome.beat() is None
    assert metronome.pending == 'metronome-tick', 'the beat stacked a tick on a running one'

    hold.set()
    service.EXECUTOR.queue.join()
    assert metronome.pending_status() == 'done'


def test_a_routine_tick_refuses_an_unprovisioned_home_and_leaves_the_book_alone(
        desk, tmp_path, monkeypatch, caplog):
    """The metronome does not provision. Verifying a workstation's whole vocabulary is minutes of
    terminal time and a person's decision, so a cadence that met an unprovisioned `DV_HOME` and
    started discovering would be doing exactly the thing nobody asked for.

    `DV_HOME` here names a directory with no `security_map.json`, which is a fresh desk. The beat
    submits through the real queue, the job refuses BEFORE it opens a session - which is what
    makes this gate reachable on a machine with no terminal at all - and the refusal names the
    home it looked in and the verb that fixes it. The book's bytes stand still, and the second
    beat is the failure discipline: exactly ONE warning line, carrying that same cause.
    """
    import logging

    # DV_HOME is the declared surface for where a desk's files live, so pointing it at a directory
    # holding no map IS an unprovisioned workstation - nothing in the package is patched
    home = tmp_path / 'unprovisioned'
    monkeypatch.setenv('DV_HOME', str(home))
    before = desk.read_bytes()
    metronome = service.Metronome(60.0)

    metronome.beat()
    service.EXECUTOR.queue.join()
    result = CLIENT.get('/results/{}'.format(metronome.pending)).json()
    outcome = result['stats']['Bloomberg']

    assert result['status'] == 'done'
    assert outcome['written'] is False
    assert any(str(home) in message and 'tick_market_from_bloomberg' in message
               for message in outcome['refused'])
    assert desk.read_bytes() == before

    # the next beat judges the one that refused: ONE warning line, and it carries the cause
    with caplog.at_level(logging.WARNING, logger='derivus.service'):
        metronome.beat()
    service.EXECUTOR.queue.join()
    warned = [record.getMessage() for record in caplog.records
              if record.name == 'derivus.service' and record.levelno == logging.WARNING]

    assert len(warned) == 1 and str(home) in warned[0]
    assert metronome.failures == 1
    assert desk.read_bytes() == before


def test_a_solve_lands_an_affine_field_in_a_handful_of_pricings(book):
    """The structuring verb on the affine case: solve a cashflow's Amount to a target value. A
    secant is exact where the value is affine in the field, so the pricing count is small, the
    residual is inside tolerance, and the result's tables are the run AT the solved value - a
    priced answer, never an extrapolated one. The book file never moves."""
    before = book.read_bytes()
    submitted = CLIENT.post('/book/solve', content=dump({
        'deal': dict(CASHFLOW, Reference='SLV1'), 'field': 'Amount',
        'target': 123_456.0}), headers=JSON).json()
    service.EXECUTOR.queue.join()
    result = CLIENT.get('/results/{}'.format(submitted['result_id'])).json()
    solved = result['stats']['Solved']

    assert result['status'] == 'done'
    assert solved['value'] == pytest.approx(123_456.0 / (SPOT * np.exp(-RATE * 2.0)), rel=1e-6)
    assert abs(solved['residual']) <= 0.01 and solved['evaluations'] <= 4
    assert mtm(submitted['result_id'])['SLV1'] == pytest.approx(123_456.0, abs=0.01)
    assert book.read_bytes() == before


def test_a_solve_brackets_a_nonlinear_strike(tmp_path):
    """The case the verb exists for: a digital's value is nonlinear and monotone in its strike,
    so brentq inside declared bounds finds the strike that marks at the target - and the pricing
    count says it genuinely iterated rather than taking the affine two-step."""
    path = tmp_path / 'book.json'
    path.write_text(json.dumps(json.loads(dump(job(factors=dict(FACTORS, **EQUITY)))), indent=2), newline='\n')
    service.BOOK = service.Book(str(path))
    try:
        submitted = CLIENT.post('/book/solve', content=dump({
            'deal': dict(BINARY, Cash_Payoff=100_000.0, Reference='SLV2'),
            'field': 'Strike_Price', 'target': 40_000.0, 'bounds': [80.0, 120.0]}),
            headers=JSON).json()
        service.EXECUTOR.queue.join()
        result = CLIENT.get('/results/{}'.format(submitted['result_id'])).json()
        solved = result['stats']['Solved']

        assert result['status'] == 'done'
        assert 80.0 < solved['value'] < 120.0 and solved['evaluations'] > 3
        assert abs(solved['residual']) <= 0.01
        assert mtm(submitted['result_id'])['SLV2'] == pytest.approx(40_000.0, abs=0.01)
    finally:
        service.BOOK = None


def test_a_solve_that_cannot_reach_its_target_says_so(book):
    """An unreachable target is an error result carrying the solver's own words, not a number
    quietly clamped to a bound."""
    submitted = CLIENT.post('/book/solve', content=dump({
        'deal': dict(CASHFLOW, Reference='SLV3'), 'field': 'Amount',
        'target': 1_000_000.0, 'bounds': [1.0, 2.0]}), headers=JSON).json()
    service.EXECUTOR.queue.join()
    result = CLIENT.get('/results/{}'.format(submitted['result_id'])).json()

    assert result['status'] == 'error' and result['error']


def test_validate_over_http_is_the_verb_verbatim():
    """Both halves of the want-list, over a job broken deliberately in both ways: a deal that
    breaks an authoring rule, and one naming a curve the market data has no block for."""
    document = job(deals=(dict(CASHFLOW, Discount_Rate='GBP'), BINARY),
                   factors=dict(FACTORS, **EQUITY))
    over_http = CLIENT.post('/validate', content=dump(document), headers=JSON).json()

    assert over_http == in_process(document).validate()
    assert over_http == {'deals': {'BIN1': ['Cash_Payoff is required']},
                         'factors': ['InterestRate.GBP']}


def test_a_browser_is_allowed_to_call_the_service_at_all():
    """Without the header a browser discards the answer before the SPA sees it, so this is the
    difference between an API a web client can use and one it cannot. Both halves are asserted:
    the preflight a POST of JSON provokes, and the header on the answer itself."""
    origin = {'Origin': 'http://localhost:4200'}
    preflight = CLIENT.options('/execute', headers=dict(
        origin, **{'Access-Control-Request-Method': 'POST'}))

    assert preflight.headers['access-control-allow-origin'] == '*'
    assert CLIENT.get('/schema/job', headers=origin).headers['access-control-allow-origin'] == '*'


def test_the_job_skeleton_is_a_job_that_loads():
    """The envelope is the one piece of contract `/schema` cannot state, so what is published has
    to BE a job rather than describe one: it goes back over `/validate` and `/execute` unedited.

    A skeleton that validates clean but does not price would pass on the want-list alone, so the
    price is asserted too — a cashflow discounted at the flat curve the skeleton carries.
    """
    skeleton = CLIENT.get('/schema/job').json()
    result_id, result = run(skeleton)
    payment = skeleton['Calc']['Deals']['Deals']['Children'][0]['Instrument']['.Deal']

    assert CLIENT.post('/validate', json=skeleton).json() == {'deals': {}, 'factors': []}
    assert result['status'] == 'done'
    assert mtm(result_id)['CF1'] == pytest.approx(
        payment['Amount'] * np.exp(-0.02 * 2.0), rel=1e-3)


def test_describe_is_the_parse_and_it_never_runs_anything():
    """What a front end shows before committing to a run: the book by type, both sides of the
    factor universe, the calculation block as loaded, and what the queue would make of it.

    Non-mutating is the claim that matters, because describing walks the deal tree and calls
    `reset` on every instrument. So it is described off a PLAN and the same plan is then executed:
    a describe that wrote to what it read would move the plan, and the id would stop agreeing with
    the one the whole document landed on. Describing twice makes the same demand of the deep copy.

    A node whose `Object` names no deal type is counted under nothing, because
    `construct_instrument` logged it and returned `{}` — the name went with the payload, and
    `/validate` is where that node is reported.
    """
    document = job(deals=(CASHFLOW, dict(CASHFLOW, Reference='CF2')), Random_Seed=23)
    described = CLIENT.post('/describe', content=dump(document), headers=JSON).json()
    plan_id = CLIENT.post('/prepare', content=dump(document), headers=JSON).json()['plan_id']
    from_document, _ = run(document)

    assert CLIENT.post('/describe', json={'plan_id': plan_id}).json() == described
    assert CLIENT.post('/describe', json={'plan_id': plan_id}).json() == described
    from_plan = CLIENT.post('/execute', json={'plan_id': plan_id}).json()
    service.EXECUTOR.queue.join()

    unknown = CLIENT.post('/describe', content=dump(
        job(deals=(CASHFLOW, dict(CASHFLOW, Object='NoSuchDeal')))), headers=JSON).json()

    assert described['deals'] == {'FixedCashflowDeal': 2}
    assert unknown['deals'] == {'FixedCashflowDeal': 1}
    assert described['factors'] == in_process(document).describe()['factors']
    assert described['factors']['resolved'] == sorted(FACTORS)
    assert described['factors']['missing'] == []
    assert described['calculation']['Object'] == 'BaseValuation'
    assert described['cost']['class'] == service.COST_CLASS['BaseValuation']
    assert from_plan['result_id'] == from_document
    assert mtm(from_document)['CF1'] == pytest.approx(AMOUNT * SPOT * np.exp(-RATE * 2.0), rel=1e-9)


def test_the_cost_estimate_counts_paths_by_grid_points():
    """The class is what the queue orders by; the estimate is the size beside it, and it has to
    move with all three fields or it is describing something else. A base valuation carries none of
    them, which is the case the `or 1` exists for."""
    heavy = {'Object': 'CreditMonteCarlo', 'Batch_Size': 512, 'Simulation_Batches': 4,
             'Time_Grid': '0d 2d 1w(1w) 3m(1m) 2y(3m)'}

    assert service.cost(heavy) == dict(
        service.cost(heavy), **{'class': service.HEAVY, 'estimate': 512 * 4 * 5})
    assert service.cost(dict(heavy, Batch_Size=256))['estimate'] == 256 * 4 * 5
    assert service.cost(dict(heavy, Time_Grid='0d 1d(1d)'))['estimate'] == 512 * 4 * 2
    assert service.cost({'Object': 'BaseValuation'}) == dict(
        service.cost(heavy), **{'class': 0, 'estimate': 1})


def test_a_plan_id_execute_is_the_document_execute():
    """Content addressing does not care how the job arrived. `/prepare` names the parse by its plan
    hash, and executing that name unpatched has to land on the id the whole document landed on —
    otherwise a client that prepared once would be reading a different run's numbers."""
    document = job(Random_Seed=13)
    prepared = CLIENT.post('/prepare', content=dump(document), headers=JSON).json()
    from_document, _ = run(document)
    from_plan = CLIENT.post('/execute', json={'plan_id': prepared['plan_id']}).json()

    assert prepared['plan_id'] == in_process(document).plan_hash()
    assert prepared['values_hash'] == in_process(document).values_hash()
    assert from_plan['result_id'] == from_document
    assert from_plan['status'] == 'done'


def test_a_patched_execute_leaves_the_plan_as_it_found_it():
    """The cache holds a PRISTINE parse and hands out deep copies, so a patch reaches one execute
    and not the plan. Asserted by patching first and then executing the same plan unpatched: a
    shared Context would return the patched id, and the numbers with it."""
    document = job(Random_Seed=17)
    plan_id = CLIENT.post('/prepare', content=dump(document), headers=JSON).json()['plan_id']
    unpatched_id, _ = run(document)

    patched = CLIENT.post('/execute', json={
        'plan_id': plan_id, 'Patch': {'FxRate.ZAR': {'Spot': SPOT * 2}}}).json()
    service.EXECUTOR.queue.join()
    after = CLIENT.post('/execute', json={'plan_id': plan_id}).json()
    service.EXECUTOR.queue.join()

    assert patched['result_id'] != unpatched_id
    assert mtm(patched['result_id'])['CF1'] == pytest.approx(2 * mtm(unpatched_id)['CF1'], rel=1e-9)
    assert after['result_id'] == unpatched_id
    assert mtm(after['result_id'])['CF1'] == pytest.approx(mtm(unpatched_id)['CF1'], rel=1e-9)


def test_a_plan_falls_out_of_the_cache_least_recently_used_first():
    """The cache is bounded because a plan is a parse, cheap to redo and never the record of
    anything. What must not happen is the wrong one being evicted: reading a plan is a USE, so the
    one read stays and the one merely older goes.

    The three jobs differ by a deal REFERENCE and not by the seed, which is a replay coordinate of
    its own and deliberately outside the plan — three seeds would have been one plan, and the gate
    would have measured nothing.
    """
    plans = [in_process(job(deals=(dict(CASHFLOW, Reference=name),)))
             for name in ('CF1', 'CF2', 'CF3')]
    cache = service.PlanCache(size=2)
    first, second = cache.put(plans[0]), cache.put(plans[1])
    cache.get(first)
    third = cache.put(plans[2])

    assert cache.get(first) is not None
    assert cache.get(second) is None
    assert cache.get(third) is not None


def test_an_unknown_plan_result_or_table_is_a_404():
    """A name the service does not hold is not an empty answer — a client that could not tell them
    apart would render a blank grid for a typo."""
    result_id, _ = run(job(Random_Seed=19))

    assert CLIENT.post('/execute', json={'plan_id': 'nosuchplan'}).status_code == 404
    assert CLIENT.post('/describe', json={'plan_id': 'nosuchplan'}).status_code == 404
    assert CLIENT.get('/results/nosuchresult').status_code == 404
    assert CLIENT.get('/results/{}/nosuchtable'.format(result_id)).status_code == 404
    assert CLIENT.get('/results/{}/mtm'.format(result_id)).status_code == 200


def test_a_result_publishes_the_shape_of_every_table_and_pages_each_one():
    """The contract this slice moved to: a summary carries shapes, never cells, and one table comes
    back a page at a time. Held through the executor because no calculation in the suite produces
    the tree that makes it interesting — a group of tables, a vector and a scalar beside a frame.

    A group is not a table and has no page, so `cashflows` arrives flattened to the path that names
    each one. The paging assertions are what stop `limit` being read as an end index.
    """
    frame = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]})
    results = {'mtm': frame, 'cashflows': {'ZAR': frame}, 'collva_t': np.arange(4.0), 'cva': 1.25}
    service.EXECUTOR.submit(service.Job('shapes', Held('shapes', [], results=results), {}),
                            service.HEAVY)
    service.EXECUTOR.queue.join()
    summary = CLIENT.get('/results/shapes').json()

    assert summary['tables'] == {'mtm': {'rows': 3, 'columns': ['a', 'b']},
                                 'cashflows/ZAR': {'rows': 3, 'columns': ['a', 'b']},
                                 'collva_t': {'rows': 4, 'columns': []},
                                 'cva': {'rows': 1, 'columns': []}}
    assert fetch('shapes', 'cashflows/ZAR')['data'] == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    assert fetch('shapes', 'collva_t') == {
        'name': 'collva_t', 'rows': 4, 'columns': [], 'offset': 0, 'index': [],
        'data': [0.0, 1.0, 2.0, 3.0]}
    assert fetch('shapes', 'cva')['data'] == [1.25]
    assert fetch('shapes', 'mtm', offset=1, limit=1) == {
        'name': 'mtm', 'rows': 3, 'columns': ['a', 'b'], 'offset': 1, 'index': [1],
        'data': [[2.0, 5.0]]}
    assert fetch('shapes', 'mtm', offset=2)['data'] == [[3.0, 6.0]]
    assert fetch('shapes', 'mtm', offset=99)['data'] == []


def test_a_patch_reaches_the_number_and_moves_the_result_id():
    """A patch is applied before the hashes are taken, so it reaches the price AND the identity.

    The two stamps say which half moved: a spot is market VALUES, so the values hash moves and the
    plan hash does not. An id taken before the patch would collide with the unpatched run and serve
    it the wrong numbers, which is exactly the failure the hash split exists to prevent.
    """
    patch = {'FxRate.ZAR': {'Spot': SPOT * 1.1}}
    plain_id, plain = run(job())
    patched_id, patched = run(dict(job(), Patch=patch))

    context = in_process(job())
    context.patch_market(patch)
    _, out = context.run_job()
    expected = json.loads(json.dumps(out['Results']['mtm'], cls=CustomJsonEncoder))['.DataFrame']

    assert patched_id != plain_id
    assert patched['values_hash'] != plain['values_hash']
    assert patched['plan_hash'] == plain['plan_hash']
    assert fetch(patched_id, 'mtm')['data'] == expected['data']
    assert mtm(patched_id)['CF1'] == pytest.approx(mtm(plain_id)['CF1'] * 1.1, rel=1e-9)


def test_an_identical_submission_is_one_result_id():
    """Retrying is free: the same job names the same result, and the second submission is already
    holding the finished one."""
    document = job(Random_Seed=11)
    first, _ = run(document)
    second = submit(document)

    assert second['result_id'] == first
    assert second['status'] == 'done'


def test_a_submission_arriving_mid_run_coalesces_onto_the_first():
    """Dedupe has to hold while the first job is still QUEUED or RUNNING, not only once it is
    filed. Counted at the executor, because an execution is observable nowhere else — a second run
    of the same job would overwrite the store with the same content and leave no trace."""
    executor = service.ComputeExecutor()
    ran, release = [], threading.Event()
    blocker = Held('blocker', ran, hold=release)
    once = Held('once', ran)

    executor.submit(service.Job('blocker', blocker, {}), service.HEAVY)
    assert blocker.started.wait(timeout=30)
    assert executor.submit(service.Job('same', once, {}), service.HEAVY) == 'queued'
    assert executor.submit(service.Job('same', once, {}), service.HEAVY) == 'queued'
    release.set()
    executor.queue.join()

    assert ran == ['blocker', 'once']


def test_a_light_job_jumps_a_heavy_one_that_is_still_waiting():
    """Cost class orders what is waiting; arrival orders within a class; the running job is left
    alone. All three are in the queue before the blocker is released, so the queue - and not the
    order they arrived in - is the only thing that can decide what runs next."""
    executor = service.ComputeExecutor()
    ran, release = [], threading.Event()
    blocker = Held('blocker', ran, hold=release)
    heavy, light = service.COST_CLASS['CreditMonteCarlo'], service.COST_CLASS['BaseValuation']

    executor.submit(service.Job('blocker', blocker, {}), heavy)
    assert blocker.started.wait(timeout=30)
    for name, cost in [('heavy_1', heavy), ('light', light), ('heavy_2', heavy)]:
        executor.submit(service.Job(name, Held(name, ran), {}), cost)
    release.set()
    executor.queue.join()

    assert ran == ['blocker', 'light', 'heavy_1', 'heavy_2']


def test_a_failing_job_is_an_error_status_and_the_worker_survives():
    """Reporting in a currency the market data has no FX rate for fails inside the run, so this is
    the engine failing rather than the wrapper refusing. The message travels, nothing else does,
    and the next job - a new one, since an identical submission would never reach the worker -
    still prices."""
    _, failed = run(job(Currency='GBP'))
    after_id, after = run(job(Random_Seed=7))

    assert failed['status'] == 'error'
    assert 'FxRate.GBP' in failed['error']
    assert 'tables' not in failed
    assert after['status'] == 'done'
    assert mtm(after_id)['CF1'] == pytest.approx(AMOUNT * SPOT * np.exp(-RATE * 2.0), rel=1e-9)


#: USDZAR as THIS book quotes it. `FxRate.ZAR.Spot` is one ZAR in the base currency's units, and
#: the suite's world sets it to 18.5, so the market pair - ZAR per USD - is its reciprocal. A gate
#: that struck the collar at 18.5 would be asking for a floor deep in the money and reading the
#: runner's bracket refusal as a service bug.
USDZAR = 1.0 / SPOT

#: The zero-cost collar as a sales desk asks for it: parameters in MARKET terms, the floor given
#: and the cap left for the recipe to solve. The keys are the structure's own declared `fields`,
#: and one it does not declare comes back as a `status: error` carrying the runner's message,
#: which is why every gate below asserts the status with that message attached rather than reading
#: past it. The floor sits 5% out of the money: the bought put is then cheap enough for a solved
#: cap to fund it well inside the bracket the runner brackets a strike solve with.
COLLAR = {'pair': 'USDZAR', 'expiry': '1Y', 'notional': AMOUNT,
          'notional_currency': 'USD', 'floor': USDZAR * 0.95}


@pytest.fixture
def quoting(tmp_path, monkeypatch):
    """A desk that can be quoted at: the one-cashflow book with the FX vol bootstrapper declared
    and a real USDZAR surface ticked into it, and `DV_HOME` pointed at the gate's own tmp.

    `DV_HOME` is the declared surface for where a desk's files live, so setting it is the honest
    way to make the pending quotes land somewhere a gate may read - and the service reads it per
    call, so the worker thread that files the quote sees the directory this fixture named.
    """
    monkeypatch.setenv('DV_HOME', str(tmp_path))
    path = tmp_path / 'book.json'
    path.write_text(json.dumps(json.loads(dump(job(sections={
        'Bootstrapper Configuration': {'FXVolSurfaceParameters': {}}}))), indent=2), newline='\n')
    service.BOOK = service.Book(str(path))
    ticked = CLIENT.post('/book/market', content=dump({'quotes': fx_vol_quotes()}),
                         headers=JSON).json()
    assert ticked['written'] is True, ticked
    yield path
    service.BOOK = None


def quote_of(structure, params, **extra):
    """Ask for a quote and wait for the one worker to give it - the outcome under `stats.Quote`,
    exactly where a solve's coordinates sit under `stats.Solved`. `extra` is the rest of the ask -
    `netting_set`, the client the quote is for - passed as the verb takes it."""
    submitted = CLIENT.post('/book/structure', content=dump(
        dict({'structure': structure, 'params': params}, **extra)), headers=JSON).json()
    service.EXECUTOR.queue.join()
    result = CLIENT.get('/results/{}'.format(submitted['result_id'])).json()
    assert result['status'] == 'done', result.get('error')
    return result['stats']['Quote']


def test_a_quoted_collar_is_filed_pending_and_books_at_zero(quoting, tmp_path):
    """The whole sales loop through one service: a structure named the way a desk names it comes
    back solved, the pending trade is on disk under its quote id, the approval books it through
    the same seam a hand-booked deal goes through, and the book then MARKS it at zero.

    The last step is the one that cannot be faked. `net` is what the runner computed while it was
    solving; the collar's value in the book is what the engine says about the composed deal that
    was actually written, priced from the file afterwards, so agreement between them is the quote
    and the booking being one trade. Zero is asserted against a leg premium rather than in
    absolute currency - a net of zero means nothing if both legs are worth nothing.
    """
    quote = quote_of('ZeroCostCollar', COLLAR)
    premium = max(abs(leg['premium']) for leg in quote['legs'])

    assert quote['structure'] == 'ZeroCostCollar'
    assert len(quote['legs']) == 2 and premium > 0.0
    assert [leg['deal_type'] for leg in quote['legs']] == ['FXOptionDeal'] * 2
    assert {leg['buy_sell'] for leg in quote['legs']} == {'Buy', 'Sell'}
    assert sum(leg['solved'] is not None for leg in quote['legs']) == 1
    assert abs(quote['net']) < premium * 1e-4

    pending = tmp_path / 'tmp' / (quote['quote_id'] + '.json')
    assert quote['files']['quote'] == str(pending)
    filed = json.loads(pending.read_text())
    assert filed['deal'] == quote['deal']
    assert filed['quote']['quote_id'] == quote['quote_id'] and 'deal' not in filed['quote']

    booked = CLIENT.post('/book/quote', json={'quote_id': quote['quote_id']}).json()
    on_disk = json.loads(quoting.read_text())
    node = deal_at(on_disk, booked['deal_path'])

    assert booked['written'] is True
    assert node['Instrument']['.Deal']['Object'] == 'StructuredDeal'
    assert [child['Instrument']['.Deal']['Reference'] for child in node['Children']] == [
        leg['reference'] for leg in quote['legs']]
    # the quote is client paper and the book holds the bank's position: the approval books the
    # MIRROR, so every booked leg carries the opposite side from the one the client was quoted
    assert [child['Instrument']['.Deal']['Buy_Sell'] for child in node['Children']] == [
        {'Buy': 'Sell', 'Sell': 'Buy'}[leg['buy_sell']] for leg in quote['legs']]
    # the file is the audit trail of what was quoted at what market - booking does not consume it
    assert pending.is_file()

    marked_id, marked = run(on_disk)
    assert marked['status'] == 'done', marked.get('error')
    assert mtm(marked_id)[node['Instrument']['.Deal']['Reference']] == pytest.approx(
        0.0, abs=premium * 1e-4)


#: A calibrated Heston-Nandi factor for the rand, as `/book/hn` writes one. The booking gate does
#: not fit it - what it is about is whether the MODEL a leg was quoted under reaches the book with
#: the trade - so these five are a stationary set (persistence 0.90) near the surface's own vol
#: rather than a fit of it, with `Gamma_Star` on the sign this pair's smile actually carries.
CALIBRATED = {'Property_Aliases': None, 'Omega': 1e-12, 'Alpha': 2.0e-6, 'Beta': 0.45,
              'Gamma_Star': -474.34, 'H0': 7.8e-5}

#: An accumulator on the RAND, which is the orientation that joins: the engine keys a spot model
#: off `Underlying_Currency`, the calibration writes the pair's non-domestic token, and an
#: accumulator - unlike a TARF - may be quoted from either side.
ACCUMULATOR = {'pair': 'USDZAR', 'expiry': '3M', 'notional': AMOUNT, 'notional_currency': 'ZAR',
               'fixing_frequency': '1M', 'knockout': USDZAR * 1.10}


def test_a_leg_quoted_under_a_model_books_into_a_book_that_marks_it(quoting):
    """The model books WITH the trade, in the same atomic write, or the desk marks a trade at a
    price it was never dealt at.

    `structures.spot_model` pins Heston-Nandi on the QUOTE's copy of the document - it has to be
    the quote's copy, since a quote is not a trade and must not touch the book. That copy is thrown
    away when the answer is published, so an approval that booked only the deal would leave a leg
    priced under a GARCH in a book whose `Valuation Configuration` says nothing, and the very next
    mark would price it as a lognormal. Nothing would say so: both numbers are plausible.

    So the quote REPORTS what it pinned, the pending file records it, and `/book/quote` merges it
    into the book inside the same edit closure that splices the deal - one lock, one validation,
    one write. What this gate reads is the FILE afterwards, because the file is what a re-mark
    prices off.

    And the same approval REFUSES where the calibration has gone away between the quote and the
    approval: an approval is validated against the book as it stands now, and booking a switch over
    a factor nobody carries any more would skip the deal inside the engine's dependency loop and
    mark the trade at nothing - the one outcome the pin exists to prevent.
    """
    document = json.loads(quoting.read_text())
    market = document['Calc']['MergeMarketData']['ExplicitMarketData']
    market['Price Factors']['HestonNandiModelParameters.ZAR'] = dict(CALIBRATED)
    document['Calc']['Calculation']['MCMC_Simulations'] = 1024
    quoting.write_text(json.dumps(document, indent=2), newline='\n')
    service.BOOK = service.Book(str(quoting))

    quote = quote_of('Accumulator', ACCUMULATOR)
    pinned = {'FXAccumulatorOptionDeal': {'SpotModel': 'HestonNandi'}}

    assert quote['legs'][0]['note'] is None, 'the leg did not join the calibration'
    assert quote['valuation_configuration'] == pinned
    # the pending file is the record of what was quoted, model included
    filed = json.loads((quoting.parent / 'tmp' / (quote['quote_id'] + '.json')).read_text())
    assert filed['quote']['valuation_configuration'] == pinned

    before = json.loads(quoting.read_text())['Calc']['MergeMarketData']['ExplicitMarketData']
    assert 'Valuation Configuration' not in before, 'the quote wrote on the book'

    # the calibration goes away between the quote and the approval: the approval refuses, the
    # book is untouched, and the message names the factor and the remedy
    dropped = json.loads(quoting.read_text())
    dropped['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors'].pop(
        'HestonNandiModelParameters.ZAR')
    quoting.write_text(json.dumps(dropped, indent=2), newline='\n')
    service.BOOK = service.Book(str(quoting))
    unbookable = json.dumps(dropped, indent=2)
    refused = CLIENT.post('/book/quote', json={'quote_id': quote['quote_id']})

    assert refused.status_code == 422
    assert 'HestonNandiModelParameters.ZAR' in refused.json()['detail']
    assert 're-quote' in refused.json()['detail'], 'a refusal without a remedy'
    assert quoting.read_text() == unbookable, 'a refused approval wrote'

    quoting.write_text(json.dumps(document, indent=2), newline='\n')
    service.BOOK = service.Book(str(quoting))
    booked = CLIENT.post('/book/quote', json={'quote_id': quote['quote_id']}).json()
    after = json.loads(quoting.read_text())['Calc']['MergeMarketData']['ExplicitMarketData']

    assert booked['written'] is True and booked['valuation_configuration'] == pinned
    assert after['Valuation Configuration'] == pinned, (
        'the book marks under a model the leg was not quoted under')


def test_a_quote_with_no_map_prices_on_the_book_and_names_the_spot_it_used(quoting, tmp_path):
    """A quote prices on this workstation's LIVE spot when the terminal is up, and the fallback
    when it is not IS the old path - said float for float rather than promised.

    This `DV_HOME` holds no security map, which is a fresh desk and the one live-spot refusal
    reachable with no terminal at all: a quote never provisions, exactly as a routine tick never
    does. So the quote runs on the book's own ticked spot, names the home it looked in, and must
    come out identical to the same structure quoted straight through `structures.quote` - the
    entry point this feature did not touch. Close would not do: a fallback that moved a price
    would be a live-spot feature firing when there is no live spot.

    `value_market` is read back off the document the legs priced against, so it is the book's
    USDZAR here and cannot be anything else. And the book file is untouched either way - a spot is
    `bind='value'` data patched onto the JOB's copy, and a quote is not a trade.
    """
    before = quoting.read_bytes()
    quote = quote_of('ZeroCostCollar', COLLAR)
    document, _ = service.BOOK.read()
    unchanged = structures.quote(document, 'ZeroCostCollar', COLLAR)

    assert quote['spot']['source'] == 'book'
    assert 'no security map in {}'.format(tmp_path) in quote['spot']['note']
    assert quote['spot']['value_market'] == USDZAR

    assert (quote['net'], quote['net_mid']) == (unchanged['net'], unchanged['net_mid'])
    for row, same in zip(quote['legs'], unchanged['legs']):
        assert (row['premium'], row['strike_market'], row['solved']) == (
            same['premium'], same['strike_market'], same['solved']), row
    assert quoting.read_bytes() == before


def test_an_unknown_quote_id_names_the_tmp_it_looked_in(quoting):
    """A quote id nobody gave is a 404 naming the directory it was looked for in - a desk whose
    `DV_HOME` is not the one the quote was filed under has to be able to READ that off the
    refusal. Nothing is written: a booking that never found its trade is not a booking."""
    before = quoting.read_bytes()
    answer = CLIENT.post('/book/quote', json={'quote_id': 'nosuchquoteid'})

    assert answer.status_code == 404
    assert 'nosuchquoteid' in answer.json()['detail']
    assert str(quoting.parent / 'tmp') in answer.json()['detail']
    assert quoting.read_bytes() == before


def test_a_pending_trade_the_book_would_refuse_is_refused_in_the_booking_wording(quoting,
                                                                                 tmp_path):
    """An approval is a BOOKING, so it is refused on what it would land in and in the same words.

    The pending file here is authored by the gate - a broken trade is data, and this is the one
    way to reach the refusal branch without asking the runner for a quote it would rightly
    decline to give. Pointing the deal at a currency the book has no market data for is exactly
    what `test_a_bad_amendment_touches_nothing` refuses, and the wording has to match it: one
    validate-before-write seam, or the two paths have drifted.
    """
    before = quoting.read_bytes()
    pending = tmp_path / 'tmp'
    pending.mkdir(parents=True, exist_ok=True)
    broken = dict(CASHFLOW, Reference='BROKEN', Currency='GBP', Discount_Rate='GBP')
    (pending / 'authored.json').write_text(json.dumps(json.loads(dump(
        {'quote': {'quote_id': 'authored'}, 'deal': broken})), indent=2), newline='\n')

    refused = CLIENT.post('/book/quote', json={'quote_id': 'authored'}).json()

    assert refused['written'] is False
    assert 'no market data for InterestRate.GBP' in refused['refused']
    assert quoting.read_bytes() == before
    # a refusal is an answer, not a deletion: the pending trade is still there to be corrected
    assert (pending / 'authored.json').is_file()


def test_the_quote_sheet_lands_beside_the_pending_trade(quoting, tmp_path):
    """The sheet is a real xlsx workbook next to the quote file, under the same id.

    Skipped rather than faked when the `quote` extra is not installed on this box: nothing is
    uninstalled and nothing is patched to reach the other branch, so what this gate says is only
    ever true of a desk that HAS the writer. `derivus.quote_sheet` owns the gates on what is
    inside the three sheets; this one owns the wiring - that the job reaches the writer, names
    the file it wrote, and put a workbook there.
    """
    pytest.importorskip('derivus.quote_sheet',
                        reason='the quote extra is not installed - there is no sheet to find')
    quote = quote_of('ZeroCostCollar', COLLAR)
    sheet = tmp_path / 'tmp' / (quote['quote_id'] + '.xlsx')

    assert quote['files']['sheet'] == str(sheet)
    assert 'sheet_note' not in quote['files']
    assert zipfile.is_zipfile(str(sheet))


def test_a_composed_candidate_prices_its_legs_not_an_empty_container(book):
    """A structure IS its legs, on the what-if verb too: a composed StructuredDeal arriving with
    node-shaped children inside its deal block prices as the sum of those legs. The killing
    mutation is `splice_deal` dropping the composed children back to an empty list - the container
    then prices 0.0 with nothing said against it, which is exactly the silent hollow quote this
    gate exists to keep dead. Non-vacuity is the legs themselves: each must carry a real value."""
    legs = [dict(CASHFLOW, Reference='CMP_A'), dict(BOOKED, Reference='CMP_B')]
    composed = {'Object': 'StructuredDeal', 'Reference': 'CMP1', 'Currency': 'USD',
                'Net_Cashflows': 'No',
                'Children': [{'Instrument': {'.Deal': json.loads(dump(leg))}} for leg in legs]}
    run = CLIENT.post('/book/price', content=dump({'deal': composed}), headers=JSON).json()
    service.EXECUTOR.queue.join()
    values = mtm(run['result_id'])

    expected = {leg['Reference']: leg['Amount'] * SPOT * np.exp(-RATE * 2.0) for leg in legs}
    for reference, value in expected.items():
        assert abs(value) > 1.0  # a leg worth nothing would make the sum check vacuous
        assert values[reference] == pytest.approx(value, rel=1e-9)
    assert values['CMP1'] == pytest.approx(sum(expected.values()), rel=1e-9)


# --------------------------------------------------------------------------------------------
# the blotter's two data views: consolidated risk, and the XVA projection
# --------------------------------------------------------------------------------------------
def test_the_consolidated_risk_is_the_book_priced_once_and_it_follows_a_booking(quoting):
    """`/book/risk` on the desk's own quoting book: coherent totals, a warm hit that changes
    nothing, and a booking that moves the etag with the answer behind it.

    Three things are held. COHERENCE: `mtm` is exactly the sum of `per_deal`, and each per-deal
    value is the value the ORDINARY run reports for that deal - `/book/price`'s `mtm` table, the
    path every other gate here prices through - so the view cannot be a second opinion about what
    the book is worth. WARMTH: the same book answers the identical object off the cache, etag and
    `as_of` included, which is what makes this pollable on the book's own beat. FOLLOWING: a
    booking changes the content the run reads, so the etag moves and the new deal is in the answer
    with the total behind it.

    The book carries a real USDZAR surface, so the gradient is not vacuous either: the FX option
    booked here puts `FXVol.USD.ZAR` rows in `greeks` on TWO tenor coordinates, which is the
    surface-node case the flattening exists for, beside the one-coordinate curve rows and the
    no-coordinate spot.
    """
    before = CLIENT.get('/book/risk').json()
    assert CLIENT.get('/book/risk').json() == before, 'a warm hit re-ran the book'
    assert before['mtm'] == pytest.approx(sum(row['value'] for row in before['per_deal']))

    booked = CLIENT.post('/book/deals', content=dump({'action': 'add', 'deal': FX_OPTION}),
                         headers=JSON).json()
    assert booked['written'] is True, booked
    after = CLIENT.get('/book/risk').json()

    assert after['etag'] != before['etag'], 'a booking left the risk etag standing'
    assert [row['reference'] for row in after['per_deal']] == ['CF1', 'OPT1']
    assert [row['deal_path'] for row in after['per_deal']] == ['0', '1']
    assert after['mtm'] == pytest.approx(sum(row['value'] for row in after['per_deal']))
    assert after['currency'] == 'USD'

    # the decisive half: the same numbers the ordinary pricing path reports for the same book
    priced = CLIENT.post('/book/price', content=dump({}), headers=JSON).json()
    service.EXECUTOR.queue.join()
    values = mtm(priced['result_id'])
    for row in after['per_deal']:
        assert abs(row['value']) > 1.0, 'a deal worth nothing makes the comparison vacuous'
        assert row['value'] == pytest.approx(values[row['reference']], rel=1e-9)

    greeks = {row['factor']: row for row in after['greeks']}
    assert 'FxRate.ZAR' in greeks and 'tenor' not in greeks['FxRate.ZAR'], 'a spot has no tenor'
    assert len(greeks['InterestRate.ZAR']['tenor']) == 1, 'a curve point is one coordinate'
    assert len(greeks['FXVol.USD.ZAR']['tenor']) == 2, 'a surface node is two'
    assert any(row['factor'] == 'FXVol.USD.ZAR' and row['value'] for row in after['greeks']), \
        'the option booked no vega - the gradient is vacuous'


#: A netting set's CSA tables, at zero thresholds - the shape the engine's own CVA fixtures use.
CSA = {'.CreditSupportList': [[0.0, 0.0]]}

#: The vanilla the sets hold: a one-year equity call, priced on the GBM the book declares - which
#: is what gives a credit Monte Carlo an exposure PROFILE rather than a constant.
VANILLA = {'Object': 'EquityOptionDeal', 'Currency': 'USD', 'Payoff_Currency': 'USD',
           'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD', 'Equity_Volatility': 'EQ',
           'Buy_Sell': 'Buy', 'Option_Type': 'Call', 'Option_Style': 'European', 'Units': 1000.0,
           'Strike_Price': EQ_SPOT, 'Expiry_Date': BASE + pd.DateOffset(years=1)}


def netting_set(reference, counterparty, deals):
    """One `NettingCollateralSet` node over its deals, uncollateralised, naming its counterparty
    where the engine reads it: `Credit_Support_Amounts.Counterparty` IS the `SurvivalProb` factor
    the CVA discounts by, which is why the recalc reads it from there and nowhere else."""
    return {'Instrument': {'.Deal': {
        'Object': 'NettingCollateralSet', 'Reference': reference, 'Netted': 'True',
        'Collateralized': 'False', 'Agreement_Currency': 'USD', 'Balance_Currency': 'USD',
        'Funding_Rate': 'USD', 'Liquidation_Period': 0.0, 'Settlement_Period': 0.0,
        'Credit_Support_Amounts': {
            'Counterparty': counterparty, 'Received_Threshold': CSA, 'Posted_Threshold': CSA,
            'Independent_Amount': CSA, 'Minimum_Received': CSA, 'Minimum_Posted': CSA}}},
        'Children': [{'Instrument': {'.Deal': deal}} for deal in deals]}


def xva_book(tmp_path, sets, counterparties=('CPTY_A', 'CPTY_B')):
    """A live book of netting sets, with a survival curve per counterparty and a GBM for the
    equity - everything a credit Monte Carlo needs and nothing it does not. The hazard rises with
    each counterparty, so the two sets' numbers are separable rather than coincidentally equal."""
    factors = dict(FACTORS, **EQUITY)
    for index, counterparty in enumerate(counterparties):
        factors['SurvivalProb.' + counterparty] = {
            'Recovery_Rate': 0.4,
            'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.2 + 0.3 * index]])}
    document = job(deals=(), factors=factors, sections={
        'Price Models': {'GBMAssetPriceModel.EQ': {'Vol': VOL, 'Drift': 0.0}},
        'Model Configuration': {'.ModelParams': {
            'modeldefaults': {'EquityPrice': 'GBMAssetPriceModel'}, 'modelfilters': {}}}})
    document['Calc']['Deals']['Deals']['Children'] = sets
    path = tmp_path / 'book.json'
    path.write_text(json.dumps(json.loads(dump(document)), indent=2), newline='\n')
    service.BOOK = service.Book(str(path))
    return path


@pytest.fixture
def desk_xva(tmp_path, monkeypatch):
    """Two counterparties, one vanilla each - the smallest book with a mosaic to keep. `DV_HOME`
    is the declared surface for where a desk's files live, so the projection lands in the gate's
    own tmp and the gate may read the file the service wrote."""
    monkeypatch.setenv('DV_HOME', str(tmp_path))
    yield xva_book(tmp_path, [
        netting_set('NS_A', 'CPTY_A', [dict(VANILLA, Reference='OPT_A')]),
        netting_set('NS_B', 'CPTY_B', [dict(VANILLA, Reference='OPT_B')])])
    service.BOOK = None


def recalc(netting_sets=None):
    """Ask for a recalc and wait for the one worker to drain every set it queued."""
    answer = CLIENT.post('/book/xva', content=json.dumps({'netting_sets': netting_sets}),
                         headers=JSON).json()
    service.EXECUTOR.queue.join()
    return answer


def xva_rows():
    return {entry['reference']: entry for entry in CLIENT.get('/book/xva').json()['sets']}


def test_the_xva_projection_is_a_mosaic_a_partial_recalc_moves_one_row_of(desk_xva, tmp_path):
    """The whole XVA lifecycle: never run, recalced, filed, and then partially recalced.

    A CMC is minutes of device time, so this view is a CACHED PROJECTION and every claim below is
    about the FILE. The sets read `never run` before anything is asked for. A full recalc queues
    one job per set and each writes its own row - a real `cva`, and the replay tuple that names the
    run it came from, so a number on the blotter is reproducible from its row alone. The file is
    what the view reads: `xva.json` on disk carries the same rows, which is what makes the
    projection survive the service restarting.

    Then the mosaic. A deal is booked into ONE set and only THAT set is recalced: its row moves -
    a new plan, a new id, a later stamp, a bigger number - and the other set's row is byte for byte
    the one it already had, `as_of` included. That is the whole design in one assertion: staleness
    is data, and a partial recalc is a partial WRITE rather than a file rebuilt from whatever
    happened to be current.
    """
    assert {reference: entry['status'] for reference, entry in xva_rows().items()} == {
        'NS_A': 'never run', 'NS_B': 'never run'}

    queued = recalc()['queued']
    assert [entry['reference'] for entry in queued] == ['NS_A', 'NS_B']
    before = xva_rows()
    for entry in queued:
        row = before[entry['reference']]
        assert row['status'] == 'done', row['error']
        assert row['cva'] > 0.0, 'a CVA of nothing makes every comparison below vacuous'
        assert row['result_id'] == entry['result_id']
        assert row['plan_hash'] and row['values_hash'] and row['seed'] == 1
    assert before['NS_A']['counterparty'] == 'CPTY_A'
    assert before['NS_A']['collateralized'] is False
    assert before['NS_B']['cva'] > before['NS_A']['cva'], 'the worse credit must cost more'

    # the projection IS the file - the view is a read of it, not of anything held in memory
    filed = json.loads((tmp_path / 'xva.json').read_text())['sets']
    assert {reference: filed[reference]['cva'] for reference in filed} == {
        reference: before[reference]['cva'] for reference in filed}
    assert filed['NS_A']['as_of'] == before['NS_A']['as_of']

    # an identical recalc over an unmoved book computes nothing and must AGE nothing: `as_of`
    # means when the number was computed, and re-aging a standing row inverts staleness-is-data
    repeat = recalc(None)['queued']
    assert {entry['result_id'] for entry in repeat} == {
        row['result_id'] for row in before.values()}
    unmoved = xva_rows()
    assert unmoved == before, 'a no-op recalc re-aged the projection'

    booked = CLIENT.post('/book/deals', content=dump(
        {'action': 'add', 'deal': dict(VANILLA, Reference='OPT_A2', Units=5000.0),
         'parent_reference': 'NS_A'}), headers=JSON).json()
    assert booked['written'] is True, booked

    partial = recalc(['NS_A'])['queued']
    assert [entry['reference'] for entry in partial] == ['NS_A']
    after = xva_rows()
    assert after['NS_A']['plan_hash'] != before['NS_A']['plan_hash']
    assert after['NS_A']['result_id'] == partial[0]['result_id']
    assert after['NS_A']['cva'] > before['NS_A']['cva'], 'a bigger position must cost more'
    assert after['NS_A']['as_of'] > before['NS_A']['as_of']
    assert after['NS_B'] == before['NS_B'], 'a partial recalc touched a row it was not asked for'


def test_an_unknown_set_refuses_by_name_and_a_missing_survival_curve_lands_in_the_row(
        tmp_path, monkeypatch):
    """The two refusals, and neither of them loses the projection.

    An unknown reference is a 422 that NAMES what was asked for and what the book actually holds,
    and it queues nothing at all - not even the set that was spelled correctly, because a desk that
    asked for two and got one would have to diff the answer to find out.

    A counterparty the market data carries no `SurvivalProb` block for is the other kind: the book
    is authored, the job is real, and it is the ENGINE that has the objection - so the row lands
    `failed` carrying the engine's own wording, which is what a desk reads to find out why its
    number is missing. The rest of the file stands: one set failing is one row, not a lost
    projection.
    """
    monkeypatch.setenv('DV_HOME', str(tmp_path))
    xva_book(tmp_path, [netting_set('NS_A', 'CPTY_A', [dict(VANILLA, Reference='OPT_A')]),
                        netting_set('NS_GHOST', 'GHOST', [dict(VANILLA, Reference='OPT_G')])])
    try:
        assert recalc(['NS_A'])['queued'][0]['reference'] == 'NS_A'
        standing = xva_rows()['NS_A']
        assert standing['status'] == 'done' and standing['cva'] > 0.0

        refused = CLIENT.post('/book/xva', content=json.dumps(
            {'netting_sets': ['NS_A', 'NOT_A_SET']}), headers=JSON)
        assert refused.status_code == 422
        assert 'NOT_A_SET' in refused.json()['detail']
        assert 'NS_GHOST' in refused.json()['detail'], 'the refusal names the sets the book holds'
        assert xva_rows()['NS_A'] == standing, 'a refused recalc queued a set anyway'

        recalc(['NS_GHOST'])
        rows = xva_rows()
        assert rows['NS_GHOST']['status'] == 'failed'
        assert rows['NS_GHOST']['cva'] is None
        assert 'GHOST' in rows['NS_GHOST']['error']
        assert rows['NS_A'] == standing, "a failed set took another set's row with it"
    finally:
        service.BOOK = None


# --------------------------------------------------------------------------------------------
# a quote is FOR someone, and it is firm for a WINDOW
# --------------------------------------------------------------------------------------------
#: The client the quoting book has opened, as a Reference. Not `CLIENT`, which is this module's
#: TestClient - and a desk's client is a `NettingCollateralSet` in any case, since that is the node
#: the counterparty and the CSA are declared on.
CLIENT_SET = 'CLIENT_A'


@pytest.fixture
def quoting_client(tmp_path, monkeypatch):
    """The quoting desk with a CLIENT on its book: the same one-cashflow book with a real USDZAR
    surface ticked in, plus an EMPTY `NettingCollateralSet` naming a counterparty the market data
    carries a survival curve for. Empty because what lands under it is the whole point.

    Built here rather than off `quoting` because a netting set is a NODE - it holds children - and
    the `job` helper wraps plain deal blocks; and kept separate from `quoting` because the existing
    gates read that book's deal paths by position.
    """
    monkeypatch.setenv('DV_HOME', str(tmp_path))
    factors = dict(FACTORS, **{'SurvivalProb.CPTY_A': {
        'Recovery_Rate': 0.4, 'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.2]])}})
    document = job(factors=factors, sections={
        'Bootstrapper Configuration': {'FXVolSurfaceParameters': {}}})
    document['Calc']['Deals']['Deals']['Children'].append(netting_set(CLIENT_SET, 'CPTY_A', []))
    path = tmp_path / 'book.json'
    path.write_text(json.dumps(json.loads(dump(document)), indent=2), newline='\n')
    service.BOOK = service.Book(str(path))
    ticked = CLIENT.post('/book/market', content=dump({'quotes': fx_vol_quotes()}),
                         headers=JSON).json()
    assert ticked['written'] is True, ticked
    yield path
    service.BOOK = None


def set_paths(path):
    """`{Reference: deal_path}` for the netting sets the book on disk carries, read through the
    service's own reading of them rather than by counting positions in the gate."""
    return {node['Instrument']['.Deal']['Reference']: deal_path
            for deal_path, node in service.netting_sets(json.loads(path.read_text()))}


def test_a_quote_for_a_client_books_under_the_clients_netting_set(quoting_client, tmp_path):
    """A quote given FOR a client books under that client's netting set, and a quote given for
    nobody books at the root exactly as it always did.

    The decisive assertion is the deal PATH: a trade booked at the root is outside every netting
    set's subtree, so the CVA projection - which prices one set's subtree per run - cannot see it.
    Beneath the set's own path is the whole feature, and the path is read off the book on disk
    through `service.netting_sets`, the same walk the XVA verb takes, rather than by counting
    positions here. The mirror still mirrors: the approval is where client paper becomes the
    bank's position, and nesting it somewhere else must not touch the side it lands on.

    The root half is in the same gate on purpose. `netting_set` absent has to be TODAY's booking,
    and a control run on the same book, in the same breath, is what says so.
    """
    quote = quote_of('ZeroCostCollar', COLLAR, netting_set=CLIENT_SET)
    assert quote['netting_set'] == CLIENT_SET

    pending = tmp_path / 'tmp' / (quote['quote_id'] + '.json')
    filed = json.loads(pending.read_text())
    assert filed['quote']['netting_set'] == CLIENT_SET, 'the pending trade forgot who it is for'
    assert filed['quoted_at'], 'the pending trade is not stamped with when it was given'

    booked = CLIENT.post('/book/quote', json={'quote_id': quote['quote_id']}).json()
    assert booked['written'] is True, booked
    on_disk = json.loads(quoting_client.read_text())
    node = deal_at(on_disk, booked['deal_path'])

    parent = set_paths(quoting_client)[CLIENT_SET]
    assert booked['deal_path'].startswith(parent + '/'), booked['deal_path']
    assert node['Instrument']['.Deal']['Object'] == 'StructuredDeal'
    assert [child['Instrument']['.Deal']['Reference'] for child in node['Children']] == [
        leg['reference'] for leg in quote['legs']]
    assert [child['Instrument']['.Deal']['Buy_Sell'] for child in node['Children']] == [
        {'Buy': 'Sell', 'Sell': 'Buy'}[leg['buy_sell']] for leg in quote['legs']]

    # and the control: no client named is the root booking, unchanged
    house = quote_of('ZeroCostCollar', COLLAR)
    assert house['netting_set'] is None
    at_root = CLIENT.post('/book/quote', json={'quote_id': house['quote_id']}).json()
    assert at_root['written'] is True, at_root
    assert '/' not in at_root['deal_path'], 'a quote for nobody nested under something'


def test_a_quote_for_a_client_the_book_never_opened_refuses_at_the_ask(quoting_client, tmp_path):
    """A netting set nobody opened refuses while the quote is being ASKED for - naming what was
    asked for and what the book actually holds, in the XVA verb's own wording, since it is the
    same question asked of the same book.

    At the ask rather than at the approval: a quote given under a set that does not exist is a
    quote nobody can book, and finding that out at the approval is finding it out after the client
    has the sheet. So nothing is priced, nothing is filed, and the book is untouched - the refusal
    is a 422 on the salesperson's own call, not an `error` status they have to poll for.
    """
    before = quoting_client.read_bytes()
    refused = CLIENT.post('/book/structure', content=dump(
        {'structure': 'ZeroCostCollar', 'params': COLLAR, 'netting_set': 'NOT_A_CLIENT'}),
        headers=JSON)

    assert refused.status_code == 422
    assert 'NOT_A_CLIENT' in refused.json()['detail']
    assert CLIENT_SET in refused.json()['detail'], 'the refusal names the sets the book holds'
    assert not list((tmp_path / 'tmp').glob('*.json')), 'a refused quote filed a pending trade'
    assert quoting_client.read_bytes() == before


def declare_policy(path, **stated):
    """The desk's mandate written onto the live book, or taken off it where nothing is stated -
    and the book's bytes as they now stand.

    A `Quote Policy` block is AUTHORED data and no verb writes one, so the gate authors it the same
    way the fixture authored the book. The live book is then REOPENED, which is what a desk that
    edits its mandate does: `Book` re-parses on mtime, and Windows' file timestamps do not resolve
    two writes landing inside the same 15ms clock tick - so a gate that changed a mandate twice in
    a row would otherwise read the first one back and pass for the wrong reason.
    """
    document = json.loads(path.read_text())
    if stated:
        document['Calc'][structures.QUOTE_POLICY] = dict(stated)
    else:
        document['Calc'].pop(structures.QUOTE_POLICY, None)
    path.write_text(json.dumps(document, indent=2), newline='\n')
    service.BOOK = service.Book(str(path))
    return path.read_bytes()


def test_a_quote_is_firm_only_for_the_window_the_book_declares(quoting, tmp_path):
    """One quote, one book, three mandates - which is what makes this about the WINDOW and nothing
    else. The quote is given once and never re-given; only the book's `Quote Policy` moves.

    With `firm_seconds: 0` an approval arriving immediately is already outside the window and is
    refused 422 naming the age, the window and the remedy - and it writes NOTHING, asserted on the
    book's bytes, because a stale quote is a refusal rather than a booking at a price nobody stands
    behind any more. With the block taken off, THE SAME pending quote books: the absence of the
    policy is the off switch here as it is for the risk-impact half, so a book that declares no
    mandate behaves exactly as it always did.

    The third mandate is the compatibility case: a pending file with no `quoted_at` - one filed
    before quotes were stamped - cannot be shown to be inside any window, and an unknown age is not
    an age inside it. So a real window treats it as AGED and the refusal says which case it is,
    rather than defaulting to fresh and booking a quote of unknown vintage.
    """
    quote = quote_of('ZeroCostCollar', COLLAR)
    pending = tmp_path / 'tmp' / (quote['quote_id'] + '.json')
    assert json.loads(pending.read_text())['quoted_at'], 'the quote was filed unstamped'

    # a quote filed before quotes were stamped, authored here because there is no other way to
    # reach a pending file with no `quoted_at` on a build that stamps them
    stale = dict(json.loads(pending.read_text()))
    stale.pop('quoted_at')
    stale['quote'] = dict(stale['quote'], quote_id='unstamped')
    (tmp_path / 'tmp' / 'unstamped.json').write_text(json.dumps(stale, indent=2), newline='\n')

    before = declare_policy(quoting, firm_seconds=0)
    refused = CLIENT.post('/book/quote', json={'quote_id': quote['quote_id']})

    assert refused.status_code == 422
    detail = refused.json()['detail']
    assert quote['quote_id'] in detail and 'firm for 0s' in detail, detail
    assert 's ago' in detail, 'the refusal never named the age'
    assert 're-quote' in detail and 'seconds to move' in detail, 'no remedy in the refusal'
    assert quoting.read_bytes() == before, 'a refused approval wrote to the book'
    assert pending.is_file(), 'a refused approval consumed the pending trade'

    # a real window, and a quote whose age cannot be established: aged, and the message says why
    declare_policy(quoting, firm_seconds=600)
    unstamped = CLIENT.post('/book/quote', json={'quote_id': 'unstamped'})
    assert unstamped.status_code == 422
    assert 'quoted_at' in unstamped.json()['detail'], unstamped.json()['detail']
    assert 'firm for 600s' in unstamped.json()['detail']

    # the same quote, the same book, no mandate: it books
    declare_policy(quoting)
    booked = CLIENT.post('/book/quote', json={'quote_id': quote['quote_id']}).json()

    assert booked['written'] is True, booked
    node = deal_at(json.loads(quoting.read_text()), booked['deal_path'])
    assert [child['Instrument']['.Deal']['Buy_Sell'] for child in node['Children']] == [
        {'Buy': 'Sell', 'Sell': 'Buy'}[leg['buy_sell']] for leg in quote['legs']]
