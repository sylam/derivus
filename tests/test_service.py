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
    untouched - a moved node is a new plan, never a tick."""
    quotes = fx_vol_quotes()
    doc = json.loads(book.read_text())
    doc['Calc']['MergeMarketData']['ExplicitMarketData'][
        'Bootstrapper Configuration'] = {'FXVolSurfaceParameters': {}}
    book.write_text(json.dumps(doc, indent=2), newline='\n')

    first = CLIENT.post('/book/market', content=dump({'quotes': quotes}), headers=JSON).json()
    assert first['installed'] == ['FXVolPrices.USD.ZAR']

    ticked = json.loads(dump(fx_vol_quotes()))
    for point in ticked['FXVolPrices.USD.ZAR']['instrument']['Points']:
        if point['Quote_Type'] == 'ATM':
            point['Quoted_Market_Value'] += 0.01
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


def quote_of(structure, params):
    """Ask for a quote and wait for the one worker to give it - the outcome under `stats.Quote`,
    exactly where a solve's coordinates sit under `stats.Solved`."""
    submitted = CLIENT.post('/book/structure', content=dump(
        {'structure': structure, 'params': params}), headers=JSON).json()
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
