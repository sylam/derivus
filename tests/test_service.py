"""One vocabulary, two bindings - and the gates are what says the second one owns no logic.

`derivus.service` is `Context` over HTTP, so the decisive gate is PARITY: the same job over HTTP
produces the same numbers as `load_json` + `run_job` in process. `/schema`, `/schema/job`,
`/validate` and `/describe` are that claim for the read verbs.

The dispatcher makes four promises. Ordering: one pricing worker, so a base valuation jumps a
simulation among the jobs still WAITING and a running job is never preempted. Identity: a
`result_id` is the hash of the replay tuple, so the same job twice is one execution - and it holds
while the first is still running. Survival: an engine failure is a result like any other. And
`plan_id`: how a job ARRIVED cannot change what it reports.

The Bloomberg verbs are gated at their seams (`discover.provision`, `security_map.stale`,
`fetch_fx_vol` monkeypatched; the job's lazy imports are what lets a patch reach it), so no blpapi,
socket or map file is needed. `--tick`'s metronome rides the same seam.

`/book/hn` is gated on the emitter, the round trip and the refusal; `/book/structure` +
`/book/quote` on the two halves being one trade - the collar nets to zero and the BOOK marks the
deal it wrote at zero. `DV_HOME` is the declared surface for where those files land.

Ordering and dedupe are deterministic without a clock: a first job blocks inside `run_job` and
announces it through an `Event`, so the others are provably queued before it is released.
`Queue.join` is the barrier everywhere else.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import threading
import time
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

#: `Cash_Payoff` IS this binary's notional, so the declaration makes it required and leaving it out
#: is an authoring message rather than a missing factor.
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
    """A job document, authored as the objects a market data file holds. Dumped through
    `CustomJsonEncoder`, so the `.Curve`/`.Timestamp` tokens the endpoint receives are the ones a
    file carries. `sections` adds further market-data sections.
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
    """Submit, wait for the one worker to drain the queue, and read the summary back with the id it
    was filed under."""
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
    """A Context as far as the executor is concerned - it calls `run_job()` and reads nothing else.

    That one verb is the whole seam between the queue and the engine, so ordering and dedupe are
    observable by handing the executor one of these, and a `Results` tree no calculation produces
    can be put through the store. Nothing is patched; `hold` makes the worker's occupancy an event
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
    """The decisive gate: identical results, table for table and cell for cell.

    Summary shapes and drill-down cells are both held against the same in-process run. The closed
    form (`amount x spot x exp(-rate x 2)`) is asserted too, so the comparison cannot pass by both
    sides being empty.
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
    """The four replay coordinates travel with the result, and the two hashes are the loaded job's
    own rather than something the service re-derived."""
    context = in_process(job())
    _, result = run(job())

    assert result['plan_hash'] == context.plan_hash()
    assert result['values_hash'] == context.values_hash()
    assert result['engine_version'] == derivus.__version__
    assert result['seed'] == 1


def test_the_schema_endpoint_is_the_declarations_plus_the_version():
    """The endpoint is `schema.mapping` plus the version that emitted it - what lets a front end
    render panels, tables and enums without restating them."""
    published = CLIENT.get('/schema').json()

    assert published.pop('engine_version') == derivus.__version__
    assert published == json.loads(json.dumps(derivus.schema.mapping, cls=CustomJsonEncoder))
    # not vacuous: this is the declaration a client reads to know which fields it may patch
    assert published['Factor']['types']['FxRate']['Spot']['bind'] == 'value'


def test_the_schema_publishes_which_deals_take_children():
    """`containers` is `Deal.accepts_children` emitted into the store, so a client answers "may this
    take children" without importing the engine. Held to the accessor over EVERY declared type,
    both directions, non-vacuously."""
    published = CLIENT.get('/schema').json()['Instrument']
    accessor = sorted(t for t in published['types'] if derivus.instruments.accepts_children(t))

    assert published['containers'] == accessor
    assert 'NettingCollateralSet' in published['containers']
    assert 'FixedCashflowDeal' not in published['containers']
    # a container the create menu does not offer is bookable over MCP and uncreatable in every UI
    menued = {t for members in published['groups'].values() for t in members}
    assert set(published['containers']) <= menued


def test_a_done_result_carries_the_run_stats():
    """`Stats` - timings, deals loaded, calibration provenance - rides the summary as a flat dict,
    never through `tables_of` (which would flatten `Calibrations` into a fake table path). A
    calculation reporting none reads as `{}` rather than a KeyError."""
    _, result = run(job())

    assert result['stats']['Deals loaded'] == 1
    assert 'stats' not in result['tables'] and 'Stats' not in result['tables']

    service.EXECUTOR.submit(service.Job('statless', Held('statless', [], results={}), {}),
                            service.HEAVY)
    service.EXECUTOR.queue.join()
    assert CLIENT.get('/results/statless').json()['stats'] == {}


def test_the_ui_is_mounted_only_when_it_is_built(tmp_path):
    """The UI is an optional CLIENT, so the mount is a flag over a directory and an empty one
    refuses. The 404 on `/ui/portfolio` is pinned deliberately: `StaticFiles(html=True)` has no SPA
    fallback, which is what the front end's no-router decision rests on."""
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
    """A live book over a temp copy of the one-cashflow job. Written at indent 2, which is what the
    formatting gate holds the rewrite to."""
    path = tmp_path / 'book.json'
    path.write_text(json.dumps(json.loads(dump(job())), indent=2), newline='\n')
    service.BOOK = service.Book(str(path))
    yield path
    service.BOOK = None


def test_a_missing_book_file_starts_blank_and_takes_its_first_booking(tmp_path):
    """`--book` at an empty path creates the blank book: no deals, dated today, the skeleton's USD
    market data aboard - which is what lets the first booking validate rather than be refused for
    market data a bare file would lack."""
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
    """A miss is a refusal naming the fix, never a book invented in memory that no file backs."""
    assert CLIENT.get('/book').status_code == 404
    assert '--book' in CLIENT.get('/book').json()['detail']


def test_a_booking_lands_in_the_file_and_every_client_sees_it(book):
    """The file is the source of truth: the booked deal is in the answer, in the file on disk and in
    the next GET, with a moved etag."""
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
    """Validate-before-write, refused on both counts at once: an authoring message and market data
    the book does not carry. File bytes and etag stand still, and the refusal is an ANSWER carrying
    the messages."""
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
    """A deal naming a curve the book has no block for would load and then be silently DROPPED by
    discovery, so the DELTA of missing factors refuses it by name. The book's pre-existing gaps do
    not block - only what this booking adds."""
    outcome = CLIENT.post('/book/deals', content=dump(
        {'action': 'add',
         'deal': dict(CASHFLOW, Reference='CF9', Currency='GBP', Discount_Rate='GBP')}),
        headers=JSON).json()

    assert outcome['written'] is False
    assert 'no market data for InterestRate.GBP' in outcome['refused']


def test_booking_then_deleting_restores_the_file_bytes(book):
    """The rewrite keeps the file's own indent, so book-then-delete is a no-op to the byte and a
    booking is reviewable as the diff of the deal and nothing else."""
    before = book.read_bytes()
    booked = CLIENT.post('/book/deals', content=dump({'action': 'add', 'deal': BOOKED}),
                         headers=JSON).json()
    deleted = CLIENT.post(
        '/book/deals', json={'action': 'delete', 'deal_path': booked['deal_path']}).json()

    assert booked['written'] and deleted['written'] and deleted['deleted'] == 'CF2'
    assert book.read_bytes() == before


def test_a_parent_must_exist_be_unique_and_take_children(book):
    """Appending under the wrong node is a mis-booked trade: a leaf parent refuses naming its type,
    an unknown one refuses naming it, and neither writes."""
    under_leaf = CLIENT.post('/book/deals', content=dump(
        {'action': 'add', 'deal': BOOKED, 'parent_reference': 'CF1'}), headers=JSON)
    unknown = CLIENT.post('/book/deals', content=dump(
        {'action': 'add', 'deal': BOOKED, 'parent_reference': 'GHOST'}), headers=JSON)

    assert under_leaf.status_code == 422 and 'FixedCashflowDeal' in under_leaf.json()['detail']
    assert unknown.status_code == 422 and 'GHOST' in unknown.json()['detail']
    assert len(CLIENT.get('/book').json()['document']['Calc']['Deals']['Deals']['Children']) == 1


def test_a_booking_nests_under_a_container(book):
    """A container books like any deal and then holds its children: the nested node lands in the
    parent's `Children` at the positional `deal_path` every client shares."""
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
    """Merge one field into a deal at its path, validated first, written atomically, etag moved."""
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
    """The same validate-delta rule as a booking, on the amend branch."""
    before = book.read_bytes()
    outcome = CLIENT.post('/book/deals', json={
        'action': 'amend', 'deal_path': '0', 'fields': {'Discount_Rate': 'GBP'}}).json()

    assert outcome['written'] is False
    assert 'no market data for InterestRate.GBP' in outcome['refused']
    assert book.read_bytes() == before


def test_amending_back_is_byte_identical(book):
    """An edit undone leaves no trace, not even a reformat."""
    before = book.read_bytes()
    original = json.loads(book.read_text())['Calc']['Deals']['Deals']['Children'][0][
        'Instrument']['.Deal']['Amount']
    CLIENT.post('/book/deals', json={
        'action': 'amend', 'deal_path': '0', 'fields': {'Amount': 1.0}})
    CLIENT.post('/book/deals', json={
        'action': 'amend', 'deal_path': '0', 'fields': {'Amount': original}})
    assert book.read_bytes() == before


def test_an_amendment_needs_a_real_path(book):
    """An unknown path is a 422 naming it; a NEGATIVE path refuses rather than resolving from the
    end, which would quietly amend a different deal."""
    unknown = CLIENT.post('/book/deals', json={
        'action': 'amend', 'deal_path': '7', 'fields': {'Amount': 1.0}})
    negative = CLIENT.post('/book/deals', json={
        'action': 'amend', 'deal_path': '-1', 'fields': {'Amount': 1.0}})
    assert unknown.status_code == 422 and '7' in unknown.json()['detail']
    assert negative.status_code == 422
    assert json.loads(book.read_text())['Calc']['Deals']['Deals']['Children'][0][
        'Instrument']['.Deal']['Amount'] == AMOUNT


def test_a_what_if_prices_the_candidate_and_writes_nothing(book):
    """The book plus a candidate priced off an in-memory copy, the file never moving. The
    candidate's value comes back through the ordinary result surface."""
    before = book.read_bytes()
    submitted = CLIENT.post('/book/price', content=dump({'deal': BOOKED}), headers=JSON).json()
    service.EXECUTOR.queue.join()
    result = CLIENT.get('/results/{}'.format(submitted['result_id'])).json()

    assert result['status'] == 'done'
    assert mtm(submitted['result_id'])['CF2'] == pytest.approx(
        BOOKED['Amount'] * SPOT * np.exp(-RATE * 2.0), rel=1e-3)
    assert book.read_bytes() == before


def fx_vol_snapshot():
    """A USDZAR snapshot through the Bloomberg package's own normalization - canned observations
    standing in for the terminal, everything downstream the real pipeline. One object, so the
    `/book/market` and `/book/bloomberg` gates tick the same numbers."""
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
    """The practical loop end to end: canned Bloomberg observations normalized, `/book/market`
    installs and bootstraps the `FXVol` surface into the book file, `/book/solve` finds the strike
    at which an option on it marks at the target premium. The before/after validate pins the
    surface as load-bearing - the option is unpriceable until the tick lands."""
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
    """The SERVED second-order route (`test_base_valuation_gamma` owns the oracles): a what-if with
    `calculation_overrides` returns `Greeks_Second`, its cells are the in-process run's to the bit,
    the spot diagonal is a live positive gamma and the vanna cross carries real weight."""
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
    """The structure guard: a re-post moving only `Quoted_Market_Value`/`Timestamp` updates; one
    moving a pillar refuses by name with the file untouched - a moved node is a new plan, never a
    tick.

    The stamp MOVES here because it is the one member of `schema.MARKET_QUOTE_VALUES` nothing else
    in the repo re-posts through the guard: a guard misspelling `Timestamp` otherwise passes 118
    tests across three files."""
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
    """The same guard with a two-way on the point: `Quoted_Bid`/`Quoted_Ask` are on the VALUE side
    of the line the mid is on, so a re-post moving bid, ask and mid together is a tick and the file
    takes it. A moved `Pillar` still refuses in the identical wording.

    The bootstrap runs on every one of these posts and builds its surface from
    `Quoted_Market_Value` alone, so a block carrying the sides ticks exactly as a mid-only one does.
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
    """A spot tick lands in the file through the engine's own values seam; a structural key is
    refused by the engine's own raise. The service adds no judgment."""
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
    """A quote block the bootstrap cannot turn into a factor refuses the WHOLE write with the
    bootstrap's own messages: a book must never carry a market its bootstrap complained about."""
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


#: A `NettingCollateralSet` authored as a DEAL compiles like any other and has no `Deal.generate`,
#: so `Deal.calculate` logs CRITICAL and marks it at nothing - a real book whose PRICING talks on
#: the channel `CapturedErrors` listens to.
SKIPPED_NETTING = 'generate in class NettingCollateralSet not implemented yet'

#: The density knob: one CRITICAL per deal per priced run, so 40 gives ~4,000 lines across the tick
#: loop below - which is what makes the overlap the gate depends on a fact rather than a hope.
NOISY_DEALS = 40


def skipped_netting_deal(reference):
    return {'Object': 'NettingCollateralSet', 'Reference': reference, 'Netted': 'True',
            'Collateralized': 'False', 'Settlement_Currency': ''}


class Chatter(logging.Handler):
    """Counts the PRICED run's CRITICAL lines by message, not by thread - `TestClient` runs a sync
    endpoint on an anyio worker thread this gate never sees. Read as a delta across each POST, so
    each tick answers for its own window.
    """

    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.count = 0

    def emit(self, record):
        if SKIPPED_NETTING in record.getMessage():
            self.count += 1


def test_a_concurrent_runs_critical_does_not_refuse_an_innocent_tick(tmp_path):
    """`market_edit` captures the root logger around `context.bootstrap()`, and the root logger is
    every thread's - so a queued `/book/price` logging a CRITICAL inside that window turned a good
    tick into `written: False` with a FOREIGN run's message as the reason. `record.thread` against
    the constructing thread's ident is the fix, and it is a comparison rather than a timing window.

    The recipe: a real book (one cashflow, the FX vol bootstrapper, a real USDZAR block, 40
    `NettingCollateralSet` deals the pricer skips), a background thread keeping `/book/price`
    queued, 25 ticks each moving the ATM.

    The OVERLAP is asserted per POST, so a quiet worker fails the gate rather than passing it
    vacuously: measured 22-24 of 25 ticks carry a foreign CRITICAL in their own window, over ~4,100
    records in ~0.5 s. MUTANT (thread test removed): 9, 7 and 8 of 25 ticks written across three
    runs; with it in place, 25 of 25 five times over.

    The negative arm is `test_a_bootstrap_that_complains_writes_nothing`: an error on the tick's OWN
    thread still refuses the whole write.
    """
    deals = [CASHFLOW] + [skipped_netting_deal('NCS{}'.format(i)) for i in range(NOISY_DEALS)]
    path = tmp_path / 'book.json'
    path.write_text(json.dumps(json.loads(dump(job(deals=deals, sections={
        'Bootstrapper Configuration': {'FXVolSurfaceParameters': {}}}))), indent=2), newline='\n')
    service.BOOK = service.Book(str(path))
    chatter = Chatter()
    stop = threading.Event()
    submitted = []

    def keep_pricing():
        """One `/book/price` after another, each at its own seed so nothing coalesces onto the
        last."""
        seed = 0
        while not stop.is_set():
            seed += 1
            submitted.append(CLIENT.post('/book/price', content=dump(
                {'calculation_overrides': {'Random_Seed': seed}}), headers=JSON).json()['status'])
            time.sleep(0.001)

    pricer = threading.Thread(target=keep_pricing, daemon=True)
    try:
        installed = CLIENT.post('/book/market', content=dump({'quotes': fx_vol_quotes()}),
                                headers=JSON).json()
        assert installed['installed'] == ['FXVolPrices.USD.ZAR'] and installed['written'] is True

        logging.getLogger().addHandler(chatter)
        pricer.start()
        # let the worker get into the book first, so the loop opens against a busy queue
        while chatter.count == 0 and not stop.is_set():
            time.sleep(0.005)

        ticks = []
        for n in range(25):
            moved = json.loads(dump(fx_vol_quotes()))
            for point in moved['FXVolPrices.USD.ZAR']['instrument']['Points']:
                if point['Quote_Type'] == 'ATM':
                    point['Quoted_Market_Value'] += 0.0001 * n
            before = chatter.count
            outcome = CLIENT.post('/book/market', content=json.dumps({'quotes': moved}),
                                  headers=JSON).json()
            ticks.append((outcome, chatter.count - before))
    finally:
        stop.set()
        pricer.join(timeout=60)
        service.EXECUTOR.queue.join()
        logging.getLogger().removeHandler(chatter)
        service.BOOK = None

    assert len(submitted) > 20 and set(submitted) <= {'queued', 'running', 'done'}, submitted
    assert chatter.count > 1000, (
        'the priced runs emitted {} CRITICAL lines - the worker was not talking and this gate is '
        'measuring nothing'.format(chatter.count))
    overlapped = [delta for _, delta in ticks if delta > 0]
    assert len(overlapped) >= 15, (
        'only {} of {} ticks had a foreign CRITICAL land inside their own window - the interleaving '
        'this gate exists for did not happen'.format(len(overlapped), len(ticks)))

    refused = [(outcome.get('refused'), delta) for outcome, delta in ticks
               if outcome.get('written') is not True]
    assert refused == [], (
        '{} of {} innocent ticks were refused by another thread\'s run: {}'.format(
            len(refused), len(ticks), refused[:2]))
    assert all(outcome['updated'] == ['FXVolPrices.USD.ZAR'] for outcome, _ in ticks), (
        'a tick wrote without moving the quote it posted')


def test_the_capture_hears_its_own_thread_and_no_other():
    """The mechanism, with no timing in it: the foreign record comes from a thread this gate JOINS
    before looking. Both halves matter - a handler that heard nothing would refuse nothing ever, so
    the same handler is required to hear THIS thread. Nothing is patched: `CapturedErrors` is the
    shipped class on the shipped channel, which is why the filter has to be on the record.
    """
    captured = service.CapturedErrors()
    foreign = threading.Thread(
        target=lambda: logging.critical('Deal FOREIGN skipped - a queued run, not this tick'))
    logging.getLogger().addHandler(captured)
    try:
        foreign.start()
        foreign.join(timeout=30)
        assert not foreign.is_alive(), 'the foreign thread never finished - nothing was measured'
        assert captured.messages == [], captured.messages
        logging.error('FXVolSurfaceParameters wrote no FXVol price factor')
    finally:
        logging.getLogger().removeHandler(captured)

    assert captured.messages == ['FXVolSurfaceParameters wrote no FXVol price factor'], (
        'the capture stopped hearing its own thread - a handler that hears nothing refuses '
        'nothing, which is the opposite defect')
    assert captured.thread == threading.get_ident()


def built_surface(path, quotes=None):
    """A live book carrying a BUILT `FXVol.USD.ZAR`: the file declares the surface bootstrapper and
    the surface arrives by POSTing a quote block to `/book/market`. So the Heston-Nandi gates start
    from a surface the engine built, never one written by hand."""
    path.write_text(json.dumps(json.loads(dump(job(sections={
        'Bootstrapper Configuration': {'FXVolSurfaceParameters': {}}}))), indent=2), newline='\n')
    service.BOOK = service.Book(str(path))
    ticked = CLIENT.post('/book/market', content=dump(
        {'quotes': quotes if quotes is not None else fx_vol_quotes()}), headers=JSON).json()
    assert ticked['written'] is True and 'FXVol.USD.ZAR' in ticked['new_factors']
    return path


def desk_smile():
    """A USDZAR smile at 1M, 2M, 3M and 6M, as the `FXVolPrices` block a quote source posts.

    FOUR PILLARS: every ladder rung the surface does not carry snaps onto one it does, and the
    canned two-pillar surface collapses ten rungs onto FOUR distinct contracts - too few for five
    parameters. Four pillars are the fewest giving the ATM term structure three expiries AND the
    two wing pairs two different ones (eight contracts, measured).

    THE RISK REVERSAL IS NEGATIVE in pair terms, the sign USDZAR trades at. On the `FxRate.ZAR`
    axis the model is fitted on that is a smile RISING with strike, which a strictly positive
    `Gamma_Star` cannot represent - so this fixture is what proves the signed leverage share.

    Expiries stop at 6M for the clock: fit cost is linear in the longest expiry's step count (252
    GARCH steps a year per L-BFGS-B evaluation), and the 3M/1Y canned surface ran past 21 minutes.
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
    """The desk's ladder off the built surface: ATM at 1M/2M/3M/6M/9M/1Y plus the 25 delta wings at
    3M and 6M, ten points, nothing past a year.

    What is asserted is what makes them the SURFACE'S points rather than a moneyness grid laid over
    it. The expiries are the surface's own (1M/2M/3M/6M here), so the 9M and 1Y rungs move to the
    nearest quoted one at or under a year and `Quote_Source` SAYS SO - the difference between a
    substitution and a silent interpolation. Ten rungs are therefore eight distinct contracts. The
    weights are normalised Black vega, so they sum to one and the back ATM outweighs the front,
    which stops an unweighted fit abandoning the front end. And the wings straddle the spot.
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

        # THE CONVENTION CHAIN: each emitted strike back through the family's OWN moneyness
        # dispatch, off the switches the block declares, must return the vol the block carries -
        # what fails first if the orientation, the forward or the inversion is wrong.
        #
        # THE VOL IS READ AT THE PILLAR, the strike hangs off the DATE. `Expiry_Date` is the pillar
        # rounded to whole days and the fit reads its accrual back off it, so the forward is that
        # accrual's; under a curve that is not ACT_365 the pillar is a different number
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
    """A `HestonNandiModelPrices.ZAR` block with nine quotes at one, two and three weeks.

    The shift gate needs the fit's ARITHMETIC, not its ladder, so the expiries are the shortest
    that still make three step counts (5, 10, 15 GARCH steps) and the gate runs in seconds instead
    of the emitter ladder's quarter hour. Everything else is the block the emitter writes.
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
    """A scenario shift must reach the fitted world exactly once. It used to reach it twice:
    `fx_surface_block` folded `Volatility_Delta` into every emitted vol and `bootstrap` added
    `vol_surface.delta` again, so a 1-vol-point scenario calibrated a 2-vol-point world.

    Two halves. The emitter is delta-BLIND: a block authored at 0.01 is the block authored at 0.0
    bit for bit, because a quote block is a QUOTE. And the fit applies the shift ONCE: fitting
    unshifted quotes under a 0.01 scenario lands on the same five parameters as fitting HAND-BUMPED
    quotes under none. The third fit rules out a shift that did nothing.
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

        # MEASURED at 7.7e-9 relative: one ulp between the two worlds' quoted vols amplified by a
        # line search, not a second application. A doubled shift moves these by percent
        assert scenario == pytest.approx(by_hand, rel=1e-6), (
            'a 1 vol point scenario did not fit the world 1 vol point away')
        assert scenario != pytest.approx(unmoved, rel=1e-3), (
            'the shift moved nothing, so the identity above is vacuous')
    finally:
        service.BOOK = None


def test_a_collapsed_ladder_refuses_and_nothing_past_a_year_is_ever_snapped_to(tmp_path):
    """The two things an unconditional argmin does not do, both silent when they happen.

    A COLLAPSED LADDER. The canned surface carries 3M and 1Y only, so ten rungs land on FOUR
    distinct contracts and a five-parameter fit is handed them as though they were ten quotes. What
    identifies `H0`, `Beta` and `Omega` is the ATM TERM STRUCTURE, which a collapse destroys - so
    the emitter refuses, naming the pillars, the ladder, the count and the remedy. The
    `expiry.size < 2` guard cannot see this: the canned surface IS a grid.

    NOTHING PAST A YEAR. Snapping is an argmin over every pillar, so a surface quoting 2Y answers
    the 1Y rung with 2Y and a sub-year fit borrows from a smile nobody quotes. Here the four-pillar
    surface carries a 2Y pillar the ladder must not touch: every emitted expiry is inside the year,
    the 9M and 1Y rungs land on 6M, and the block says so.
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

    # where NOTHING is admissible every rung is DROPPED rather than snapped past the cap, and the
    # refusal says which rung did what
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
    bootstraps, the five parameters land in the book FILE, and the model has to reprice the ten
    quotes it was fitted to - in the family's OWN objective, recomputed here off the written
    parameters.

    It runs on the NEGATIVE risk reversal, the sign USDZAR trades at and the one the family could
    not fit until the leverage share carried a sign. So the gate holds the shape rather than the
    numbers: `Gamma_Star` NEGATIVE and the optimum INTERIOR - no parameter on a box bound, which is
    what a fit that cannot represent its data does.

    MEASURED on `desk_smile`'s four pillars: 288 s on a quiet box, 549 s with the suite beside it,
    the same five numbers BIT FOR BIT (deterministic L-BFGS-B). `Omega` 2.757e-6, `Alpha` 7.784e-8,
    `Beta` 1.079e-3, `Gamma_Star` -3529.45, `H0` 7.027e-5 - persistence 0.9708, signed leverage
    share -0.9989, initial vol 13.31%, long-run 15.64% against a 3M ATM quote of 14.50%. Worst
    point 4.73% (the 3M 25 delta put), weighted residual 6.21e-5.

    THE MUTANT the bounds sit against: the fit with the LEVERAGE CHANNEL removed and nothing else
    moved (`Alpha` 0, `Beta` = psi, `Omega` = omega + alpha, holding persistence and stationary
    variance where the fit put them) reads worst point 13.13% and residual 2.83e-4 - 4.6x, on the
    same ten quotes.
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
        # the tick does not refit: the family was borrowed for this run and handed back, so no
        # later bootstrap re-enters a minutes-long least squares
        assert list(market['Bootstrapper Configuration']) == ['FXVolSurfaceParameters']
        assert [key for key in utils.HN_PARAM_NAMES if key not in written] == []
        assert all(np.isfinite(written[key]) for key in utils.HN_PARAM_NAMES)
        assert 0.0 < utils.hn_persistence(*(written[key] for key in (
            'Alpha', 'Beta', 'Gamma_Star'))) < 1.0, 'a non-stationary fit'
        # THE DEGENERATE FIT: leverage off (`Alpha` zero, flat smile) with `Gamma_Star` pinned at
        # its bound - what an inadmissible skew sign produces and the family still calls converged
        assert written['Alpha'] > 0.0, 'the leverage channel is off - the fitted smile is flat'
        # THE SIGN: this smile RISES with strike on the axis the model is fitted on, and only a
        # negative Gamma_Star says so
        assert written['Gamma_Star'] < 0.0, 'a rising smile fitted with equity-leverage skew'
        # AND THE OPTIMUM IS INTERIOR. `Gamma_Star`'s box is +-[1, 5000] in magnitude
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

        # MEASURED: worst 4.73%, weighted residual 6.21e-5, against a no-leverage mutant at 13.13%
        # and 2.83e-4. A one-factor GARCH is not asked to fit a smile exactly, only at all
        assert worst < 0.08, 'the model does not reprice its own quotes'
        assert weighted < 1.2e-4
    finally:
        service.BOOK = None


def test_a_pair_with_no_built_surface_refuses_at_the_verb(tmp_path):
    """A calibration against a surface the book does not carry is refused ON THE REQUEST THREAD, by
    name and with the remedy - never queued as a minutes-long job whose answer is a typo."""
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
    """A live book declaring the `FXVolSurfaceParameters` bootstrapper - what a market tick needs to
    turn quotes into price factors."""
    path = tmp_path / 'book.json'
    path.write_text(json.dumps(json.loads(dump(job(sections={
        'Bootstrapper Configuration': {'FXVolSurfaceParameters': {}}}))), indent=2), newline='\n')
    service.BOOK = service.Book(str(path))
    yield path
    service.BOOK = None


def canned_map():
    """The verified security map `discover.provision` hands back: one USDZAR block with its
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
    provision, freshness check and fetch are all seams the gates drive. blpapi is never reached."""

    def __init__(self, **options):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *error):
        return False


def bloomberg_seams(monkeypatch, provision=None, stale=None):
    """Every seam between the verb and the terminal, replaced. The lazy imports inside
    `BloombergJob.run_job` are what lets a patch reach the job: each name is bound off the package
    when the WORKER runs. Returns the definitions the fetch was handed."""
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
    """POST the verb, drain the worker, read the outcome off the result the way a poller does - the
    book write rides the run's own Stats, as a solve's coordinates do."""
    submitted = CLIENT.post('/book/bloomberg', json=request).json()
    service.EXECUTOR.queue.join()
    result = CLIENT.get('/results/{}'.format(submitted['result_id'])).json()
    return result, result.get('stats', {}).get('Bloomberg', {})


def test_the_bloomberg_verb_provisions_fetches_and_ticks_the_book(desk, monkeypatch):
    """The verb end to end on a machine with no terminal: the map is provisioned, its scope (every
    fx_vol pair at the expiries it verified, at the default pillar) is what the fetch is asked for,
    and what comes back is installed and bootstrapped in one atomic write - so the file carries the
    `FXVol` surface a pricer reads, not just the quotes."""
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
    """A retired series keeps answering with a plausible price, so the update date is the only thing
    that says so: one late quote refuses the WHOLE tick by name, before anything is fetched, and
    the book's bytes stand still."""
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
    """A terminal round trip is minutes behind one `result_id`, so the job publishes progress and
    `/results/{id}` merges it while the job waits or runs. Held on an Event, so the worker is
    provably mid-provision at the poll; the entry is gone once the result carries its outcome."""
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


def test_a_tick_a_bootstrapper_refuses_lands_refused_rather_than_raising(desk, monkeypatch):
    """A REFUSAL IS AN OUTCOME OF THE JOB, never an exception out of it.

    `Config.bootstrap` wraps only the CONSTRUCTION of a family, so a family that constructs, runs
    and then refuses by name raises out of the bootstrap and out of `market_edit`. `/book/market`
    catches that into a 422; the queued path did not.

    `Metronome.cause` reads `refused` off `stats.Bloomberg` and logs one warning; an `error` status
    is a different branch. So the SHAPE is pinned - a `done` job, `written` false, the engine's own
    wording carried through - and the book's bytes standing still.
    """
    document = json.loads(desk.read_text())
    market = document['Calc']['MergeMarketData']['ExplicitMarketData']
    market['Bootstrapper Configuration']['HestonNandiModelParameters'] = {}
    market.setdefault('Market Prices', {})['HestonNandiModelPrices.USD.ZAR'] = {
        'instrument': {'Quote_Type': 'Nonsense', 'Underlying': 'ZAR', 'Discount_Rate': 'USD',
                       'Volatility': 'USD.ZAR', 'European_Options': []}}
    desk.write_text(json.dumps(document, indent=2), newline='\n')
    before = desk.read_bytes()

    bloomberg_seams(monkeypatch)
    result, outcome = ticked()

    assert result['status'] == 'done', result
    assert 'error' not in result, result
    assert outcome['written'] is False
    assert any('Nonsense' in message and 'HestonNandiModelPrices.USD.ZAR' in message
               for message in outcome['refused']), outcome
    assert desk.read_bytes() == before, 'a refused tick moved the book'


def test_the_bloomberg_verb_needs_a_book_and_a_bootstrapper(book):
    """The market verbs' own refusals, in the same words: no book is a 404 naming the flag that
    opens one, no bootstrapper is a 422. Neither reaches the terminal or the queue."""
    bare = CLIENT.post('/book/bloomberg', json={})
    service.BOOK = None
    missing = CLIENT.post('/book/bloomberg', json={})

    assert bare.status_code == 422
    assert 'Bootstrapper Configuration' in bare.json()['detail']
    assert missing.status_code == 404 and '--book' in missing.json()['detail']


def test_the_metronome_skips_the_beat_its_last_tick_is_still_in_flight():
    """A terminal round trip can outlast an interval, and the result id's clock stamp means two
    ticks never coalesce - so only the metronome stops a slow terminal accumulating a queue.

    The decision is read off the executor's REAL store, so this needs no terminal and no patch: a
    job holding the worker is queued, then running, then done, and `pending_status` is that word
    each time. Non-vacuous by construction - no book is open, so a beat that did NOT skip would
    reach `live_book()` and raise the 404.
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
    """The metronome does not provision: verifying a workstation's vocabulary is minutes of terminal
    time and a person's decision.

    `DV_HOME` names a directory with no `security_map.json`. The beat submits through the real
    queue, the job refuses BEFORE it opens a session - which is what makes this reachable with no
    terminal - naming the home it looked in and the verb that fixes it. The book's bytes stand
    still, and the second beat is the failure discipline: exactly ONE warning carrying that cause.
    """
    import logging

    # DV_HOME is the declared surface for a desk's files, so pointing it at a directory holding no
    # map IS an unprovisioned workstation - nothing is patched
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
    """Solve a cashflow's Amount to a target. A secant is exact where the value is affine in the
    field, so the pricing count is small, the residual is inside tolerance, and the tables are the
    run AT the solved value rather than an extrapolation. The book file never moves."""
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
    """A digital's value is nonlinear and monotone in its strike, so brentq inside declared bounds
    finds the strike that marks at the target - and the pricing count says it genuinely iterated
    rather than taking the affine two-step."""
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
    """An unreachable target is an error result carrying the solver's words, never a number clamped
    to a bound."""
    submitted = CLIENT.post('/book/solve', content=dump({
        'deal': dict(CASHFLOW, Reference='SLV3'), 'field': 'Amount',
        'target': 1_000_000.0, 'bounds': [1.0, 2.0]}), headers=JSON).json()
    service.EXECUTOR.queue.join()
    result = CLIENT.get('/results/{}'.format(submitted['result_id'])).json()

    assert result['status'] == 'error' and result['error']


def test_validate_over_http_is_the_verb_verbatim():
    """Both halves of the want-list: a deal that breaks an authoring rule, and one naming a curve
    the market data has no block for."""
    document = job(deals=(dict(CASHFLOW, Discount_Rate='GBP'), BINARY),
                   factors=dict(FACTORS, **EQUITY))
    over_http = CLIENT.post('/validate', content=dump(document), headers=JSON).json()

    assert over_http == in_process(document).validate()
    assert over_http == {'deals': {'BIN1': ['Cash_Payoff is required']},
                         'factors': ['InterestRate.GBP']}


def test_a_browser_is_allowed_to_call_the_service_at_all():
    """Without the CORS header a browser discards the answer before the SPA sees it. Both halves:
    the preflight a POST of JSON provokes, and the header on the answer itself."""
    origin = {'Origin': 'http://localhost:4200'}
    preflight = CLIENT.options('/execute', headers=dict(
        origin, **{'Access-Control-Request-Method': 'POST'}))

    assert preflight.headers['access-control-allow-origin'] == '*'
    assert CLIENT.get('/schema/job', headers=origin).headers['access-control-allow-origin'] == '*'


def test_the_job_skeleton_is_a_job_that_loads():
    """The envelope is the one piece of contract `/schema` cannot state, so what is published has to
    BE a job: it goes back over `/validate` and `/execute` unedited. The price is asserted too - a
    skeleton that validates clean but does not price would pass on the want-list alone.
    """
    skeleton = CLIENT.get('/schema/job').json()
    result_id, result = run(skeleton)
    payment = skeleton['Calc']['Deals']['Deals']['Children'][0]['Instrument']['.Deal']

    assert CLIENT.post('/validate', json=skeleton).json() == {'deals': {}, 'factors': []}
    assert result['status'] == 'done'
    assert mtm(result_id)['CF1'] == pytest.approx(
        payment['Amount'] * np.exp(-0.02 * 2.0), rel=1e-3)


def test_describe_is_the_parse_and_it_never_runs_anything():
    """The book by type, both sides of the factor universe, the calculation block as loaded, and
    what the queue would make of it.

    NON-MUTATING is the claim that matters, because describing walks the deal tree and calls
    `reset` on every instrument. So it is described off a PLAN and that plan is then executed: a
    describe that wrote to what it read would move the plan and the id would stop agreeing.

    A node whose `Object` names no deal type is counted under nothing - `construct_instrument`
    logged it and returned `{}`, and `/validate` is where that node is reported.
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
    """The class orders the queue; the estimate is the size beside it and must move with all three
    fields. A base valuation carries none of them - the case the `or 1` exists for."""
    heavy = {'Object': 'CreditMonteCarlo', 'Batch_Size': 512, 'Simulation_Batches': 4,
             'Time_Grid': '0d 2d 1w(1w) 3m(1m) 2y(3m)'}

    assert service.cost(heavy) == dict(
        service.cost(heavy), **{'class': service.HEAVY, 'estimate': 512 * 4 * 5})
    assert service.cost(dict(heavy, Batch_Size=256))['estimate'] == 256 * 4 * 5
    assert service.cost(dict(heavy, Time_Grid='0d 1d(1d)'))['estimate'] == 512 * 4 * 2
    assert service.cost({'Object': 'BaseValuation'}) == dict(
        service.cost(heavy), **{'class': 0, 'estimate': 1})


def test_a_plan_id_execute_is_the_document_execute():
    """Content addressing does not care how the job arrived: `/prepare` names the parse by its plan
    hash, and executing that name unpatched lands on the id the whole document landed on."""
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
    and not the plan - asserted by executing the same plan unpatched afterwards, which a shared
    Context would answer with the patched id."""
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
    """Reading a plan is a USE, so the one read stays and the one merely older goes.

    The three jobs differ by a deal REFERENCE and not by the seed, which is a replay coordinate
    deliberately outside the plan - three seeds would have been one plan and measured nothing.
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
    """A name the service does not hold is a 404, never an empty answer a client would render as a
    blank grid."""
    result_id, _ = run(job(Random_Seed=19))

    assert CLIENT.post('/execute', json={'plan_id': 'nosuchplan'}).status_code == 404
    assert CLIENT.post('/describe', json={'plan_id': 'nosuchplan'}).status_code == 404
    assert CLIENT.get('/results/nosuchresult').status_code == 404
    assert CLIENT.get('/results/{}/nosuchtable'.format(result_id)).status_code == 404
    assert CLIENT.get('/results/{}/mtm'.format(result_id)).status_code == 200


def test_a_result_publishes_the_shape_of_every_table_and_pages_each_one():
    """A summary carries shapes, never cells; one table comes back a page at a time. Held through
    the executor because no calculation in the suite produces the interesting tree - a group of
    tables, a vector and a scalar beside a frame.

    A group is not a table and has no page, so `cashflows` arrives flattened to the path naming
    each one. The paging assertions stop `limit` being read as an end index.
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
    """A patch is applied before the hashes are taken, so it reaches the price AND the identity. A
    spot is market VALUES, so the values hash moves and the plan hash does not; an id taken before
    the patch would collide with the unpatched run.
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
    """The same job names the same result, and the second submission already holds the finished
    one."""
    document = job(Random_Seed=11)
    first, _ = run(document)
    second = submit(document)

    assert second['result_id'] == first
    assert second['status'] == 'done'


def test_a_submission_arriving_mid_run_coalesces_onto_the_first():
    """Dedupe holds while the first job is still QUEUED or RUNNING, not only once it is filed.
    Counted at the executor, because a second run would overwrite the store with the same content
    and leave no trace."""
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
    """Cost class orders what is waiting, arrival orders within a class, the running job is left
    alone. All three are queued before the blocker is released, so the queue and not arrival order
    decides what runs next."""
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
    """Reporting in a currency with no FX rate fails inside the run, so this is the engine failing
    rather than the wrapper refusing. The message travels, nothing else does, and the next job
    still prices."""
    _, failed = run(job(Currency='GBP'))
    after_id, after = run(job(Random_Seed=7))

    assert failed['status'] == 'error'
    assert 'FxRate.GBP' in failed['error']
    assert 'tables' not in failed
    assert after['status'] == 'done'
    assert mtm(after_id)['CF1'] == pytest.approx(AMOUNT * SPOT * np.exp(-RATE * 2.0), rel=1e-9)


#: USDZAR as THIS book quotes it. `FxRate.ZAR.Spot` is one ZAR in base-currency units (18.5 here),
#: so the market pair - ZAR per USD - is its reciprocal. Striking the collar at 18.5 instead would
#: ask for a floor deep in the money and read the runner's bracket refusal as a service bug.
USDZAR = 1.0 / SPOT

#: The zero-cost collar as a sales desk asks for it: parameters in MARKET terms, the floor given
#: and the cap left for the recipe to solve. The keys are the structure's own declared `fields`.
#: The floor sits 5% out of the money, so the bought put is cheap enough for a solved cap to fund
#: it well inside the runner's bracket.
COLLAR = {'pair': 'USDZAR', 'expiry': '1Y', 'notional': AMOUNT,
          'notional_currency': 'USD', 'floor': USDZAR * 0.95}


@pytest.fixture
def quoting(tmp_path, monkeypatch):
    """A desk that can be quoted at: the one-cashflow book with the FX vol bootstrapper declared, a
    real USDZAR surface ticked in, and `DV_HOME` at the gate's own tmp. The service reads `DV_HOME`
    per call, so the worker thread that files the quote sees this directory.
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
    """Ask for a quote and drain the worker - the outcome under `stats.Quote`, where a solve's
    coordinates sit under `stats.Solved`. `extra` is the rest of the ask."""
    submitted = CLIENT.post('/book/structure', content=dump(
        dict({'structure': structure, 'params': params}, **extra)), headers=JSON).json()
    service.EXECUTOR.queue.join()
    result = CLIENT.get('/results/{}'.format(submitted['result_id'])).json()
    assert result['status'] == 'done', result.get('error')
    return result['stats']['Quote']


def test_a_quoted_collar_is_filed_pending_and_books_at_zero(quoting, tmp_path):
    """The sales loop through one service: a structure comes back solved, the pending trade is on
    disk under its quote id, the approval books it through the hand-booking seam, and the book then
    MARKS it at zero.

    The last step cannot be faked: `net` is what the runner computed while solving, and the book's
    value is the engine on the composed deal actually written, priced from the file. Zero is
    asserted against a leg premium - a net of zero means nothing if both legs are worth nothing.
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
    # the quote is client paper and the book holds the bank's position, so the approval books the
    # MIRROR: every booked leg carries the opposite side from the one quoted
    assert [child['Instrument']['.Deal']['Buy_Sell'] for child in node['Children']] == [
        {'Buy': 'Sell', 'Sell': 'Buy'}[leg['buy_sell']] for leg in quote['legs']]
    # the file is the audit trail of what was quoted at what market - booking does not consume it
    assert pending.is_file()

    marked_id, marked = run(on_disk)
    assert marked['status'] == 'done', marked.get('error')
    assert mtm(marked_id)[node['Instrument']['.Deal']['Reference']] == pytest.approx(
        0.0, abs=premium * 1e-4)


#: A calibrated Heston-Nandi factor for the rand, as `/book/hn` writes one. Not a fit: a stationary
#: set (persistence 0.90) near the surface's own vol, with `Gamma_Star` on the sign this pair's
#: smile carries. The gate is about the MODEL reaching the book with the trade.
CALIBRATED = {'Property_Aliases': None, 'Omega': 1e-12, 'Alpha': 2.0e-6, 'Beta': 0.45,
              'Gamma_Star': -474.34, 'H0': 7.8e-5}

#: An accumulator on the RAND: the orientation whose underlying IS the token a spot model is keyed
#: on, so it rides the fit as written and crosses no axis. The keying's own gates are in
#: `test_structures.py` and `test_fx_accumulator_json.py`.
ACCUMULATOR = {'pair': 'USDZAR', 'expiry': '3M', 'notional': AMOUNT, 'notional_currency': 'ZAR',
               'fixing_frequency': '1M', 'knockout': USDZAR * 1.10}


def test_a_leg_quoted_under_a_model_books_into_a_book_that_marks_it(quoting):
    """The model books WITH the trade, in the same atomic write, or the desk marks a trade at a
    price it was never dealt at.

    `structures.spot_model` pins Heston-Nandi on the QUOTE's copy of the document, and that copy is
    thrown away when the answer is published - so an approval booking only the deal would leave a
    leg priced under a GARCH in a book whose `Valuation Configuration` says nothing, and the next
    mark would price it as a lognormal. So the quote REPORTS what it pinned, the pending file
    records it, and `/book/quote` merges it inside the same edit closure that splices the deal.
    What this gate reads is the FILE afterwards.

    And the approval REFUSES where the calibration went away between quote and approval: booking a
    switch over a factor nobody carries would skip the deal in the dependency loop and mark the
    trade at nothing.
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
    """A quote prices on the LIVE spot when the terminal is up; the fallback when it is not IS the
    old path, said float for float.

    This `DV_HOME` holds no security map - a fresh desk, and the one live-spot refusal reachable
    with no terminal, since a quote never provisions. So the quote runs on the book's own ticked
    spot, names the home it looked in, and comes out identical to the same structure quoted through
    `structures.quote`. Close would not do: a fallback that moved a price would be the live-spot
    feature firing where there is no live spot.

    The book file is untouched either way - a spot is `bind='value'` data patched onto the JOB's
    copy, and a quote is not a trade.
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
    """A quote id nobody gave is a 404 naming the directory it was looked for in, so a desk whose
    `DV_HOME` is not the one it was filed under can read that off the refusal. Nothing is
    written."""
    before = quoting.read_bytes()
    answer = CLIENT.post('/book/quote', json={'quote_id': 'nosuchquoteid'})

    assert answer.status_code == 404
    assert 'nosuchquoteid' in answer.json()['detail']
    assert str(quoting.parent / 'tmp') in answer.json()['detail']
    assert quoting.read_bytes() == before


def test_a_pending_trade_the_book_would_refuse_is_refused_in_the_booking_wording(quoting,
                                                                                 tmp_path):
    """An approval is a BOOKING, so it is refused on what it would land in and in the same words as
    `test_a_bad_amendment_touches_nothing` - one validate-before-write seam, or the two paths have
    drifted. The pending file is authored here because that is the only way to reach the refusal
    branch without asking the runner for a quote it would rightly decline.
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
    """The sheet is a real xlsx workbook next to the quote file, under the same id. Skipped rather
    than faked when the `quote` extra is absent. `derivus.quote_sheet` owns what is inside the
    three sheets; this owns the wiring - the job reaches the writer, names the file, wrote a
    workbook.
    """
    pytest.importorskip('derivus.quote_sheet',
                        reason='the quote extra is not installed - there is no sheet to find')
    quote = quote_of('ZeroCostCollar', COLLAR)
    sheet = tmp_path / 'tmp' / (quote['quote_id'] + '.xlsx')

    assert quote['files']['sheet'] == str(sheet)
    assert 'sheet_note' not in quote['files']
    assert zipfile.is_zipfile(str(sheet))


def test_a_composed_candidate_prices_its_legs_not_an_empty_container(book):
    """A composed StructuredDeal arriving with node-shaped children prices as the sum of those legs.
    MUTATION: `splice_deal` dropping the composed children back to an empty list prices the
    container at 0.0 with nothing said against it. Non-vacuity is each leg carrying a real value."""
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
    """`/book/risk` on the quoting book. COHERENCE: `mtm` is the sum of `per_deal`, and each
    per-deal value is what the ORDINARY `/book/price` run reports, so the view is not a second
    opinion. WARMTH: the same book answers the identical object off the cache, etag and `as_of`
    included. FOLLOWING: a booking moves the etag with the answer behind it.

    The book carries a real USDZAR surface, so the gradient is not vacuous: the FX option puts
    `FXVol.USD.ZAR` rows in `greeks` on TWO tenor coordinates - the surface-node case the
    flattening exists for - beside one-coordinate curve rows and a no-coordinate spot.
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


def netting_set(reference, counterparty, deals, funding_rate='USD'):
    """One uncollateralised `NettingCollateralSet` over its deals.
    `Credit_Support_Amounts.Counterparty` IS the `SurvivalProb` factor the CVA discounts by, which
    is why the recalc reads it from there and nowhere else. `Funding_Rate` is what the FVA's
    funding leg is priced off; the default `'USD'` is also the deflation curve, so it declares no
    spread and therefore no adjustment."""
    return {'Instrument': {'.Deal': {
        'Object': 'NettingCollateralSet', 'Reference': reference, 'Netted': 'True',
        'Collateralized': 'False', 'Agreement_Currency': 'USD', 'Balance_Currency': 'USD',
        'Funding_Rate': funding_rate, 'Liquidation_Period': 0.0, 'Settlement_Period': 0.0,
        'Credit_Support_Amounts': {
            'Counterparty': counterparty, 'Received_Threshold': CSA, 'Posted_Threshold': CSA,
            'Independent_Amount': CSA, 'Minimum_Received': CSA, 'Minimum_Posted': CSA}}},
        'Children': [{'Instrument': {'.Deal': deal}} for deal in deals]}


#: A funding curve ABOVE the book's USD curve, so a set funding at it carries a real spread over
#: risk-free. Without one, FCA and FBA are zero by construction and the FVA column cannot be told
#: from an unimplemented one.
FUNDING_RATE = 0.05


def xva_book(tmp_path, sets, counterparties=('CPTY_A', 'CPTY_B')):
    """A live book of netting sets: a survival curve per counterparty, a GBM for the equity and a
    funding curve above risk-free. The hazard rises with each counterparty, so the two sets'
    numbers are separable rather than coincidentally equal."""
    factors = dict(FACTORS, **EQUITY)
    factors['InterestRate.FUND'] = {
        'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
        'Curve': utils.Curve([], [[0.0, FUNDING_RATE], [5.0, FUNDING_RATE]])}
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
    """Two counterparties, one vanilla each - the smallest book with a mosaic to keep, with
    `DV_HOME` at the gate's tmp so the projection is readable.

    The sets fund DIFFERENTLY on purpose: NS_A at `FUND`, a spread over risk-free, and NS_B at the
    risk-free curve itself - a real number beside an exact zero in one projection, each traceable
    to the set's own `Funding_Rate`."""
    monkeypatch.setenv('DV_HOME', str(tmp_path))
    yield xva_book(tmp_path, [
        netting_set('NS_A', 'CPTY_A', [dict(VANILLA, Reference='OPT_A')], funding_rate='FUND'),
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
    """The XVA lifecycle: never run, recalced, filed, then partially recalced.

    A CMC is minutes of device time, so this view is a CACHED PROJECTION and every claim is about
    the FILE. A full recalc queues one job per set, each writing a real `cva` and the replay tuple
    that names the run it came from, and `xva.json` on disk carries the same rows.

    Then the mosaic: a deal booked into ONE set with only THAT set recalced moves its row - new
    plan, new id, later stamp, bigger number - and leaves the other byte for byte, `as_of`
    included. Staleness is data, and a partial recalc is a partial WRITE.
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

    # the projection IS the file, not anything held in memory
    filed = json.loads((tmp_path / 'xva.json').read_text())['sets']
    assert {reference: filed[reference]['cva'] for reference in filed} == {
        reference: before[reference]['cva'] for reference in filed}
    assert filed['NS_A']['as_of'] == before['NS_A']['as_of']

    # an identical recalc over an unmoved book computes nothing and must AGE nothing: `as_of` is
    # when the number was computed
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
    """Two refusals, neither losing the projection.

    An unknown reference is a 422 NAMING what was asked for and what the book holds, and it queues
    nothing at all - not even the set spelled correctly, since a desk asking for two and getting
    one would have to diff the answer to find out.

    A counterparty with no `SurvivalProb` block is the other kind: the ENGINE has the objection, so
    the row lands `failed` carrying its wording. One set failing is one row, not a lost projection.
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
        assert rows['NS_GHOST']['fva'] is None, 'a failed run filed a funding number anyway'
        assert 'GHOST' in rows['NS_GHOST']['error']
        assert rows['NS_A'] == standing, "a failed set took another set's row with it"
    finally:
        service.BOOK = None


def test_fva_is_a_column_of_the_same_row_off_the_same_run(desk_xva, tmp_path):
    """CVA and FVA are two columns of ONE row from ONE credit Monte Carlo, sharing the row's
    identity: one `as_of`, one `result_id`, one `plan_hash`. That is what makes them addable - two
    reductions of the same exposure cube at the same market.

    The NUMBER is the set's own declaration. NS_A funds at `FUND` and carries a real adjustment;
    NS_B funds at risk-free and carries exactly 0.0, since FCA and FBA are each a spread OVER
    risk-free. That exact zero is a computed number, not a column standing in for one.

    MEASURED, 1024 paths, seed 1: NS_A cva 120.85 / fva 302.82, NS_B cva 297.74 / fva 0.0.

    Then the mosaic on both columns: a deal booked into NS_A with only NS_A recalced moves cva AND
    fva together under one new stamp, and leaves NS_B byte for byte.
    """
    queued = recalc()['queued']
    before = xva_rows()
    for entry in queued:
        row = before[entry['reference']]
        assert row['status'] == 'done', row['error']
        assert row['result_id'] == entry['result_id'], 'the row names a run it did not come from'
        assert row['cva'] > 0.0 and row['fva'] is not None

    assert before['NS_A']['fva'] > 0.0, 'a set funding above risk-free paid nothing to fund'
    assert before['NS_B']['fva'] == 0.0, 'a set funding AT risk-free was charged a spread anyway'

    # the two columns are one run: the same stamp, and the same stamp on disk
    filed = json.loads((tmp_path / 'xva.json').read_text())['sets']
    for reference, row in before.items():
        assert filed[reference]['fva'] == row['fva'] and filed[reference]['cva'] == row['cva']
        assert filed[reference]['as_of'] == row['as_of'] == before[reference]['as_of']

    booked = CLIENT.post('/book/deals', content=dump(
        {'action': 'add', 'deal': dict(VANILLA, Reference='OPT_A2', Units=5000.0),
         'parent_reference': 'NS_A'}), headers=JSON).json()
    assert booked['written'] is True, booked

    recalc(['NS_A'])
    after = xva_rows()
    assert after['NS_A']['cva'] > before['NS_A']['cva'], 'a bigger position must cost more credit'
    assert after['NS_A']['fva'] > before['NS_A']['fva'], 'a bigger position must cost more to fund'
    assert after['NS_A']['as_of'] > before['NS_A']['as_of']
    assert after['NS_B'] == before['NS_B'], 'a partial recalc touched a row it was not asked for'


def test_the_cva_column_reads_the_same_whether_or_not_fva_ran(desk_xva):
    """One run serving both adjustments must not make the credit one a different number.

    `Funding_Valuation_Adjustment.Calculate` sets `CMC_State.scale_survival`, so every set reports
    its MTM already multiplied by the counterparty's survival probability. That factor is positive
    and deterministic per bucket, so `relu` commutes with it and the CVA integrand divides it back
    out. Both of the projection's sets, 1024 paths, seed 1: 0 ULP. Un-divided the same two runs read
    119.676 against 120.845 (-0.967%, 2% hazard) and 290.646 against 297.741 (-2.383%, 5% hazard),
    tracking the hazard as a survival scaling must.

    `fva` is the other half and is NOT compared, because there is nothing to compare it against: a
    CVA-only run reports no `fva` key at all. It keeps the scaled cube it is defined on.
    """
    with open(service.BOOK.path) as handle:
        document = json.load(handle)
    for node in document['Calc']['Deals']['Deals']['Children']:
        deal = node['Instrument']['.Deal']
        both = service.xva_document(document, node, deal['Credit_Support_Amounts']['Counterparty'])
        both['Calc']['Calculation'].update(Batch_Size=1024, Simulation_Batches=1, Random_Seed=1)
        off = json.loads(json.dumps(both))
        off['Calc']['Calculation']['Funding_Valuation_Adjustment']['Calculate'] = 'No'

        on_results = in_process(both).run_job()[1]['Results']
        off_results = in_process(off).run_job()[1]['Results']
        assert float(on_results['cva']) == float(off_results['cva']), (
            deal['Reference'], float(on_results['cva']), float(off_results['cva']))
        assert float(on_results['cva']) > 0.0, (
            'the set priced no credit exposure at all', deal['Reference'])
        assert 'fva' in on_results and 'fva' not in off_results, (
            'the FVA-off run reports an `fva` the gate would have to hold', deal['Reference'])


def test_a_row_filed_before_the_fva_column_existed_still_reads(desk_xva, tmp_path):
    """STALENESS IS DATA IN TIME AS WELL AS ACROSS SETS. A row written when CVA was the only
    adjustment has no `fva` key, and the view reads it as what it is - a done run with a real
    credit number and no funding number - rather than refusing the file or calling the row stale.

    The old row is AUTHORED here because there is no other way to reach one on a build that writes
    the column. It carries a `plan_hash` and `result_id` from a plan with no funding block, which
    is what makes the recalc rewrite it rather than recognise it as already filed.

    The remedy is the one an old `as_of` takes: recalculate that set. The column fills in, and the
    OTHER set's old-shape row is untouched.
    """
    recalc()
    projection = json.loads((tmp_path / 'xva.json').read_text())
    standing = {reference: dict(row) for reference, row in projection['sets'].items()}
    for reference, row in projection['sets'].items():
        row.pop('fva')
        row['plan_hash'] = row['result_id'] = 'a-plan-with-no-funding-block'
    (tmp_path / 'xva.json').write_text(json.dumps(projection, indent=2), newline='\n')

    old = xva_rows()
    for reference, row in old.items():
        assert row['status'] == 'done', 'an old-shape row was read as a failure'
        assert row['cva'] == pytest.approx(standing[reference]['cva'], rel=1e-12)
        assert row['fva'] is None, 'a column that was never run reported a number'

    fresh = recalc(['NS_A'])['queued']
    rows = xva_rows()
    assert rows['NS_A']['result_id'] == fresh[0]['result_id'], 'the old row was left standing'
    assert rows['NS_A']['fva'] == pytest.approx(standing['NS_A']['fva'], rel=1e-12)
    assert rows['NS_A']['cva'] == pytest.approx(standing['NS_A']['cva'], rel=1e-12)
    assert rows['NS_B'] == old['NS_B'], 'a partial recalc upgraded a row it was not asked for'


def test_a_missing_funding_table_is_age_on_a_stored_row_and_a_defect_on_a_live_run():
    """ONE absent column, two readings - and the gate is the DIFFERENCE between them, over the same
    two mappings.

    A run the store already holds is written UP rather than paid for again, and one filed before
    this column existed carries no `fva`, so the STORED reading is lenient and lands a null.

    A live run is not stale by construction - the job composes both adjustments over ONE exposure
    cube - so results with no `fva` are a defect of that run. Filing a null would put 'no funding
    cost' on the blotter under a fresh stamp, so the fresh path raises and the row lands `failed`.
    """
    whole, aged = {'cva': 119.68, 'fva': 302.82}, {'cva': 119.68}

    assert service.XvaJob.adjustments(whole) == {'cva': 119.68, 'fva': 302.82}
    assert service.XvaJob.adjustments(whole, stored=True) == service.XvaJob.adjustments(whole)
    assert service.XvaJob.adjustments(aged, stored=True) == {'cva': 119.68, 'fva': None}
    with pytest.raises(KeyError, match='fva'):
        service.XvaJob.adjustments(aged)

    # the row each reading lands: the stored one is a DONE row carrying a null
    filed = service.XvaJob(None, 'NS_A', ('CP_A', True), 'a-result-id',
                           {'plan_hash': 'p', 'values_hash': 'v', 'seed': 1}).landed(
        {'status': 'done', 'tables': aged})
    assert filed['status'] == 'done' and filed['cva'] == 119.68 and filed['fva'] is None


# --------------------------------------------------------------------------------------------
# a quote is FOR someone, and it is firm for a WINDOW
# --------------------------------------------------------------------------------------------
#: The client the quoting book has opened, as a Reference. A desk's client is a
#: `NettingCollateralSet` - the node the counterparty and the CSA are declared on.
CLIENT_SET = 'CLIENT_A'


@pytest.fixture
def quoting_client(tmp_path, monkeypatch):
    """The quoting desk with a CLIENT on its book: the one-cashflow book with a real USDZAR surface
    plus an EMPTY `NettingCollateralSet` naming a counterparty with a survival curve. Empty because
    what lands under it is the point.

    Built here rather than off `quoting` because a netting set is a NODE and the `job` helper wraps
    plain deal blocks; kept separate because the existing gates read that book's paths by position.
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
    """`{Reference: deal_path}` for the book's netting sets, through the service's own walk rather
    than by counting positions here."""
    return {node['Instrument']['.Deal']['Reference']: deal_path
            for deal_path, node in service.netting_sets(json.loads(path.read_text()))}


def test_a_quote_for_a_client_books_under_the_clients_netting_set(quoting_client, tmp_path):
    """A quote given FOR a client books under that client's netting set; a quote for nobody books at
    the root as it always did.

    The decisive assertion is the deal PATH: a trade at the root is outside every netting set's
    subtree, so the CVA projection - which prices one subtree per run - cannot see it. The path is
    read through `service.netting_sets`, the walk the XVA verb takes. The mirror still mirrors:
    nesting must not touch the side the approval lands on.

    The root half is in the same gate on purpose - `netting_set` absent has to be TODAY's booking.
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
    """A netting set nobody opened refuses while the quote is being ASKED for, naming what was asked
    for and what the book holds, in the XVA verb's own wording.

    At the ask rather than the approval: finding out at the approval is finding out after the
    client has the sheet. Nothing is priced, nothing is filed, the book is untouched, and the
    refusal is a 422 on the salesperson's own call rather than an `error` status to poll for.
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
    """The desk's mandate written onto the live book, or taken off where nothing is stated, plus
    the book's bytes as they now stand.

    The live book is REOPENED because `Book` re-parses on mtime and Windows' file timestamps do not
    resolve two writes inside the same 15ms tick - a gate changing a mandate twice in a row would
    otherwise read the first one back and pass for the wrong reason.
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
    """One quote, one book, three mandates - so this is about the WINDOW and nothing else. The quote
    is given once; only the book's `Quote Policy` moves.

    With `firm_seconds: 0` an approval arriving immediately is outside the window and is refused
    422 naming the age, the window and the remedy, writing NOTHING. With the block taken off, THE
    SAME pending quote books - absence of the policy is the off switch.

    The third mandate is the compatibility case: a pending file with no `quoted_at` cannot be shown
    to be inside any window, and an unknown age is not an age inside it. So a real window treats it
    as AGED and says which case it is, rather than booking a quote of unknown vintage.
    """
    quote = quote_of('ZeroCostCollar', COLLAR)
    pending = tmp_path / 'tmp' / (quote['quote_id'] + '.json')
    assert json.loads(pending.read_text())['quoted_at'], 'the quote was filed unstamped'

    # authored here because there is no other way to reach a pending file with no `quoted_at`
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
