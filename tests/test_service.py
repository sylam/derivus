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

`/book/model` is gated on the emitter, the round trip and the refusal; `/book/structure` +
`/book/quote` on the two halves being one trade - the collar nets to zero and the BOOK marks the
deal it wrote at zero. `DV_HOME` is the declared surface for where those files land.

Ordering and dedupe are deterministic without a clock: a first job blocks inside `run_job` and
announces it through an `Event`, so the others are provably queued before it is released.
`Queue.join` is the barrier everywhere else.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
import json
import logging
import re
import threading
import time
import zipfile

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import derivus
from derivus import service, structures, utils
from derivus.config import CustomJsonEncoder
from derivus.schema import deal_at

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


#: One client call in `web/src/api.ts` - the method, and the path up to its query string, with the
#: template holes left in. `call<T>` itself is not a call site, so the lookbehind drops it.
API_CALL = re.compile(r"(?<!function )call<[^(]*\(\s*'(GET|POST|PUT|DELETE)',\s*['`]([^'`?]*)")

#: A path PARAMETER on either side - `${id}` as the client writes it, `{table:path}` as the route
#: declares it - so the two spellings of the same hole compare equal.
PATH_HOLE = re.compile(r'\$\{[^}]*\}|\{[^}]*\}')


def test_every_path_the_web_client_names_is_a_route_this_service_declares():
    """The client renders from `/schema`, so the ONE thing it cannot read off a declaration is the
    endpoint list - which makes a renamed route the one drift nothing catches until a desk clicks.
    Here it is a gate: every `call(method, path)` in `api.ts` against `app.routes`, path parameters
    normalised so the client's `/results/${id}` is the declared `/results/{result_id}`.

    The web tree is optional to the library - a wheel carries the BUILD, not the source - so this
    skips where it is absent rather than failing a package that never had it.
    """
    source = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'web', 'src', 'api.ts')
    if not os.path.exists(source):
        pytest.skip('web/src is not in this tree: the UI ships as a built directory')
    with open(source, encoding='utf-8') as handle:
        text = handle.read()

    called = {(method, PATH_HOLE.sub('{}', path)) for method, path in API_CALL.findall(text)}
    # a regex that stopped matching would pass an empty set against anything
    assert len(called) == len(re.findall(r'(?<!function )call<', text))
    declared = {(method, PATH_HOLE.sub('{}', route.path)) for route in service.app.routes
                for method in getattr(route, 'methods', None) or ()}
    assert sorted(called - declared) == []


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


#: A ZAR swap on the book's own curve, stating its TERMS and no convention at all: what it is, in
#: which currency, between which dates, which way round, at what rate and for how much.
SWAP = {'Object': 'SwapInterestDeal', 'Reference': 'SW1', 'Currency': 'ZAR',
        'Discount_Rate': 'ZAR', 'Interest_Rate': 'ZAR', 'Effective_Date': BASE,
        'Maturity_Date': BASE + pd.DateOffset(years=2), 'Pay_Rate_Type': 'Fixed',
        'Swap_Rate': 8.0, 'Principal': 1_000_000.0}


@pytest.mark.parametrize('key', ['Swap_Rate', 'Pay_Rate_Type', 'Principal'])
def test_a_deal_missing_a_term_is_refused_by_name_and_a_convention_is_not(book, key):
    """A DECLARED DEFAULT IS A CONVENTION OR A PLACEHOLDER, and the booking is where the difference
    is paid. A swap states seven things and inherits forty-one; drop one of the seven and the
    booking refuses BY NAME, the file untouched - and a `null` under that key says the same
    nothing, which is what a form and a host both round-trip for one. Drop every convention and it
    books and prices, because the declaration says what each one means.

    KILLING MUTATION: the absence check dropped from `schema.validate_instrument`. The swap books
    with no fixed rate, compiles, prices at 0% and the job reports success - which is the failure
    the whole flag exists to end. The swap that states its rate marks -2,168,937.57 here and the
    same swap at a zero rate marks +725,395.38, so a completed placeholder does not even keep its
    sign.

    SECOND KILLING MUTATION: the check reading FALSITY rather than what the document says
    (`not deal.field.get(key)`). Every refusal above still fires, and the benchmark below stops
    booking - `Swap_Rate: 0.0` is the rate every curve strip's swap carries, so a verdict that
    refuses it refuses the bootstrap's own instruments.
    """
    before = book.read_bytes()
    for absent in ({k: v for k, v in SWAP.items() if k != key}, dict(SWAP, **{key: None})):
        outcome = CLIENT.post('/book/deals', content=dump({'action': 'add', 'deal': absent}),
                              headers=JSON).json()
        assert outcome['written'] is False, outcome
        assert '{} is not stated'.format(key) in outcome['refused'], outcome['refused']
        assert book.read_bytes() == before

    # a STATED zero is a statement: it is what a benchmark's fixed leg carries, and it books
    zero = CLIENT.post('/book/deals', content=dump(
        {'action': 'add', 'deal': dict(SWAP, Reference='SW0', Swap_Rate=0.0)}), headers=JSON).json()
    assert zero['written'] is True, zero

    # and an AMENDMENT cannot put the nothing back that the booking refused
    amended = CLIENT.post('/book/deals', json={
        'action': 'amend', 'deal_path': zero['deal_path'], 'reference': 'SW0',
        'fields': {key: None}}).json()
    assert amended['written'] is False
    assert '{} is not stated'.format(key) in amended['refused'], amended['refused']

    CLIENT.post('/book/deals', json={'action': 'delete', 'deal_path': zero['deal_path'],
                                     'reference': 'SW0'})
    assert book.read_bytes() == before

    # and the same swap stating its terms alone - every convention left unsaid - books and prices
    booked = CLIENT.post('/book/deals', content=dump({'action': 'add', 'deal': SWAP}),
                         headers=JSON).json()
    assert booked['written'] is True
    assert json.loads(book.read_text())['Calc']['Deals']['Deals']['Children'][1][
        'Instrument']['.Deal'].keys() == SWAP.keys()

    submitted = CLIENT.post('/book/price', content=dump({}), headers=JSON).json()
    service.EXECUTOR.queue.join()
    priced = CLIENT.get('/results/{}'.format(submitted['result_id'])).json()
    assert priced['status'] == 'done', priced
    assert mtm(submitted['result_id'])['SW1'] == pytest.approx(-2_168_937.571277)


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


def test_a_malformed_booking_is_refused_by_name_and_writes_nothing(book):
    """Nothing malformed reaches the file. A misspelt `Object`, a missing one, an amount authored
    as text and a date authored as a bare string are each an ANSWER naming the field and the form
    it takes, and the file stands still through all four.

    KILLING MUTATION: every one of these was WRITTEN with `book_issues: 1` - the first two loading
    as a node carrying no deal at all, the string amount making every later validate of the book
    die `can only concatenate str (not "float") to str`, and the bare date answering a 500 out of
    `discover_factors` comparing a Timestamp with a str.
    """
    before = book.read_bytes()
    refused = {}
    for label, deal in [('typo', dict(BOOKED, Object='FixedCashflwDeal')),
                        ('nameless', {'Reference': 'CF3', 'Amount': 1.0}),
                        ('text amount', dict(BOOKED, Amount='1e6')),
                        ('bare date', dict(BOOKED, Payment_Date='2027-01-15'))]:
        answer = CLIENT.post('/book/deals', content=dump({'action': 'add', 'deal': deal}),
                             headers=JSON)
        assert answer.status_code == 200, (label, answer.text)
        refused[label] = answer.json()

    assert not any(outcome['written'] for outcome in refused.values())
    assert "Object is 'FixedCashflwDeal'" in refused['typo']['refused'][0]
    assert 'names no deal type' in refused['nameless']['refused'][0]
    assert refused['text amount']['refused'] == ["Amount must be a number, not '1e6'"]
    assert refused['bare date']['refused'] == [
        'Payment_Date must be {".Timestamp": "2027-01-15"}, not \'2027-01-15\'']
    # A DEAL IS AN OBJECT: no deal at all, a string and a list are each refused by name rather
    # than reaching the splice - a 500 out of `dict.update`, or a 200 that wrote nothing
    for body in ({'deal': None}, {'deal': 'EURZAR'}, {'deal': []}):
        answer = CLIENT.post('/book/deals', json=body)
        assert answer.status_code == 422 and answer.json()['detail'] == (
            'a booking is a deal - post it under `deal`'), body
    assert book.read_bytes() == before


def test_a_container_booked_with_one_unnamed_child_refuses_whole(book):
    """A legacy import lands as ONE node - a set carrying a converted export - so a misspelt type
    among its children refuses the whole batch, naming that child by its walk position.

    KILLING MUTATION: a 500, `AttributeError: 'dict' object has no attribute 'field'`, out of
    `compress_deal_data`, which walks a container's children as the document loads.
    """
    before = book.read_bytes()
    batch = {'Object': 'StructuredDeal', 'Reference': 'BATCH', 'Currency': 'ZAR', 'Children': [
        {'Instrument': {'.Deal': BOOKED}},
        {'Instrument': {'.Deal': dict(BOOKED, Reference='CF3', Object='FixedCashflwDeal')}}]}
    answer = CLIENT.post('/book/deals', content=dump({'action': 'add', 'deal': batch}),
                         headers=JSON)

    assert answer.status_code == 200, answer.text
    assert answer.json()['written'] is False
    assert answer.json()['refused'] == [
        "deal at position 3 names no deal type - Object is 'FixedCashflwDeal'; name a declared "
        'type or delete the deal']
    assert book.read_bytes() == before


def test_a_book_carrying_an_unnamed_deal_names_it_rather_than_dying(tmp_path):
    """A file a booking can no longer write - a legacy import, a hand edit - is NAMED at the first
    verb that compiles it, with the walk position and the `Object` as authored.

    KILLING MUTATION: `AttributeError: 'dict' object has no attribute 'base_currency' and no
    __dict__ for setting new attributes` in `Calculation.set_deal_structures` - the whole book
    unpriceable, under every verb, with nothing naming the deal that did it.
    """
    document = json.loads(dump(job(deals=(CASHFLOW, dict(
        CASHFLOW, Reference='CF3', Object='FixedCashflwDeal')))))
    path = tmp_path / 'book.json'
    path.write_text(json.dumps(document, indent=2), newline='\n')
    service.BOOK = service.Book(str(path))
    try:
        submitted = CLIENT.post('/book/price', content=dump({}), headers=JSON).json()
        service.EXECUTOR.queue.join()
        priced = CLIENT.get('/results/{}'.format(submitted['result_id'])).json()
        risk = CLIENT.get('/book/risk')
        named = ("deal at position 1 names no deal type - Object is 'FixedCashflwDeal'; name a "
                 'declared type or delete the deal')

        assert priced['status'] == 'error' and priced['error'] == named
        assert risk.status_code == 422 and named in risk.json()['detail']
        assert CLIENT.post('/validate', content=dump(document), headers=JSON).json() == {
            'deals': {'#1': [named]}, 'factors': []}
    finally:
        service.BOOK = None


def test_a_value_outside_its_own_declaration_is_an_authoring_message():
    """A field declaring `values` accepts nothing else, so a misspelt menu value is a message
    rather than a deal priced as something.

    KILLING MUTATION: `validate_instrument` read the REQUIRED fields and nothing else, so
    `Option_Type: 'Putt'` was written and priced, the pricer's own `== 'Call'` making it a put.
    """
    document = job(deals=[dict(BINARY, Cash_Payoff=100.0, Option_Type='Putt')],
                   factors=dict(FACTORS, **EQUITY))

    assert CLIENT.post('/validate', content=dump(document), headers=JSON).json()['deals'] == {
        'BIN1': ["Option_Type is 'Putt', not one of Call, Put"]}


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


def test_a_calculation_override_is_judged_against_its_own_declarations(book):
    """An override is a dial of a declared calculation, so a field it does not declare and a value
    outside that field's menu each refuse 422 naming it - and the book still prices with the dials
    it does declare.

    KILLING MUTATION: both RAN, the value simply ignored, so a model asking for Greeks got a run
    with no gradient in it and no way to tell.
    """
    unknown = CLIENT.post('/book/price', content=dump(
        {'calculation_overrides': {'Nope': 1}}), headers=JSON)
    outside = CLIENT.post('/book/price', content=dump(
        {'calculation_overrides': {'Greeks': 'Maybe'}}), headers=JSON)
    declared = CLIENT.post('/book/price', content=dump(
        {'calculation_overrides': {'Greeks': 'First'}}), headers=JSON)
    service.EXECUTOR.queue.join()

    assert unknown.status_code == 422 and 'BaseValuation declares no Nope' in unknown.json()[
        'detail']
    assert outside.status_code == 422
    assert outside.json()['detail'] == "Greeks is 'Maybe', not one of All, First, No"
    assert declared.status_code == 200
    assert CLIENT.get('/results/{}'.format(
        declared.json()['result_id'])).json()['status'] == 'done'


def test_a_credit_monte_carlo_this_book_cannot_frame_refuses_before_it_draws(book):
    """A book with no `Price Models` gives a credit Monte Carlo nothing to simulate, and a book
    whose deals all matured gives it no grid to grow: each refuses naming which.

    KILLING MUTATION: `RuntimeError: cannot reshape tensor of 0 elements into shape [0, 16, -1]`
    for the first and `ValueError: max() iterable argument is empty` for the second - the roadmap's
    three books, dying on the tensor with nothing said about the book.
    """
    submitted = CLIENT.post('/book/price', content=dump(
        {'calculation_overrides': {'Object': 'CreditMonteCarlo'}}), headers=JSON).json()
    matured = CLIENT.post('/book/price', content=dump(
        {'calculation_overrides': {'Object': 'CreditMonteCarlo',
                                   'Base_Date': BASE + pd.DateOffset(years=5)}}),
        headers=JSON).json()
    service.EXECUTOR.queue.join()
    modelless = CLIENT.get('/results/{}'.format(submitted['result_id'])).json()
    gridless = CLIENT.get('/results/{}'.format(matured['result_id'])).json()

    assert modelless['status'] == 'error'
    assert modelless['error'].startswith('no factor of this book has a model')
    assert gridless['status'] == 'error'
    assert gridless['error'].startswith('no deal of this book has a date after the base date')


def test_a_candidate_naming_market_data_the_book_lacks_is_refused(book):
    """The what-if validates its candidate before it queues, in the booking's own words.

    A candidate naming a curve the book has no block for LOADS and is then dropped by discovery, so
    the run came back `done` with `Deals Skipped: 1` and no mtm row - the candidate's absence was a
    count and nothing else. The same candidate on a curve the book carries queues and prices, which
    is what says the check reads the DELTA: a what-if is no more blocked by the book's own gaps
    than a booking is.
    """
    candidate = dict(CASHFLOW, Reference='CF9', Amount=100_000.0)
    absent = CLIENT.post('/book/price', content=dump(
        {'deal': dict(candidate, Discount_Rate='ZAR-SWAP')}), headers=JSON)
    carried = CLIENT.post('/book/price', content=dump({'deal': candidate}), headers=JSON)
    service.EXECUTOR.queue.join()

    assert absent.status_code == 422
    assert absent.json()['detail'] == 'no market data for InterestRate.ZAR-SWAP'
    assert carried.status_code == 200
    assert mtm(carried.json()['result_id'])['CF9'] == pytest.approx(
        candidate['Amount'] * SPOT * np.exp(-RATE * 2.0), rel=1e-3)


def fx_vol_snapshot(pair='USDZAR'):
    """A snapshot through the Bloomberg package's own normalization - canned observations standing
    in for the terminal, everything downstream the real pipeline. One object, so the
    `/book/market` and `/book/bloomberg` gates tick the same numbers; `pair` is what a set-up
    fetches a surface for a second pair with."""
    from derivus_bloomberg import (FXQuoteSecurity, FXVolDefinition, RawBloombergObservation,
                                   normalize_fx_vol)
    raw = {('3M', 'ATM', None): 14.0, ('3M', 'RR', 0.25): -1.2, ('3M', 'BF', 0.25): 0.35,
           ('1Y', 'ATM', None): 15.0, ('1Y', 'RR', 0.25): -1.6, ('1Y', 'BF', 0.25): 0.45}
    definition = FXVolDefinition(
        pair=pair, surface_name=pair[:3] + '.' + pair[3:], currency=pair[:3],
        expiries={'3M': 0.25, '1Y': 1.0}, pillars=(0.25,),
        securities={coordinate: FXQuoteSecurity('{} {} {} {}'.format(pair, *coordinate))
                    for coordinate in raw})
    observations = [
        RawBloombergObservation(expiry, quote_type, pillar,
                                '{} {} {} {}'.format(pair, expiry, quote_type, pillar),
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

    # KILLING MUTATION: the patch was WRITTEN and the book's spot became the string 'high', which
    # every pricer then read as a number
    typed = CLIENT.post('/book/market', json={'patch': {'FxRate.ZAR': {'Spot': 'high'}}})
    assert typed.status_code == 422
    assert typed.json()['detail'] == "FxRate.ZAR: Spot must be a number, not 'high'"
    assert book.read_bytes() == before


def test_a_bootstrap_that_complains_writes_nothing(book):
    """A bootstrap that reports an ERROR refuses the WHOLE write with its own messages - the good
    half of the tick with it, because a book must never carry a market its bootstrap complained
    about. Both refusals, and the file untouched by either.

    THE COMPLAINT is a configured family that writes no factor: the USDZAR surface ticks in
    perfectly well and the Hull-White fit declared beside it has no block, so the run says so and
    the whole edit is dropped.

    THE ORPHAN is a block no configured family READS, which never reaches a bootstrapper at all -
    `Config.bootstrap` refuses it by name, with the families it does read, and the endpoint answers
    422 rather than a refusal outcome.
    """
    doc = json.loads(book.read_text())
    doc['Calc']['MergeMarketData']['ExplicitMarketData']['Bootstrapper Configuration'] = {
        'FXVolSurfaceParameters': {}, 'HullWhite2FactorModelParameters': {}}
    book.write_text(json.dumps(doc, indent=2), newline='\n')
    before = book.read_bytes()

    outcome = CLIENT.post('/book/market', content=dump(
        {'quotes': fx_vol_quotes()}), headers=JSON).json()

    assert outcome['written'] is False
    assert any('wrote no' in message for message in outcome['refused'])
    assert book.read_bytes() == before

    ghost = {'GhostPrices.NOWHERE': {'instrument': {'Points': []}}}
    orphan = CLIENT.post('/book/market', content=dump({'quotes': ghost}), headers=JSON)

    assert orphan.status_code == 422
    assert 'GhostPrices' in orphan.json()['detail'], 'a refusal that does not name the block'
    assert 'FXVolPrices' in orphan.json()['detail'], 'a refusal that does not say what is read'
    assert book.read_bytes() == before


def configured_book(path, entries):
    """A live book carrying the USDZAR smile as a quote block that states NO `Grid_Tolerance`, so
    the SECTION's own dial is what the surface is refined to, plus the entries the gate configures.
    The Bloomberg emitter writes that field into every block it emits, which is what a section dial
    is read over, so the gate drops it rather than authoring a second smile."""
    quotes = json.loads(dump(fx_vol_quotes()))
    del quotes['FXVolPrices.USD.ZAR']['instrument']['Grid_Tolerance']
    path.write_text(json.dumps(json.loads(dump(job(sections={
        'Bootstrapper Configuration': entries, 'Market Prices': quotes}))), indent=2),
        newline='\n')
    service.BOOK = service.Book(str(path))


def surface_nodes(path):
    """How many (moneyness, expiry, vol) rows the written `FXVol` surface carries - what
    `Grid_Tolerance` SIZES, and the only thing on the file that moves when it does."""
    surface = json.loads(path.read_text())['Calc']['MergeMarketData']['ExplicitMarketData'][
        'Price Factors'].get('FXVol.USD.ZAR')
    return surface and len(surface['Surface']['.Curve']['data'])


def test_a_configured_dial_re_bootstraps_the_factor_it_sizes(tmp_path):
    """The write half of `/book/configure`: a dial merged into the entry the BOOK spells, the
    market re-bootstrapped in the same atomic write, and the answer naming what moved.

    `Grid_Tolerance` sizes the log-moneyness grid the FX surface is refined to, so it is the one
    dial whose effect is countable on the file. The book declares the entry by its old class name
    and the request names the factor the family writes: the same family either way, so the dial
    lands in the entry that is there rather than a second one beside it - a desk file is never
    renamed under a desk.
    """
    path = tmp_path / 'book.json'
    configured_book(path, {'FXVolSurfaceParameters': {}})
    try:
        fine = CLIENT.post('/book/configure', json={
            'section': 'Bootstrapper Configuration', 'entry': 'FXVol',
            'fields': {'Grid_Tolerance': 1e-4}}).json()

        assert fine['written'] is True and fine['entry'] == 'FXVolSurfaceParameters'
        assert fine['rewrote'] == ['FXVol.USD.ZAR'], 'the re-bootstrap did not reach the surface'
        assert fine['dials'] == {'Prices': 'FXVol', 'Grid_Tolerance': 1e-4}
        refined = surface_nodes(path)

        coarse = CLIENT.post('/book/configure', json={
            'section': 'Bootstrapper Configuration', 'entry': 'FXVol',
            'fields': {'Grid_Tolerance': 0.5}}).json()

        assert coarse['rewrote'] == ['FXVol.USD.ZAR']
        assert surface_nodes(path) < refined, (
            'a tenfold looser tolerance refined the same grid - the dial did not reach the fit')
        assert json.loads(path.read_text())['Calc']['MergeMarketData']['ExplicitMarketData'][
            'Bootstrapper Configuration'] == {
                'FXVolSurfaceParameters': {'Prices': 'FXVol', 'Grid_Tolerance': 0.5}}
    finally:
        service.BOOK = None


def test_a_malformed_dial_refuses_by_name_before_a_quote_is_read(tmp_path):
    """Constructing the family IS the validation, which is why the refusal is the family's own
    sentence and not the bootstrap's: `Sigma_L_Bounds` reversed is refused at construction, before
    a quote is read, and the file is untouched - entry and all, the entry being created only if
    what it says builds."""
    path = tmp_path / 'book.json'
    configured_book(path, {'FXVolSurfaceParameters': {}})
    try:
        before = path.read_bytes()
        refused = CLIENT.post('/book/configure', json={
            'section': 'Bootstrapper Configuration', 'entry': 'LogVar2FJModelParameters',
            'fields': {'Sigma_L_Bounds': '2.0,0.3'}})

        assert refused.status_code == 422
        assert refused.json()['detail'].startswith('Sigma_L_Bounds'), (
            'the refusal is the bootstrap\'s, so a quote was read before the dial was judged')
        assert 'lower < upper' in refused.json()['detail']
        assert path.read_bytes() == before

        # KILLING MUTATION: `could not convert string to float: 'tight'` out of the cast, naming
        # neither the dial nor the entry it was posted to
        typed = CLIENT.post('/book/configure', json={
            'section': 'Bootstrapper Configuration', 'entry': 'InterestRate',
            'fields': {'Tol': 'tight'}})
        assert typed.status_code == 422
        assert typed.json()['detail'].endswith("Tol must be a number, not 'tight'")
        assert path.read_bytes() == before
    finally:
        service.BOOK = None


def test_an_interpolation_the_engine_routes_nowhere_refuses_by_name(tmp_path):
    """The second section, keyed by the routed factor TYPE: `method` is what every factor of it is
    built with, and an `id` names ONE factor to carry a scheme of its own - a `modelfilters` rule
    the engine's own `ModelParams.search` resolves AHEAD of the default, which is what lets two
    curves be built two ways. A blank method clears the rule. A type nothing routes and a method
    nothing implements are both refused by name, because neither raises anywhere downstream - the
    factor is silently interpolated the default way instead.

    Killing mutations: the rule written into `modeldefaults` beside the type's own method, which
    `search` then answers for every curve; a blank method leaving the rule standing, which is a
    curve no screen can put back onto the default; an entry stating neither, which would write an
    empty rule nothing resolves.
    """
    path = tmp_path / 'book.json'
    configured_book(path, {'FXVolSurfaceParameters': {}})
    try:
        written = CLIENT.post('/book/configure', json={
            'section': 'Price Factor Interpolation', 'entry': 'InterestRate',
            'fields': {'method': 'HermiteRT'}}).json()
        ruled = CLIENT.post('/book/configure', json={
            'section': 'Price Factor Interpolation', 'entry': 'InterestRate',
            'fields': {'id': 'ZAR-ZARONIA', 'method': 'LinearRT'}}).json()

        assert written['written'] is True and written['dials'] == {'method': 'HermiteRT',
                                                                   'rules': {}}
        assert ruled['dials'] == {'method': 'HermiteRT', 'rules': {'ZAR-ZARONIA': 'LinearRT'}}
        document = json.loads(path.read_text())
        assert document['Calc']['MergeMarketData']['ExplicitMarketData'][
            'Price Factor Interpolation'] == {'.ModelParams': {
                'modeldefaults': {'InterestRate': 'HermiteRT'},
                'modelfilters': {'InterestRate': [[['id', 'ZAR-ZARONIA'], 'LinearRT']]}}}
        section = in_process(document).current_cfg.params['Price Factor Interpolation']
        assert section.search(utils.Factor('InterestRate', ('ZAR',)), {}, True) == 'HermiteRT'
        assert section.search(
            utils.Factor('InterestRate', ('ZAR-ZARONIA',)), {}, True) == 'LinearRT'

        cleared = CLIENT.post('/book/configure', json={
            'section': 'Price Factor Interpolation', 'entry': 'InterestRate',
            'fields': {'id': 'ZAR-ZARONIA'}}).json()
        assert cleared['dials'] == {'method': 'HermiteRT', 'rules': {}}
        assert json.loads(path.read_text())['Calc']['MergeMarketData']['ExplicitMarketData'][
            'Price Factor Interpolation']['.ModelParams']['modelfilters'] == {}

        before = path.read_bytes()
        for entry, fields, named in (('FxRate', {'method': 'HermiteRT'}, 'FxRate'),
                                     ('InterestRate', {'method': 'Cubic'}, 'Cubic'),
                                     ('InterestRate', {'id': 'ZAR', 'method': 'Cubic'}, 'Cubic'),
                                     ('InterestRate', {}, 'id')):
            refused = CLIENT.post('/book/configure', json={
                'section': 'Price Factor Interpolation', 'entry': entry, 'fields': fields})
            assert refused.status_code == 422 and named in refused.json()['detail']
            assert 'InterestRate' in refused.json()['detail'], 'a refusal naming no menu'
        assert path.read_bytes() == before
    finally:
        service.BOOK = None


#: A ZAR curve as a desk states it: a 3M JIBAR deposit at the front, a FRA beside it and the swap
#: strip beyond, each row naming the security it is quoted off. The LEVELS are invented and only
#: have to be plausibly shaped - what is asserted is the authoring, never the market.
CURVE_ROWS = [{'tenor': '3M', 'security': 'JIBA3M Index', 'quote': 7.41},
              {'tenor': '1Mx4M', 'security': 'SAFR0AD Curncy', 'quote': 7.35},
              {'tenor': '1Y', 'security': 'SASW1 BGN Curncy', 'quote': 7.62},
              {'tenor': '2Y', 'security': 'SASW2 BGN Curncy', 'quote': 7.94},
              {'tenor': '5Y', 'security': 'SASW5 BGN Curncy', 'quote': 8.55},
              {'tenor': '10Y', 'security': 'SASW10 BGN Curncy', 'quote': 8.83}]

CURVE_BLOCK = 'InterestRatePrices.ZAR'

#: A second curve for the gates that need two on one book: the seed's USD entry is an overnight
#: front and an OIS strip, and every row here is quoted BY HAND, so no terminal is in the picture.
USD_ROWS = [{'tenor': 'ON', 'quote': 5.33}, {'tenor': '1Y', 'quote': 5.05},
            {'tenor': '2Y', 'quote': 4.55}, {'tenor': '5Y', 'quote': 4.05}]


def set_up_curve(rows=None, **request):
    return CLIENT.post('/book/curve', content=dump(dict(
        {'curve': 'ZAR', 'currency': 'ZAR', 'rows': CURVE_ROWS if rows is None else rows},
        **request)), headers=JSON)


def curve_block(path, curve='ZAR'):
    return json.loads(path.read_text())['Calc']['MergeMarketData']['ExplicitMarketData'][
        'Market Prices']['InterestRatePrices.{}'.format(curve)]['instrument']


def par_residuals(path, curve='ZAR', interp=None):
    """Every benchmark of the written block repriced off the curve the bootstrap solved, AS OF THE
    DAY THE BOOK IS DATED - the par vector, which is zero iff the knot grid is square and the dates
    the emitter rolled are the ones the solve used. A million of notional, so 1e-6 is a thousandth
    of a basis point of principal.

    Read under the book's OWN `Price Factor Interpolation` unless `interp` names another, so a
    curve solved under one scheme and read under a second says so in the residual."""
    import copy

    import torch
    from derivus.bootstrappers import (BenchmarkInstruments, author_quote, completed, quote_node)
    from derivus.config import Config, ModelParams

    market = Config().read_json(str(path))['Calc']['MergeMarketData']['ExplicitMarketData']
    block = market['Market Prices']['InterestRatePrices.{}'.format(curve)]['instrument']
    nodes = []
    for point in block['Points']:
        if point['Use'] != 'Yes':      # a held-out row was not solved for and never reprices
            continue
        # `completed` because a quote WRITER reads the block's conventions and a benchmark states
        # only what differs from its declaration - the same seam `quote_nodes` builds through
        deal = completed(dict(copy.deepcopy(point['Deal']), Object=point['DealType']))
        author_quote(deal, point['Quoted_Market_Value'], curve)
        nodes.append(quote_node(deal, {}))
    return BenchmarkInstruments(
        nodes, market['Price Factors'],
        interp or market.get('Price Factor Interpolation') or ModelParams(),
        market['System Parameters']['Base_Date'], block['Currency'], {}, [],
        torch.device('cpu'))({}).detach().numpy()


def test_a_curve_set_up_from_its_rows_solves_at_par(book):
    """THE VERB IS THE AUTHORING ACT. A desk states a tenor, the security it is quoted off and a
    number; the block is authored from the seed's conventions for that curve, the `InterestRate`
    bootstrapper entry is added because this book configures none, and the whole market is
    re-bootstrapped in ONE write - so the file carries the solved curve and the block that made it.

    THE BLOCK IS THE CURVE'S DEFINITION, so the conventions ride beside the quotes and every row
    carries its tenor and its security; and EVERY BENCHMARK REPRICES AT PAR off what the run wrote,
    which is what says the knot grid is square and the rolled dates are the ones that were solved.
    """
    answer = set_up_curve()
    outcome = answer.json()
    market = json.loads(book.read_text())['Calc']['MergeMarketData']['ExplicitMarketData']
    block = curve_block(book)

    assert answer.status_code == 200, outcome
    assert outcome['written'] is True and outcome['block'] == CURVE_BLOCK
    assert outcome['knots'] == [row['tenor'] for row in CURVE_ROWS]
    assert outcome['rewrote'] == ['InterestRate.ZAR']
    assert market['Bootstrapper Configuration']['InterestRate']['Prices'] == 'InterestRate'
    assert 'Price Factor Interpolation' not in market, 'a curve stating no scheme wrote a section'
    assert block['Spot_Days'] == 0 and block['Compounding'] == 'None'
    assert block['Fixed_Frequency'] == {'.DateOffset': '3M'} and block['Day_Count'] == 'ACT_365'
    assert [row['Tenor'] for row in block['Points']] == [row['tenor'] for row in CURVE_ROWS]
    assert [row['Security'] for row in block['Points']] == [row['security'] for row in CURVE_ROWS]
    # the shape of each row came off its own TENOR - nothing in the request said `deposit`
    assert [row['DealType'] for row in block['Points']] == [
        'DepositDeal', 'FRADeal'] + ['SwapInterestDeal'] * 4
    assert abs(par_residuals(book)).max() < 1e-6, 'a benchmark the solved curve does not reprice'


def test_a_curve_states_its_own_interpolation_and_the_read_verb_resolves_it(book):
    """A CURVE'S SCHEME IS A RULE IN `Price Factor Interpolation`, not a field of its block. The
    verb writes the rule in the same atomic write as the block and BEFORE the bootstrap, so the
    curve is SOLVED under what it will be read under; a second curve set up beside it with no
    scheme takes the routed type's own. The read verb resolves both the way `construct_factor`
    does and says which of the two answered.

    MEASURED, on a million of notional: the ZAR strip solved under its HermiteRT rule reprices its
    own benchmarks to **5.82e-11** and, read under the Linear default instead, to **690.14** - the
    swap rows, whose every coupon reads the curve between the knots the ladder carries.

    Killing mutations: the rule written after the bootstrap, which leaves the curve fitted under
    the default and off par by that second number; the read verb answering the type default alone,
    which cannot tell the two curves apart; a blank `interpolation` leaving the rule standing.
    """
    from derivus.config import ModelParams

    assert set_up_curve(interpolation='HermiteRT').status_code == 200
    assert set_up_curve(rows=USD_ROWS, curve='USD', currency='USD').status_code == 200
    market = json.loads(book.read_text())['Calc']['MergeMarketData']['ExplicitMarketData']
    answer = CLIENT.get('/book/curve').json()['curves']

    assert market['Price Factor Interpolation']['.ModelParams'] == {
        'modeldefaults': {}, 'modelfilters': {'InterestRate': [[['id', 'ZAR'], 'HermiteRT']]}}
    assert (answer[CURVE_BLOCK]['interpolation'],
            answer[CURVE_BLOCK]['interpolation_source']) == ('HermiteRT', 'curve')
    assert answer[CURVE_BLOCK]['conventions']['interpolation'] == 'HermiteRT'
    assert (answer['InterestRatePrices.USD']['interpolation'],
            answer['InterestRatePrices.USD']['interpolation_source']) == ('Linear', 'default')

    assert abs(par_residuals(book)).max() < 1e-6, 'the rule was not what the curve was solved under'
    assert abs(par_residuals(book, interp=ModelParams())).max() > 1e-4

    # the rows re-stated with nothing said about the scheme: the rule stands (killing mutation: an
    # absent field read as a blank one, which clears the rule under every re-statement)
    assert set_up_curve().status_code == 200
    standing = CLIENT.get('/book/curve', params={'curve': 'ZAR'}).json()['curves'][CURVE_BLOCK]
    assert (standing['interpolation'], standing['interpolation_source']) == ('HermiteRT', 'curve')

    assert set_up_curve(interpolation='').status_code == 200
    assert json.loads(book.read_text())['Calc']['MergeMarketData']['ExplicitMarketData'][
        'Price Factor Interpolation']['.ModelParams']['modelfilters'] == {}
    cleared = CLIENT.get('/book/curve', params={'curve': 'ZAR'}).json()['curves'][CURVE_BLOCK]
    assert (cleared['interpolation'], cleared['interpolation_source']) == ('Linear', 'default')


def test_the_status_verb_says_what_the_desk_is_set_up_with(book, tmp_path, monkeypatch):
    """The read a client opens with, composed out of the readers beside it: the book's date and
    currency, what it holds, every curve with the knots it solved on, the latest print its rows
    carry and any benchmark held out, and what this workstation could fetch if asked.

    `DV_HOME` is the gate's own tmp, so `provisioned` and the XVA rows are this book's and not
    the workstation's, and `blpapi` is made ABSENT - no session is opened either way, the import
    being the whole question.

    Killing mutations: `held_out` partitioning on the rows in USE, which the read-back block
    carries either way so only the split says which; `terminal.present` reading anything but this
    workstation's own import, which the absent module then does not move.
    """
    from derivus_bloomberg import session
    from derivus_bloomberg.errors import BloombergUnavailable

    def absent():
        raise BloombergUnavailable('no blpapi on this workstation')

    monkeypatch.setenv('DV_HOME', str(tmp_path / 'home'))
    monkeypatch.setattr(session, 'blpapi_module', absent)
    bare = CLIENT.get('/book/status').json()

    assert set(bare) == {'etag', 'base_date', 'base_currency', 'calculation', 'deals',
                         'netting_sets', 'curves', 'surfaces', 'models', 'xva', 'spine',
                         'terminal'}
    assert bare['spine'] is None, 'a box that records nothing has no position to report'
    assert bare['base_date'] == '2024-06-28' and bare['base_currency'] == 'USD'
    assert {key: bare['calculation'][key] for key in ('Object', 'Currency', 'paths')} == {
        'Object': 'BaseValuation', 'Currency': 'USD', 'paths': 1}
    # a served book whose count cannot price a simulated deal is NAMED, with the field, the number
    # the declaration states and where a desk sets it - never healed behind the desk
    assert 'MCMC_Simulations is 1' in bare['calculation']['paths_note']
    assert '{:,}'.format(structures.declared_paths()) in bare['calculation']['paths_note']
    assert 'marks on its own' in bare['calculation']['paths_note']
    assert structures.thin_paths({}) is None, 'a book stating nothing takes the declaration'
    assert structures.thin_paths(
        {'MCMC_Simulations': structures.declared_paths()}) is None
    assert (bare['deals'], bare['netting_sets']) == (1, [])
    assert (bare['curves'], bare['surfaces'], bare['models'], bare['xva']) == ([], [], [], [])
    assert bare['terminal'] == {'present': False, 'ticking': None, 'provisioned': False}
    assert bare['etag'] == CLIENT.get('/book').json()['etag']

    held = [dict(row, use='No') if row['tenor'] == '10Y' else row for row in CURVE_ROWS]
    assert set_up_curve(held).status_code == 200
    status = CLIENT.get('/book/status').json()

    assert status['curves'] == [{
        'curve': 'ZAR', 'currency': 'ZAR', 'snapped': '2024-06-28', 'interpolation': 'Linear',
        'knots': [row['tenor'] for row in CURVE_ROWS[:-1]], 'held_out': ['10Y']}]


def test_the_curve_verb_refuses_by_name_and_the_file_stands_still(book, monkeypatch):
    """THREE REFUSALS, each naming the thing to fix, each writing nothing.

    An unquoted row and no terminal to price it off - `BloombergSession` is the seam and a
    workstation without one is what `BloombergUnavailable` says. A tenor the emitter's grammar
    cannot read, which is the one thing a desk can spell wrong that no convention would catch. And
    a bootstrap that complains, which refuses the WHOLE write the way a tick's does: the
    `HullWhite2FactorModelParameters` entry configured beside the curve has no block to fit.
    """
    from derivus_bloomberg import session
    from derivus_bloomberg.errors import BloombergUnavailable

    def absent(**options):
        raise BloombergUnavailable('no blpapi module on this workstation')

    monkeypatch.setattr(session, 'BloombergSession', absent)
    before = book.read_bytes()

    unquoted = set_up_curve(rows=[dict(CURVE_ROWS[0], quote=None)] + CURVE_ROWS[1:])
    assert unquoted.status_code == 422
    assert '3M' in unquoted.json()['detail'] and 'no terminal' in unquoted.json()['detail']
    assert book.read_bytes() == before

    unreadable = set_up_curve(rows=[dict(row, tenor='3Q') if row['tenor'] == '5Y' else row
                                    for row in CURVE_ROWS])
    assert unreadable.status_code == 422 and "'3Q'" in unreadable.json()['detail']
    assert book.read_bytes() == before

    document = json.loads(book.read_text())
    document['Calc']['MergeMarketData']['ExplicitMarketData']['Bootstrapper Configuration'] = {
        'HullWhite2FactorModelParameters': {}}
    book.write_text(json.dumps(document, indent=2), newline='\n')
    before = book.read_bytes()

    complained = set_up_curve()
    assert complained.status_code == 422 and 'wrote no' in complained.json()['detail']
    assert book.read_bytes() == before, 'a bootstrap that complained still wrote the block'


#: The OIS strip the three gates below tick: long enough that its solve is seconds rather than
#: milliseconds, which is what a desk's own curve costs and what makes a bootstrap worth not
#: holding a lock across.
OIS_ROWS = ([{'tenor': 'ON', 'quote': 7.30}]
            + [{'tenor': tenor, 'quote': 7.35 + 0.01 * n}
               for n, tenor in enumerate(('1M', '2M', '3M', '6M', '9M'), 1)]
            + [{'tenor': tenor, 'quote': 7.55 + 0.03 * n}
               for n, tenor in enumerate(('1Y', '2Y', '3Y', '4Y', '5Y', '6Y', '7Y', '8Y', '9Y',
                                          '10Y', '12Y', '15Y', '20Y', '25Y', '30Y'), 1)])

#: What a verb answers inside while a bootstrap runs beside it. A blotter polls the book on a
#: cadence a person can feel, and the solve it polls beside is measured in seconds.
LOCK_BUDGET = 0.2


@pytest.fixture
def desk_curves(book, monkeypatch):
    """A desk-shaped curve set on one book: `ZAR-ZARONIA` off the OIS strip, `ZAR` discounting on
    it - `CURVE_ROWS` without its deposit, which a foreign discount curve does not identify - and
    `USD`, which nothing here reads. Every row is quoted by hand and `DV_HOME` is the gate's own
    tmp, so neither a terminal nor a desk file is in the picture."""
    monkeypatch.setenv('DV_HOME', str(book.parent))
    assert set_up_curve(rows=OIS_ROWS, curve='ZAR-ZARONIA', currency='ZAR').status_code == 200
    assert set_up_curve(rows=CURVE_ROWS[1:], discount_rate='ZAR-ZARONIA').status_code == 200
    assert set_up_curve(rows=USD_ROWS, curve='USD', currency='USD').status_code == 200
    return book


def moved_quotes(path, curve, by=0.005):
    """One `InterestRatePrices` block as it stands in the book with every quote moved - the body of
    a values tick, which is the one thing a desk posts between authorings."""
    block = {'instrument': curve_block(path, curve)}
    for point in block['instrument']['Points']:
        point['Quoted_Market_Value'] = round(point['Quoted_Market_Value'] + by, 6)
    return {'InterestRatePrices.{}'.format(curve): block}


def written_factors(path):
    """Every price factor of the book as the BYTES it stands in the file as."""
    return {name: json.dumps(block, sort_keys=True) for name, block in json.loads(
        path.read_text())['Calc']['MergeMarketData']['ExplicitMarketData'][
            'Price Factors'].items()}


def under_a_tick(quotes, during):
    """Post a values tick from another thread and run `during` while it bootstraps: the tick's
    `(start, end)`, its answer, and whatever `during` gave back.

    The window is what the callers assert the overlap against, so a tick too fast to overlap fails
    the gate rather than passing it vacuously. The sleep is the tick's head start and nothing is
    asserted on it.
    """
    window, answered = [], []

    def tick():
        start = time.perf_counter()
        posted = CLIENT.post('/book/market', content=dump({'quotes': quotes}), headers=JSON)
        window.append((start, time.perf_counter()))
        answered.append(posted.json())

    thread = threading.Thread(target=tick)
    thread.start()
    time.sleep(0.4)
    ran = during()
    thread.join()
    return window[0], answered[0], ran


def test_a_read_and_a_booking_answer_while_a_tick_bootstraps(desk_curves):
    """A READ NEVER WAITS BEHIND A BOOTSTRAP. `Book.mutate` reads under the lock, runs the edit
    outside it and takes the lock again only to write, so the seconds a curve solve costs are not
    seconds every other verb waits: a `GET /book` and a `POST /book/deals` issued while the tick is
    bootstrapping both answer inside the budget, and both are asserted to have run INSIDE the
    tick's own window - a tick that answered first fails this gate rather than passing it.

    Killing mutation: the lock held across the edit again - the read waits the whole solve out,
    0.86s against the 0.007s and the booking's 0.022s here, and both miss the tick's window.
    """
    def read_and_book():
        started = time.perf_counter()
        read = CLIENT.get('/book')
        halfway = time.perf_counter()
        booked = CLIENT.post('/book/deals', content=dump({'action': 'add', 'deal': BOOKED}),
                             headers=JSON)
        return started, halfway, time.perf_counter(), read, booked

    window, ticked, (started, halfway, done, read, booked) = under_a_tick(
        moved_quotes(desk_curves, 'ZAR-ZARONIA'), read_and_book)

    assert ticked['written'] is True and read.status_code == 200
    assert booked.json()['written'] is True
    assert halfway - started < LOCK_BUDGET, 'the read waited on the bootstrap'
    assert done - halfway < LOCK_BUDGET, 'the booking waited on the bootstrap'
    assert window[0] < started and done < window[1], 'the calls did not overlap the bootstrap'


def test_a_booking_landing_mid_tick_costs_the_tick_a_redo_and_both_edits_land(desk_curves):
    """TWO EDITS, ONE FILE, BOTH LANDING. The booking lands inside the tick's window, so the tick's
    edit ran against a document that no longer stands: its write is refused by the etag it read and
    the edit RE-RUNS on the document the booking wrote. The file carries both afterwards - the
    booked deal and the moved quotes - and the tick's answer names the etag the book now has.

    Killing mutation: the etag check dropped in `Book.mutate` - the tick writes the document it
    read and the deal booked inside its window is gone from the file it lands in.
    """
    def book_a_deal():
        started = time.perf_counter()
        answer = CLIENT.post('/book/deals', content=dump({'action': 'add', 'deal': BOOKED}),
                             headers=JSON)
        return started, time.perf_counter(), answer.json()

    moved = moved_quotes(desk_curves, 'ZAR-ZARONIA')
    window, ticked, (started, done, booked) = under_a_tick(moved, book_a_deal)
    document = json.loads(desk_curves.read_text())

    assert window[0] < started and done < window[1], 'the booking did not land inside the tick'
    assert booked['written'] is True and ticked['written'] is True
    assert booked['etag'] != ticked['etag'] == CLIENT.get('/book').json()['etag']
    assert [node['Instrument']['.Deal']['Reference']
            for node in document['Calc']['Deals']['Deals']['Children']] == ['CF1', 'CF2']
    assert [row['Quoted_Market_Value'] for row in curve_block(desk_curves, 'ZAR-ZARONIA')[
        'Points']] == [row['Quoted_Market_Value'] for row in
                       moved['InterestRatePrices.ZAR-ZARONIA']['instrument']['Points']]


def test_a_tick_that_keeps_losing_to_bookings_lands_on_its_last_pass(desk_curves):
    """AN EDIT THAT KEEPS LOSING TO SHORTER ONES LANDS. Three hosts booking and deleting as fast as
    the service answers rewrite the file faster than a curve solve can re-run, so the tick's write
    is refused by its etag pass after pass; the last pass runs under the lock and lands, with every
    booking landed beside it - measured under three concurrent hosts, where a tick refused after
    three open passes.

    Each host deletes by the path its booking answered WITH THE REFERENCE beside it: the other
    hosts move that position under it, and a path that no longer holds the deal refuses by name
    rather than deleting whoever sits there, so the host reads the book and deletes where the deal
    now stands. The file ends with the one deal it started with.

    Killing mutations: the last pass run outside the lock like the others - the tick refuses after
    its passes and the moved quotes never reach the file; the reference guard dropped - a delete
    removes whichever deal now sits at the path, and fifty of the hosts' own deals are left behind.
    """
    def delete(deal_path, reference):
        # a path that moved refuses (or points past the end): read where the deal stands now, and
        # a deal nobody can find any more was deleted by someone else
        while True:
            answer = CLIENT.post('/book/deals', content=dump(
                {'action': 'delete', 'deal_path': deal_path, 'reference': reference}),
                headers=JSON)
            if answer.status_code == 200:
                return answer.json()['deleted']
            assert answer.status_code == 422, answer.text
            children = CLIENT.get('/book').json()['document']['Calc']['Deals']['Deals']['Children']
            deal_path = next((str(position) for position, node in enumerate(children)
                              if node['Instrument']['.Deal'].get('Reference') == reference), None)
            if deal_path is None:
                return None

    def hammer(host, tally):
        deadline = time.perf_counter() + 6.0
        try:
            while time.perf_counter() < deadline:
                reference = 'H{}_{}'.format(host, tally[host])
                booked = CLIENT.post('/book/deals', content=dump({'action': 'add', 'deal': dict(
                    BOOKED, Reference=reference)}), headers=JSON).json()
                tally[host] += 1
                if booked.get('written') and delete(booked['deal_path'], reference) != reference:
                    tally['wrong'] += 1
        except Exception as error:  # a thread's failure is the gate's, not the log's
            tally['errors'].append(repr(error))

    def three_hosts():
        tally = {0: 0, 1: 0, 2: 0, 'wrong': 0, 'errors': []}
        threads = [threading.Thread(target=hammer, args=(host, tally)) for host in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return tally

    moved = moved_quotes(desk_curves, 'ZAR-ZARONIA')
    window, ticked, tally = under_a_tick(moved, three_hosts)

    assert ticked['written'] is True, ticked
    assert ticked['bootstrapped'] == ['InterestRatePrices.ZAR', 'InterestRatePrices.ZAR-ZARONIA']
    assert tally[0] + tally[1] + tally[2] > 10 and tally['wrong'] == 0, tally
    assert tally['errors'] == [], tally['errors']
    assert [row['Quoted_Market_Value'] for row in curve_block(desk_curves, 'ZAR-ZARONIA')[
        'Points']] == [row['Quoted_Market_Value'] for row in
                       moved['InterestRatePrices.ZAR-ZARONIA']['instrument']['Points']]
    assert [node['Instrument']['.Deal']['Reference'] for node in json.loads(
        desk_curves.read_text())['Calc']['Deals']['Deals']['Children']] == ['CF1']


def test_a_moved_curve_re_solves_what_reads_it_and_nothing_else(desk_curves):
    """A VALUE TICK RE-SOLVES WHAT IT MOVED, and what reads what it moved.

    The OIS curve's quotes move: it re-solves, and so does the projection curve naming it in
    `Discount_Rate` - the coupling read off the block itself, never a list kept beside it. The
    third curve is left exactly as it stands, asserted on the BYTES of its price factor while its
    own quotes stand moved and UNSOLVED, posted with `bootstrap: 'No'`: a run that covered it would
    solve those quotes and move the factor. Ticking that block then re-solves it alone.

    Killing mutation: the selection dropped at either end - `market_edit` not narrowing, or
    `Config.bootstrap` running its whole section - and the deferred curve re-solves under the OIS
    tick, so its bytes move and `bootstrapped` names all three blocks.
    """
    deferred = CLIENT.post('/book/market', content=dump(
        {'quotes': moved_quotes(desk_curves, 'USD'), 'bootstrap': 'No'}), headers=JSON).json()
    standing = written_factors(desk_curves)
    ticked = CLIENT.post('/book/market', content=dump(
        {'quotes': moved_quotes(desk_curves, 'ZAR-ZARONIA')}), headers=JSON).json()
    after = written_factors(desk_curves)

    assert deferred['updated'] == ['InterestRatePrices.USD'] and 'bootstrapped' not in deferred
    assert ticked['bootstrapped'] == ['InterestRatePrices.ZAR', 'InterestRatePrices.ZAR-ZARONIA']
    assert after['InterestRate.USD'] == standing['InterestRate.USD'], 'a curve nothing reads solved'
    assert after['InterestRate.ZAR'] != standing['InterestRate.ZAR'], 'what discounts on it stood'
    assert after['InterestRate.ZAR-ZARONIA'] != standing['InterestRate.ZAR-ZARONIA']

    standing = written_factors(desk_curves)
    alone = CLIENT.post('/book/market', content=dump(
        {'quotes': moved_quotes(desk_curves, 'USD')}), headers=JSON).json()
    after = written_factors(desk_curves)

    assert alone['bootstrapped'] == ['InterestRatePrices.USD']
    assert after.pop('InterestRate.USD') != standing.pop('InterestRate.USD')
    assert after == standing, 'a tick of one curve rewrote another'


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
    the surface arrives by POSTing a quote block to `/book/market`. So the spot-model gates start
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


def spot_model_block(path):
    """The emitter, run the way the verb runs it: off a Context over the book file on disk."""
    from derivus.bootstrappers import LogVar2FJModelParameters

    params = in_process(json.loads(path.read_text())).current_cfg.params
    return LogVar2FJModelParameters.fx_surface_block(
        'USD.ZAR', params['Price Factors'], params['System Parameters'],
        params['Price Factor Interpolation'])


def test_the_fx_ladder_is_vega_weighted_points_on_the_surfaces_own_strikes(tmp_path):
    """The desk's ladder off the built surface: one ATM rung per `ATM_Expiries` plus both wings at
    each `Wing_Pillars` on each `Wing_Expiries`, nothing past a year.

    THE COUNT IS THE FAMILY'S OWN DECLARATION, not a number written here - a family that widens its
    wings widens this gate with it. What is asserted is what makes them the SURFACE'S points rather
    than a moneyness grid laid over it. The expiries are the surface's own (1M/2M/3M/6M here), so
    the rungs past that move to the nearest quoted one at or under a year and `Quote_Source` SAYS
    SO - the difference between a substitution and a silent interpolation, which is also why the
    distinct-contract count is BELOW the rung count. The weights are normalised Black vega, so they
    sum to one and the back ATM outweighs the front, which stops an unweighted fit abandoning the
    front end. And the wings straddle the spot.
    """
    from derivus.bootstrappers import LogVar2FJModelParameters as Family

    name, block = spot_model_block(built_surface(tmp_path / 'book.json',
                                                 json.loads(dump(desk_smile()))))
    try:
        instrument = block['instrument']
        points = instrument['European_Options']
        expiries = sorted({point['Expiry_Date'] for point in points})

        assert name == 'LogVar2FJModelPrices.ZAR'
        ladder = Family.fx_ladder()
        rungs = len(ladder.atm) + 2 * len(ladder.wings) * len(ladder.pillars)
        assert len(points) == rungs, 'the emitter did not write its own declared ladder'
        # the surface carries 1/12, 2/12, 0.25 and 0.5 in years, and nothing else
        assert [str(x.date()) for x in expiries] == [
            '2024-07-28', '2024-08-28', '2024-09-27', '2024-12-27']
        contracts = len({(point['Expiry_Date'], point['Strike']) for point in points})
        assert ladder.minimum <= contracts < rungs, (
            'the ladder collapsed further than the fixture says it does, or not at all')
        assert 'moved to the nearest quoted' in instrument['Quote_Source']
        for moved in ('ATM 0.75 -> 0.5', 'ATM 1 -> 0.5'):
            assert moved in instrument['Quote_Source'], moved
        # the surface's own as-of travels onto the block, to the minute - staleness is data
        assert '2024-06-28' in str(instrument['Quote_Timestamp'])
        assert '16:30' in str(instrument['Quote_Timestamp'])

        # the ATM rungs are emitted in ladder order, then the wing pairs
        n_atm = len(ladder.atm)
        atm, wings = points[:n_atm], points[n_atm:]
        assert sum(point['Weight'] for point in points) == pytest.approx(1.0)
        assert {point['Expiry_Date'] for point in atm} == set(expiries)
        assert atm[0]['Weight'] < atm[-1]['Weight'], 'the front ATM outweighs the back one'
        assert atm[0]['Expiry_Date'] == expiries[0] and atm[-1]['Expiry_Date'] == expiries[-1]

        below = [point for point in wings if point['Option_Type'] == 'Put']
        above = [point for point in wings if point['Option_Type'] == 'Call']
        assert len(below) == len(above) == len(wings) // 2, 'a wing lost its pair'
        assert all(point['Strike'] < SPOT for point in below)
        assert all(point['Strike'] > SPOT for point in above)
        # the wings carry the smile, not the ATM vol repeated - and this pair's RISES with strike
        # in the underlying's own units, which is the sign a one-signed Gamma_Star could not fit
        assert below[0]['Quoted_Market_Value'] < above[0]['Quoted_Market_Value']

        # the block resolves: every reference the fit hard-reads types off the book's own factors
        from derivus import riskfactors

        params = in_process(json.loads((tmp_path / 'book.json').read_text())).current_cfg.params
        factors, interp = params['Price Factors'], params['Price Factor Interpolation']
        assert [Family.resolve(instrument, field, factors)
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
            pillar, _ = Family.fx_surface_expiry(
                surface, days / ladder.days, max(ladder.atm), ladder.tolerance)
            forward = spot * np.exp(
                (float(discount.current_value(t)) - float(carry.current_value(t))) * t)
            moneyness = Family.moneyness(
                point['Strike'], spot, forward, surface, True, True)
            assert float(surface.current_value([[moneyness, pillar]])[0]) == pytest.approx(
                point['Quoted_Market_Value'], rel=1e-9)
    finally:
        service.BOOK = None


def hand_authored_block(vols):
    """A `LogVar2FJModelPrices.ZAR` block with nine quotes at one, two and three weeks.

    The shift gate needs the fit's ARITHMETIC, not its ladder, so the expiries are the shortest
    that still make three step counts and the gate runs in seconds instead of the emitter ladder's
    quarter hour. `Stationary_Spread` is `Floor` because a three-week smile fits a vol-of-vol the
    VIX band does not carry (sd 0.287 against 0.4), and the band is not what is measured here.
    Everything else is the block the emitter writes.
    """
    return {'instrument': {
        'Underlying': 'ZAR', 'Underlying_Type': 'FxRate',
        'Volatility': 'USD.ZAR', 'Volatility_Type': 'FXVol',
        'Discount_Rate': 'USD', 'Discount_Rate_Type': 'InterestRate',
        'Yield': 'ZAR', 'Yield_Type': 'InterestRate',
        'Quote_Type': 'Implied_Volatility', 'Use_Forward': 'Yes', 'Invert_Moneyness': 'Yes',
        'Stationary_Spread': 'Floor',
        'Steps_Per_Year': 252.0, 'Max_Iterations': 2, 'Paths': 1024,
        'European_Options': [
            {'Expiry_Date': BASE + pd.DateOffset(days=days), 'Strike': SPOT * ratio,
             'Option_Type': 'Call' if ratio >= 1.0 else 'Put', 'Units': 1.0, 'Weight': 1.0 / 9.0,
             'Quoted_Market_Value': vol}
            for days in (7, 14, 21)
            for ratio, vol in zip((0.95, 1.0, 1.05), vols)]}}


def fitted_scalars(path, block, delta):
    """Every fitted curve value a book lands on for `block` at `Volatility_Delta` `delta` - through
    the market seam, off the file, exactly as a tick calibrates.

    THE FIT IS THE CURVES. `LogVar2FJ.PARAM_NAMES` are structural priors a ladder does not identify and
    sit at their declared defaults whatever the quotes say, so what is read is `LogVar2FJ.CURVE_NAMES`.
    And the previously written factor is DROPPED first, because a fit warm starts off one: three
    comparable fits have to be three cold ones.
    """
    document = json.loads(path.read_text())
    market = document['Calc']['MergeMarketData']['ExplicitMarketData']
    market['System Parameters']['Volatility_Delta'] = delta
    market['Bootstrapper Configuration'] = {'LogVar2FJModelParameters': {}}
    market.get('Market Prices', {}).pop('LogVar2FJModelPrices.ZAR', None)
    market['Price Factors'].pop('LogVar2FJModelParameters.ZAR', None)
    path.write_text(json.dumps(document, indent=2), newline='\n')
    service.BOOK = service.Book(str(path))
    written = CLIENT.post('/book/market', content=dump(
        {'quotes': {'LogVar2FJModelPrices.ZAR': block}}), headers=JSON).json()
    assert written['written'] is True, written
    factor = derivus.Context().load_json(str(path)).current_cfg.params[
        'Price Factors']['LogVar2FJModelParameters.ZAR']
    return [float(value) for name in utils.LogVar2FJ.CURVE_NAMES
            for value in factor[name].array[:, 1]]


def test_a_volatility_delta_moves_the_fitted_world_once(tmp_path):
    """A scenario shift must reach the fitted world exactly once. It used to reach it twice:
    `fx_surface_block` folded `Volatility_Delta` into every emitted vol and `bootstrap` added
    `vol_surface.delta` again, so a 1-vol-point scenario calibrated a 2-vol-point world.

    Two halves. The emitter is delta-BLIND: a block authored at 0.01 is the block authored at 0.0
    bit for bit, because a quote block is a QUOTE. And the fit applies the shift ONCE: fitting
    unshifted quotes under a 0.01 scenario lands on the same fitted curves as fitting HAND-BUMPED
    quotes under none. The third fit rules out a shift that did nothing.
    """
    path = built_surface(tmp_path / 'book.json', json.loads(dump(desk_smile())))
    try:
        unshifted = spot_model_block(path)[1]['instrument']['European_Options']
        document = json.loads(path.read_text())
        document['Calc']['MergeMarketData']['ExplicitMarketData'][
            'System Parameters']['Volatility_Delta'] = 0.01
        path.write_text(json.dumps(document, indent=2), newline='\n')
        service.BOOK = service.Book(str(path))
        shifted = spot_model_block(path)[1]['instrument']['European_Options']

        assert [point['Quoted_Market_Value'] for point in shifted] == [
            point['Quoted_Market_Value'] for point in unshifted], (
            'the emitted block moved with a scenario shift - the block is a quote')
        assert [point['Strike'] for point in shifted] == [
            point['Strike'] for point in unshifted], 'the strikes moved with the shift'

        vols = (0.14, 0.145, 0.15)
        scenario = fitted_scalars(path, hand_authored_block(vols), 0.01)
        by_hand = fitted_scalars(path, hand_authored_block(
            tuple(vol + 0.01 for vol in vols)), 0.0)
        unmoved = fitted_scalars(path, hand_authored_block(vols), 0.0)

        # MEASURED bit-identical across the seven curve values, the shift itself moving each of
        # them by 8.1 to 26.1%. The band is a fit's noise floor, not what was read
        assert scenario == pytest.approx(by_hand, rel=1e-6), (
            'a 1 vol point scenario did not fit the world 1 vol point away')
        assert scenario != pytest.approx(unmoved, rel=1e-3), (
            'the shift moved nothing, so the identity above is vacuous')
    finally:
        service.BOOK = None


def test_a_collapsed_ladder_refuses_and_nothing_past_a_year_is_ever_snapped_to(tmp_path):
    """The two things an unconditional argmin does not do, both silent when they happen.

    A COLLAPSED LADDER. The canned surface carries 3M and 1Y; move the 1Y pillar to 2Y and 3M is
    the only one inside the ladder's own cap, so twenty-two rungs land on FIVE distinct contracts -
    one ATM and one expiry's four wings - and the later buckets of `Rho_S` and `Beta` are
    identified by nothing. The emitter refuses, naming the pillars, the ladder, the count and the
    remedy. The `expiry.size < 2` guard cannot see this: the surface IS a grid.

    NOTHING PAST A YEAR. Snapping is an argmin over every pillar, so a surface quoting 2Y answers
    the 1Y rung with 2Y and a sub-year fit borrows from a smile nobody quotes. Here the four-pillar
    surface carries a 2Y pillar the ladder must not touch: every emitted expiry is inside the year,
    the 9M and 1Y rungs land on 6M, and the block says so.
    """
    from derivus.bootstrappers import LogVar2FJModelParameters as Family

    collapsed = json.loads(dump(fx_vol_quotes()))
    for point in collapsed['FXVolPrices.USD.ZAR']['instrument']['Points']:
        if point['Expiry'] == 1.0:
            point['Expiry'] = 2.0
    with pytest.raises(ValueError) as refusal:
        spot_model_block(built_surface(tmp_path / 'canned.json', collapsed))
    service.BOOK = None
    assert '5 distinct contracts' in str(refusal.value)
    assert 'FXVol.USD.ZAR carries pillars 0.25/2' in str(refusal.value)
    assert 'term structure' in str(refusal.value), 'a refusal that does not say what was lost'
    assert 'more expiries' in str(refusal.value), 'a refusal without a remedy'

    long_dated = json.loads(dump(desk_smile()))
    points = long_dated['FXVolPrices.USD.ZAR']['instrument']['Points']
    for pillar, quote_type, value in ((0.0, 'ATM', 0.170), (0.25, 'RR', -0.020),
                                      (0.25, 'BF', 0.0055)):
        points.append(dict(points[0], Expiry=2.0, Pillar=pillar, Quote_Type=quote_type,
                           Quoted_Market_Value=value))

    name, block = spot_model_block(built_surface(tmp_path / 'book.json', long_dated))
    try:
        instrument = block['instrument']
        emitted = sorted({point['Expiry_Date'] for point in instrument['European_Options']})
        year = BASE + pd.DateOffset(days=int(Family.fx_ladder().days))

        assert name == 'LogVar2FJModelPrices.ZAR'
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
        spot_model_block(built_surface(tmp_path / 'past.json', past_the_cap))
    service.BOOK = None
    assert '0 distinct contracts' in str(dropped.value)
    assert 'ATM 1 DROPPED - no pillar at or under 1' in str(dropped.value)
    assert '0.25/0.1d 0.5 DROPPED' in str(dropped.value)


# THE ROUND TRIP - the verb authoring a block, installing it through the market seam,
# bootstrapping, and the model repricing the quotes it was fitted to - is a MINUTES-LONG fit, so
# it is read as a document outside the suite: one `/book/model` call landing
# `LogVar2FJModelParameters.ZAR` off the banked USDZAR surface.


def test_a_pair_with_no_built_surface_refuses_at_the_verb(tmp_path):
    """A calibration against a surface the book does not carry is refused ON THE REQUEST THREAD, by
    name and with the remedy - never queued as a minutes-long job whose answer is a typo."""
    path = built_surface(tmp_path / 'book.json')
    try:
        before = path.read_bytes()
        refused = CLIENT.post('/book/model', json={'pair': 'EUR.USD'})
        unnamed = CLIENT.post('/book/model', json={})

        assert refused.status_code == 422
        assert 'FXVol.EUR.USD' in refused.json()['detail']
        assert 'FXVolPrices' in refused.json()['detail'], 'the refusal must name the remedy'
        assert unnamed.status_code == 422 and 'pair' in unnamed.json()['detail']
        # ONE PAIR GRAMMAR across the verbs: the quote verbs' USDZAR and USD/ZAR name the same
        # surface the factor spells USD.ZAR (killing mutation: read the pair as a factor name
        # alone, and 'EURUSD' is refused as the surface FXVol.EURUSD, which nothing writes)
        for spelling in ('EURUSD', 'EUR/USD'):
            other = CLIENT.post('/book/model', json={'pair': spelling})
            assert other.status_code == 422 and 'FXVol.EUR.USD' in other.json()['detail'], spelling
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


def bloomberg_seams(monkeypatch, provision=None, stale=None, terminal=None):
    """Every seam between the verb and the terminal, replaced. The lazy imports inside
    `BloombergJob.run_job` are what lets a patch reach the job: each name is bound off the package
    when the WORKER runs. Returns the definitions the fetch was handed."""
    import derivus_bloomberg
    from derivus_bloomberg import discover, security_map, session

    asked = []

    def fetch_fx_vol(source, definition):
        asked.append(definition)
        return fx_vol_snapshot()

    monkeypatch.setattr(session, 'BloombergSession', terminal or FakeTerminal)
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
    market['Bootstrapper Configuration']['LogVar2FJModelParameters'] = {}
    # a ONE-TOKEN name, the rand priced in this book's base: a two-token block name says what its
    # underlying is priced in and is refused where its `Priced_In` does not say the same, which
    # would refuse before the Quote_Type this gate is about
    market.setdefault('Market Prices', {})['LogVar2FJModelPrices.ZAR'] = {
        'instrument': {'Quote_Type': 'Nonsense', 'Underlying': 'ZAR', 'Discount_Rate': 'USD',
                       'Volatility': 'USD.ZAR', 'European_Options': []}}
    desk.write_text(json.dumps(document, indent=2), newline='\n')
    before = desk.read_bytes()

    bloomberg_seams(monkeypatch)
    result, outcome = ticked()

    assert result['status'] == 'done', result
    assert 'error' not in result, result
    assert outcome['written'] is False
    assert any('Nonsense' in message and 'LogVar2FJModelPrices.ZAR' in message
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


class CannedTerminal:
    """A terminal answering `reference_data_report` in the shape a discovery run records - one row
    per security carrying `ok`, `error` and `fields`, the mid with both sides and the print's own
    date. It is the SESSION as well, so the job's `BloombergSession(...)` lands on it and blpapi is
    never reached; `asked` is every security it was handed, batch by batch, and `stamp` is the day
    every print claims - which is the SNAP a curve is dated by, and what a freshness check reads.
    `stamps` dates ONE security differently, which is how a dead series is put among live ones."""

    def __init__(self, prints, dead=(), stamp='2024-06-27', stamps={}):
        self.prints, self.dead, self.asked, self.stamp = prints, set(dead), [], stamp
        self.stamps = dict(stamps)

    def __call__(self, **options):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *error):
        return False

    def reference_data_report(self, securities, fields):
        self.asked += list(securities)
        return {name: {'ok': name not in self.dead, 'error': None,
                       'fields': {} if name in self.dead else {
                           'PX_LAST': self.prints[name], 'PX_BID': self.prints[name] - 0.01,
                           'PX_ASK': self.prints[name] + 0.01,
                           'LAST_UPDATE_DT': self.stamps.get(name, self.stamp)}}
                for name in securities}


def curve_map():
    """A verified map carrying no `fx_vol` pair at all, so the only work a tick has is the book's
    own curve blocks - which is what makes this reachable with one canned answer."""
    return dict(canned_map(), blocks={'fx_vol': {}})


def curve_tick(monkeypatch, terminal):
    bloomberg_seams(monkeypatch, terminal=terminal,
                    provision=lambda source, as_of, on_batch=None: (curve_map(), False))
    return ticked()


@pytest.fixture
def curve_desk(book):
    """A live book carrying a solved ZAR curve - the verb's own output, which is what a tick keeps
    valued."""
    assert set_up_curve().status_code == 200
    return book


def test_the_tick_values_the_books_curve_rows_and_holds_a_dead_one_out(curve_desk, monkeypatch):
    """THE TICK COVERS THE CURVES, through the same queued job the metronome beats. It asks the
    terminal about the securities the book's OWN rows name, moves the numbers as VALUES - the plan
    standing, so the block is `updated` and nothing is re-authored - and the curve is re-solved in
    the same atomic write, which is why the benchmarks still reprice at par afterwards.

    A security the screen refuses is a different act: the row keeps the number it had, is held out
    with `Use` No and is NAMED in `held_out`, and THAT is a re-authoring, because `Use` is
    structure and no tick may move it. The rest of the strip still solves, one dead ticker being
    one knot fewer where a refused strip would be no curve at all.
    """
    prints = {row['security']: row['quote'] + 0.1 for row in CURVE_ROWS}
    terminal = CannedTerminal(prints)
    before = service.quote_plan(json.loads(curve_desk.read_text())['Calc']['MergeMarketData'][
        'ExplicitMarketData']['Market Prices'][CURVE_BLOCK])

    result, outcome = curve_tick(monkeypatch, terminal)
    block = curve_block(curve_desk)

    assert result['status'] == 'done' and outcome['written'] is True
    assert outcome['updated'] == [CURVE_BLOCK] and outcome['reauthored'] == []
    assert outcome['held_out'] == [] and outcome['rewrote'] == ['InterestRate.ZAR']
    assert sorted(terminal.asked) == sorted(prints), 'the rows name what the tick asks about'
    assert [row['Quoted_Market_Value'] for row in block['Points']] == list(prints.values())
    assert all(row['Timestamp'] == {'.Timestamp': '2024-06-27'} for row in block['Points'])
    assert service.quote_plan({'instrument': block}) == before, 'a value tick moved the plan'
    assert abs(par_residuals(curve_desk)).max() < 1e-6

    dead = CannedTerminal(prints, dead=['SASW5 BGN Curncy'])
    result, outcome = curve_tick(monkeypatch, dead)
    block = curve_block(curve_desk)

    assert result['status'] == 'done' and outcome['written'] is True
    assert outcome['reauthored'] == [CURVE_BLOCK]
    assert any('SASW5 BGN Curncy' in message and 'invalid' in message
               for message in outcome['held_out']), outcome
    assert [(row['Tenor'], row['Use']) for row in block['Points']] == [
        (row['tenor'], 'No' if row['tenor'] == '5Y' else 'Yes') for row in CURVE_ROWS]
    assert outcome['rewrote'] == ['InterestRate.ZAR'], 'the strip without it did not re-solve'

    # a block this emitter cannot read - an older book's, authored before the conventions rode
    # beside the quotes - is NAMED and left standing rather than taking the whole tick down
    values = [row['Quoted_Market_Value'] for row in block['Points']]
    document = json.loads(curve_desk.read_text())
    del document['Calc']['MergeMarketData']['ExplicitMarketData']['Market Prices'][
        CURVE_BLOCK]['instrument']['Spot_Days']
    curve_desk.write_text(json.dumps(document, indent=2), newline='\n')

    moved = CannedTerminal({security: value + 1.0 for security, value in prints.items()})
    result, outcome = curve_tick(monkeypatch, moved)

    assert result['status'] == 'done' and 'error' not in result, result
    assert outcome['held_out'] == ["{} left as it stands - 'Spot_Days'".format(CURVE_BLOCK)]
    assert moved.asked == [], 'a block with no conventions to re-roll under was still asked about'
    assert [row['Quoted_Market_Value'] for row in curve_block(curve_desk)['Points']] == values


def test_a_rolled_base_date_re_authors_the_curve_rather_than_refusing(curve_desk, monkeypatch):
    """A TICK CANNOT CARRY A ROLLED DATE. `Effective_Date` and `Maturity_Date` are structure, so
    the value guard refuses them - rightly, since it cannot tell a rolled date from a mis-authored
    one - and the block is RE-AUTHORED from its own rows and conventions instead, which is what
    putting the conventions on the block bought. The strip three days later is the same benchmarks
    on new dates, and it solves.
    """
    document = json.loads(curve_desk.read_text())
    market = document['Calc']['MergeMarketData']['ExplicitMarketData']
    before = [row['Deal']['Maturity_Date'] for row in market['Market Prices'][
        CURVE_BLOCK]['instrument']['Points']]
    market['System Parameters']['Base_Date'] = {'.Timestamp': '2024-07-01'}
    document['Calc']['Calculation']['Base_Date'] = {'.Timestamp': '2024-07-01'}
    curve_desk.write_text(json.dumps(document, indent=2), newline='\n')

    result, outcome = curve_tick(monkeypatch, CannedTerminal(
        {row['security']: row['quote'] for row in CURVE_ROWS}))
    block = curve_block(curve_desk)

    assert result['status'] == 'done' and outcome['written'] is True
    assert outcome['reauthored'] == [CURVE_BLOCK] and outcome['held_out'] == []
    assert [row['Deal']['Maturity_Date'] for row in block['Points']] != before
    assert [row['Deal']['Effective_Date'] for row in block['Points'][2:]] == [
        {'.Timestamp': '2024-07-01'}] * 4, 'a spot swap that did not re-roll onto the new base date'
    assert outcome['rewrote'] == ['InterestRate.ZAR']


def test_the_date_verb_sets_both_dates_and_re_rolls_every_curve_onto_them(curve_desk):
    """A DESK SETS THE DATE TO ANYTHING. The book carries it twice - the day a curve's benchmarks
    roll off and the day the pricers run on - and this verb moves both together, forward as readily
    as back, re-authoring every curve block on the new day from its own rows and conventions. The
    same benchmarks and the same quotes on new dates, and the strip still solves at par as of the
    day the book now claims, which is what says the re-roll and the re-solve agree.

    A block too old to carry the conventions it was authored under has nothing to re-roll with: it
    is NAMED and left exactly as it stands, the refusal the tick answers with.

    Killing mutations: stamping one of the two dates and not the other (`Calculation.Base_Date` or
    `System Parameters.Base_Date` stands at the old day); handing `curve_quotes` no date, so the
    blocks are re-authored on the day the book already had and the maturities never move.
    """
    before = [row['Deal']['Maturity_Date'] for row in curve_block(curve_desk)['Points']]

    for day in ('2024-09-30', '2024-01-15'):
        answer = CLIENT.post('/book/date', json={'base_date': day})
        outcome = answer.json()
        document = json.loads(curve_desk.read_text())
        block = curve_block(curve_desk)

        assert answer.status_code == 200, outcome
        assert outcome['written'] is True and outcome['base_date'] == day
        assert outcome['reauthored'] == [CURVE_BLOCK] and outcome['held_out'] == []
        assert outcome['rewrote'] == ['InterestRate.ZAR']
        assert document['Calc']['Calculation']['Base_Date'] == {'.Timestamp': day}
        assert document['Calc']['MergeMarketData']['ExplicitMarketData'][
            'System Parameters']['Base_Date'] == {'.Timestamp': day}
        assert [row['Tenor'] for row in block['Points']] == [row['tenor'] for row in CURVE_ROWS]
        assert [row['Quoted_Market_Value'] for row in block['Points']] == [
            row['quote'] for row in CURVE_ROWS], 'a re-roll that moved a quote'
        assert [row['Deal']['Maturity_Date'] for row in block['Points']] != before
        assert abs(par_residuals(curve_desk)).max() < 1e-6

    points = curve_block(curve_desk)['Points']
    document = json.loads(curve_desk.read_text())
    del document['Calc']['MergeMarketData']['ExplicitMarketData']['Market Prices'][
        CURVE_BLOCK]['instrument']['Spot_Days']
    curve_desk.write_text(json.dumps(document, indent=2), newline='\n')
    standing = CLIENT.post('/book/date', json={'base_date': {'.Timestamp': '2024-01-10'}}).json()

    assert standing['held_out'] == ["{} left as it stands - 'Spot_Days'".format(CURVE_BLOCK)]
    assert standing['reauthored'] == [] and standing['base_date'] == '2024-01-10'
    assert curve_block(curve_desk)['Points'] == points, 'a block with no conventions was re-rolled'


def test_a_later_print_rolls_the_books_date_through_the_tick_and_an_older_one_does_not(
        curve_desk, monkeypatch):
    """WHEN WE BOOTSTRAP THE CURVE WE KNOW WHEN IT WAS SNAPPED. The tick authors on the latest
    print it came back with: where that is later than the day the book stands at, both of the
    book's dates roll onto it and the block is RE-AUTHORED there rather than value-ticked, so a
    book fetched with Friday's quotes is dated Friday and its benchmarks mature off Friday.

    An older print is evidence about a QUOTE and never a valuation date, so the second tick moves
    the values alone and leaves the dates where the first one put them - the block `updated`, its
    plan standing.

    Killing mutations: dropping the book's own date from `snap_date`'s floor, so the second tick
    rolls the book back onto 2024-07-02; authoring on the book's date rather than on the snap, so
    the first tick leaves both dates at 2024-06-28 and reports no re-authoring at all.
    """
    prints = {row['security']: row['quote'] for row in CURVE_ROWS}
    before = [row['Deal']['Maturity_Date'] for row in curve_block(curve_desk)['Points']]

    result, outcome = curve_tick(monkeypatch, CannedTerminal(prints, stamp='2024-07-05'))
    document = json.loads(curve_desk.read_text())
    block = curve_block(curve_desk)

    assert result['status'] == 'done' and outcome['written'] is True
    assert outcome['base_date'] == '2024-07-05' and outcome['reauthored'] == [CURVE_BLOCK]
    assert document['Calc']['Calculation']['Base_Date'] == {'.Timestamp': '2024-07-05'}
    assert document['Calc']['MergeMarketData']['ExplicitMarketData'][
        'System Parameters']['Base_Date'] == {'.Timestamp': '2024-07-05'}
    assert all(row['Timestamp'] == {'.Timestamp': '2024-07-05'} for row in block['Points'])
    assert [row['Deal']['Maturity_Date'] for row in block['Points']] != before
    assert abs(par_residuals(curve_desk)).max() < 1e-6

    moved = {security: value + 0.1 for security, value in prints.items()}
    result, outcome = curve_tick(monkeypatch, CannedTerminal(moved, stamp='2024-07-02'))

    assert result['status'] == 'done' and outcome['written'] is True
    assert outcome['base_date'] == '2024-07-05' and outcome['reauthored'] == []
    assert outcome['updated'] == [CURVE_BLOCK], 'a print older than the book re-authored it'
    assert [row['Quoted_Market_Value'] for row in curve_block(curve_desk)['Points']] == list(
        moved.values())
    assert json.loads(curve_desk.read_text())['Calc']['Calculation']['Base_Date'] == {
        '.Timestamp': '2024-07-05'}


def test_a_curve_set_up_off_a_later_print_rolls_the_book_and_every_other_curve(
        curve_desk, monkeypatch):
    """A SNAP IS A DATE, through the verb as through the tick. The desk sets the ZAR curve up
    again quoting no row itself, so the terminal prices every one of them; the prints come back
    dated after the day the book stands at, the book rolls onto them, and the USD curve beside it
    is RE-AUTHORED there too in the same write - two curves dated differently being one of them
    solved for a day nobody asked about. Both still reprice at par afterwards.

    Killing mutations: leaving the other blocks where they stood on a roll, so the USD strip keeps
    its June maturities under a July book; taking the snap off the book's date rather than off the
    prints, so nothing rolls and the ZAR block is authored in June.
    """
    from derivus_bloomberg import session

    assert set_up_curve(rows=USD_ROWS, curve='USD', currency='USD').status_code == 200
    dollars = [row['Deal']['Maturity_Date'] for row in curve_block(curve_desk, 'USD')['Points']]
    monkeypatch.setattr(session, 'BloombergSession', CannedTerminal(
        {row['security']: row['quote'] for row in CURVE_ROWS}, stamp='2024-07-05'))

    answer = set_up_curve(rows=[{'tenor': row['tenor'], 'security': row['security']}
                                for row in CURVE_ROWS])
    outcome = answer.json()
    document = json.loads(curve_desk.read_text())

    assert answer.status_code == 200, outcome
    assert outcome['base_date'] == '2024-07-05' and outcome['block'] == CURVE_BLOCK
    assert outcome['reauthored'] == ['InterestRatePrices.USD', CURVE_BLOCK]
    # the USD FACTOR is unmoved: a week's roll leaves an ON/1Y/2Y/5Y strip on the same year
    # fractions off the same quotes, so a curve is a function of its tenors and not of its dates
    assert outcome['rewrote'] == ['InterestRate.ZAR'] and outcome['held_out'] == []
    assert document['Calc']['Calculation']['Base_Date'] == {'.Timestamp': '2024-07-05'}
    assert [row['Deal']['Maturity_Date']
            for row in curve_block(curve_desk, 'USD')['Points']] != dollars
    assert abs(par_residuals(curve_desk, 'USD')).max() < 1e-6
    assert abs(par_residuals(curve_desk)).max() < 1e-6

    # the read verb says both: the day every block is authored on, and the day each was snapped
    read = CLIENT.get('/book/curve', params={'curve': 'USD'}).json()
    assert read['base_date'] == '2024-07-05'
    assert read['curves']['InterestRatePrices.USD']['snapped'] == '2024-06-28'


def test_a_tick_landing_while_the_terminal_priced_the_rows_refuses_rather_than_rolling_back(
        curve_desk, monkeypatch):
    """A SNAP NEVER ROLLS A BOOK BACK, and the terminal round trip is the one window where it
    could: the verb reads the book, spends minutes pricing its rows, and writes against a book a
    tick may have rolled since. Rather than stamping the day it read, the write refuses BY NAME
    with both days in it and touches nothing - a curve dated behind its own book has a front
    whose accrual already started, which is no benchmark at all.

    The seam is the terminal itself: this one rolls the book forward while it is being asked about,
    which is a tick landing mid-trip with nothing threaded.

    Killing mutation: stamp the day the rows were authored on, and both of the book's dates go
    back to 2024-07-05 with every other block re-rolled onto it.
    """
    from derivus_bloomberg import session

    class TickingTerminal(CannedTerminal):
        def reference_data_report(self, securities, fields):
            document = json.loads(curve_desk.read_text())
            stamp = {'.Timestamp': '2024-08-01'}
            document['Calc']['MergeMarketData']['ExplicitMarketData'][
                'System Parameters']['Base_Date'] = stamp
            document['Calc']['Calculation']['Base_Date'] = stamp
            curve_desk.write_text(json.dumps(document, indent=2), newline='\n')
            return super().reference_data_report(securities, fields)

    monkeypatch.setattr(session, 'BloombergSession', TickingTerminal(
        {row['security']: row['quote'] for row in CURVE_ROWS}, stamp='2024-07-05'))
    points = curve_block(curve_desk)['Points']
    answer = set_up_curve(rows=[{'tenor': row['tenor'], 'security': row['security']}
                                for row in CURVE_ROWS])
    detail = answer.json()['detail']
    document = json.loads(curve_desk.read_text())

    assert answer.status_code == 422, answer.json()
    assert '2024-08-01' in detail and '2024-07-05' in detail and 'post it again' in detail
    # the book stands where the TICK left it, the refused set-up having written nothing
    assert document['Calc']['Calculation']['Base_Date'] == {'.Timestamp': '2024-08-01'}
    assert curve_block(curve_desk)['Points'] == points


#: A curve nothing packaged seeds, so a gate's own scope is its own: two swap years and an
#: overnight fixing, each spelled the way the strip grammar spells one.
GATE_CURVE = {'prefix': 'GATE', 'expect': 'GATE', 'currency': 'ZAR', 'years': [1, 2],
              'overnight': {'security': 'GATEON Index', 'expect': 'GATEON'}}


@pytest.fixture
def vocabulary(tmp_path, monkeypatch):
    """A `DV_HOME` of the gate's own carrying neither seed nor map - so what is read is the
    PACKAGED questionnaire and an unverified home, never this workstation's own files."""
    monkeypatch.setenv('DV_HOME', str(tmp_path / 'home'))
    return tmp_path / 'home'


def author_map(home, verified, rejected={}):
    """A security map the gate authors: `{path: security}` entered under its own path carrying its
    evidence, and a ledger keyed by ticker - the shape `discover.build_map` writes."""
    document = {'schema': 'derivus-bloomberg-map/1', 'generated': '2024-06-28',
                'blocks': {}, 'rejected': rejected}
    for path, security in verified.items():
        node = document['blocks']
        for part in path.split('/')[:-1]:
            node = node.setdefault(part, {})
        node[path.rsplit('/', 1)[-1]] = {'security': security, 'name': security,
                                         'last_update': '2024-06-27', 'verified': '2024-06-28'}
    os.makedirs(home, exist_ok=True)
    (home / 'security_map.json').write_text(json.dumps(document, indent=1), newline='\n')
    return document


def test_the_vocabulary_reads_back_on_a_home_that_has_never_been_verified(book, vocabulary):
    """THE VOCABULARY IS READABLE WITHOUT A TERMINAL. A home carrying neither file reads the
    packaged questionnaire completed the way a curve's conventions are, an EMPTY map rather than a
    refusal, and `provisioned` False - which is what says the candidates are a desk's claim and
    nothing has evidenced them.

    Killing mutations: `provisioned` read off the seed rather than off the map on disk, which a
    packaged-only home then reads as verified; the seed read off the desk's own file alone, which
    a home with none answers empty.
    """
    answer = CLIENT.get('/book/securities').json()

    assert answer['provisioned'] is False and answer['home'] == str(vocabulary)
    assert answer['map'] == {'generated': None, 'blocks': {}, 'rejected': {}}
    assert answer['used'] == [], 'a book with no curve block has no knot to evidence'
    assert answer['seed']['rates']['ZAR']['conventions']['front'] == 'fixings/3M'
    assert 'ZAR-ZARONIA' in answer['seed']['rates'] and 'USDZAR' in answer['seed']['fx_vol']['pairs']

    narrowed = CLIENT.get('/book/securities', params={'block': 'rates'}).json()

    assert set(narrowed['seed']) == {'rates'} and set(narrowed['map']['blocks']) == {'rates'}
    assert narrowed['seed']['rates'] == answer['seed']['rates']


def test_every_knot_names_the_print_it_was_solved_from(book, vocabulary):
    """THE IPV JOIN: a curve set up from hand-quoted rows, read back against a map authored here,
    so each knot carries the security it is quoted off, the print's own timestamp and whatever the
    terminal ever answered about that security - an entry's evidence, the verdict that rejected it,
    or `unmapped` where the map has never heard of it.

    Killing mutations: the join keyed by tenor rather than by security, which the rejected ledger
    then cannot reach since it is keyed by ticker; `unmapped` collapsed into an empty evidence
    block, which reads as a verified quote with nothing recorded.
    """
    assert set_up_curve().status_code == 200
    author_map(vocabulary,
               {'rates/ZAR/fixings/3M': 'JIBA3M Index', 'rates/ZAR/strip/1Y': 'SASW1 BGN Curncy'},
               {'SASW5 BGN Curncy': {'verdict': 'dead', 'name': 'ZAR SWAP QTR (VS 3M) 5Y',
                                     'last_update': '2007-03-26', 'error': None}})
    answer = CLIENT.get('/book/securities').json()
    used = {row['security']: row for row in answer['used']}

    assert answer['provisioned'] is True
    assert [row['tenor'] for row in answer['used']] == [row['tenor'] for row in CURVE_ROWS]
    assert used['JIBA3M Index'] == {
        'curve': 'ZAR', 'tenor': '3M', 'security': 'JIBA3M Index', 'quote': 7.41,
        'timestamp': '2024-06-28', 'use': 'Yes',
        'evidence': {'name': 'JIBA3M Index', 'last_update': '2024-06-27',
                     'verified': '2024-06-28'}}
    assert used['SASW5 BGN Curncy']['evidence'] == {
        'verdict': 'dead', 'name': 'ZAR SWAP QTR (VS 3M) 5Y', 'last_update': '2007-03-26',
        'error': None}
    assert used['SASW10 BGN Curncy']['evidence'] == {'verdict': 'unmapped'}
    assert answer['map']['blocks']['rates']['ZAR']['strip']['1Y']['security'] == 'SASW1 BGN Curncy'


def test_a_vocabulary_entry_is_merged_kept_and_refused_by_name(vocabulary):
    """A SEED ENTRY IS AN AUTHORING ACT on the desk's own file: the packaged questionnaire is the
    base a desk with none starts from, the entry is merged into it, and the answer carries the
    tickers that block now spells. The file a write replaces is kept beside it, and a spec the
    grammar cannot spell refuses BY NAME with the file untouched.

    Killing mutations: the merge validated by json alone, which lets `{'prefix': 'X'}` through and
    breaks the next verification instead; the write landing on the packaged file, which the second
    read then answers as the desk's own.
    """
    written = CLIENT.post('/book/securities', json={'block': 'rates', 'key': 'GATE',
                                                    'entry': GATE_CURVE}).json()
    seed = json.loads((vocabulary / 'seed.json').read_text())

    assert written['written'] is True and written['backup'] is None
    assert {'GATE1 BGN Curncy', 'GATE2 BGN Curncy', 'GATEON Index'} <= set(written['candidates'])
    assert seed['rates']['GATE'] == GATE_CURVE
    assert seed['rates']['ZAR']['prefix'] == 'SASW', 'the packaged questionnaire is the base'

    moved = CLIENT.post('/book/securities', json={'block': 'fx_vol', 'key': 'pillars',
                                                  'entry': [0.25]}).json()
    kept = json.loads(open(moved['backup'], encoding='utf-8').read())

    assert json.loads((vocabulary / 'seed.json').read_text())['fx_vol']['pillars'] == [0.25]
    assert kept['rates']['GATE'] == GATE_CURVE and kept['fx_vol']['pillars'] == [0.1, 0.25]

    before = (vocabulary / 'seed.json').read_bytes()
    malformed = CLIENT.post('/book/securities', json={'block': 'rates', 'key': 'BAD',
                                                      'entry': {'prefix': 'X', 'years': [1]}})
    unknown = CLIENT.post('/book/securities', json={'block': 'curves', 'key': 'ZAR', 'entry': {}})
    nameless = CLIENT.post('/book/securities', json={'entry': {}})

    assert malformed.status_code == 422 and 'rates/BAD' in malformed.json()['detail']
    assert "'expect'" in malformed.json()['detail'], 'a refusal naming no field'
    assert unknown.status_code == 422 and 'fx_vol' in unknown.json()['detail']
    assert nameless.status_code == 422 and '`block`' in nameless.json()['detail']
    assert (vocabulary / 'seed.json').read_bytes() == before

    removed = CLIENT.post('/book/securities', json={'block': 'rates', 'key': 'GATE'}).json()

    assert 'GATE' not in json.loads((vocabulary / 'seed.json').read_text())['rates']
    assert not [name for name in removed['candidates'] if name.startswith('GATE')]


class NamingTerminal(CannedTerminal):
    """The canned terminal answering `NAME` beside the price - what a VERIFICATION reads a drift
    off, where a tick reads the number alone. `names` renames one security, which is the drift a
    recorded entry cannot see for itself."""

    def __init__(self, prints, names={}, **rest):
        super().__init__(prints, **rest)
        self.names = names

    def reference_data_report(self, securities, fields):
        return {name: dict(row, fields=dict(row['fields'], NAME=self.names.get(name, name)))
                for name, row in super().reference_data_report(securities, fields).items()}


def verified(request={}):
    """POST the verify verb, drain the worker, read the outcome off the result the way a poller
    does - the map write rides the run's own Stats, as a tick's book write does."""
    submitted = CLIENT.post('/book/securities/verify', json=request).json()
    service.EXECUTOR.queue.join()
    result = CLIENT.get('/results/{}'.format(submitted['result_id'])).json()
    return result, result.get('stats', {}).get('Securities', {})


def test_a_verification_needs_a_terminal_and_re_verifies_the_scope_it_is_given(
        vocabulary, monkeypatch):
    """ONLY A TERMINAL WRITES THE MAP, so a workstation whose blpapi does not import refuses at
    submission naming the read that still works - no session, no queue, no file.

    With one answering, the scope is the whole act: the entry the map already carries is re-probed
    and its DRIFT named by its own path, the two names the seed spells that the map has never heard
    of are probed once and entered with their evidence, and nothing outside the scope is asked
    about at all. A second run naming securities re-asks about those and grows nothing.

    Killing mutations: the scope dropped, and the whole packaged vocabulary is re-probed - which
    `asked` measures; the recheck reading the entry's own recorded name rather than the terminal's,
    which then never drifts.
    """
    import datetime

    from derivus_bloomberg import session
    from derivus_bloomberg.errors import BloombergUnavailable

    def absent():
        raise BloombergUnavailable('no blpapi on this workstation')

    monkeypatch.setattr(session, 'blpapi_module', absent)
    refused = CLIENT.post('/book/securities/verify', json={'block': 'rates'})

    assert refused.status_code == 422 and 'blpapi' in refused.json()['detail']
    assert 'GET /book/securities' in refused.json()['detail']
    assert not (vocabulary / 'security_map.json').exists()

    monkeypatch.setattr(session, 'blpapi_module', lambda: True)
    assert CLIENT.post('/book/securities', json={'block': 'rates', 'key': 'GATE',
                                                 'entry': GATE_CURVE}).status_code == 200
    author_map(vocabulary, {'rates/GATE/strip/1Y': 'GATE1 BGN Curncy'})
    terminal = NamingTerminal({'GATE1 BGN Curncy': 7.6, 'GATE2 BGN Curncy': 7.9,
                               'GATEON Index': 7.3}, names={'GATE1 BGN Curncy': 'GATE 1Y RENAMED'},
                              stamp=datetime.date.today().isoformat())
    monkeypatch.setattr(session, 'BloombergSession', terminal)
    result, outcome = verified({'block': 'rates', 'key': 'GATE'})
    document = json.loads((vocabulary / 'security_map.json').read_text())
    strip = document['blocks']['rates']['GATE']

    assert result['status'] == 'done' and 'error' not in result, result
    assert outcome['written'] is True and outcome['verified'] == ['GATE1 BGN Curncy']
    assert outcome['added'] == {'GATE2 BGN Curncy': 'live', 'GATEON Index': 'live'}
    assert 'renamed' in outcome['drifted']['rates/GATE/strip/1Y']['drift']
    assert sorted(terminal.asked) == ['GATE1 BGN Curncy', 'GATE2 BGN Curncy', 'GATEON Index']
    assert strip['strip']['2Y']['name'] == 'GATE2 BGN Curncy'
    assert strip['strip']['2Y']['verified'] == datetime.date.today().isoformat()
    assert strip['strip']['1Y']['name'] == 'GATE1 BGN Curncy', 'a recheck rewrote the evidence'
    assert outcome['map'] == str(vocabulary / 'security_map.json')

    terminal.asked.clear()
    result, outcome = verified({'securities': ['GATE2 BGN Curncy', 'NOPE Index']})

    assert result['status'] == 'done' and outcome['added'] == {}
    assert outcome['verified'] == ['GATE2 BGN Curncy'] and outcome['unknown'] == ['NOPE Index']
    assert terminal.asked == ['GATE2 BGN Curncy'], 'a named ticker grew the map'


def test_a_rejected_ticker_is_asked_again_and_a_revived_one_lands_in_the_map(
        vocabulary, monkeypatch):
    """A REJECTION IS ONE DAY'S ANSWER AND NOT A RETIREMENT - a terminal that priced nothing that
    morning, or a name since fixed on its own side, would otherwise never be asked about again.

    So the ledger is re-probed beside the map's own entries, in the same scope: a name that prices
    now moves off the ledger into `blocks` under the path its seed spells, carrying the evidence
    that put it there and named under `revived`, and one that still fails keeps its row rewritten
    from the FRESH answer rather than the one it was rejected on.

    Killing mutations: the recheck scoped to `entries(document)` alone, which leaves `revived`
    empty and the ticker on the ledger; the ledger asked UNSCOPED, which the ZAR row outside the
    scope measures.
    """
    import datetime

    from derivus_bloomberg import session

    today = datetime.date.today().isoformat()
    stale = {'verdict': 'invalid', 'name': None, 'last_update': None, 'error': 'nothing that day'}
    monkeypatch.setattr(session, 'blpapi_module', lambda: True)
    assert CLIENT.post('/book/securities', json={'block': 'rates', 'key': 'GATE',
                                                 'entry': GATE_CURVE}).status_code == 200
    author_map(vocabulary, {'rates/GATE/strip/1Y': 'GATE1 BGN Curncy'},
               rejected={'GATE2 BGN Curncy': dict(stale), 'GATEON Index': dict(stale),
                         'SASW30 BGN Curncy': dict(stale)})
    terminal = NamingTerminal({'GATE1 BGN Curncy': 7.6, 'GATE2 BGN Curncy': 7.9},
                              dead=['GATEON Index'], stamp=today)
    monkeypatch.setattr(session, 'BloombergSession', terminal)
    result, outcome = verified({'block': 'rates', 'key': 'GATE'})
    document = json.loads((vocabulary / 'security_map.json').read_text())

    assert result['status'] == 'done' and 'error' not in result, result
    assert outcome['revived'] == {'rates/GATE/strip/2Y': 'GATE2 BGN Curncy'}
    assert document['blocks']['rates']['GATE']['strip']['2Y'] == {
        'security': 'GATE2 BGN Curncy', 'name': 'GATE2 BGN Curncy',
        'last_update': today, 'verified': today}
    assert 'GATE2 BGN Curncy' not in document['rejected'], 'revived and still rejected'
    assert document['rejected']['GATEON Index'] == {
        'verdict': 'invalid', 'name': 'GATEON Index', 'last_update': None,
        'error': None}, 'a stale ledger row survived its own probe'
    assert document['rejected']['SASW30 BGN Curncy'] == stale
    assert 'SASW30 BGN Curncy' not in terminal.asked, 'the ledger was asked outside the scope'
    # the entry half is quiet while the ledger moves, and a revived name is not then grown again
    assert outcome['drifted'] == {} and outcome['added'] == {}


#: A EUR/JPY cross-currency swap on a USD book: every factor it reaches is one the book lacks, and
#: none of them is a factor the book's own cashflow reaches - which is what a candidate walk has to
#: answer for and a whole-book walk cannot.
XCCY = {'Object': 'MtMCrossCurrencySwapDeal', 'Reference': 'XC1', 'MtM_Side': 'Pay',
        'Pay_Currency': 'EUR', 'Pay_Interest_Rate': 'EUR', 'Pay_Discount_Rate': 'EUR',
        'Pay_Rate_Type': 'Floating', 'Receive_Currency': 'JPY', 'Receive_Interest_Rate': 'JPY',
        'Receive_Discount_Rate': 'JPY', 'Receive_Rate_Type': 'Floating',
        'Principal_Exchange': 'Start_Maturity', 'Effective_Date': BASE,
        'Maturity_Date': BASE + pd.DateOffset(years=3)}

#: The equity binary completed, so it is a deal the booking would take - which is what makes its
#: three unseeded factors a VOCABULARY answer rather than an authoring one.
EQUITY_DEAL = dict(BINARY, Cash_Payoff=AMOUNT)


def dependencies(**request):
    """The walk over a CANDIDATE, as a model asks for it."""
    return CLIENT.post('/book/dependencies', content=dump(request), headers=JSON).json()


def factor_rows(answer):
    return {row['factor']: row for row in answer['factors']}


def test_the_walk_names_what_a_trade_needs(book, vocabulary):
    """THE WALK IS THE ENGINE'S OWN DISCOVERY, re-emitted: a candidate spliced the way the what-if
    splices one, its subtree walked against the book's market data, and every factor answered with
    the seed entry that would supply it - the block, the key, how many securities the seed spells
    and how many a terminal has verified. Nothing is priced and nothing is written.

    A factor type this desk's vocabulary spells nothing for is reported as honestly: `supply` null
    and a note, never a guess at an equity's ticker. And a candidate the BOOKING would refuse is
    refused here in its own words rather than walked - a misspelt type reaches nothing, so a walk
    that answered it would tell a model the market is fine.

    Killing mutations: the walk reading the whole book instead of the candidate - `InterestRate.ZAR`
    and `FxRate.ZAR` are the book's own cashflow's; the pair looked up in one spelling only, which
    leaves `FxRate.JPY` unsuppliable because the seed spells `USDJPY` and not `JPYUSD`; the
    authoring verdict skipped, which walks a `CrossCurrencySwap` clean.
    """
    before = book.read_bytes()
    answer = dependencies(deal=XCCY)
    rows = factor_rows(answer)

    assert answer['base_currency'] == 'USD' and answer['deal_path'] == '1'
    assert [name for name, row in rows.items() if row['status'] == 'missing'] == [
        'FxRate.EUR', 'FxRate.JPY', 'InterestRate.EUR', 'InterestRate.JPY']
    assert rows['InterestRate.EUR']['supply'] == {
        'block': 'rates', 'key': 'EUR', 'securities': 24, 'verified': 0, 'conventions': True}
    # the pair is the one that ROUTES the currency against the base, either spelling of it
    assert rows['FxRate.EUR']['supply'] == {
        'block': 'fx_spot', 'key': 'EURUSD', 'securities': 1, 'verified': 0}
    assert rows['FxRate.JPY']['supply']['key'] == 'USDJPY'
    assert rows['FxRate.USD'] == {'factor': 'FxRate.USD', 'status': 'resolved', 'supply': None}
    assert 'InterestRate.ZAR' not in rows, 'the walk answered for the book, not the candidate'
    assert book.read_bytes() == before

    equity = factor_rows(dependencies(deal=json.loads(dump(EQUITY_DEAL))))

    assert equity['EquityPrice.EQ']['supply'] is None
    assert 'EquityPrice' in equity['EquityPrice.EQ']['note']

    # a curve entry a desk spells without its conventions cannot be authored, and the walk says so
    # rather than answering `true` for a questionnaire nobody wrote
    assert CLIENT.post('/book/securities', json={
        'block': 'rates', 'key': 'GATE', 'entry': GATE_CURVE}).status_code == 200
    gated = dict(json.loads(dump(XCCY)), Pay_Interest_Rate='GATE', Pay_Discount_Rate='GATE')
    supply = factor_rows(dependencies(deal=gated))['InterestRate.GATE']['supply']

    assert supply['conventions'] is False and supply['block'] == 'rates'
    assert book.read_bytes() == before


def test_the_walk_refuses_by_name_what_it_cannot_walk(book, vocabulary):
    """A REFUSAL NAMES THE THING AND THE REMEDY, and a 500 is neither. The candidate body is the
    whole surface: no `deal`, a null one, a `deal_path` beside it, and a deal whose `Object` names
    no type - which `POST /book/deals` refuses in words this verb hands back verbatim.

    Killing mutations: `request['deal']` unguarded, which is a `TypeError` out of the walk and a
    bare `'deal'` for an empty body; the `deal_path` in a POST body ignored, which makes two verbs
    a model uses back to back disagree about one request.
    """
    before = book.read_bytes()
    malformed = {'Object': 'CrossCurrencySwap', 'Reference': 'T', 'Pay_Currency': 'EUR'}
    answers = {name: CLIENT.post('/book/dependencies', content=dump(body), headers=JSON)
               for name, body in (('empty', {}), ('null', {'deal': None}),
                                  ('both', {'deal': XCCY, 'deal_path': '0'}),
                                  ('nameless', {'deal': malformed}))}

    for name, answer in answers.items():
        assert answer.status_code == 422, (name, answer.json())
    assert '`deal`' in answers['empty'].json()['detail']
    assert 'GET /book/dependencies' in answers['null'].json()['detail']
    assert 'names both' in answers['both'].json()['detail']
    assert answers['nameless'].json()['detail'] == CLIENT.post(
        '/book/deals', content=dump({'deal': malformed}), headers=JSON).json()['refused'][0]
    assert 'CrossCurrencySwap' in answers['nameless'].json()['detail']
    assert book.read_bytes() == before


def test_a_subtree_is_a_portfolio(tmp_path, vocabulary):
    """`?deal_path=` IS THE PORTFOLIO. One netting set of a two-set book answers that set's factors
    and no others, which is what makes the walk usable per counterparty rather than per book.

    Killing mutation: the path ignored - both sets then answer the union, and the ZAR set claims
    the USD deal's discount curve.
    """
    document = json.loads(dump(job(deals=())))
    document['Calc']['Deals']['Deals']['Children'] = [
        {'Instrument': {'.Deal': skipped_netting_deal(reference)},
         'Children': [{'Instrument': {'.Deal': json.loads(dump(deal))}}]}
        for reference, deal in (('CP1', CASHFLOW),
                                ('CP2', dict(CASHFLOW, Reference='CF2', Currency='USD',
                                             Discount_Rate='USD')))]
    path = tmp_path / 'book.json'
    path.write_text(json.dumps(document, indent=2), newline='\n')
    service.BOOK = service.Book(str(path))
    try:
        zar = CLIENT.get('/book/dependencies', params={'deal_path': '0'}).json()
        usd = CLIENT.get('/book/dependencies', params={'deal_path': '1'}).json()
        whole = CLIENT.get('/book/dependencies').json()

        assert sorted(factor_rows(zar)) == ['FxRate.USD', 'FxRate.ZAR', 'InterestRate.USD',
                                            'InterestRate.ZAR']
        assert sorted(factor_rows(usd)) == ['FxRate.USD', 'InterestRate.USD']
        assert sorted(factor_rows(whole)) == sorted(set(factor_rows(zar)) | set(factor_rows(usd)))
        assert all(row['status'] == 'resolved' for row in whole['factors'])
    finally:
        service.BOOK = None


def seeded_map(home, *scopes, rejected={}):
    """A security map carrying every candidate the seed spells for each `(block, key)` scope,
    built by the emitter a discovery run builds one with - a workstation whose terminal has already
    answered for exactly those names, so a set-up over them probes nothing. `rejected` puts named
    candidates on the ledger instead, which is what a re-ask is measured against."""
    from derivus_bloomberg import discover

    document = {'schema': 'derivus-bloomberg-map/1', 'generated': '2024-06-28',
                'blocks': {}, 'rejected': {}}
    for scope in scopes:
        seed = service.scoped_seed(service.desk_seed(), *scope)
        grown = discover.build_map(seed, [
            discover.Verdict(candidate, 'rejected' if candidate.security in rejected else 'live',
                             candidate.security, '2024-06-27', None)
            for candidate in discover.candidates_from_seed(seed)], '2024-06-28')
        for block, section in grown['blocks'].items():
            document['blocks'].setdefault(block, {}).update(section)
        document['rejected'].update(grown['rejected'])
    os.makedirs(home, exist_ok=True)
    (home / 'security_map.json').write_text(json.dumps(document, indent=1), newline='\n')
    return document


def seeded_candidates(*scopes):
    """Every security the seed spells in the named scopes, with the NAME a terminal would have to
    answer for each to verify - the `expect` fragments `discover.verify` holds a name to."""
    from derivus_bloomberg import discover

    return {candidate.security: ' '.join(candidate.expect) for scope in scopes
            for candidate in discover.candidates_from_seed(
                service.scoped_seed(service.desk_seed(), *scope))}


def seeded_rows_of(curve):
    """The seeded benchmark rows of one curve, in the order the strip is spelled - what a gate
    names when it wants a ticker of its own to kill."""
    from derivus_bloomberg import ir_curve

    return ir_curve.seeded_rows(service.desk_seed(), curve)


def setup_terminal(monkeypatch, *scopes, stamps={}, **prints):
    """The seams a set-up meets, canned: a terminal answering every security the seed spells in
    `scopes` - at the NAME each candidate expects, so a probe verifies rather than mismatching, and
    at today's clock unless `stamps` dates one of them back - with `prints` naming the spots, and
    the surface fetch handed back the gate's own snapshot for the pair it was asked about.

    THE FRESHNESS CHECK IS NOT STUBBED: `security_map.stale` runs against this terminal's own
    `LAST_UPDATE_DT`, so a set-up gate pays for the check a late print has to trip. blpapi is never
    reached; `asked` is every security the job handed over and `fetched` every surface it took.
    """
    import datetime

    import derivus_bloomberg
    from derivus_bloomberg import session

    named = seeded_candidates(*scopes)
    terminal = NamingTerminal(dict({security: 3.0 for security in named}, **prints), names=named,
                              stamp=datetime.date.today().isoformat(), stamps=stamps)
    terminal.fetched = []
    monkeypatch.setattr(session, 'blpapi_module', lambda: True)
    monkeypatch.setattr(session, 'BloombergSession', terminal)
    monkeypatch.setattr(derivus_bloomberg, 'fetch_fx_vol', lambda source, definition: (
        terminal.fetched.append(definition.pair) or fx_vol_snapshot(definition.pair)))
    return terminal


@pytest.fixture
def desk_setup(tmp_path, monkeypatch):
    """A USD-base book a market can be set up onto: the one-cashflow market with the FX vol
    bootstrapper declared, and a `DV_HOME` of the gate's own carrying neither seed nor map - so the
    vocabulary read is the PACKAGED questionnaire and this workstation's own files are never
    touched."""
    monkeypatch.setenv('DV_HOME', str(tmp_path / 'home'))
    path = tmp_path / 'book.json'
    path.write_text(json.dumps(json.loads(dump(job(sections={
        'Bootstrapper Configuration': {'FXVolSurfaceParameters': {}}}))), indent=2), newline='\n')
    service.BOOK = service.Book(str(path))
    yield path
    service.BOOK = None


def curve_only(path):
    """The gate book's bootstrapper section cut to the curve family. A book declaring a surface
    family it carries no block for is one `Config.bootstrap` complains about - a different gate's
    subject, and noise in any set-up that installs no surface."""
    document = json.loads(path.read_text())
    document['Calc']['MergeMarketData']['ExplicitMarketData']['Bootstrapper Configuration'] = {
        'InterestRate': {'Prices': 'InterestRate'}}
    path.write_text(json.dumps(document, indent=2), newline='\n')


def set_up(request):
    """POST the set-up verb, drain the worker, read the outcome off the result the way a poller
    does - the book write rides the run's own Stats, as a tick's does."""
    submitted = CLIENT.post('/book/setup', json=request).json()
    service.EXECUTOR.queue.join()
    result = CLIENT.get('/results/{}'.format(submitted['result_id'])).json()
    return result, result.get('stats', {}).get('Setup', {})


def test_a_pair_is_set_up_and_then_quotes(desk_setup, tmp_path, monkeypatch):
    """THE WHOLE MOVE IN ONE VERB. A desk names a pair the book has no market for; what comes back
    is a book that quotes it - the surface, the new currency's spot and its curve installed and
    bootstrapped in ONE write, and a collar on that pair solved to zero off what was written.

    THE SPOT IS ON THE ENGINE'S AXIS, and both directions are gated: `EURUSD` prices one euro in
    dollars, so `FxRate.EUR.Spot` IS the print, while `USDJPY` prices one dollar in yen, so
    `FxRate.JPY.Spot` is its reciprocal. `check` names what a trader should look at - a curve set
    up on conventions nobody here declared, and a surface the base currency is neither leg of.

    A second set-up of the same pair asks no terminal and writes nothing: the want-list is empty.
    A pair stated in any spelling of it reaches the same seed entry.

    Killing mutations: the spot installed un-crossed, which puts 157.25 on `FxRate.JPY` and makes
    a yen worth 157 dollars; the walk not narrowed to the pair, which installs the whole book's
    want-list; the check rows dropped, which is a curve priced off conventions nobody read; the
    pair not normalised, which sends `eurzar` to a seed that spells `EURZAR`.
    """
    home = tmp_path / 'home'
    seeded_map(home, ('fx_vol', 'EURZAR'), ('fx_vol', 'USDJPY'), ('fx_spot', 'EURUSD'),
               ('fx_spot', 'USDJPY'), ('rates', 'EUR'), ('rates', 'JPY'))
    terminal = setup_terminal(
        monkeypatch, ('fx_vol', 'EURZAR'), ('fx_vol', 'USDJPY'), ('fx_spot', 'EURUSD'),
        ('fx_spot', 'USDJPY'), ('rates', 'EUR'), ('rates', 'JPY'),
        **{'EURUSD BGN Curncy': 1.0855, 'USDJPY BGN Curncy': 157.25})
    before = CLIENT.get('/book').json()['etag']

    result, outcome = set_up({'pair': 'eur/zar'})
    market = json.loads(desk_setup.read_text())['Calc']['MergeMarketData']['ExplicitMarketData']

    assert result['status'] == 'done' and 'error' not in result, result
    assert outcome['written'] is True and outcome['refused'] == []
    assert outcome['installed'] == ['FXVol.EUR.ZAR', 'FxRate.EUR', 'InterestRate.EUR']
    assert outcome['not_supplied'] == [] and outcome['held_out'] == []
    assert outcome['discovered'] == {'added': {}, 'revived': {}}
    assert sorted(market['Market Prices']) == ['FXVolPrices.EUR.ZAR', 'InterestRatePrices.EUR']
    assert market['Price Factors']['FxRate.EUR'] == {
        'Domestic_Currency': None, 'Spot': 1.0855, 'Interest_Rate': 'EUR'}
    assert market['Bootstrapper Configuration']['InterestRate']['Prices'] == 'InterestRate'
    assert abs(par_residuals(desk_setup, 'EUR')).max() < 1e-6, 'a benchmark that does not reprice'
    assert any('EUR was set up on the conventions this build ships' in row
               for row in outcome['check']), outcome['check']
    assert any(row.startswith('FXVol.EUR.ZAR is a cross') for row in outcome['check'])
    # ONE write: the etag the job answers with is the file's, and nothing else moved it
    assert outcome['etag'] == CLIENT.get('/book').json()['etag'] != before

    result, outcome = set_up({'pair': 'USDJPY'})
    market = json.loads(desk_setup.read_text())['Calc']['MergeMarketData']['ExplicitMarketData']

    assert result['status'] == 'done' and outcome['written'] is True
    assert outcome['installed'] == ['FXVol.USD.JPY', 'FxRate.JPY', 'InterestRate.JPY']
    assert market['Price Factors']['FxRate.JPY']['Spot'] == pytest.approx(1.0 / 157.25)
    assert not [row for row in outcome['check'] if 'cross' in row], 'a USD leg read as a cross'

    terminal.asked.clear()
    standing = desk_setup.read_bytes()
    result, quiet = set_up({'pair': 'EUR.ZAR'})

    assert result['status'] == 'done' and quiet['written'] is False and quiet['installed'] == []
    # EVERY answer carries the same keys, so a refusal is read exactly as a landing is
    assert quiet['refused'] == [] and quiet['not_supplied'] == [] and quiet['held_out'] == []
    assert quiet['check'] == [] and quiet['discovered'] == {'added': {}, 'revived': {}}
    assert terminal.asked == [], 'a set-up with nothing missing asked the terminal'
    assert desk_setup.read_bytes() == standing

    # and the market that was set up prices the trade it was set up for
    spot = (market['Price Factors']['FxRate.EUR']['Spot']
            / market['Price Factors']['FxRate.ZAR']['Spot'])
    quote = quote_of('ZeroCostCollar', {'pair': 'EURZAR', 'expiry': '1Y', 'notional': AMOUNT,
                                        'notional_currency': 'EUR', 'floor': spot * 0.95})

    assert len(quote['legs']) == 2 and quote['legs'][0]['strike_market'] == pytest.approx(
        spot * 0.95)
    assert abs(quote['net']) < max(abs(leg['premium']) for leg in quote['legs']) * 1e-4


def test_a_set_up_a_bootstrap_complains_about_writes_nothing(desk_setup, tmp_path, monkeypatch):
    """A COMPLAINT REFUSES THE WHOLE WRITE and hands its messages back verbatim - the surface, the
    spot and the curve all fetched and authored, and the file byte-identical afterwards. Which is
    also what says the install is ONE write: a set-up that landed the spots or the curve before it
    bootstrapped would leave that half on disk.

    The complaint is a configured family whose quote block the engine cannot read, so the run says
    so in its own words. The map is not rewritten either - this desk's terminal had already
    verified every name in scope, and a map whose content would not change is left alone, which is
    read off its MTIME because two writers of the same document produce the same bytes.

    AND A BARE `KeyError` IS STILL A SENTENCE: a standing block the emitter cannot read is held out
    by name and still reaches the bootstrap, which asks for a key nothing wrote.

    Killing mutation: that error handed back as its repr, which names a model nothing to act on.
    """
    home = tmp_path / 'home'
    mapped = (home / 'security_map.json')
    seeded_map(home, ('fx_vol', 'EURZAR'), ('fx_spot', 'EURUSD'), ('rates', 'EUR'))
    setup_terminal(monkeypatch, ('fx_vol', 'EURZAR'), ('fx_spot', 'EURUSD'), ('rates', 'EUR'),
                   **{'EURUSD BGN Curncy': 1.0855})
    document = json.loads(desk_setup.read_text())
    market = document['Calc']['MergeMarketData']['ExplicitMarketData']
    market['Bootstrapper Configuration']['LogVar2FJModelParameters'] = {}
    market.setdefault('Market Prices', {})['LogVar2FJModelPrices.ZAR'] = {
        'instrument': {'Quote_Type': 'Nonsense', 'Underlying': 'ZAR', 'Discount_Rate': 'USD',
                       'Volatility': 'USD.ZAR', 'European_Options': []}}
    desk_setup.write_text(json.dumps(document, indent=2), newline='\n')
    before, stamped = desk_setup.read_bytes(), mapped.stat().st_mtime_ns

    result, outcome = set_up({'pair': 'EURZAR'})

    assert result['status'] == 'done' and 'error' not in result, result
    assert outcome['written'] is False and outcome['installed'] == []
    assert any('Nonsense' in message and 'LogVar2FJModelPrices.ZAR' in message
               for message in outcome['refused']), outcome
    assert outcome['check'] == [], 'a set-up that wrote nothing told a trader to go and look'
    assert desk_setup.read_bytes() == before, 'a refused set-up moved the book'
    assert mapped.stat().st_mtime_ns == stamped, 'a map with nothing to add was rewritten'

    del market['Bootstrapper Configuration']['LogVar2FJModelParameters']
    del market['Market Prices']['LogVar2FJModelPrices.ZAR']
    # an older book's curve block: no conventions beside the quotes and no deal under the row
    market['Market Prices']['InterestRatePrices.OLD'] = {'instrument': {
        'Currency': 'USD', 'Discount_Rate': '', 'Points': [
            {'Tenor': '1Y', 'Security': 'NOSUCH Index', 'Quoted_Market_Value': 1.0,
             'Use': 'Yes'}]}}
    desk_setup.write_text(json.dumps(document, indent=2), newline='\n')
    before = desk_setup.read_bytes()

    result, outcome = set_up({'pair': 'EURZAR'})

    assert result['status'] == 'done' and outcome['written'] is False, result
    assert outcome['refused'] == ['the bootstrap asked for Deal and this market does not carry it']
    assert [row for row in outcome['held_out'] if row.startswith('InterestRatePrices.OLD left as')]
    assert desk_setup.read_bytes() == before, 'a refused set-up moved the book'


def test_a_late_print_refuses_the_whole_set_up(desk_setup, tmp_path, monkeypatch):
    """A LATE PRINT REFUSES THE TRIP, as it does on the tick. The check runs over every wanted
    surface AND every wanted spot BEFORE anything is fetched, so a dead series never reaches a
    book: `written` false, the refusal naming the security and its date, nothing fetched, and the
    file byte-identical - not a curve and a spot installed beside a surface that never arrived.

    The spot half is gated the same way, since a spot is the number every deal in the book
    reprices off.

    Killing mutation: the freshness check deleted, or moved after the fetch - the curve and the
    spot then install and the answer reads `written: true` with a `refused` list beside it.
    """
    home = tmp_path / 'home'
    scopes = (('fx_vol', 'EURZAR'), ('fx_spot', 'EURUSD'), ('rates', 'EUR'))
    seeded_map(home, *scopes)
    terminal = setup_terminal(monkeypatch, *scopes, stamps={'EURZAR25R1Y BGN Curncy': '2019-01-01'},
                              **{'EURUSD BGN Curncy': 1.0855})
    before = desk_setup.read_bytes()

    result, outcome = set_up({'pair': 'EURZAR'})

    assert result['status'] == 'done' and 'error' not in result, result
    assert outcome['written'] is False and outcome['installed'] == []
    assert outcome['refused'] == ['EURZAR25R1Y BGN Curncy is stale - 2019-01-01']
    assert terminal.fetched == [], 'a surface was fetched off a book that was never written'
    assert desk_setup.read_bytes() == before

    terminal.stamps = {'EURUSD BGN Curncy': '2019-01-01'}
    result, outcome = set_up({'pair': 'EURZAR'})

    assert result['status'] == 'done' and outcome['written'] is False
    assert outcome['refused'] == ['EURUSD BGN Curncy is stale - 2019-01-01']
    assert terminal.fetched == [] and desk_setup.read_bytes() == before


def test_a_want_nothing_seeds_is_named_beside_what_was_installed(desk_setup, tmp_path,
                                                                 monkeypatch):
    """THE ANSWER'S THREE FACTS ARE INDEPENDENT. A deal reaching a euro cashflow and an equity:
    the euro half can be supplied and is installed, the equity half cannot and is named under
    `not_supplied` with the factor and the reason, and `refused` stays empty because the WRITE was
    not refused. `installed` is exactly what landed - the equity is not in it.

    THE TWO KINDS OF `not_supplied` ARE BOTH HERE: a factor nothing in the vocabulary spells at
    all, and one it spells but cannot author - a curve entry a desk wrote as a spelling with no
    conventions, which the seed's own reader refuses naming every field it lacks. The second is
    the one `installed` can lie about, because it HAS a supply and so is on the want-list.

    Killing mutations: `installed` listing the whole want-list, which claims a curve the book does
    not carry; the unsuppliable rows folded into `refused`, which tells a model the file did not
    move when it did.
    """
    assert CLIENT.post('/book/securities', json={
        'block': 'rates', 'key': 'GATE', 'entry': GATE_CURVE}).status_code == 200
    scopes = (('fx_spot', 'EURUSD'), ('rates', 'EUR'), ('rates', 'GATE'))
    seeded_map(tmp_path / 'home', *scopes)
    setup_terminal(monkeypatch, *scopes, **{'EURUSD BGN Curncy': 1.0855})
    curve_only(desk_setup)
    euro = dict(json.loads(dump(CASHFLOW)), Reference='CFE', Currency='EUR', Discount_Rate='EUR')
    gated = dict(json.loads(dump(CASHFLOW)), Reference='CFG', Discount_Rate='GATE')
    mixed = {'Object': 'NettingCollateralSet', 'Reference': 'MIX', 'Netted': 'True',
             'Collateralized': 'False', 'Settlement_Currency': '',
             'Children': [{'Instrument': {'.Deal': deal}} for deal in
                          (euro, gated, json.loads(dump(EQUITY_DEAL)))]}

    result, outcome = set_up({'deal': mixed})
    market = json.loads(desk_setup.read_text())['Calc']['MergeMarketData']['ExplicitMarketData']
    reasons = {row['factor']: row['reason'] for row in outcome['not_supplied']}

    assert result['status'] == 'done' and 'error' not in result, result
    assert outcome['written'] is True and outcome['refused'] == []
    assert outcome['installed'] == ['FxRate.EUR', 'InterestRate.EUR']
    assert sorted(reasons) == ['DividendRate.EQ', 'EquityPrice.EQ', 'EquityPriceVol.EQ',
                               'InterestRate.GATE']
    assert 'this desk seeds no EquityPrice' in reasons['EquityPrice.EQ']
    assert 'GATE' in reasons['InterestRate.GATE'] and 'spot_days' in reasons['InterestRate.GATE']
    assert 'EquityPrice.EQ' not in market['Price Factors']
    assert 'InterestRate.GATE' not in market['Price Factors'], 'a curve nobody could author'
    assert sorted(market['Market Prices']) == ['InterestRatePrices.EUR']


def test_a_set_up_holds_out_the_benchmark_the_screen_refuses(desk_setup, tmp_path, monkeypatch):
    """ONE DEAD TICKER IS ONE KNOT FEWER, never a refused strip: the row keeps its place, is held
    out by name in `held_out`, and the curve solves on what is left - the curve verb's own rule,
    and the reason a benchmark is screened rather than checked for freshness with the surfaces.

    Killing mutation: `held_out` dropped from the answer, which leaves a desk no way to know the
    curve it just set up is a knot short.
    """
    scopes = (('fx_vol', 'EURZAR'), ('fx_spot', 'EURUSD'), ('rates', 'EUR'))
    seeded_map(tmp_path / 'home', *scopes)
    terminal = setup_terminal(monkeypatch, *scopes, **{'EURUSD BGN Curncy': 1.0855})
    terminal.dead = {'EESWE10 BGN Curncy'}

    result, outcome = set_up({'pair': 'EURZAR'})
    block = curve_block(desk_setup, 'EUR')

    assert result['status'] == 'done' and outcome['written'] is True, outcome
    assert outcome['installed'] == ['FXVol.EUR.ZAR', 'FxRate.EUR', 'InterestRate.EUR']
    assert [row for row in outcome['held_out'] if 'EESWE10 BGN Curncy' in row], outcome['held_out']
    assert [(row['Tenor'], row['Use']) for row in block['Points'] if row['Use'] != 'Yes'] == [
        ('10Y', 'No')]
    assert any('held out' in row for row in outcome['check'])
    assert abs(par_residuals(desk_setup, 'EUR')).max() < 1e-6, 'the strip without it did not solve'


def test_the_snap_sets_the_date_for_a_set_up(desk_setup, tmp_path, monkeypatch):
    """THE SNAP SETS THE DATE. A curve authored off a terminal carries the print's own clock, so a
    book standing BEHIND those prints rolls onto them - both of its dates - and every curve block
    it already carries is re-authored there in the same write. A book standing AHEAD of them does
    not roll back: an old print is evidence about a quote, not a valuation date.

    And a book that rolls forward WHILE the terminal is pricing refuses by name rather than
    stamping the day it read, which is the one window where a set-up could date a curve behind its
    own book.

    Killing mutations: the snap never advancing the date, so the block is authored on the day the
    book stood at; the roll-back guard removed, which stamps 2024-06-28 onto a book at 2030.
    """
    from derivus_bloomberg import session

    scopes = (('fx_vol', 'EURZAR'), ('fx_spot', 'EURUSD'), ('rates', 'EUR'))
    wider = scopes + (('fx_vol', 'USDZAR'), ('fx_vol', 'USDJPY'), ('fx_spot', 'USDJPY'),
                      ('rates', 'JPY'), ('rates', 'ZAR'))
    seeded_map(tmp_path / 'home', *scopes, ('fx_vol', 'USDZAR'))
    terminal = setup_terminal(monkeypatch, *wider, **{'EURUSD BGN Curncy': 1.0855,
                                                      'USDJPY BGN Curncy': 157.25})
    curve_only(desk_setup)
    assert set_up_curve().status_code == 200, 'a standing curve for the roll to re-author'
    snapped = terminal.stamp

    result, outcome = set_up({'pair': 'EURZAR'})
    document = json.loads(desk_setup.read_text())
    stamps = document['Calc']['MergeMarketData']['ExplicitMarketData']['System Parameters']

    assert result['status'] == 'done' and outcome['written'] is True, result
    assert outcome['base_date'] == snapped
    assert stamps['Base_Date'] == {'.Timestamp': snapped}
    assert document['Calc']['Calculation']['Base_Date'] == {'.Timestamp': snapped}
    assert 'InterestRatePrices.ZAR' in outcome['reauthored'], 'a standing curve left behind'

    # a book that rolls FORWARD while the terminal is pricing: the write refuses by name with both
    # days in it rather than stamping the day it read
    class Rolling(NamingTerminal):
        def reference_data_report(self, securities, fields):
            CLIENT.post('/book/date', json={'base_date': '2030-01-02'})
            return super().reference_data_report(securities, fields)

    seeded_map(tmp_path / 'home', *wider)
    monkeypatch.setattr(session, 'BloombergSession', Rolling(
        terminal.prints, names=terminal.names, stamp=terminal.stamp))
    result, outcome = set_up({'pair': 'USDJPY'})

    assert result['status'] == 'done' and outcome['written'] is False, outcome
    assert any('2030-01-02' in message and snapped in message
               for message in outcome['refused']), outcome['refused']

    # and a book standing AHEAD of the prints keeps its own day
    monkeypatch.setattr(session, 'BloombergSession', terminal)
    result, outcome = set_up({'pair': 'USDZAR'})

    assert result['status'] == 'done' and outcome['written'] is True, outcome
    assert outcome['base_date'] == '2030-01-02', 'a snap rolled the book backwards'


def test_a_set_up_reads_the_curve_its_own_seed_verified(desk_setup, tmp_path, monkeypatch):
    """ONE SEED READER PER JOB. A desk that restates a curve's SPELLING without restating its
    conventions is the legacy case the seed's own fallback exists for, and it is exactly where two
    readings diverge: the want, the count, the discovery and the strip that is PRICED must all be
    the one merged entry - the desk's spelling over the packaged conventions - or the map is grown
    with one family of tickers and the curve built out of another, which the Securities screen
    then reads as `unmapped` for every knot.

    Killing mutation: the strip read off `seeded_rates`, which takes the desk entry only where it
    declares conventions - the map holds `EUSA*` and the block is authored from `EESWE*`.
    """
    assert CLIENT.post('/book/securities', json={
        'block': 'rates', 'key': 'EUR', 'entry': {'prefix': 'EUSA', 'expect': 'EUR SWAP (ESTR)',
                                                  'years': [1, 2, 5, 10], 'weeks': ['1W'],
                                                  'overnight': {'security': 'ESTRON Index',
                                                                'expect': 'ESTR'}}}).status_code \
        == 200
    scopes = (('fx_vol', 'EURZAR'), ('fx_spot', 'EURUSD'), ('rates', 'EUR'))
    seeded_map(tmp_path / 'home', *scopes)
    terminal = setup_terminal(monkeypatch, *scopes, **{'EURUSD BGN Curncy': 1.0855})

    result, outcome = set_up({'pair': 'EURZAR'})
    rows = curve_block(desk_setup, 'EUR')['Points']

    assert result['status'] == 'done' and outcome['written'] is True, outcome
    assert {row['Security'] for row in rows} <= set(terminal.prints), 'a ticker never verified'
    assert [row['Security'] for row in rows if row['Tenor'] == '10Y'] == ['EUSA10 BGN Curncy']
    assert not [row for row in rows if row['Security'].startswith('EESWE')]
    # the conventions are the packaged ones the desk did not restate, which is what makes the
    # spelling it DID restate usable at all
    assert curve_block(desk_setup, 'EUR')['Compounding'] == 'OIS'


def test_a_set_up_installs_no_curve_the_book_does_not_discount_on(tmp_path, monkeypatch):
    """A PAIR'S WANT-LIST IS THE WALK, not a name. The book discounts its rand on `ZAR-ZARONIA`,
    so a EURZAR set-up needs the euro's market and nothing else: reading `InterestRate.ZAR` off the
    currency code would install a second rand curve nothing reads and pay the terminal for a whole
    strip to do it.

    Killing mutation: the want-list computed from `Price Factors` membership rather than from the
    engine's own walk of a vanilla option on the pair.
    """
    factors = dict(FACTORS, **{
        'FxRate.ZAR': dict(FACTORS['FxRate.ZAR'], Interest_Rate='ZAR-ZARONIA'),
        'InterestRate.ZAR-ZARONIA': FACTORS['InterestRate.ZAR']})
    del factors['InterestRate.ZAR']
    monkeypatch.setenv('DV_HOME', str(tmp_path / 'home'))
    path = tmp_path / 'book.json'
    path.write_text(json.dumps(json.loads(dump(job(deals=(), factors=factors, sections={
        'Bootstrapper Configuration': {'FXVolSurfaceParameters': {}}}))), indent=2), newline='\n')
    service.BOOK = service.Book(str(path))
    try:
        scopes = (('fx_vol', 'EURZAR'), ('fx_spot', 'EURUSD'), ('rates', 'EUR'), ('rates', 'ZAR'))
        seeded_map(tmp_path / 'home', *scopes)
        terminal = setup_terminal(monkeypatch, *scopes, **{'EURUSD BGN Curncy': 1.0855})

        result, outcome = set_up({'pair': 'EURZAR'})
        market = json.loads(path.read_text())['Calc']['MergeMarketData']['ExplicitMarketData']

        assert result['status'] == 'done' and outcome['written'] is True, outcome
        assert outcome['installed'] == ['FXVol.EUR.ZAR', 'FxRate.EUR', 'InterestRate.EUR']
        assert 'InterestRate.ZAR' not in market['Price Factors'], 'a second rand curve'
        assert sorted(market['Market Prices']) == ['FXVolPrices.EUR.ZAR',
                                                   'InterestRatePrices.EUR']
        assert not [name for name in terminal.asked if name.startswith('SASW')], \
            'the terminal was asked for a whole rand strip nothing reads'
    finally:
        service.BOOK = None


def test_a_set_up_discovers_only_what_the_map_has_never_heard_of(desk_setup, tmp_path,
                                                                 monkeypatch):
    """DISCOVERY IS SCOPED TO WHAT WOULD SUPPLY THE WANT. A home whose map has never heard of the
    pair probes exactly the names the seed spells for its three supplying entries - the surface,
    the spot pair and the curve - and no others, which is the difference between one pair's names
    and the six hundred the packaged questionnaire spells.

    A home that already holds them probes NONE and does not rewrite the map; one holding a REJECTED
    name in scope re-asks exactly that name, because a rejection is one day's answer.

    Killing mutations: the unscoped seed handed to `discover.extend` - `asked` then carries
    `USDZARV1W BGN Curncy` and every swaption the questionnaire names; the map rewritten when
    nothing was discovered, which is invisible in the bytes and plain in the mtime.
    """
    scopes = (('fx_vol', 'EURZAR'), ('fx_spot', 'EURUSD'), ('rates', 'EUR'))
    scoped = seeded_candidates(*scopes)
    terminal = setup_terminal(monkeypatch, ('fx_vol', 'USDZAR'), *scopes,
                              **{'EURUSD BGN Curncy': 1.0855})
    # the home already carries USDZAR: what the EURZAR set-up probes is its OWN scope regardless
    mapped = tmp_path / 'home' / 'security_map.json'
    seeded_map(tmp_path / 'home', ('fx_vol', 'USDZAR'))

    result, outcome = set_up({'pair': 'EURZAR'})
    grown = json.loads(mapped.read_text())

    assert result['status'] == 'done' and outcome['written'] is True, result
    assert set(terminal.asked) == set(scoped), 'the unscoped vocabulary reached the terminal'
    assert len(outcome['discovered']['added']) == len(scoped)
    assert set(grown['blocks']['fx_vol']['EURZAR']['quotes']['1Y']) == {
        'ATM', 'RR_0.10', 'BF_0.10', 'RR_0.25', 'BF_0.25'}
    # a grown pair lands READABLE: `expiries` is the block's own metadata rather than an entry, and
    # a surface fetch reads it before it reads a quote
    assert grown['blocks']['fx_vol']['EURZAR']['expiries']['1Y'] == 1.0

    terminal.asked.clear()
    stamped = mapped.stat().st_mtime_ns
    result, again = set_up({'pair': 'USDZAR'})

    assert result['status'] == 'done' and again['installed'] == ['FXVol.USD.ZAR'], again
    assert again['discovered'] == {'added': {}, 'revived': {}}, 'a verified map was re-probed'
    assert [name for name in terminal.asked if name.startswith('USDZARV')], 'no freshness check'
    assert not [name for name in terminal.asked if name.endswith('Curncy')
                and 'USDZAR' not in name], 'a name outside the scope was probed'
    assert mapped.stat().st_mtime_ns == stamped, 'a map with nothing to add was rewritten'


def test_a_rejected_name_in_scope_is_asked_again(desk_setup, tmp_path, monkeypatch):
    """A REJECTION IS ONE DAY'S ANSWER. A map that holds every name of the scope but carries one on
    its ledger re-asks exactly that one - not the verified entries beside it, and nothing outside
    the scope - and a name that prices now lands in the map under `revived`.

    Killing mutation: the ledger asked only where the entries are incomplete, which leaves a
    rejection standing forever on a scope that is otherwise fully verified.
    """
    scopes = (('fx_vol', 'EURZAR'), ('fx_spot', 'EURUSD'), ('rates', 'EUR'))
    dead = 'EESWE10 BGN Curncy'
    mapped = tmp_path / 'home' / 'security_map.json'
    seeded_map(tmp_path / 'home', *scopes, ('fx_vol', 'USDZAR'), rejected={dead})
    terminal = setup_terminal(monkeypatch, ('fx_vol', 'USDZAR'), *scopes,
                              **{'EURUSD BGN Curncy': 1.0855})

    result, outcome = set_up({'pair': 'EURZAR'})
    grown = json.loads(mapped.read_text())

    assert result['status'] == 'done' and outcome['written'] is True, outcome
    assert outcome['discovered']['revived'] == {'rates/EUR/strip/10Y': dead}
    assert dead not in grown['rejected'] and outcome['discovered']['added'] == {}
    assert dead in terminal.asked
    assert not [name for name in terminal.asked if name.startswith('USDZAR')], 'out of scope'


def test_a_set_up_refuses_at_submission_and_a_routine_tick_never_discovers(
        desk_setup, tmp_path, monkeypatch):
    """ONLY A TERMINAL SETS A MARKET UP, so a workstation whose blpapi does not import refuses at
    SUBMISSION by name - no session, no queue, nothing written. A pair this desk's vocabulary does
    not spell refuses there too, naming the verb that adds it, and one stated BACKWARDS is told the
    spelling the seed carries rather than sent to add a duplicate. A `deal` the booking verb would
    refuse is refused there, in its words.

    AND A ROUTINE TICK NEVER GROWS THE MAP. The cadence values the book's spots off the pairs the
    map already carries; a currency it verified none for is named under `unrouted` and keeps the
    spot it had, and the map file does not move.
    """
    from derivus_bloomberg import session
    from derivus_bloomberg.errors import BloombergUnavailable

    def absent():
        raise BloombergUnavailable('no blpapi on this workstation')

    home = tmp_path / 'home'
    seeded_map(home, ('fx_vol', 'EURZAR'))
    before, stamped = desk_setup.read_bytes(), (home / 'security_map.json').stat().st_mtime_ns
    monkeypatch.setattr(session, 'blpapi_module', absent)
    refused = CLIENT.post('/book/setup', json={'pair': 'EURZAR'})

    assert refused.status_code == 422 and 'blpapi' in refused.json()['detail']
    assert desk_setup.read_bytes() == before

    monkeypatch.setattr(session, 'blpapi_module', lambda: True)
    unseeded = CLIENT.post('/book/setup', json={'pair': 'EURNOK'})
    backwards = CLIENT.post('/book/setup', json={'pair': 'ZAREUR'})
    ambiguous = CLIENT.post('/book/setup', json={'pair': 'EURZAR', 'deal_path': '0'})
    nameless = CLIENT.post('/book/setup', content=dump(
        {'deal': {'Object': 'CrossCurrencySwap', 'Reference': 'T'}}), headers=JSON)

    assert unseeded.status_code == 422 and 'EURNOK' in unseeded.json()['detail']
    assert '/book/securities' in unseeded.json()['detail']
    assert backwards.status_code == 422
    assert backwards.json()['detail'] == 'this desk seeds EURZAR rather than ZAREUR - ask for that'
    assert ambiguous.status_code == 422 and 'deal_path' in ambiguous.json()['detail']
    assert nameless.status_code == 422 and 'CrossCurrencySwap' in nameless.json()['detail']

    bloomberg_seams(monkeypatch, terminal=CannedTerminal({}),
                    provision=lambda source, as_of, on_batch=None: (
                        json.loads((home / 'security_map.json').read_text()), False))
    metronome = service.Metronome(60.0, {'pairs': []})
    metronome.beat()
    service.EXECUTOR.queue.join()
    outcome = CLIENT.get('/results/{}'.format(metronome.pending)).json()['stats']['Bloomberg']

    assert outcome['written'] is True and outcome['spots'] == {}
    assert any('ZAR against USD' in message for message in outcome['unrouted']), outcome
    assert (home / 'security_map.json').stat().st_mtime_ns == stamped


def routed_prints(pair='USDZAR', value=19.0):
    """A number for every security `routed_map` verifies - the freshness check asks about the
    surface's names beside the route's, and a terminal answers for what it is asked."""
    from derivus_bloomberg.security_map import entries

    return {entry['security']: value for _, entry in entries(routed_map(pair))}


def today():
    import datetime

    return datetime.date.today().isoformat()



def test_nothing_lands_that_would_depend_on_a_block_the_write_will_not_carry(
        desk_setup, tmp_path, monkeypatch):
    """THE HOLD RULE. A curve the terminal leaves under the emitter's floor is `not_supplied` in
    the SCREEN's own words; the currency's spot is held with it, and the surface that leg belongs
    to is held with the spot. Nothing lands, because nothing left could stand on its own: a book
    that took the spot would carry an `FxRate` pointing at a curve it does not have, and the deal
    the set-up was asked to build a market for still would not book.

    Killing mutations: the hold not carried from the curve to the spot, which installs a dangling
    `FxRate.EUR`; the sub-floor strip reported as a refusal of the write rather than as a row, which
    loses the reason with it.
    """
    scopes = (('fx_vol', 'EURZAR'), ('fx_spot', 'EURUSD'), ('rates', 'EUR'))
    seeded_map(tmp_path / 'home', *scopes)
    terminal = setup_terminal(monkeypatch, *scopes, **{'EURUSD BGN Curncy': 1.0855})
    # every EUR benchmark but one is dead: one believed point against a floor of two
    terminal.dead = {row['security'] for row in seeded_rows_of('EUR')[1:]}
    before = desk_setup.read_bytes()

    result, outcome = set_up({'pair': 'EURZAR'})
    market = json.loads(desk_setup.read_text())['Calc']['MergeMarketData']['ExplicitMarketData']
    reasons = {row['factor']: row['reason'] for row in outcome['not_supplied']}

    assert result['status'] == 'done' and 'error' not in result, result
    assert outcome['written'] is False and outcome['installed'] == []
    assert outcome['refused'] == [], 'a want that cannot be supplied is not a refused write'
    assert sorted(reasons) == ['FXVol.EUR.ZAR', 'FxRate.EUR', 'InterestRate.EUR']
    assert 'kept 1 of 24 seeded benchmarks' in reasons['InterestRate.EUR']
    assert reasons['FxRate.EUR'] == 'held with InterestRate.EUR'
    assert reasons['FXVol.EUR.ZAR'] == 'held with FxRate.EUR'
    assert 'FxRate.EUR' not in market['Price Factors'], 'a spot against a currency with no curve'
    assert 'FXVolPrices.EUR.ZAR' not in market.get('Market Prices', {})
    assert desk_setup.read_bytes() == before


def test_what_the_write_can_complete_still_lands(desk_setup, tmp_path, monkeypatch):
    """AND THE OTHER HALF OF THE RULE: a deal wanting two currencies where one strip is dead lands
    the other one COMPLETE. The yen's curve and spot install; the euro's curve is `not_supplied`
    and its spot is held with it, so the book gains a market it can price rather than two halves.

    Killing mutation: the hold taken as a refusal of the whole write, which leaves the yen market
    unbuilt because the euro's strip was dead.
    """
    scopes = (('fx_spot', 'EURUSD'), ('rates', 'EUR'), ('fx_spot', 'USDJPY'), ('rates', 'JPY'))
    seeded_map(tmp_path / 'home', *scopes)
    terminal = setup_terminal(monkeypatch, *scopes, **{'EURUSD BGN Curncy': 1.0855,
                                                       'USDJPY BGN Curncy': 157.25})
    terminal.dead = {row['security'] for row in seeded_rows_of('EUR')[1:]}
    curve_only(desk_setup)
    deals = [dict(json.loads(dump(CASHFLOW)), Reference='CF' + currency, Currency=currency,
                  Discount_Rate=currency) for currency in ('EUR', 'JPY')]

    result, outcome = set_up({'deal': {
        'Object': 'NettingCollateralSet', 'Reference': 'TWO', 'Netted': 'True',
        'Collateralized': 'False', 'Settlement_Currency': '',
        'Children': [{'Instrument': {'.Deal': deal}} for deal in deals]}})
    market = json.loads(desk_setup.read_text())['Calc']['MergeMarketData']['ExplicitMarketData']
    reasons = {row['factor']: row['reason'] for row in outcome['not_supplied']}

    assert result['status'] == 'done' and 'error' not in result, result
    assert outcome['written'] is True and outcome['refused'] == []
    assert outcome['installed'] == ['FxRate.JPY', 'InterestRate.JPY']
    assert sorted(reasons) == ['FxRate.EUR', 'InterestRate.EUR']
    assert reasons['FxRate.EUR'] == 'held with InterestRate.EUR'
    assert sorted(market['Market Prices']) == ['InterestRatePrices.JPY']
    assert 'FxRate.EUR' not in market['Price Factors']
    assert market['Price Factors']['FxRate.JPY']['Interest_Rate'] == 'JPY'


def test_a_surface_is_held_with_the_leg_it_cannot_stand_on(desk_setup, tmp_path, monkeypatch):
    """THE HOLD READS THE WHOLE WANT-LIST, not only what this call trimmed, and A NEW CURRENCY IS
    A PAIR: a leg's spot can be missing before any route exists - the map verified no pair for it,
    or this desk's seed spells none - and then its CURVE cannot be fitted either, because the fit
    values that strip's own deals in the book's base and reads the `FxRate` block to do it. Spot,
    curve and the surface over them are held together, each naming what it waits on.

    Each half holds the other: on a book carrying the yen spot and no yen curve the surface waits
    on the curve, on one carrying the euro curve and no euro spot it waits on the spot, and the row
    says which.

    Killing mutations: the hold gathered from the trimmed spots alone, which lands the curve and
    the surface against a spot that is not coming - the fit then indexes `FxRate.EUR`, and the
    whole write refuses with a bare `KeyError` naming a key rather than a sentence; the curve half
    of the surface's hold dropped, which lands a surface on a leg nothing discounts.
    """
    scopes = (('fx_vol', 'EURZAR'), ('rates', 'EUR'))
    seeded_map(tmp_path / 'home', *scopes)
    # the terminal answers for EURUSD under a name no candidate expects, so the probe rejects it
    # and the map verifies no spot for the pair - the shape a desk meets before any verification
    terminal = setup_terminal(monkeypatch, *scopes, **{'EURUSD BGN Curncy': 1.0855})
    curve_only(desk_setup)
    before = desk_setup.read_bytes()

    result, outcome = set_up({'pair': 'EURZAR'})
    market = json.loads(desk_setup.read_text())['Calc']['MergeMarketData']['ExplicitMarketData']
    reasons = {row['factor']: row['reason'] for row in outcome['not_supplied']}

    assert result['status'] == 'done' and 'error' not in result, result
    assert outcome['written'] is False and outcome['installed'] == [], outcome
    assert outcome['refused'] == [], 'a want that cannot be supplied is not a refused write'
    assert sorted(reasons) == ['FXVol.EUR.ZAR', 'FxRate.EUR', 'InterestRate.EUR']
    assert 'the map verified no spot for EURUSD' in reasons['FxRate.EUR']
    assert reasons['InterestRate.EUR'] == 'held with FxRate.EUR'
    assert reasons['FXVol.EUR.ZAR'] == 'held with FxRate.EUR'
    assert 'InterestRatePrices.EUR' not in market.get('Market Prices', {})
    assert desk_setup.read_bytes() == before

    # the other way in: the seed spells no pair for that leg at all, so the walk itself diverts it
    assert CLIENT.post('/book/securities', json={
        'block': 'fx_spot', 'key': 'pairs', 'entry': ['USDJPY', 'USDZAR']}).status_code == 200
    result, outcome = set_up({'pair': 'EURZAR'})
    reasons = {row['factor']: row['reason'] for row in outcome['not_supplied']}

    assert result['status'] == 'done' and outcome['written'] is False, result
    assert outcome['refused'] == [] and outcome['installed'] == []
    assert sorted(reasons) == ['FXVol.EUR.ZAR', 'FxRate.EUR', 'InterestRate.EUR']
    assert 'the seed spells no fx_spot pair for EURUSD or USDEUR' in reasons['FxRate.EUR']
    assert reasons['FXVol.EUR.ZAR'] == 'held with FxRate.EUR'
    assert desk_setup.read_bytes() == before

    # a book carrying one half of each pair: a yen spot whose curve is dead, a euro curve whose
    # spot nothing supplies - so the surface over each is held with the other half by name
    document = json.loads(before)
    carried = document['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors']
    carried['FxRate.JPY'] = {'Domestic_Currency': None, 'Interest_Rate': 'JPY',
                             'Spot': 1.0 / 157.25}
    carried['InterestRate.EUR'] = dict(carried['InterestRate.ZAR'], Currency='EUR')
    desk_setup.write_text(json.dumps(document, indent=2), newline='\n')
    scopes = (('fx_vol', 'USDJPY'), ('rates', 'JPY'), ('fx_vol', 'EURZAR'))
    seeded_map(tmp_path / 'home', *scopes)
    terminal = setup_terminal(monkeypatch, *scopes, **{'EURUSD BGN Curncy': 1.0855})
    terminal.dead = {row['security'] for row in seeded_rows_of('JPY')[1:]}
    standing = desk_setup.read_bytes()

    result, outcome = set_up({'pair': 'USDJPY'})
    reasons = {row['factor']: row['reason'] for row in outcome['not_supplied']}

    assert result['status'] == 'done' and outcome['written'] is False, result
    assert outcome['refused'] == [], 'the surface reached a bootstrap it cannot stand in'
    assert sorted(reasons) == ['FXVol.USD.JPY', 'InterestRate.JPY']
    assert reasons['FXVol.USD.JPY'] == 'held with InterestRate.JPY'
    assert desk_setup.read_bytes() == standing

    expiry = {'.Timestamp': (datetime.date.today() + datetime.timedelta(365)).isoformat()}
    result, outcome = set_up({'deal': dict(
        service.PAIR_PROBE, Reference='OPT', Currency='ZAR', Underlying_Currency='EUR',
        FX_Volatility='EUR.ZAR', Discount_Rate='EUR', Expiry_Date=expiry,
        Settlement_Date=expiry)})
    reasons = {row['factor']: row['reason'] for row in outcome['not_supplied']}

    assert result['status'] == 'done' and outcome['written'] is False, result
    assert outcome['refused'] == [], 'the surface reached a bootstrap it cannot stand in'
    assert sorted(reasons) == ['FXVol.EUR.ZAR', 'FxRate.EUR']
    assert reasons['FXVol.EUR.ZAR'] == 'held with FxRate.EUR'
    assert desk_setup.read_bytes() == standing


def test_two_wanted_curves_of_one_currency_answer_an_outcome(desk_setup, tmp_path, monkeypatch):
    """TWO WANTED CURVES CAN BE ONE CURRENCY - this desk's own vocabulary spells three for the rand
    - and what a queued job owes its caller is an OUTCOME. Both strips are built and land, the spot
    the book already carries keeps the curve IT names, and `check` names both. Where that spot
    cannot be supplied, BOTH are held with it and nothing lands.

    Killing mutations: each curve's currency read off a map keyed by CURRENCY, which collapses the
    two - as a repr it lost `written`, `refused` and `not_supplied` to a bare `KeyError` the job
    could not answer with at all, and as a lookup it leaves the collapsed curve unheld, landing it
    against a spot the book will not carry.
    """
    scopes = (('fx_spot', 'USDZAR'), ('rates', 'ZAR'), ('rates', 'ZAR-ZARONIA'))
    seeded_map(tmp_path / 'home', *scopes)
    setup_terminal(monkeypatch, *scopes, **{'USDZAR BGN Curncy': 18.5})
    curve_only(desk_setup)
    document = json.loads(desk_setup.read_text())
    document['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors'].pop(
        'InterestRate.ZAR')
    desk_setup.write_text(json.dumps(document, indent=2), newline='\n')
    legs = [dict(json.loads(dump(CASHFLOW)), Reference=reference, Discount_Rate=curve)
            for reference, curve in (('A', 'ZAR'), ('B', 'ZAR-ZARONIA'))]
    both = {'deal': {'Object': 'NettingCollateralSet', 'Reference': 'TWO', 'Netted': 'True',
                     'Collateralized': 'False', 'Settlement_Currency': '',
                     'Children': [{'Instrument': {'.Deal': leg}} for leg in legs]}}

    result, outcome = set_up(both)
    market = json.loads(desk_setup.read_text())['Calc']['MergeMarketData']['ExplicitMarketData']

    assert result['status'] == 'done' and 'error' not in result, result
    assert outcome['written'] is True and outcome['refused'] == [], outcome
    assert outcome['installed'] == ['InterestRate.ZAR', 'InterestRate.ZAR-ZARONIA']
    assert outcome['not_supplied'] == [] and outcome['held_out'] == []
    assert sorted(market['Market Prices']) == ['InterestRatePrices.ZAR',
                                               'InterestRatePrices.ZAR-ZARONIA']
    assert market['Price Factors']['FxRate.ZAR']['Interest_Rate'] == 'ZAR'
    assert len([row for row in outcome['check'] if 'conventions this build ships' in row]) == 2

    assert CLIENT.post('/book/securities', json={
        'block': 'fx_spot', 'key': 'pairs', 'entry': ['EURUSD']}).status_code == 200
    document['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors'].pop('FxRate.ZAR')
    desk_setup.write_text(json.dumps(document, indent=2), newline='\n')
    before = desk_setup.read_bytes()

    result, outcome = set_up(both)
    reasons = {row['factor']: row['reason'] for row in outcome['not_supplied']}

    assert result['status'] == 'done' and outcome['written'] is False, result
    assert outcome['refused'] == [] and outcome['installed'] == []
    assert sorted(reasons) == ['FxRate.ZAR', 'InterestRate.ZAR', 'InterestRate.ZAR-ZARONIA']
    assert reasons['InterestRate.ZAR'] == 'held with FxRate.ZAR'
    assert reasons['InterestRate.ZAR-ZARONIA'] == 'held with FxRate.ZAR'
    assert desk_setup.read_bytes() == before


def test_a_surface_is_held_with_the_curve_its_leg_actually_discounts_on(desk_setup, tmp_path,
                                                                        monkeypatch):
    """A LEG'S CURVE IS THE ONE ITS OWN `FxRate` BLOCK NAMES, not the one a set-up would have built
    it against. A book whose rand spot discounts on `ZAR-ZARONIA` and whose `ZAR-ZARONIA` entry
    declares no conventions cannot have that curve supplied, so the EURZAR surface is held with it
    by name and the book stands still.

    Killing mutation: the leg's curve read off the currency alone, which asks whether
    `InterestRate.ZAR` is missing while `InterestRate.ZAR-ZARONIA` is the one that is - the surface
    lands against a currency with no curve, and the book then prices a EURZAR collar with the
    engine quietly skipping the missing discount curve.
    """
    from derivus_bloomberg import security_map

    packaged = security_map.read_seed(security_map.packaged_seed())['rates']['ZAR-ZARONIA']
    assert CLIENT.post('/book/securities', json={
        'block': 'rates', 'key': 'ZAR-ZARONIA',
        'entry': dict(packaged, conventions={})}).status_code == 200
    document = json.loads(desk_setup.read_text())
    factors = document['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors']
    factors.pop('InterestRate.ZAR')
    factors['FxRate.ZAR'] = dict(factors['FxRate.ZAR'], Interest_Rate='ZAR-ZARONIA')
    document['Calc']['Deals']['Deals']['Children'] = []
    desk_setup.write_text(json.dumps(document, indent=2), newline='\n')
    scopes = (('fx_vol', 'EURZAR'), ('fx_spot', 'EURUSD'), ('rates', 'EUR'),
              ('rates', 'ZAR-ZARONIA'))
    seeded_map(tmp_path / 'home', *scopes)
    setup_terminal(monkeypatch, *scopes, **{'EURUSD BGN Curncy': 1.0855,
                                            'USDZAR BGN Curncy': 18.5})
    before = desk_setup.read_bytes()

    result, outcome = set_up({'pair': 'EURZAR'})
    market = json.loads(desk_setup.read_text())['Calc']['MergeMarketData']['ExplicitMarketData']
    reasons = {row['factor']: row['reason'] for row in outcome['not_supplied']}

    assert result['status'] == 'done' and 'error' not in result, result
    assert outcome['written'] is False and outcome['installed'] == []
    assert reasons['FXVol.EUR.ZAR'] == 'held with InterestRate.ZAR-ZARONIA'
    assert 'ZAR-ZARONIA declares no curve_day_count' in reasons['InterestRate.ZAR-ZARONIA']
    assert 'FXVol.EUR.ZAR' not in market['Price Factors'], 'a surface with no rand curve under it'
    assert desk_setup.read_bytes() == before


def test_a_book_ahead_of_its_prints_is_told_which_day_to_roll(desk_setup, tmp_path, monkeypatch):
    """A BOOK VALUED FORWARD OF ITS MARKET IS A LEGAL STATE, and the set-up screens its prints
    exactly as the tick screens the book's own rows - against the book's date. So a book six days
    ahead of its prints keeps every benchmark out, and what the answer owes is the WAY OUT: the
    book's date, the print's date and the verb that moves one. It never reads "nothing refused"
    beside two dozen held-out rows, which is what the emitter's own census says when the rows were
    held rather than rejected.

    Killing mutation: the emitter's sentence handed back verbatim, which counts the rejections it
    never made.
    """
    scopes = (('fx_vol', 'EURZAR'), ('fx_spot', 'EURUSD'), ('rates', 'EUR'))
    seeded_map(tmp_path / 'home', *scopes)
    setup_terminal(monkeypatch, *scopes, **{'EURUSD BGN Curncy': 1.0855})
    curve_only(desk_setup)
    ahead = (datetime.date.today() + datetime.timedelta(6)).isoformat()
    assert CLIENT.post('/book/date', json={'base_date': ahead}).status_code == 200

    result, outcome = set_up({'pair': 'EURZAR'})
    reason = {row['factor']: row['reason'] for row in outcome['not_supplied']}['InterestRate.EUR']

    assert result['status'] == 'done' and outcome['written'] is False, outcome
    assert 'kept 0 of 24 seeded benchmarks' in reason and '24 stale' in reason
    assert ahead in reason and 'POST /book/date' in reason
    # the screen does not keep a rejected print's own date, so the window it judged by stands in
    assert 'no print more than 5 days older than that' in reason
    assert 'nothing refused' not in reason, 'the emitter counted rejections it never made'


def test_an_installed_spot_names_the_curve_that_was_built(desk_setup, tmp_path, monkeypatch):
    """A SPOT DISCOUNTS ON THE CURVE THAT LANDED, not on its own currency code. The shipped
    `ZAR-ZARONIA` entry is the shape that tells them apart - a curve keyed by its own name and
    declaring `currency: ZAR` - so a book set up through it must carry an `FxRate.ZAR` pointing at
    `ZAR-ZARONIA`, the block the write actually made.

    Killing mutation: the block authored with `Interest_Rate: currency`, which points a rand spot
    at a rand curve nothing wrote.
    """
    factors = {name: block for name, block in FACTORS.items() if 'ZAR' not in name}
    monkeypatch.setenv('DV_HOME', str(tmp_path / 'home'))
    path = tmp_path / 'book.json'
    path.write_text(json.dumps(json.loads(dump(job(deals=(), factors=factors, sections={
        'Bootstrapper Configuration': {'InterestRate': {'Prices': 'InterestRate'}}}))),
        indent=2), newline='\n')
    service.BOOK = service.Book(str(path))
    try:
        scopes = (('fx_spot', 'USDZAR'), ('rates', 'ZAR-ZARONIA'))
        seeded_map(tmp_path / 'home', *scopes)
        setup_terminal(monkeypatch, *scopes, **{'USDZAR BGN Curncy': 18.5})
        rand = dict(json.loads(dump(CASHFLOW)), Discount_Rate='ZAR-ZARONIA')

        result, outcome = set_up({'deal': rand})
        market = json.loads(path.read_text())['Calc']['MergeMarketData']['ExplicitMarketData']

        assert result['status'] == 'done' and outcome['written'] is True, outcome
        assert outcome['installed'] == ['FxRate.ZAR', 'InterestRate.ZAR-ZARONIA']
        assert market['Price Factors']['FxRate.ZAR']['Interest_Rate'] == 'ZAR-ZARONIA'
        assert sorted(market['Market Prices']) == ['InterestRatePrices.ZAR-ZARONIA']
    finally:
        service.BOOK = None


def test_a_roll_re_prices_the_market_the_book_already_carried(desk_setup, tmp_path, monkeypatch):
    """WHEN THE PRINTS MOVE THE DATE, THE SET-UP IS A TICK AS WELL. A book ten days behind its
    prints gains a new market AND has the one it already carried re-priced off the terminal in the
    same session, landing in the same write - the standing surface, the standing curve's rows and
    the routed spots - otherwise it would hold two days of quotes under one date, the standing
    curve re-authored forward on numbers nobody re-fetched.

    THE NUMBER IS WHAT SAYS SO: the euro benchmarks print somewhere new, and the block that lands
    carries the new print rather than the one it was authored on. The write is dated by the whole
    trip, the roll's own snap included - later here than the day the new strip printed - the roll's
    held-out rows travel in `held_out`, `check` names every standing block that moved under the
    desk's feet, and a standing currency's `FxRate` keeps the curve IT discounts on rather than the
    one this set-up built.

    A book already standing on the prints' day asks the terminal about nothing standing: only the
    wants are fetched.

    Killing mutations: the roll dropped, or fetched and never merged into the write, which
    re-authors the standing curve on stale numbers while `check` announces it was re-priced; the
    write's date ignoring the day the roll's own prints were snapped; the roll's held rows dropped;
    no standing SURFACE re-priced at all; every spot authored as a new block, which re-points a
    standing spot at the curve this set-up happened to install.
    """
    scopes = (('fx_vol', 'EURZAR'), ('fx_spot', 'EURUSD'), ('rates', 'EUR'),
              ('fx_spot', 'USDZAR'), ('rates', 'ZAR-ZARONIA'), ('fx_spot', 'USDCHF'),
              ('rates', 'CHF'))
    seeded_map(tmp_path / 'home', *scopes)
    # the rand strip prints three days back, so the day the ROLL snapped is the later one
    rand = (datetime.date.today() - datetime.timedelta(3)).isoformat()
    terminal = setup_terminal(
        monkeypatch, *scopes,
        stamps={row['security']: rand for row in seeded_rows_of('ZAR-ZARONIA')},
        **{'EURUSD BGN Curncy': 1.0855, 'USDZAR BGN Curncy': 18.5, 'USDCHF BGN Curncy': 0.79})
    assert set_up({'pair': 'EURZAR'})[1]['written'] is True
    behind = (datetime.date.today() - datetime.timedelta(10)).isoformat()
    assert CLIENT.post('/book/date', json={'base_date': behind}).status_code == 200
    euro = [row['security'] for row in seeded_rows_of('EUR')]
    terminal.prints.update(dict.fromkeys(euro, 3.25))
    terminal.dead = {euro[-1]}
    terminal.asked.clear()

    result, outcome = set_up({'deal': dict(json.loads(dump(CASHFLOW)),
                                           Discount_Rate='ZAR-ZARONIA')})
    factors = json.loads(desk_setup.read_text())['Calc']['MergeMarketData'][
        'ExplicitMarketData']['Price Factors']
    points = curve_block(desk_setup, 'EUR')['Points']

    assert result['status'] == 'done' and outcome['written'] is True, result
    assert outcome['installed'] == ['InterestRate.ZAR-ZARONIA']
    assert set(euro) <= set(terminal.asked), 'the standing euro curve was not re-priced'
    # THE MERGE IS THE POINT: the standing block carries the print the roll came back with, and
    # the one row the screen refused keeps the number it had under `Use` No
    assert {row['Quoted_Market_Value'] for row in points if row['Use'] == 'Yes'} == {3.25}
    assert outcome['base_date'] == datetime.date.today().isoformat(), 'the roll did not date it'
    assert 'FXVolPrices.EUR.ZAR' in outcome['updated']
    assert 'InterestRatePrices.EUR' in outcome['reauthored']
    assert sorted(row.split(' ', 1)[0] for row in outcome['check'] if 're-priced onto' in row) == [
        'FXVolPrices.EUR.ZAR', 'InterestRatePrices.EUR'], outcome['check']
    assert [row for row in outcome['held_out'] if euro[-1] in row], outcome['held_out']
    assert factors['FxRate.ZAR']['Interest_Rate'] == 'ZAR', 'a standing spot took the new curve'
    assert factors['FxRate.ZAR']['Spot'] == pytest.approx(1.0 / 18.5)

    terminal.asked.clear()
    result, again = set_up({'deal': dict(json.loads(dump(CASHFLOW)), Reference='CFC',
                                         Currency='CHF', Discount_Rate='CHF')})

    assert result['status'] == 'done' and again['written'] is True, result
    assert again['installed'] == ['FxRate.CHF', 'InterestRate.CHF']
    assert not (set(euro) & set(terminal.asked)), 'a book on its prints re-priced what stood anyway'
    assert not [row for row in again['check'] if 're-priced onto' in row]


def routed_map(pair='USDZAR'):
    """The canned map with an `fx_spot` block beside its surface - a desk whose terminal verified
    the one pair that prices the book's rand against its dollars."""
    document = dict(canned_map())
    document['blocks'] = dict(document['blocks'], fx_spot={pair: {
        'security': '{} BGN Curncy'.format(pair), 'name': pair, 'last_update': '2024-06-28',
        'verified': '2024-06-28'}})
    return document


def test_the_tick_moves_the_books_spots(desk, monkeypatch):
    """THE TICK VALUES THE BOOK'S SPOTS, in the same atomic write as the surfaces it fetches. The
    map says which verified pair prices each currency against the base, and the print is crossed
    onto the ENGINE's axis - `USDZAR` is rand per dollar, so `FxRate.ZAR` is its reciprocal, one
    rand in dollars.

    A currency the map verified no pair for is named under `unrouted` and keeps the spot it had: a
    spot is never triangulated through a third currency, that being a market view rather than a
    tick.

    Killing mutation: the spot patched un-crossed, which writes 19.0 onto `FxRate.ZAR` and marks
    one rand at nineteen dollars.
    """
    document = json.loads(desk.read_text())
    market = document['Calc']['MergeMarketData']['ExplicitMarketData']
    market['Price Factors']['FxRate.CHF'] = {'Domestic_Currency': None, 'Interest_Rate': 'USD',
                                             'Spot': 1.1}
    desk.write_text(json.dumps(document, indent=2), newline='\n')
    terminal = CannedTerminal({'USDZAR BGN Curncy': 19.0})
    bloomberg_seams(monkeypatch, terminal=terminal,
                    provision=lambda source, as_of, on_batch=None: (routed_map(), False))

    result, outcome = ticked()
    factors = json.loads(desk.read_text())['Calc']['MergeMarketData']['ExplicitMarketData'][
        'Price Factors']

    assert result['status'] == 'done' and outcome['written'] is True
    assert outcome['spots'] == {'FxRate.ZAR': 1.0 / 19.0}
    assert factors['FxRate.ZAR']['Spot'] == pytest.approx(1.0 / 19.0)
    assert factors['FxRate.USD']['Spot'] == 1.0, 'the base leg moved'
    assert factors['FxRate.CHF']['Spot'] == 1.1, 'an unrouted currency was moved anyway'
    assert any('CHF against USD' in message for message in outcome['unrouted']), outcome
    assert terminal.asked == ['USDZAR BGN Curncy'], 'the tick asked about more than the routes'
    # the same write as the surface: one outcome, one etag, both on the file
    assert 'FXVol.USD.ZAR' in factors and outcome['installed'] == ['FXVolPrices.USD.ZAR']
    assert outcome['etag'] == CLIENT.get('/book').json()['etag']


def test_a_spot_print_the_terminal_will_not_stand_behind_refuses_the_tick(desk, monkeypatch):
    """A SPOT GOES THROUGH THE SAME SCREEN AS A SURFACE. A vol quote five days old refuses the
    whole tick by name; a spot print from 2019 is the same trap and the same refusal, because a
    dead series keeps answering with a plausible number and every deal in the book reprices off a
    spot. A ticker that does not answer at all is a named refusal too - `written: false` with the
    engine's own words - never an `error` status the metronome cannot read a cause off.

    Killing mutations: the spot securities left out of the freshness call, which writes a
    two-year-old print straight onto the book; the dead-print error left to escape the job, which
    reaches a poller as `status: error` and a blank outcome.
    """
    from derivus_bloomberg import security_map

    document = json.loads(desk.read_text())
    document['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors']['FxRate.CHF'] = {
        'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Spot': 1.1}
    desk.write_text(json.dumps(document, indent=2), newline='\n')
    before = desk.read_bytes()
    stale = bloomberg_seams(
        monkeypatch, stale=security_map.stale,
        terminal=CannedTerminal(routed_prints(), stamp=today(),
                                stamps={'USDZAR BGN Curncy': '2019-01-01'}),
        provision=lambda source, as_of, on_batch=None: (routed_map(), False)) is not None

    result, outcome = ticked()

    assert stale and result['status'] == 'done' and 'error' not in result, result
    assert outcome['written'] is False
    assert outcome['refused'] == ['USDZAR BGN Curncy is stale - 2019-01-01']
    # the answer carries the same keys on this path as on a landing: a desk reading `unrouted`
    # must not have to know which branch answered it
    assert any('CHF against USD' in message for message in outcome['unrouted']), outcome
    assert outcome['held_out'] == [] and outcome['spots'] == {}
    assert desk.read_bytes() == before

    bloomberg_seams(monkeypatch, provision=lambda source, as_of, on_batch=None: (
        routed_map(), False), terminal=CannedTerminal(
            routed_prints(), dead=['USDZAR BGN Curncy'], stamp=today()))
    result, outcome = ticked()

    assert result['status'] == 'done' and 'error' not in result, result
    assert outcome['written'] is False
    assert any('USDZAR BGN Curncy' in message for message in outcome['refused']), outcome
    assert service.Metronome(60.0).cause(result), 'the metronome could read no cause'
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


def test_a_solve_refuses_a_field_and_a_bounds_the_deal_cannot_carry(book):
    """What a model gets wrong about a solve is named instead of being handed to scipy: a field the
    deal's type does not declare, a `bounds` that is not a pair, and a target the field cannot
    reach - the first two before anything prices at all.

    KILLING MUTATION: `RuntimeError: Tolerance ... Failed to converge` for the unknown field and
    the unreachable target, and `IndexError: list index out of range` for the one-ended bounds -
    three sentences naming neither the field nor the deal.
    """
    answers = {}
    for label, solve in [('field', {'field': 'Nope', 'target': 0.0}),
                         ('bounds', {'field': 'Amount', 'target': 0.0, 'bounds': [1.0]}),
                         ('reach', {'field': 'Amount', 'target': 1e300})]:
        submitted = CLIENT.post('/book/solve', content=dump(dict(
            solve, deal=dict(CASHFLOW, Reference='SLV_' + label))), headers=JSON).json()
        service.EXECUTOR.queue.join()
        answers[label] = CLIENT.get('/results/{}'.format(submitted['result_id'])).json()

    assert all(answer['status'] == 'error' for answer in answers.values())
    assert answers['field']['error'].startswith('Nope is not a field of FixedCashflowDeal')
    assert 'Payment_Date' in answers['field']['error']
    assert answers['bounds']['error'] == (
        'bounds is a pair [low, high] of numbers with low below high, not [1.0]')
    assert answers['reach']['error'].startswith('solve for Amount never reached 1e+300')


def test_a_solve_on_a_candidate_naming_market_data_the_book_lacks_is_refused(book):
    """The solve validates its candidate before it queues, in the booking's own words - the
    what-if's check. Without it the candidate LOADS, discovery drops it on the first iterate, the
    marks frame carries no row under its reference and the loop dies reading one: an error result
    saying `single positional indexer is out-of-bounds`, the market data never named."""
    absent = CLIENT.post('/book/solve', content=dump({
        'deal': dict(CASHFLOW, Reference='SLV4', Discount_Rate='ZAR-SWAP'), 'field': 'Amount',
        'target': 1.0}), headers=JSON)

    assert absent.status_code == 422
    assert absent.json()['detail'] == 'no market data for InterestRate.ZAR-SWAP'


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
    # the level the ticket named is what says which way round a collar is dealt, and the answer
    # reports the variation it priced rather than leaving a reader to infer it from the legs
    assert quote['variation'] == 'floor' and quote['client'] == {'buys': 'ZAR', 'sells': 'USD'}
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


#: The IMPORTER's forward extra as a desk asks for it: the cap the client will pay for dollars,
#: with no direction stated at all. The level is what says which way round it is dealt, and a
#: booking is what it has to come back as.
IMPORTER = {'pair': 'USDZAR', 'expiry': '1Y', 'notional': AMOUNT, 'notional_currency': 'USD',
            'cap': USDZAR * 1.03}


def test_a_quoted_forward_extra_says_which_way_it_was_dealt_and_books_that(quoting, tmp_path):
    """One structure, two BOOKINGS, and the whole loop on the one the ticket meant.

    A cap is the importer's forward extra: the client buys the dollars, so they buy the call and
    sell the knock-in put below it. Nothing in the ask says a direction - the level says it - and
    the answer reports back both the `variation` it priced and the client's own two cashflows, so
    a salesperson reading the sheet and the engine pricing the legs cannot disagree about which
    trade this is. The pending file carries the same two facts, being the audit trail.

    Then the approval books the MIRROR of exactly those legs, which is the claim a price alone
    cannot make: the bank sells the call it was asked for and buys the down-and-in put.
    """
    quote = quote_of('ForwardExtra', IMPORTER)

    assert quote['variation'] == 'cap'
    assert quote['client'] == {'buys': 'USD', 'sells': 'ZAR'}
    assert [(leg['role'], leg['deal_type'], leg['buy_sell']) for leg in quote['legs']] == [
        ('protection', 'FXOptionDeal', 'Buy'), ('reversion', 'FXBarrierOption', 'Sell')]
    assert quote['legs'][1]['barrier_market'] < IMPORTER['cap'], (
        'the knock-in that reverts an importer to their cap sits BELOW it')

    filed = json.loads((tmp_path / 'tmp' / (quote['quote_id'] + '.json')).read_text())
    assert (filed['quote']['variation'], filed['quote']['client']) == ('cap', quote['client'])

    booked = CLIENT.post('/book/quote', json={'quote_id': quote['quote_id']}).json()
    node = deal_at(json.loads(quoting.read_text()), booked['deal_path'])
    legs = [child['Instrument']['.Deal'] for child in node['Children']]

    assert booked['written'] is True
    assert [(block['Object'], block['Option_Type'], block.get('Barrier_Type'), block['Buy_Sell'])
            for block in legs] == [('FXOptionDeal', 'Call', None, 'Sell'),
                                   ('FXBarrierOption', 'Put', 'Down_And_In', 'Buy')]


#: The desk's charge through the verb, and the notional it is quoted against. This book's
#: `FxRate.ZAR` carries 18.5 DOLLARS per rand, so 50,000 rand is 925,000 dollars - which is a sales
#: margin against the notional below and the whole trade against `COLLAR`'s.
MARGIN = {'amount': 50_000.0, 'currency': 'ZAR'}
MARGIN_COLLAR = dict(COLLAR, notional=AMOUNT * 200)


def test_a_collar_quoted_at_a_margin_books_the_bank_at_plus_it(quoting, tmp_path):
    """The margin through the whole loop, and the sign read from both ends of it.

    The quote is CLIENT paper, so the cap the recipe solves funds the bought put and the charge
    together and `net` comes back at minus the margin - converted at the quote's own spot, minus
    50,000 rand. What BOOKS is the mirror, and the book then marks it at plus the margin's dollar
    value, which is the engine on the deal actually written rather than anything the runner said.
    The ticket records what was agreed, in the currency it was agreed in.
    """
    plain = quote_of('ZeroCostCollar', MARGIN_COLLAR)
    quote = quote_of('ZeroCostCollar', MARGIN_COLLAR, margin=MARGIN)
    charge = quote['margin']

    assert charge == {'amount': 50_000.0, 'currency': 'ZAR', 'pricing_currency': 'USD',
                      'value': pytest.approx(50_000.0 * SPOT, rel=1e-12)}
    assert quote['net'] * quote['spot']['value_market'] == pytest.approx(-50_000.0, abs=0.01)
    assert quote['legs'][1]['strike_market'] < plain['legs'][1]['strike_market'], (
        'the financing strike did not move for the margin')
    assert (quote['deal']['Sales_Margin'],
            quote['deal']['Sales_Margin_Currency']) == (50_000.0, 'ZAR')

    booked = CLIENT.post('/book/quote', json={'quote_id': quote['quote_id']}).json()
    on_disk = json.loads(quoting.read_text())
    node = deal_at(on_disk, booked['deal_path'])

    assert booked['written'] is True
    assert node['Instrument']['.Deal']['Sales_Margin'] == 50_000.0
    assert [child['Instrument']['.Deal']['Buy_Sell'] for child in node['Children']] == [
        {'Buy': 'Sell', 'Sell': 'Buy'}[leg['buy_sell']] for leg in quote['legs']]

    marked_id, marked = run(on_disk)
    assert marked['status'] == 'done', marked.get('error')
    assert mtm(marked_id)[node['Instrument']['.Deal']['Reference']] == pytest.approx(
        charge['value'], abs=1.0), 'the book does not hold the margin the desk charged'


#: The desk's two-way as a book carries one: 0.4 vol points on the ATM rows and half that on the
#: wings, written around each quote's own mid. The bootstrap never reads these sides - the surface
#: stays the one the mid built - so what they buy is the quote's own CHARGE.
QUOTE_SPREAD = 0.004


@pytest.fixture
def quoting_two_way(quoting):
    """The same desk, quoting a two-way: bid and ask around every quote's mid on the served book."""
    document = json.loads(quoting.read_text())
    points = document['Calc']['MergeMarketData']['ExplicitMarketData'][
        'Market Prices']['FXVolPrices.USD.ZAR']['instrument']['Points']
    for point in points:
        half = 0.5 * QUOTE_SPREAD * (1.0 if point['Quote_Type'] == 'ATM' else 0.5)
        point['Quoted_Bid'] = point['Quoted_Market_Value'] - half
        point['Quoted_Ask'] = point['Quoted_Market_Value'] + half
    quoting.write_text(json.dumps(document, indent=2), newline='\n')
    service.BOOK = service.Book(str(quoting))
    return quoting


def test_a_two_way_quote_books_the_mirror_at_the_margin_and_the_spread(quoting_two_way):
    """The quoting day on a book that quotes a two-way, with the desk's whole take read off the
    ENGINE rather than off the runner.

    Every leg is priced at the mid and what the market charges for the spread is levied on the
    coordinate the recipe solves, so the client is quoted minus the MARGIN alone - the spread being
    inside the terms rather than on the ticket - while the trade MARKS at minus the margin and the
    edge together. The book holds the bank's side of that paper, so once the approval books the
    mirror a base valuation of the book reports plus both, to the solve's own tolerance.

    And `edge` is the charge and nothing else: the sum of the legs' own `spread_charge`, each of
    them a number rather than a null, each saying where its vega was read.
    """
    quote = quote_of('ZeroCostCollar', MARGIN_COLLAR, margin=MARGIN)
    charge = quote['margin']['value']

    assert quote['edge'] > 0.0
    assert quote['edge'] == pytest.approx(
        sum(leg['spread_charge'] for leg in quote['legs']), rel=1e-12)
    assert {leg['spread_source'] for leg in quote['legs']} == {'surface'}
    assert quote['spread_note'] is None, 'the book quotes a two-way; there is no absence to name'
    assert quote['net'] == pytest.approx(-charge, abs=0.01)
    assert quote['net_mid'] == pytest.approx(-(charge + quote['edge']), abs=0.01)

    booked = CLIENT.post('/book/quote', json={'quote_id': quote['quote_id']}).json()
    on_disk = json.loads(quoting_two_way.read_text())
    node = deal_at(on_disk, booked['deal_path'])
    assert booked['written'] is True

    marked_id, marked = run(on_disk)
    assert marked['status'] == 'done', marked.get('error')
    assert mtm(marked_id)[node['Instrument']['.Deal']['Reference']] == pytest.approx(
        charge + quote['edge'], abs=1.0), (
        'the book does not hold the margin and the spread the desk charged')


def test_a_fitted_structure_books_and_marks_at_the_margin_and_the_spread(quoting_two_way):
    """The one path where the PIN is load-bearing for the mark, walked end to end.

    A strip quoted on a fitted book prices under the fitted law, and its two-way charge comes off
    the lognormal reading of the same leg because a fitted leg publishes no quote sensitivity. Book
    it, and the book has to mark it the way it was dealt: `/book/quote` merges the quote's own
    `valuation_configuration` inside the same edit closure that splices the deal, and only then
    does a base valuation of the book report the bank at plus the margin and the edge.

    Without that merge the mirror is marked as a lognormal and the number is not the desk's take at
    all - a plausible mark on a trade nobody dealt at it.

    THE BOOK STATES THE DECLARED PATH COUNT, because the mark is an ordinary base valuation of the
    book and a strip walking a fitted law solves a strike 2.8e-2 per path wide: the identity below
    holds when the two readings share a count and a seed, and a book stating fewer than the quote
    is floored onto marks its own trade on a different estimator - at 1,024 paths this mark lands
    4.6% off the take it was quoted at, which is what `/book/status` names.
    """
    document = json.loads(quoting_two_way.read_text())
    market = document['Calc']['MergeMarketData']['ExplicitMarketData']
    market['Price Factors']['LogVar2FJModelParameters.ZAR'] = json.loads(dump(CALIBRATED))
    document['Calc']['Calculation']['MCMC_Simulations'] = structures.declared_paths()
    quoting_two_way.write_text(json.dumps(document, indent=2), newline='\n')
    service.BOOK = service.Book(str(quoting_two_way))

    quote = quote_of('Accumulator', ACCUMULATOR, margin=MARGIN)
    charge = quote['margin']['value']
    take = charge + quote['edge']

    assert quote['legs'][0]['note'] is None, 'the leg did not join the calibration'
    assert quote['legs'][0]['spread_source'] == 'lognormal reading'
    assert quote['valuation_configuration'] == {
        'FXAccumulatorOptionDeal': {'SpotModel': 'LogVar2FJ'}}
    assert quote['edge'] > 0.0 and quote['charged_on'] == 'Strike_Price'
    assert quote['net'] == pytest.approx(-charge, abs=0.01)

    booked = CLIENT.post('/book/quote', json={'quote_id': quote['quote_id']}).json()
    on_disk = json.loads(quoting_two_way.read_text())
    node = deal_at(on_disk, booked['deal_path'])

    assert booked['written'] is True
    assert on_disk['Calc']['MergeMarketData']['ExplicitMarketData'][
        'Valuation Configuration'] == quote['valuation_configuration']

    marked_id, marked = run(on_disk)
    assert marked['status'] == 'done', marked.get('error')
    assert mtm(marked_id)[node['Instrument']['.Deal']['Reference']] == pytest.approx(
        take, rel=1e-3), 'the fitted book does not hold the margin and the spread it charged'


def test_a_margin_in_a_currency_the_book_cannot_cross_refuses_at_the_verb(quoting):
    """A margin nobody can value is a price nobody can quote, and the salesperson finds out on the
    call rather than off a failed job: the verb refuses 422, naming the rate the book would need."""
    refused = CLIENT.post('/book/structure', content=dump(
        {'structure': 'ZeroCostCollar', 'params': MARGIN_COLLAR,
         'margin': {'amount': 50_000.0, 'currency': 'JPY'}}), headers=JSON)

    assert refused.status_code == 422
    assert 'FxRate.JPY' in refused.json()['detail']


def test_a_solve_takes_its_target_as_money_in_a_currency_the_book_carries(quoting):
    """The same margin on the deal verb. `target` is a number in the run's own currency or an
    amount in one the book carries a rate for; this book reports dollars and the ask is in rand, so
    the verb crosses it at the book's own spot and solves against THAT. The deal being solved is
    the one the desk books, so it marks at PLUS the margin - the opposite side from the client
    paper a structure quote hands back, and the same number."""
    submitted = CLIENT.post('/book/solve', content=dump({
        'deal': FX_OPTION, 'field': 'Strike_Price', 'target': MARGIN,
        'bounds': [12.0, 30.0]}), headers=JSON).json()
    service.EXECUTOR.queue.join()
    result = CLIENT.get('/results/{}'.format(submitted['result_id'])).json()
    solved = result['stats']['Solved']

    assert result['status'] == 'done', result.get('error')
    assert solved['target'] == pytest.approx(50_000.0 * SPOT, rel=1e-12), (
        'the solve ran against the amount rather than the money')
    assert solved['margin'] == {'amount': 50_000.0, 'currency': 'ZAR',
                                'pricing_currency': 'USD', 'value': solved['target']}
    assert mtm(submitted['result_id'])['OPT1'] == pytest.approx(50_000.0 * SPOT, abs=0.01)


#: A calibrated spot-model factor for the rand, as `/book/model` writes one: the ladder
#: `fx_surface_block` authors off this file's own bootstrapped `FXVol.USD.ZAR`, fitted through
#: `Config.bootstrap` at 2,048 paths and pasted here. The gate is about the MODEL reaching the book
#: with the trade, so what it needs of the fit is that the engine wrote it.
CALIBRATED = {
    'Property_Aliases': None, 'Kappa_L': 0.5, 'Sigma_L': 0.5, 'Rho_L': 0.2, 'Kappa_S': 6.0,
    'Cap_A': 4.605170185988092, 'Steps_Per_Year': 252.0, 'C_Min': 0.12,
    'Residual_Law': 'NIG', 'On_Guard': '', 'Stickiness_Band': 0.5,
    'Skew_Gradient': '-0.0222277544361,-2.12183436316',
    'Xi_Curve': utils.Curve([], [[0.0, 0.020733491013238004],
                                 [0.2493150684931507, 0.02438547422614177]]),
    'Rho_S': utils.Curve([], [[0.0, 0.08094527234766719]]),
    'Beta': utils.Curve([], [[0.0, -10.934407669066678]]),
    'Sigma_S': utils.Curve([], [[0.0, 2.256886996797385]]),
    'Alpha': utils.Curve([], [[0.0, 60.24372735960779]])}

#: An accumulator on the RAND: the orientation whose underlying IS the token a spot model is keyed
#: on, so it rides the fit as written and crosses no axis. The keying's own gates are in
#: `test_structures.py` and `test_fx_accumulator_json.py`.
#: An accrual strip is dealt both ways off one knock-out level, so the ask states its direction:
#: buying USD at every fixing is the accumulator, selling it the decumulator.
ACCUMULATOR = {'pair': 'USDZAR', 'expiry': '3M', 'notional': AMOUNT, 'notional_currency': 'ZAR',
               'buy_currency': 'USD', 'fixing_frequency': '1M', 'knockout': USDZAR * 1.10}


def test_a_leg_quoted_under_a_model_books_into_a_book_that_marks_it(quoting):
    """The model books WITH the trade, in the same atomic write, or the desk marks a trade at a
    price it was never dealt at.

    `structures.spot_model` pins the desk's family on the QUOTE's copy of the document, and that copy is
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
    market['Price Factors']['LogVar2FJModelParameters.ZAR'] = json.loads(dump(CALIBRATED))
    document['Calc']['Calculation']['MCMC_Simulations'] = structures.declared_paths()
    quoting.write_text(json.dumps(document, indent=2), newline='\n')
    service.BOOK = service.Book(str(quoting))

    quote = quote_of('Accumulator', ACCUMULATOR)
    pinned = {'FXAccumulatorOptionDeal': {'SpotModel': 'LogVar2FJ'}}

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
        'LogVar2FJModelParameters.ZAR')
    quoting.write_text(json.dumps(dropped, indent=2), newline='\n')
    service.BOOK = service.Book(str(quoting))
    unbookable = json.dumps(dropped, indent=2)
    refused = CLIENT.post('/book/quote', json={'quote_id': quote['quote_id']})

    assert refused.status_code == 422
    assert 'LogVar2FJModelParameters.ZAR' in refused.json()['detail']
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


def test_a_pending_quote_and_its_sheet_are_read_back_by_id(quoting, tmp_path):
    """A quote is filed as a FILE on the service's disk, so a client that is not on that machine
    needs the two reads: the pending trade as JSON, and the sheet as the spreadsheet itself rather
    than a path nothing else can open.

    Killing mutations: the sheet verb answering the JSON's bytes (the zip signature is what says
    it is a workbook); the id checked by anything but a basename, which the backslashed path then
    reads out of the desk's own tmp.
    """
    pytest.importorskip('derivus.quote_sheet',
                        reason='the quote extra is not installed - there is no sheet to read')
    quote = quote_of('ZeroCostCollar', COLLAR)

    pending = CLIENT.get('/book/quote/{}'.format(quote['quote_id'])).json()
    sheet = CLIENT.get('/book/quote/{}/sheet'.format(quote['quote_id']))

    assert pending == json.loads((tmp_path / 'tmp' / (quote['quote_id'] + '.json')).read_text())
    assert pending['deal'] == quote['deal'] and pending['quoted_at']
    assert sheet.status_code == 200 and sheet.content[:2] == b'PK'
    assert sheet.headers['content-type'] == service.QUOTE_SHEET_MIME

    unknown = CLIENT.get('/book/quote/nosuchquoteid')
    assert unknown.status_code == 404 and str(tmp_path / 'tmp') in unknown.json()['detail']
    escaping = CLIENT.get('/book/quote/..%5C..%5Cbook')
    assert escaping.status_code == 422 and 'is not a quote id' in escaping.json()['detail']


def test_a_quote_given_with_no_sheet_writer_names_the_install(quoting, tmp_path):
    """A sheet is the `quote` extra's and a quote never fails for want of one, so the read refuses
    404 carrying the quote's OWN `sheet_note` - the install, not a bare miss. Authored here
    because that is the only way to reach the branch on a workstation that has the writer.

    Killing mutation: the refusal saying 'no sheet' without the note, which leaves a desk with
    nothing to do about it.
    """
    filed = tmp_path / 'tmp'
    filed.mkdir(parents=True, exist_ok=True)
    (filed / 'sheetless.json').write_text(json.dumps({'quote': {'files': {
        'sheet': None, 'sheet_note': 'no quote sheet - pip install derivus[quote]'}}}),
        newline='\n')

    answer = CLIENT.get('/book/quote/sheetless/sheet')

    assert answer.status_code == 404
    assert 'pip install derivus[quote]' in answer.json()['detail']
    assert CLIENT.get('/book/quote/sheetless').json()['quote']['files']['sheet'] is None


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


def test_a_curve_authored_before_it_carried_its_definition_reads_back_with_a_note(tmp_path):
    """An older book's curve block has quotes and no definition: no tenor on its rows, no
    conventions on the block. The read verb answers it as far as it goes, blank tenors and the
    quotes, with a note naming what is missing and the verb that re-authors it, rather than dying
    on the first missing key."""
    old = {'instrument': {'Currency': 'ZAR', 'Day_Count': 'ACT_365', 'Discount_Rate': '', 'Points': [
        {'Use': 'Yes', 'Descriptor': 'ZAR 3M', 'DealType': 'DepositDeal', 'Quote_Type': 'Par_Rate',
         'Quoted_Market_Value': 7.1, 'Deal': {}}]}}
    path = tmp_path / 'book.json'
    path.write_text(json.dumps(json.loads(dump(job(
        sections={'Market Prices': {'InterestRatePrices.ZAR': old}}))), indent=2), newline='\n')
    service.BOOK = service.Book(str(path))
    try:
        answer = CLIENT.get('/book/curve')
        assert answer.status_code == 200
        curve = answer.json()['curves']['InterestRatePrices.ZAR']
        assert 'Tenor' in curve['note'] and '/book/curve' in curve['note']
        assert curve['rows'] == [{'tenor': '', 'security': '', 'quote': 7.1, 'use': 'Yes'}]
        assert 'conventions' not in curve
    finally:
        service.BOOK = None
