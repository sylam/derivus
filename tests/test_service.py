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

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import derivus
from derivus import service, utils
from derivus.config import CustomJsonEncoder

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


def job(deals=(CASHFLOW,), factors=FACTORS, **calculation):
    """A job document, authored as the objects a market data file holds. Dumping it through
    `CustomJsonEncoder` is what a client posts, so the `.Curve` and `.Timestamp` tokens the endpoint
    receives are exactly the ones a file carries — and the decoder that reads them is the same one.
    """
    return {'Calc': {
        'Calculation': dict({'Object': 'BaseValuation', 'Base_Date': BASE, 'Currency': 'USD',
                             'MCMC_Simulations': 1, 'Random_Seed': 1}, **calculation),
        'Deals': {'Tag_Titles': '', 'Reference': 'service',
                  'Deals': {'Children': [{'Instrument': {'.Deal': deal}} for deal in deals]}},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': {
            'System Parameters': {'Base_Currency': 'USD', 'Base_Date': BASE},
            'Price Factors': factors}}}}


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
    path.write_text(json.dumps(json.loads(dump(job())), indent=2))
    service.BOOK = service.Book(str(path))
    yield path
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
