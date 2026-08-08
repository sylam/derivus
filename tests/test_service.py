"""One vocabulary, two bindings — and the gates are what says the second one owns no logic.

`derivus.service` is `Context` over HTTP. Every endpoint builds a Context from the posted job,
calls one of its verbs and serialises the answer, so the decisive gate is PARITY: the same job
submitted over HTTP has to produce the same numbers as loading it in process and calling `run_job`.
Anything the wrapper did of its own would show up there as a difference. `/schema` and `/validate`
are the same claim for the two read verbs.

The part that is genuinely new is the dispatcher, and it makes three promises worth holding to.
Ordering: pricing goes through ONE worker, so a base valuation jumps a simulation among the jobs
still WAITING and a running job is never preempted. Identity: a `result_id` is the hash of the
replay tuple, so submitting the same job twice is one execution — dedupe and retry-idempotency are
the same feature, and it has to hold while the first is still running, not just after it finishes.
Survival: a job that fails in the engine is a result like any other, and the next job still runs.

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
    """Submit, wait for the one worker to drain the queue, and read the result back."""
    submitted = submit(document)
    service.EXECUTOR.queue.join()
    return submitted, CLIENT.get('/results/{}'.format(submitted['result_id'])).json()


def mtm(result):
    """`{Reference: Value}` out of the serialized `mtm` table."""
    table = result['results']['mtm']['.DataFrame']
    reference, value = table['columns'].index('Reference'), table['columns'].index('Value')
    return {row[reference]: row[value] for row in table['data']}


class Held:
    """A Context as far as the executor is concerned — it calls `run_job()` and reads nothing else.

    That one verb is the whole seam between the queue and the engine, so the ordering and dedupe
    promises can be observed by handing the executor one of these. Nothing in the package is
    patched, and `hold` is what makes the worker's occupancy an event rather than a race.
    """

    def __init__(self, name, ran, hold=None):
        self.name, self.ran, self.hold = name, ran, hold
        self.started = threading.Event()

    def run_job(self):
        self.started.set()
        if self.hold is not None:
            self.hold.wait(timeout=30)
        self.ran.append(self.name)
        return None, {'Results': {}}


def test_a_job_priced_over_http_is_the_job_priced_in_process():
    """The decisive gate. Identical results, table for table and cell for cell — which is what
    "the wrapper owns no logic" means when it is asserted rather than asserted about.

    The closed form is here so the comparison cannot pass by both sides being empty: a serialised
    table equal to another serialised table says nothing if neither carries a price.
    """
    document = job()
    _, result = run(document)
    _, out = in_process(document).run_job()

    assert result['status'] == 'done'
    assert result['results'] == json.loads(json.dumps(out['Results'], cls=CustomJsonEncoder))
    assert mtm(result)['CF1'] == pytest.approx(AMOUNT * SPOT * np.exp(-RATE * 2.0), rel=1e-9)


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
    """What makes a front end thin: it renders panels, tables and enums from `fields.mapping`
    rather than restating them, so the endpoint is that store and the version that emitted it."""
    published = CLIENT.get('/schema').json()

    assert published.pop('engine_version') == derivus.__version__
    assert published == json.loads(json.dumps(derivus.fields.mapping, cls=CustomJsonEncoder))
    # not vacuous: this is the declaration a client reads to know which fields it may patch
    assert published['Factor']['types']['FxRate']['Spot']['bind'] == 'value'


def test_validate_over_http_is_the_verb_verbatim():
    """Both halves of the want-list, over a job broken deliberately in both ways: a deal that
    breaks an authoring rule, and one naming a curve the market data has no block for."""
    document = job(deals=(dict(CASHFLOW, Discount_Rate='GBP'), BINARY),
                   factors=dict(FACTORS, **EQUITY))
    over_http = CLIENT.post('/validate', content=dump(document), headers=JSON).json()

    assert over_http == in_process(document).validate()
    assert over_http == {'deals': {'BIN1': ['Cash_Payoff is required']},
                         'factors': ['InterestRate.GBP']}


def test_a_patch_reaches_the_number_and_moves_the_result_id():
    """A patch is applied before the hashes are taken, so it reaches the price AND the identity.

    The two stamps say which half moved: a spot is market VALUES, so the values hash moves and the
    plan hash does not. An id taken before the patch would collide with the unpatched run and serve
    it the wrong numbers, which is exactly the failure the hash split exists to prevent.
    """
    patch = {'FxRate.ZAR': {'Spot': SPOT * 1.1}}
    unpatched, plain = run(job())
    submitted, patched = run(dict(job(), Patch=patch))

    context = in_process(job())
    context.patch_market(patch)
    _, out = context.run_job()

    assert submitted['result_id'] != unpatched['result_id']
    assert patched['values_hash'] != plain['values_hash']
    assert patched['plan_hash'] == plain['plan_hash']
    assert patched['results'] == json.loads(json.dumps(out['Results'], cls=CustomJsonEncoder))
    assert mtm(patched)['CF1'] == pytest.approx(mtm(plain)['CF1'] * 1.1, rel=1e-9)


def test_an_identical_submission_is_one_result_id():
    """Retrying is free: the same job names the same result, and the second submission is already
    holding the finished one."""
    document = job(Random_Seed=11)
    first, _ = run(document)
    second = submit(document)

    assert second['result_id'] == first['result_id']
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
    _, after = run(job(Random_Seed=7))

    assert failed['status'] == 'error'
    assert 'FxRate.GBP' in failed['error']
    assert 'results' not in failed
    assert after['status'] == 'done'
    assert mtm(after)['CF1'] == pytest.approx(AMOUNT * SPOT * np.exp(-RATE * 2.0), rel=1e-9)
