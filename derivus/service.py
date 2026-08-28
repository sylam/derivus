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

"""One vocabulary, two bindings: the Context verbs, over HTTP.

`Context` is the in-process binding — `load_json`, `validate`, `describe`, `market_patch` /
`patch_market`, `plan_hash` / `values_hash`, `run_job`. This module is the same verbs over HTTP, and
it owns no logic of its own: every endpoint builds a Context, calls one of those methods, and
serialises what comes back. A web SPA, a marimo notebook and an Excel add-in are all clients of the
same endpoints, so nothing client-specific may enter the surface — and anything a client needs that
the verbs cannot answer is a missing verb on `Context`, not an endpoint that reaches inside.

| | |
| --- | --- |
| `GET /schema` | every declaration a front end renders from, plus the engine version |
| `GET /schema/job` | the job ENVELOPE the declarations sit inside, as a skeleton that loads |
| `POST /validate` | what would stop this job running, without running it |
| `POST /describe` | what the engine parsed, and what running it would cost, without running it |
| `POST /prepare` | `{plan_id}` — the parsed job, named by its plan hash and cached under it |
| `POST /execute` | `{result_id, status}` — always, for every calculation |
| `GET /results/{result_id}` | status, the replay tuple, the run's stats, and the SHAPE of each table |
| `GET /results/{result_id}/{table}` | one table, paged |
| `GET /ui` | a built web UI, when `DV_Service --ui` mounted one - a client, not a verb |
| `GET /book` | the live job document `DV_Service --book` serves, and the etag naming its state |
| `POST /book/deals` | book or delete one deal - validated BEFORE an atomic write, refusal writes nothing |
| `POST /book/price` | price the book, optionally with a candidate deal spliced in - a what-if, writes nothing |
| `POST /book/solve` | solve one field of a candidate deal to a target value - a root find over base valuations, writes nothing |
| `POST /book/market` | tick the book's market: quote blocks installed or value-updated, a values patch applied, the bootstrap run - one atomic write |
| `POST /book/bloomberg` | provision the security map, fetch the desk's FX vol surfaces off the terminal and tick the book |
| `POST /book/structure` | quote a named structure against the book - legs solved, the pending trade filed under its quote id |
| `POST /book/quote` | book a quote already given - the approval half, refused exactly as a booking is |

Every POST body is either a job document or `{"plan_id": ...}` naming one already prepared, and
nothing downstream can tell the two apart: the plan cache holds a pristine parse and every read of
it is a deep copy. A plan-id execute and a full-document execute of the same job therefore report
the same `result_id` — content addressing does not care how the job arrived.

A posted job is a job FILE: `Config.read_json` takes `(text, name)` as readily as a path, so the
same decoder builds the Curves, Timestamps and DateLists, and there is no second parser to keep in
step with the first.

There is no sync/async split at the API level. `/execute` always answers with a `result_id` and a
status, and a base valuation is simply `done` by the first poll — one contract an Excel RTD cell
and a browser poll loop can both be written against. A finished run answers with the shape of its
tables and never their cells: a CMC's exposure is a cube, and a client fetches one table and one
page of it. `fastapi` and `uvicorn` are the `service` extra and are imported only here, so
`import derivus` never needs them.

There is no auth, and CORS is open by default, because the service is a TRUSTED-NETWORK deployment:
put it behind something that terminates both, or narrow the origins with `DV_Service --origin`.
"""

import json
import logging
import os
import queue
import threading
import time

from collections import namedtuple, OrderedDict
from copy import deepcopy
from itertools import count

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import Context, content_hash, solve_deal_field
from .schema import mapping
from ._version import __version__
from .config import (CustomJsonEncoder, deal_at, remove_deal, sniff_indent, splice_deal,
                     update_market_quote)

LOG = logging.getLogger(__name__)

#: Cost class off `Calculation['Object']`: a light job jumps a heavy one among those still WAITING.
#: Anything unnamed is heavy, and `run_job` is the one that names it when it turns out not to run.
COST_CLASS = {'BaseValuation': 0, 'CreditMonteCarlo': 1, 'HedgeMonteCarlo': 1}
HEAVY = 1

#: `--tick` with no value: a cadence a desk can watch a spot move on, without asking a terminal
#: for a whole surface more often than a surface moves.
TICK_SECONDS = 30.0

#: The metronome's backoff: this many failed ticks in a row stretch the interval by this factor
#: until one succeeds. A terminal that went away, a market that closed - keep beating, stop asking
#: every thirty seconds.
TICK_FAILURES = 3
TICK_STRETCH = 5

#: A quote prices on a LIVE spot when this workstation's terminal is up. That fetch rides in front
#: of EVERY quote rather than on the cadence, so it gets a budget of its own: a terminal that is not
#: there must cost a salesperson nothing they can feel.
SPOT_TIMEOUT_MS = 2000

#: And the cost is paid once. A failure that reached the terminal is remembered process-wide for
#: this long, so consecutive quotes skip the attempt instead of each re-paying the timeout. A plain
#: clock and no thread machinery: the worst a race can do here is one extra request.
SPOT_BACKOFF_SECONDS = 30.0

#: Why a quote fell back to the book's ticked spot, in ONE wording. A quote NEVER fails for want of
#: a live one - the book's is a real market, ticked on the cadence - so every branch below ends as
#: a named note beside the price rather than as a refusal.
NO_LIVE_SPOT = "priced on the book's ticked spot - {}"

#: `(monotonic stamp, note)` of the last live-spot attempt that REACHED the terminal and failed.
_SPOT_FAILURE = (0.0, None)

#: Browsers refuse a cross-origin fetch the server does not invite, so an SPA served from anywhere
#: but the service itself cannot call it at all without this. Read when the middleware stack is
#: built, which is the first request - long after `main` has parsed `--origin`.
ORIGINS = ['*']

#: The one piece of contract `/schema` does not describe: the ENVELOPE the declarations sit inside.
#: `Config.read_json` is the source of truth, and this is a skeleton of exactly what it reads - a
#: complete job that loads, prices and validates clean, with the dates and the amount as
#: placeholders. What is not guessable from the declarations is the shape: market data lives under
#: `MergeMarketData.ExplicitMarketData` (or behind a `MarketDataFile` path instead), and a deal is a
#: `.Deal` token inside `Deals.Deals.Children[].Instrument`, nested by each node's own `Children`.
JOB_SKELETON = {'Calc': {
    'Calculation': {'Object': 'BaseValuation', 'Base_Date': {'.Timestamp': '2024-06-28'},
                    'Currency': 'USD', 'MCMC_Simulations': 1, 'Random_Seed': 1},
    'Deals': {'Tag_Titles': '', 'Reference': 'skeleton', 'Deals': {'Children': [
        {'Instrument': {'.Deal': {
            'Object': 'FixedCashflowDeal', 'Reference': 'CF1', 'Currency': 'USD',
            'Discount_Rate': 'USD', 'Calendars': None, 'Amount': 1000000.0,
            'Payment_Date': {'.Timestamp': '2026-06-28'}}}}]}},
    'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': {
        'System Parameters': {'Base_Currency': 'USD', 'Base_Date': {'.Timestamp': '2024-06-28'}},
        'Price Factors': {
            'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Spot': 1.0},
            'InterestRate.USD': {
                'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                'Curve': {'.Curve': {'meta': [], 'data': [[0.0, 0.02], [5.0, 0.02]]}}}}}}}}

#: What the worker needs off the request thread: the id to file under, the Context to run, and the
#: replay tuple that id was hashed from.
Job = namedtuple('Job', 'result_id context replay')


def as_json(obj):
    """Plain JSON, through the one encoder the codebase already has for what JSON has no form for —
    a Curve, a Timestamp, a results table."""
    return json.loads(json.dumps(obj, cls=CustomJsonEncoder))


def load(job):
    """A Context over a posted job, built by the decoder that reads a job file."""
    return Context().load_json((json.dumps(job), 'posted'))


def replay(context):
    """The four coordinates a reported number replays from. Hashed, they are also its `result_id`,
    which is what makes an identical submission one execution: the tuple names the numbers."""
    return {'plan_hash': context.plan_hash(), 'values_hash': context.values_hash(),
            'engine_version': __version__,
            'seed': context.current_cfg.deals['Calculation'].get('Random_Seed')}


def cost(calculation):
    """The queue's cost class for this job, and a crude size ESTIMATE beside it.

    The class is what the dispatcher orders by. The estimate is paths times grid points -
    `Batch_Size x Simulation_Batches x` the number of segments `Time_Grid` declares - and the last
    factor is a proxy, not the grid: the real one needs the horizon the deal walk finds. A field a
    calculation does not carry counts as one, so a base valuation estimates 1.
    """
    grid = len(str(calculation.get('Time_Grid', '')).split()) or 1
    return {'class': COST_CLASS.get(calculation['Object'], HEAVY),
            'estimate': calculation.get('Batch_Size', 1) * calculation.get(
                'Simulation_Batches', 1) * grid,
            'basis': 'Batch_Size x Simulation_Batches x Time_Grid segments - an estimate'}


def tables_of(results):
    """Every table in a `Results` tree, flat, under the path that names it.

    `cashflows` and `scenarios` are dicts of tables rather than tables, so they arrive as
    `cashflows/ZAR` and `scenarios/FxRate.ZAR`: a client fetches one table, and a tree has no page.
    """
    tables = {}
    for name, value in results.items():
        if isinstance(value, dict) and '.DataFrame' not in value:
            tables.update({'{}/{}'.format(name, path): table
                           for path, table in tables_of(value).items()})
        else:
            tables[name] = value
    return tables


def shape(table):
    """What a client renders a header and a pager from: how many rows a table has and what its
    columns are called. A serialised ndarray has rows and no column labels; a scalar is one row."""
    frame = table.get('.DataFrame') if isinstance(table, dict) else None
    if frame is not None:
        return {'rows': len(frame['data']), 'columns': frame['columns']}
    return {'rows': len(table) if isinstance(table, list) else 1, 'columns': []}


def rows_of(table):
    """The rows a table pages along, and the index labelling them — a frame's `data` and `index`,
    an ndarray's own list, or a scalar as the single row it is."""
    frame = table.get('.DataFrame') if isinstance(table, dict) else None
    if frame is not None:
        return frame['data'], frame['index']
    return (table if isinstance(table, list) else [table]), []


class PlanCache:
    """The parsed job, kept under the name of its plan, so EXECUTE can be plan plus values patch.

    What is cached today is the PARSE — decoding a job document and building its Deals, Curves and
    DateLists. Caching the COMPILE arrives behind the same verb with the live refill; `plan_id` is
    the content hash either way, so a client written against this one does not move.

    The entry is PRISTINE and every read is a deep copy, because an execute patches the market and
    a describe walks the deal tree calling `reset` on every instrument. Two executes off one plan
    therefore cannot contaminate each other or the plan they came from. Bounded and
    least-recently-used: a plan is a parse, cheap to redo, and never the record of anything — the
    replay tuple is.
    """

    def __init__(self, size=32):
        self.size = size
        self.plans = OrderedDict()
        self.lock = threading.Lock()

    def put(self, context):
        with self.lock:
            plan_id = context.plan_hash()
            self.plans[plan_id] = context
            self.plans.move_to_end(plan_id)
            if len(self.plans) > self.size:
                self.plans.popitem(last=False)
            return plan_id

    def get(self, plan_id):
        with self.lock:
            if plan_id not in self.plans:
                return None
            self.plans.move_to_end(plan_id)
            return deepcopy(self.plans[plan_id])


class ComputeExecutor:
    """One worker thread over one priority queue, and the result store it alone writes.

    There is no cpu lane. A base valuation IS a Monte Carlo for an autocall or a TARF book, so
    every priced job takes the same road and device selection stays where it already lives — in
    the engine. What the queue orders is COST CLASS: a base valuation jumps a simulation among the
    jobs still waiting, within a class it is first in first out, and a running job is never
    preempted.

    A job is filed under the hash of its replay tuple, so submitting the same job twice is one
    execution and one result — dedupe and retry-idempotency are the same feature. A submission
    arriving while the first is still queued or running coalesces onto it rather than enqueueing a
    second copy, because the store is checked and written under the same lock that enqueues.

    A stored result holds the run's tables under `tables`, keyed by the path that names each one.
    That is what the two result endpoints project: one serves their shapes, the other serves one of
    them, a page at a time.
    """

    def __init__(self):
        self.results = {}
        self.lock = threading.Lock()
        self.queue = queue.PriorityQueue()
        # arrival order breaks ties within a cost class, and keeps the queue from ever comparing
        # two Jobs
        self.arrival = count()
        threading.Thread(target=self.work, daemon=True).start()

    def submit(self, job, cost):
        """File and enqueue the job unless its `result_id` is already known, and return the status
        the caller sees immediately — `queued`, or wherever an identical earlier submission got to.
        """
        with self.lock:
            if job.result_id not in self.results:
                self.results[job.result_id] = {'status': 'queued'}
                self.queue.put((cost, next(self.arrival), job))
            return self.results[job.result_id]['status']

    def result(self, result_id):
        with self.lock:
            return self.results.get(result_id)

    def work(self):
        """Run one job at a time, forever. A job that fails is a result like any other, so the
        thread survives it and the next job runs."""
        while True:
            _, _, job = self.queue.get()
            with self.lock:
                self.results[job.result_id] = {'status': 'running'}
            try:
                _, out = job.context.run_job()
                # Stats is a flat dict of timings and counts (plus Calibrations provenance), not a
                # table - through `tables_of` it would flatten into fake table paths
                result = dict(status='done', tables=tables_of(as_json(out['Results'])),
                              stats=as_json(out.get('Stats', {})), **job.replay)
            except Exception as error:
                result = {'status': 'error', 'error': str(error)}
            with self.lock:
                self.results[job.result_id] = result
            self.queue.task_done()


class Book:
    """One live job document on disk - the FILE is the source of truth.

    This is the state MCP, the SPA and Excel meet in: a booking lands as a file write, and every
    client sees it on its next read. Reads check mtime and re-parse on change, so an external edit
    is picked up too; the etag is a hash of the text, so a client polls one small GET and
    re-renders only when the book actually moved. Writes are atomic (write-temp-then-replace) in
    the file's own indent, and `mutate` holds one lock across read-edit-validate-write so two
    bookings cannot interleave. Every read parses fresh, so an edit a refusal abandons never
    reaches the cache.
    """

    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self._cache = (None, None, None)  # (mtime_ns, etag, text)

    def _read(self):
        stamp = os.stat(self.path).st_mtime_ns
        if stamp != self._cache[0]:
            # utf-8 named on both sides: Windows' locale default is cp1252, which misreads any
            # non-ascii book; newline translation on read makes the etag convention-independent
            with open(self.path, encoding='utf-8') as handle:
                text = handle.read()
            self._cache = (stamp, content_hash(text), text)
        return json.loads(self._cache[2]), self._cache[1]

    def read(self):
        """`(wire document, etag)` - a fresh parse, safe for the caller to mutate."""
        with self.lock:
            return self._read()

    def mutate(self, edit):
        """Read-modify-write under one lock. `edit(document)` returns `(write, outcome)`; a False
        first half leaves the file untouched - a refused booking is a read, not a write."""
        with self.lock:
            document, _ = self._read()
            write, outcome = edit(document)
            if write:
                text = json.dumps(document, indent=sniff_indent(self._cache[2]))
                temporary = self.path + '.tmp'
                with open(temporary, 'w', encoding='utf-8', newline='') as handle:
                    handle.write(text)
                os.replace(temporary, self.path)
                self._cache = (os.stat(self.path).st_mtime_ns, content_hash(text), text)
            return dict(outcome, etag=self._cache[1])


#: The live book, when `DV_Service --book` opened one. None is a 404 on every /book verb.
BOOK = None


app = FastAPI(title='derivus', version=__version__)
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=['*'], allow_headers=['*'])
EXECUTOR = ComputeExecutor()
PLANS = PlanCache()

#: Where a long job has got to, under the result id it will be filed as: `{done, total, note}`.
#: A terminal round trip is minutes of work behind one `result_id`, so the worker publishes its own
#: progress here and `/results/{result_id}` merges it while the job is queued or running. That
#: thread is the only writer and drops its entry as the job leaves, so a reader tolerates absence -
#: a finished result carries its outcome, never a stale count.
PROGRESS = {}


def context_for(body):
    """The Context an endpoint works on: the posted document parsed, or a deep copy of the plan it
    names. Nothing downstream can tell the two apart, which is the whole point of `plan_id`."""
    if 'plan_id' in body:
        context = PLANS.get(body['plan_id'])
        if context is None:
            raise HTTPException(404, 'Unknown plan {}'.format(body['plan_id']))
        return context
    return load(body)


@app.get('/schema', summary='Every declaration a front end renders from')
def schema():
    """Every declaration in the engine, and the version that emitted them.

    This is what makes a front end thin: panels, tables and enums are rendered from `schema.mapping`
    rather than restated, so a field gaining a `bind`, a valid value or a new deal type reaches the
    UI by being declared on the class.
    """
    return dict(as_json(mapping), engine_version=__version__)


@app.get('/schema/job', summary='The job envelope, as a skeleton that loads')
def schema_job():
    """The ENVELOPE the declarations sit inside — the one piece of contract `/schema` cannot state.

    `/schema` describes what goes in a `Calculation` block, a `Price Factors` block and a deal; this
    describes where those blocks go, which is not guessable from them: market data under
    `MergeMarketData.ExplicitMarketData` (or behind a `MarketDataFile` path instead of it), and a
    deal as a `.Deal` token inside `Deals.Deals.Children[].Instrument`, nested by each node's own
    `Children`. `Config.read_json` is the source of truth and this is a skeleton of what it reads:
    post it to `/validate` or `/execute` unchanged and it loads, prices and validates clean.
    """
    return JOB_SKELETON


@app.post('/validate', summary='What would stop this job running')
def validate(job: dict):
    """What would stop this job running, verbatim from `cx.validate()` — the authoring messages of
    every deal in the book, and the price factors it names that the market data has no block for.
    Answered inline: it runs nothing, prices nothing and never reaches the queue."""
    return context_for(job).validate()


@app.post('/describe', summary='What the engine parsed, and what running it would cost')
def describe(job: dict):
    """What the engine made of this job, without running any of it: the book counted by deal type,
    the price factors it reaches on both sides of the want-list, and the `Calculation` block as
    loaded — `cx.describe()` verbatim.

    The dispatcher's own reading rides alongside under `cost`: the class this job would be queued
    at, and a crude estimate of its size. Nothing here writes, so describing a job and then
    executing it reports exactly what executing it alone reports.
    """
    context = context_for(job)
    return dict(as_json(context.describe()),
                cost=cost(context.current_cfg.deals['Calculation']))


@app.post('/prepare', summary='Parse a job and name it by its plan')
def prepare(job: dict):
    """Parse the job, name it by its plan hash and keep the parse under that name.

    `/execute` then takes `{"plan_id": ..., "Patch": {...}}` in place of the document, which is what
    a client streaming market values wants: the plan is sent once and a tick is a delta. The parse
    is cached, not the compile — that arrives behind this same verb with the live refill.
    """
    context = load(job)
    return {'plan_id': PLANS.put(context), 'values_hash': context.values_hash(),
            'engine_version': __version__}


@app.post('/execute', summary='Submit a job, and get the id its numbers will be filed under')
def execute(job: dict):
    """Submit a job — a document or a `plan_id` — optionally over a values `Patch`, a delta exactly
    as `patch_market` takes it.

    The patch is applied BEFORE the hashes are taken, so `values_hash` describes what actually runs
    and two clients patching to the same market get one execution. The answer is always
    `{result_id, status}`: poll `/results/{result_id}` for the numbers, however cheap the job.
    """
    context = context_for(job)
    context.patch_market(job.get('Patch', {}))
    stamp = replay(context)
    submitted = Job(content_hash(stamp), context, stamp)
    calculation = context.current_cfg.deals['Calculation']
    return {'result_id': submitted.result_id,
            'status': EXECUTOR.submit(submitted, cost(calculation)['class'])}


@app.get('/results/{result_id}', summary='Status, the replay tuple, and the shape of each table')
def results(result_id: str):
    """`queued` / `running` while it waits, then the run's SUMMARY: the replay tuple its numbers
    reproduce from, and every table it produced named with its row count and column labels.

    No cells. A credit Monte Carlo's exposure is dates by scenarios and does not fit in an answer
    anyone wants to hold, so a client reads the shapes here and fetches the one table it is showing
    from `/results/{result_id}/{table}`. A failed run carries the message and nothing else.

    A job that publishes its own progress - a Bloomberg round trip is minutes of terminal time -
    carries it under `progress` while it waits or runs, so one poll loop serves a cell that wants
    a number and a screen that wants a bar.
    """
    result = EXECUTOR.result(result_id)
    if result is None:
        raise HTTPException(404, 'Unknown result {}'.format(result_id))
    tables = result.get('tables')
    answer = result if tables is None else dict(
        result, tables={name: shape(table) for name, table in tables.items()})
    # a done result carries its outcome instead, and the worker has already dropped the entry
    progress = PROGRESS.get(result_id) if result['status'] in ('queued', 'running') else None
    return answer if progress is None else dict(answer, progress=progress)


@app.get('/results/{result_id}/{table:path}', summary='One table of a finished run, paged')
def table(result_id: str, table: str, offset: int = 0, limit: int = None):
    """One table, a page at a time: `rows` and `columns` are the whole table's, `data` is the rows
    from `offset` on, and `limit` caps how many of them travel. The default is the whole table.

    The name is the path `/results/{result_id}` published, so a table inside a group is fetched as
    `cashflows/ZAR`. An offset past the end is an empty page, not an error.
    """
    result = EXECUTOR.result(result_id)
    if result is None or table not in result.get('tables', {}):
        raise HTTPException(404, 'Unknown table {} of result {}'.format(table, result_id))
    found = result['tables'][table]
    rows, index = rows_of(found)
    end = len(rows) if limit is None else offset + limit
    return dict(shape(found), name=table, offset=offset,
                index=index[offset:end], data=rows[offset:end])


def live_book():
    if BOOK is None:
        raise HTTPException(404, 'No book is being served - start DV_Service --book <job file>')
    return BOOK


@app.get('/book', summary='The live job document this service serves')
def book():
    """The book, whole and in wire form, with the etag naming its current state. A client renders
    the document and then polls only the etag question - re-fetching when it moves - so a deal
    booked by any other client appears within a poll tick."""
    document, etag = live_book().read()
    return {'document': document, 'etag': etag, 'path': live_book().path}


def booked_node(deal):
    """A booked deal as the NODE a job document holds: `{'Instrument': {'.Deal': fields}}` with
    `Children` hanging off the node beside `Instrument`.

    A structure IS its legs, so `/book/quote` books a container and everything under it as ONE
    trade - and a composed deal arrives with the legs written INTO the container, which is the
    natural way to say "this deal, with these legs" and the one placement `Config` does not walk
    (`node['Children']`, never `deal['Children']`). Lifting them here is what makes a composed
    structure bookable as it stands rather than leg by leg; the deal block is copied rather than
    edited, because what a quote file holds is the record of what was quoted.
    """
    if 'Instrument' in deal:
        return deal
    deal = dict(deal)
    children = deal.pop('Children', None)
    node = {'Instrument': {'.Deal': deal}}
    if children:
        node['Children'] = children
    return node


def deal_references(node):
    """Every `Reference` in a booked subtree: the deal itself, then each leg under it. What the
    verdict has to read - a container whose LEG is misauthored is a refused booking, not a
    booking whose container had nothing said about it."""
    return [node['Instrument']['.Deal'].get('Reference')] + [
        reference for child in node.get('Children', [])
        for reference in deal_references(child)]


def deal_verdict(document, references, deal_path, already_missing):
    """The validate-before-write verdict on a document a change has already landed on:
    `(write, outcome)` for `Book.mutate`, written iff nothing is said against the CHANGED deals.

    What counts against them is their own authoring messages plus market data the book did not
    ALREADY lack - `already_missing` is that baseline, taken before the change - so a book failing
    elsewhere cannot block a correct change. Both mutating actions of `/book/deals` and every
    approved quote end here, which is what makes a refusal one wording rather than three.
    """
    verdict = load(document).validate()
    refused = [message for reference in references
               for message in verdict['deals'].get(reference, [])]
    refused += ['no market data for {}'.format(name)
                for name in sorted(set(verdict['factors']) - already_missing)]
    if refused:
        return False, {'written': False, 'refused': refused, 'validate': verdict}
    return True, {'written': True, 'deal_path': deal_path, 'validate': verdict}


def deal_edit(document, deal, parent_reference=None):
    """One deal - or one whole structured subtree - added to a wire document, as the ONE edit
    closure body `/book/deals` books through: the baseline taken, the node spliced in, the verdict
    read off the whole document.

    `splice_deal` gives a container an EMPTY `Children`, since a hand-booking fills it one deal at
    a time; a quoted structure arrives with its legs already composed, so they land with it - one
    booking, one atomic write, one verdict over the container and every leg.

    `/book/quote` books an approved quote through this same function rather than a second write
    path, so a structured deal is refused in exactly the wording - and by exactly the reading - a
    hand-booked one is.
    """
    already_missing = set(load(document).validate()['factors'])
    node = booked_node(deal)
    deal_path = splice_deal(document, node['Instrument']['.Deal'], parent_reference)
    if node.get('Children'):
        deal_at(document, deal_path)['Children'] = node['Children']
    return deal_verdict(document, deal_references(node), deal_path, already_missing)


@app.post('/book/deals', summary='Book, amend or delete one deal - validated, then written atomically')
def book_deals(request: dict):
    """`{action: 'add', deal, parent_reference?}`, `{action: 'amend', deal_path, fields}` or
    `{action: 'delete', deal_path}`. An amendment MERGES `fields` into the deal at `deal_path`.

    `deal` is a deal block, or a whole node - a container with its `Children` - which is how a
    quoted structure books: the legs land with the container, in one write and under one verdict.

    The contract is validate-before-write, one spelling for both mutating actions: the change
    lands on a copy, the whole document is validated, and the file is rewritten only if nothing is
    said against the CHANGED deal - its own authoring messages, or market data the book did not
    already lack. A book already failing elsewhere cannot block a correct change, and the caller
    sees the whole verdict either way. A refusal is `{written: False, ...}` and touches nothing -
    it is an answer, not an error.
    """
    action = request.get('action', 'add')
    if action not in ('add', 'amend', 'delete'):
        raise HTTPException(422, 'action must be add, amend or delete, not {!r}'.format(action))

    def edit(document):
        if action == 'delete':
            removed = remove_deal(document, request['deal_path'])
            return True, {'written': True,
                          'deleted': removed['Instrument']['.Deal'].get('Reference')}
        if action == 'amend':
            already_missing = set(load(document).validate()['factors'])
            deal = deal_at(document, request['deal_path'])['Instrument']['.Deal']
            deal.update(request['fields'])
            return deal_verdict(document, [deal.get('Reference')], request['deal_path'],
                                already_missing)
        return deal_edit(document, request['deal'], request.get('parent_reference'))

    try:
        return live_book().mutate(edit)
    except ValueError as error:
        raise HTTPException(422, str(error))


@app.post('/book/price', summary='Price the book, with an optional candidate deal - writes nothing')
def book_price(request: dict):
    """The what-if verb: `{deal?, parent_reference?, calculation_overrides?}` prices the book plus
    an optional candidate on an in-memory copy - the file never moves. Overrides merge into
    `Calc.Calculation`, so "with Greeks" or "as a CMC" needs no second write surface. Answers
    `{result_id, status}` exactly like `/execute`, and the same content addressing applies: the
    same what-if twice is one run."""
    document, _ = live_book().read()
    try:
        if request.get('deal') is not None:
            splice_deal(document, request['deal'], request.get('parent_reference'))
    except ValueError as error:
        raise HTTPException(422, str(error))
    document['Calc']['Calculation'].update(request.get('calculation_overrides', {}))
    context = load(document)
    stamp = replay(context)
    submitted = Job(content_hash(stamp), context, stamp)
    calculation = context.current_cfg.deals['Calculation']
    return {'result_id': submitted.result_id,
            'status': EXECUTOR.submit(submitted, cost(calculation)['class'])}


class CapturedErrors(logging.Handler):
    """What the bootstrap has to say for itself: `Config.bootstrap` reports a family that could
    not run or wrote nothing at ERROR and carries on, so a market update captures that channel
    and refuses the write when anything landed on it - a book must never carry a market its own
    bootstrap complained about."""

    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


#: What a tick cannot do without, in the one wording both market verbs refuse in.
NO_BOOTSTRAPPER = ('the book declares no Bootstrapper Configuration - nothing can turn quotes '
                   'into price factors')

#: What a ROUTINE tick refuses on when this workstation has never been verified. The metronome
#: does not provision - that is a person's act, minutes of terminal time - so it says so, names
#: the home it looked in, and keeps beating.
UNPROVISIONED = ('no security map in {} - a routine tick never provisions. Run '
                 'tick_market_from_bloomberg once (or DV_Bloomberg discover) to verify this '
                 "workstation's securities, and the cadence picks up from the next beat")


def market_edit(document, quotes, patch, bootstrap=None):
    """The market tick as ONE edit closure over a wire document, for `Book.mutate`: quote blocks
    installed or value-updated, a values patch applied, the bootstrap run, `(write, outcome)` back.

    `/book/market` and `/book/bloomberg` differ only in where the quotes came from, so the whole
    install-and-refuse semantic lives here once rather than twice: a bootstrap that reports an
    ERROR writes NOTHING and hands its messages back, because a book must never carry a market its
    own bootstrap complained about. `bootstrap` is `'Yes'`/`'No'`, or None for the default - run it
    iff quotes arrived.
    """
    outcome = {'installed': [], 'updated': []}
    for name in sorted(quotes):
        outcome[update_market_quote(document, name, quotes[name])].append(name)
    wants_bootstrap = (bootstrap if bootstrap is not None else
                       ('Yes' if quotes else 'No')) == 'Yes'
    market = document['Calc']['MergeMarketData']['ExplicitMarketData']
    if wants_bootstrap and not market.get('Bootstrapper Configuration'):
        raise ValueError(NO_BOOTSTRAPPER)
    if not (wants_bootstrap or patch):
        return bool(outcome['installed'] or outcome['updated']), dict(outcome, written=True)

    context = load(document)
    context.patch_market(patch)
    before = set(market.get('Price Factors', {}))
    if wants_bootstrap:
        captured = CapturedErrors()
        logging.getLogger().addHandler(captured)
        try:
            context.bootstrap()
        finally:
            logging.getLogger().removeHandler(captured)
        if captured.messages:
            return False, {'written': False, 'refused': captured.messages}
    params = context.current_cfg.params
    market['Price Factors'] = as_json(params['Price Factors'])
    market['Market Prices'] = as_json(params['Market Prices'])
    return True, dict(outcome, written=True, patched=sorted(patch),
                      new_factors=sorted(set(market['Price Factors']) - before))


@app.post('/book/market', summary="Tick the book's market - quotes in, values patched, bootstrapped")
def book_market(request: dict):
    """`{quotes?: {name: block}, patch?: {factor: {field: value}}, bootstrap?: 'Yes'|'No'}`.

    The practical tick path: a quote source (`derivus_bloomberg.to_market_prices_block`, a desk
    script, an MCP tool) posts `Market Prices` blocks; an update may move only each point's
    `Quoted_Market_Value` and `Timestamp` - structure is a re-authoring, refused by name. `patch`
    is the values delta exactly as `patch_market` takes it, so the engine's own refusal guards
    the structural half. The bootstrap (default: run iff quotes arrived) turns the quotes into
    the price factors the pricers read, and the book file gains the whole result in one atomic
    write - a bootstrap that reports an error writes NOTHING and hands the messages back.
    """
    live = live_book()

    def edit(document):
        return market_edit(document, request.get('quotes', {}), request.get('patch', {}),
                           request.get('bootstrap'))

    try:
        return live.mutate(edit)
    except (ValueError, KeyError) as error:
        raise HTTPException(422, str(error))


class BloombergJob:
    """A terminal round trip as ONE unit of queued work: the security map provisioned, every
    requested surface fetched and checked, and the whole lot installed and bootstrapped in one
    atomic write.

    `derivus_bloomberg` is imported INSIDE `run_job`. blpapi lives only on a terminal workstation,
    and the service has to start and serve every other verb on a machine that has never heard of
    it - so the dependency is reached at the moment a desk asks for a fetch, and nowhere else.

    The outcome is a book WRITE rather than tables, so it rides the run's own Stats under
    `Bloomberg`, exactly as a solve's coordinates ride `Solved`: the worker files every result the
    one way, and a job with no tables adds no second shape for a client to learn. Progress rides
    `PROGRESS` under the result id and is dropped in a `finally`, so a poller sees a count while
    the terminal is answering and the outcome once it has.

    `routine` is the metronome's mode: the interactive verb PROVISIONS a workstation that has no
    security map, and a cadence must not - verifying a whole vocabulary is minutes of terminal
    time and something a person asked for. So a routine job checks for the map first and refuses
    by name without opening a session at all, which is also what keeps the refusal reachable on a
    machine with no terminal.
    """

    def __init__(self, book, scope, result_id, routine=False):
        self.book, self.scope, self.result_id = book, scope, result_id
        self.routine = routine

    def note(self, note, done=0, total=0):
        PROGRESS[self.result_id] = {'done': done, 'total': total, 'note': note}

    def outcome(self, started, **fields):
        """The write's account of itself, with the trip's wall time beside it - what a poller
        reads off `stats.Bloomberg`, and what the metronome logs a landed tick by."""
        return None, {'Results': {}, 'Stats': {
            'Bloomberg': dict(fields, seconds=round(time.perf_counter() - started, 2))}}

    def run_job(self):
        import datetime

        from derivus_bloomberg import discover, fetch_fx_vol, security_map, to_market_prices_block
        from derivus_bloomberg.session import BloombergSession

        started = time.perf_counter()
        surface = {key: self.scope[key] for key in ('expiries', 'pillars') if key in self.scope}
        try:
            if self.routine and discover.provisioned() is None:
                return self.outcome(started, written=False, refused=[UNPROVISIONED.format(
                    security_map.home())])
            self.note('provisioning the security map')
            with BloombergSession(timeout_ms=30000) as session:
                def on_batch(done, total):
                    self.note('verifying securities', done, total)

                map_document, created = discover.provision(
                    session, datetime.date.today(), on_batch=on_batch)
                pairs = self.scope.get('pairs') or sorted(map_document['blocks']['fx_vol'])
                quotes, late = {}, {}
                for index, pair in enumerate(pairs, 1):
                    self.note('fetching {} {}/{}'.format(pair, index, len(pairs)),
                              index - 1, len(pairs))
                    definition = security_map.fx_vol_definition(map_document, pair, **surface)
                    late.update(security_map.stale(session, sorted(
                        {quote.security for quote in definition.securities.values()})))
                    # one late quote refuses the whole tick - and the rest are still checked, so
                    # the refusal names every dead security rather than the first one found
                    if not late:
                        snapshot = fetch_fx_vol(session, definition)
                        quotes['FXVolPrices.' + snapshot.surface_name] = as_json(
                            to_market_prices_block(snapshot))
            if late:
                return self.outcome(started, written=False, refused=[
                    '{} is stale - {}'.format(name, why) for name, why in sorted(late.items())])

            self.note('installing and bootstrapping', len(pairs), len(pairs))
            written = self.book.mutate(
                lambda document: market_edit(document, quotes, {}, 'Yes'))
            return self.outcome(started, provisioned=created, **written)
        finally:
            PROGRESS.pop(self.result_id, None)


def submit_bloomberg(scope, routine=False):
    """The terminal round trip as a queued job - the ONE seam `/book/bloomberg` and the metronome
    both ride.

    Same job class, same single-worker queue, same result store: a posted tick and a beat of the
    cadence are indistinguishable downstream, so a tick's write serialises with every pricing and
    lands atomically exactly as a posted one does. `routine` is the only thing that separates
    them, and all it says is "do not provision".
    """
    live = live_book()
    document, etag = live.read()
    if not document['Calc']['MergeMarketData']['ExplicitMarketData'].get(
            'Bootstrapper Configuration'):
        raise HTTPException(422, NO_BOOTSTRAPPER)
    # a fetch is an ACT against the terminal rather than a function of the book, so the submission
    # clock names it: two ticks against an unmoved book are two trips, never one coalesced result
    result_id = content_hash({'book': etag, 'bloomberg': scope, 'at': time.perf_counter()})
    submitted = Job(result_id, BloombergJob(live, scope, result_id, routine), {})
    return {'result_id': result_id, 'status': EXECUTOR.submit(submitted, HEAVY)}


@app.post('/book/bloomberg', summary="Fetch the desk's FX vol surfaces off Bloomberg and tick the book")
def book_bloomberg(request: dict):
    """`{pairs?: [PAIR], expiries?: [LABEL], pillars?: [DELTA]}` - the whole tick, terminal to
    book, as one queued job.

    `DV_Service --tick SECONDS` submits this same job on a cadence, so on a ticking service this
    verb is what FORCES a refresh between beats - and what provisions the workstation, since the
    metronome deliberately will not.

    The scope defaults to the desk's own: every `fx_vol` pair the security map carries, at the
    expiries it verified and the delta pillars `security_map.fx_vol_definition` defaults to. The
    map is provisioned first - discovered and written when this workstation has none - then every
    requested surface is checked for staleness and fetched, and what came back is installed and
    bootstrapped through the same seam `/book/market` uses, in ONE atomic write. A quote whose
    last print is late refuses the whole tick BY NAME and writes nothing: a dead series keeps
    answering with a plausible number, and a book must never carry one unremarked.

    Answers `{result_id, status}` exactly like `/execute`, because a terminal round trip is
    minutes of work: `/results/{result_id}` carries `progress` while it runs and the outcome under
    `stats.Bloomberg` when it is done. The id names the ACT rather than the numbers - a fetch is a
    trip to the terminal, so two ticks against an unmoved book are two fetches, never one.
    """
    return submit_bloomberg(
        {key: request[key] for key in ('pairs', 'expiries', 'pillars') if key in request})


class Metronome:
    """`DV_Service --tick SECONDS` - the background market ticker, as a thread that SUBMITS.

    It owns no fetching of its own: every beat goes through `submit_bloomberg`, the same seam
    `/book/bloomberg` rides, so a tick is the same job class on the same single-worker queue and
    its write serialises with pricings and lands atomically like any other. What the thread adds
    is a clock and three disciplines.

    NEVER STACK. A terminal round trip can outlast an interval, and the result id carries a clock
    stamp precisely so that two ticks never coalesce onto one result - which means nothing
    downstream would stop a slow terminal accumulating a queue of them. So the beat that finds its
    predecessor still `queued` or `running` is skipped, on a reading taken straight off the
    executor's own store rather than off bookkeeping of the metronome's own.

    NEVER PROVISION. An unprovisioned `DV_HOME` refuses by name and the cadence carries on:
    verifying a workstation's whole vocabulary is minutes of terminal time, and stays the
    interactive verb's act.

    NEVER DIE. A failed tick - the terminal down, the market closed, a refusal - is ONE warning
    line naming the cause, and the book is untouched by construction, since a job that refuses
    writes nothing. Three failures in a row stretch the interval fivefold until one succeeds, so a
    workstation that lost its terminal overnight is not still asking every thirty seconds by
    morning. Nothing a beat can raise reaches the loop.
    """

    def __init__(self, interval, scope=None):
        self.interval = interval
        self.scope = scope or {}
        self.pending = None
        self.failures = 0
        self.stretched = False

    def pending_status(self):
        """Where the tick this thread last submitted got to - `queued`, `running`, `done`,
        `error`, or None when there is nothing to wait on. The whole skip decision is this."""
        outcome = EXECUTOR.result(self.pending) if self.pending is not None else None
        return None if outcome is None else outcome['status']

    def cause(self, outcome):
        """Why the last tick failed, or None if it did not. A job that raised carries its message;
        a job that refused carries `refused` under its own Stats - and anything that did not write
        is a failure whatever else it says, because the write is the point of a tick."""
        if outcome.get('status') == 'error':
            return outcome.get('error')
        bloomberg = outcome.get('stats', {}).get('Bloomberg', {})
        if bloomberg.get('written'):
            return None
        return '; '.join(bloomberg.get('refused', [])) or 'the tick wrote nothing'

    def failed(self, cause):
        self.failures += 1
        LOG.warning('bloomberg tick failed (%d in a row): %s - the book is untouched',
                    self.failures, cause)
        # `==`, not `>=`: the stretch is announced once, not once per failure after it
        if self.failures == TICK_FAILURES:
            self.stretched = True
            LOG.warning('bloomberg ticks stretched to every %gs until one succeeds',
                        self.interval * TICK_STRETCH)

    def succeeded(self, outcome):
        if self.stretched:
            LOG.warning('bloomberg tick recovered - back to every %gs', self.interval)
        self.failures, self.stretched = 0, False
        LOG.debug('bloomberg tick landed in %ss',
                  outcome.get('stats', {}).get('Bloomberg', {}).get('seconds'))

    def beat(self):
        """One beat: skip if the last tick is still in flight, else judge it and submit the next.

        Judging at the TOP rather than after submitting is what keeps a beat a submission - the
        metronome never waits on a terminal, it only reads what the worker has already filed.
        """
        if self.pending_status() in ('queued', 'running'):
            LOG.debug('bloomberg tick skipped - the last one is still in flight')
            return
        settled = EXECUTOR.result(self.pending) if self.pending is not None else None
        # cleared BEFORE the submit, so a submit that refuses cannot leave a settled result to be
        # judged - and counted - a second time on the next beat
        self.pending = None
        if settled is not None:
            cause = self.cause(settled)
            if cause is None:
                self.succeeded(settled)
            else:
                self.failed(cause)
        self.pending = submit_bloomberg(self.scope, routine=True)['result_id']

    def run(self):
        while True:
            time.sleep(self.interval * (TICK_STRETCH if self.stretched else 1))
            try:
                self.beat()
            except Exception as error:
                # the thread outlives every beat: a book that went away, a terminal that did, a
                # bug in a fetch. `detail` is an HTTPException's message - the refusals the shared
                # seam already spells, in the wording a POST would have got back
                self.failed(getattr(error, 'detail', None) or error)

    def start(self):
        threading.Thread(target=self.run, daemon=True).start()
        return self


class SolveJob:
    """A solve as ONE unit of queued work: the loop runs on the worker like any pricing, so it
    cannot interleave with other jobs, and its answer files under a result_id like any run. The
    loop itself is `derivus.solve_deal_field` - this only packages what came back, with the
    solved coordinates riding the run's own Stats."""

    def __init__(self, document, deal_path, solve):
        self.document, self.deal_path, self.solve = document, deal_path, solve

    def run_job(self):
        solved, evaluations, residual, out = solve_deal_field(
            self.document, self.deal_path, **self.solve)
        out['Stats'] = dict(out.get('Stats', {}), Solved=dict(
            self.solve, value=solved, evaluations=evaluations, residual=residual))
        return None, out


@app.post('/book/solve', summary='Solve one field of a candidate deal to a target value')
def book_solve(request: dict):
    """The structuring verb: `{deal, field, target?, bounds?, tolerance?,
    calculation_overrides?}` finds the value of `deal[field]` at which the deal's own base
    valuation marks at `target` (default 0 - par), against the book's market data, on an
    in-memory copy - the file never moves.

    Not a calculation type: a root find over ordinary base valuations (brentq inside declared
    bounds, else a secant - exact in two pricings for a field the value is affine in). The
    candidate prices ALONE on the book's market data, since a deal's own base-valuation row does
    not depend on its siblings and a lone deal compiles faster per iterate. Answers
    `{result_id, status}`; the solved value, pricing count and residual arrive under the
    result's `stats.Solved`, and the result's tables are the run AT the solved value.
    """
    document, etag = live_book().read()
    try:
        document['Calc']['Deals']['Deals']['Children'] = []
        deal_path = splice_deal(document, request['deal'])
    except ValueError as error:
        raise HTTPException(422, str(error))
    document['Calc']['Calculation'].update(request.get('calculation_overrides', {}))
    document['Calc']['Calculation']['Object'] = 'BaseValuation'
    solve = {key: request[key] for key in ('field', 'target', 'bounds', 'tolerance')
             if key in request}
    if 'field' not in solve:
        raise HTTPException(422, 'a solve names the field it moves')
    # the identity is the request against this exact book state - the same solve twice is one run
    submitted = Job(content_hash({'book': etag, 'solve': solve, 'deal': request['deal'],
                                  'calculation': document['Calc']['Calculation']}),
                    SolveJob(document, deal_path, solve), {})
    return {'result_id': submitted.result_id, 'status': EXECUTOR.submit(submitted, HEAVY)}


def dv_home():
    """The desk's own user-data directory: `DV_HOME`, or `~/.derivus` when it is unset.

    One env var names where a desk's files live, the same way `RF_SERVICE_URL` names the service
    for every client. Read on every call rather than captured at import, because the book `main`
    opens and the quotes a running service files are the same setting answered at two different
    moments - and a client that moves it moves both.
    """
    return os.path.expanduser(os.environ.get('DV_HOME', os.path.join('~', '.derivus')))


def quote_dir():
    """Where a pending trade waits for its approval: `DV_HOME/tmp`, created on first use.

    A quote is not a booking. `/book/structure` files the whole pending trade here under its own
    `quote_id` and `/book/quote` books it from here, so the desk that gave the quote and the desk
    that approves it meet in a FILE - and the file stays after the booking, because what was
    quoted at what market is the audit trail of why the book carries what it carries.
    """
    return os.path.join(dv_home(), 'tmp')


def spot_failure():
    """The remembered live-spot failure while it is still believed, or None.

    Read before every attempt: a dead terminal is learned ONCE and costs the next half-minute of
    quotes nothing at all, which is the whole difference between a feature and a two-second tax on
    every price a desk asks for.
    """
    stamp, note = _SPOT_FAILURE
    return note if note is not None and time.monotonic() - stamp < SPOT_BACKOFF_SECONDS else None


def remember_spot_failure(note):
    """Remember a live-spot failure that REACHED the terminal - only those. A home with no map or a
    pair the map never verified costs a stat and a dict lookup, and the home a note names can change
    between two quotes, so remembering those would buy nothing and make the note lie."""
    global _SPOT_FAILURE
    _SPOT_FAILURE = (time.monotonic(), note)


def live_spot_crosses(base_currency, currencies, host=None, port=None):
    """`({PAIR: cross}, None)` off this workstation's terminal, or `(None, note)` naming why not.

    ONE request for every pair at once, through the same `derivus_bloomberg` machinery a tick
    fetches a surface with: the security map says which verified ticker prices each currency
    against the book's own base, and the session's tolerant reader answers them together.
    `derivus_bloomberg` is imported HERE for the reason `BloombergJob` imports it inside `run_job` -
    blpapi lives only on a terminal workstation, and the service serves every other verb on machines
    that have never heard of it.

    An unprovisioned home, a missing blpapi, a pair the map never verified: all of them end as a
    named note, never as an error and never as provisioning - verifying a workstation's vocabulary
    is minutes of terminal time and a person's act, exactly as the metronome has it.

    `host`/`port` are the session's own endpoint arguments threaded through rather than hard-coded,
    so a workstation whose Desktop API is not on the default socket reaches this one seam too.
    """
    remembered = spot_failure()
    if remembered is not None:
        return None, remembered
    try:
        from derivus_bloomberg import security_map
        from derivus_bloomberg.discover import provisioned
    except ImportError as error:
        return None, NO_LIVE_SPOT.format('derivus_bloomberg is not importable ({})'.format(error))
    try:
        path = provisioned()
        if path is None:
            return None, NO_LIVE_SPOT.format(
                'no security map in {} - a quote never provisions'.format(security_map.home()))
        map_document = security_map.load(path)
        pairs = {}
        for currency in currencies:
            pair, security = security_map.fx_spot_route(map_document, currency, base_currency)
            if pair is not None:
                pairs[pair] = security
    except Exception as error:
        return None, NO_LIVE_SPOT.format(error)
    if not pairs:
        return None, NO_LIVE_SPOT.format(
            'nothing to fetch - the pair is quoted in {} against itself'.format(base_currency))

    try:
        from derivus_bloomberg.session import BloombergSession
        # the connect is bounded as tightly as the request: a terminal that is not listening spends
        # its whole budget in the socket, where `timeout_ms` alone would never reach
        endpoint = {name: value for name, value in (('host', host), ('port', port))
                    if value is not None}
        with BloombergSession(timeout_ms=SPOT_TIMEOUT_MS, connect_timeout_ms=SPOT_TIMEOUT_MS,
                              **endpoint) as session:
            values = security_map.fetch_fx_spot(session, pairs.values())
    except Exception as error:
        # this one reached the terminal, so this is the one worth remembering
        note = NO_LIVE_SPOT.format(error)
        remember_spot_failure(note)
        return None, note
    return {pair: values[security] for pair, security in pairs.items()}, None


def patch_live_spot(document, params):
    """This workstation's LIVE spot put onto the document a quote is about to price, and the
    account of which spot that turned out to be.

    A spot is `bind='value'` data and `document` is the job's OWN copy - `/book/structure` hands
    every job a fresh parse - so the book FILE is never written by a quote. Only the spot moves:
    the vol surface and the curves stay the book's, because the cadence owns those and a
    delta-quoted FX surface is meant to be read at whatever spot is standing. A spot, alone, is
    stale within seconds of a quote being given.

    Answers the `source`/`note` half of the outcome's `spot` block; the runner fills in the value
    it actually priced on, so the two cannot disagree.
    """
    from . import structures

    try:
        currencies = structures.split_pair(params['pair'])
    except (KeyError, TypeError, ValueError) as error:
        return {'source': 'book', 'note': NO_LIVE_SPOT.format(error)}
    base = structures.base_currency(document)
    if not base:
        return {'source': 'book', 'note': NO_LIVE_SPOT.format(
            'the book declares no Base_Currency to read a cross against')}
    crosses, note = live_spot_crosses(base, currencies)
    if crosses is None:
        return {'source': 'book', 'note': note}
    try:
        structures.with_live_spots(document, crosses)
    except ValueError as error:
        return {'source': 'book', 'note': NO_LIVE_SPOT.format(error)}
    return {'source': 'terminal', 'note': None}


class StructureJob:
    """A structured quote as ONE unit of queued work: the recipe's solves run on the worker like
    any pricing, and the pending trade is filed before the answer is published.

    `derivus.structures` is imported INSIDE `run_job`, the way `BloombergJob` reaches its own
    package: the runner prices and solves leg by leg, so it belongs to the moment a desk asks for
    a quote rather than to import time, and every other verb still serves on a tree whose
    structures module is missing or broken. The sheet writer is reached the same way and is
    genuinely optional - `xlsxwriter` is the `quote` extra, and a desk that has not installed it
    gets the quote with `files['sheet_note']` naming the install, never a refusal.

    The outcome is the whole quote rather than tables, so it rides the run's own Stats under
    `Quote`, exactly as a solve's coordinates ride `Solved` and a fetch's write rides `Bloomberg`.

    THE DOCUMENT TRAVELS WHOLE, which is what lets the quote read its own mandate: the book's
    `Quote Policy` block rides the same copy the legs price against, and `structures.quote` reads
    it there - so the risk-impact half is a property of the BOOK rather than of the service, and
    the library verb behaves identically with no service anywhere near it.

    THE SPOT IS LIVE. Before the recipe runs, `patch_live_spot` puts this workstation's terminal
    spot onto THIS job's copy of the book - the surface and the curves stay the book's, since the
    cadence owns those and a delta-quoted surface reads at whatever spot is standing. The book file
    is untouched either way, and the outcome's `spot` block says which market was used, so a
    fallback is a note beside a price rather than a quote that did not happen.
    """

    def __init__(self, document, structure, params):
        self.document, self.structure, self.params = document, structure, params

    def sheet(self, directory, outcome):
        """The quote sheet beside the quote file, or the note saying why there is none. The import
        is what fails when `xlsxwriter` is absent, so it is the only thing guarded: a writer that
        raises is a real failure and travels."""
        try:
            from . import quote_sheet
        except ImportError as error:
            return {'sheet': None,
                    'sheet_note': 'no quote sheet ({}) - pip install derivus[quote]'.format(error)}
        path = os.path.join(directory, outcome['quote_id'] + '.xlsx')
        quote_sheet.write_sheet(path, outcome, self.document)
        return {'sheet': path}

    def run_job(self):
        from . import structures

        # the live spot lands BEFORE anything is priced, so `engine_spot`, every solve bracket and
        # every leg - and the sheet written from this same document - read one market
        spot_source = patch_live_spot(self.document, self.params)
        outcome = structures.quote(self.document, self.structure, self.params, spot_source)
        directory = quote_dir()
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, outcome['quote_id'] + '.json')
        outcome['files'] = dict(self.sheet(directory, outcome), quote=path)
        # one file IS the pending trade: what was quoted, and the deal that books it
        record = {'quote': {name: value for name, value in outcome.items() if name != 'deal'},
                  'deal': outcome['deal']}
        with open(path, 'w', encoding='utf-8', newline='') as handle:
            json.dump(as_json(record), handle, indent=2)
        return None, {'Results': {}, 'Stats': {'Quote': as_json(outcome)}}


@app.post('/book/structure', summary='Quote a named structure against the book')
def book_structure(request: dict):
    """`{structure: name, params: {...}}` - the sales verb: a structure named the way a desk names
    it, its parameters in MARKET terms, and back comes the whole quote with every leg priced and
    the solved ones solved.

    The recipe runs against the book's market data on an in-memory copy - the book file never
    moves, because a quote is not a trade. The SPOT on that copy is this workstation's live one
    when the terminal is up, and the book's last ticked one with a named reason when it is not;
    the surface and the curves are always the book's, and the outcome's `spot` block says which
    market the legs were struck on. What it DOES write is the pending trade:
    `DV_HOME/tmp/<quote_id>.json` carries the quote and the composed deal, and a `.xlsx` sheet
    lands beside it when the `quote` extra is installed. `/book/quote` with that id is the
    approval.

    Answers `{result_id, status}` exactly like `/execute`, because a recipe is a solve or three:
    `/results/{result_id}` carries the outcome under `stats.Quote` when it is done, files and all.
    The id names the ACT rather than the numbers - two identical asks are two quotes, never one
    coalesced result, the same reading `/book/bloomberg` takes of a trip to the terminal.
    """
    document, etag = live_book().read()
    structure = request.get('structure')
    if not structure:
        raise HTTPException(422, 'a quote names the structure it prices')
    params = request.get('params', {})
    # a quote is an ACT, not a function of the book: asking twice is two quotes, both filed
    result_id = content_hash({'book': etag, 'structure': structure, 'params': params,
                              'at': time.perf_counter()})
    submitted = Job(result_id, StructureJob(document, structure, params), {})
    return {'result_id': result_id, 'status': EXECUTOR.submit(submitted, HEAVY)}


@app.post('/book/quote', summary='Book a quote already given - the approval half of a structure')
def book_quote(request: dict):
    """`{quote_id}` - approve a quote and book it. The pending trade is read back from
    `DV_HOME/tmp/<quote_id>.json` and its deal goes through `deal_edit`, which is the same
    validate-before-write seam `/book/deals` books through: one atomic write, and a refusal that
    writes nothing and reads `{written: False, refused: [...]}` in the identical wording.

    The market may have moved since the quote was given, so the validation is against the book as
    it is NOW rather than as it was quoted - an approval is a booking, and a booking is refused on
    what it would land in. The file is not deleted: what was quoted, at what market, under what
    id, is why the book carries what it carries.

    What books is the MIRROR of the pending deal: the quote is client paper, and the book holds
    the bank's position, so the approval is where the side flips - `structures.mirror`, the same
    verb the risk-impact step will price the book-plus-candidate through, so the risk measured
    and the trade booked cannot disagree by a sign. The file keeps the client frame it was quoted
    in; the flip is the booking's act, not the quote's.
    """
    live = live_book()
    quote_id = request.get('quote_id')
    if not quote_id:
        raise HTTPException(422, 'an approval names the quote_id it books')
    # a quote id names a file in the desk's own tmp - never a path out of it
    if quote_id != os.path.basename(quote_id):
        raise HTTPException(422, '{!r} is not a quote id'.format(quote_id))
    path = os.path.join(quote_dir(), quote_id + '.json')
    if not os.path.isfile(path):
        raise HTTPException(404, 'Unknown quote {} - nothing under that id in {}'.format(
            quote_id, quote_dir()))
    with open(path, encoding='utf-8') as handle:
        pending = json.load(handle)

    from . import structures
    try:
        return live.mutate(lambda document: deal_edit(document, structures.mirror(pending['deal'])))
    except ValueError as error:
        raise HTTPException(422, str(error))


def blank_book():
    """A blank book: the job skeleton's envelope and market data with NO deals, dated today.

    A fresh desk has no job file, so `--book` pointed at a path that does not exist starts one -
    empty enough to mean nothing, furnished enough to work: the skeleton's USD factors stay so
    the first booking has market data to validate against (the missing-factor delta would refuse
    every deal on truly bare market data), and the dates are stamped to today because a starting
    point is authored at creation, unlike the skeleton itself, whose fixed date is a gated
    contract. The two bootstrapper families are declared for the same reason: a fresh desk's
    first market tick (`/book/bloomberg` provisioning included) must find something able to turn
    quotes into price factors, or first use dead-ends at a 422 no newcomer can act on."""
    import datetime

    document = json.loads(json.dumps(JOB_SKELETON))
    document['Calc']['Deals']['Deals']['Children'] = []
    stamp = {'.Timestamp': datetime.date.today().strftime('%Y-%m-%d')}
    document['Calc']['Calculation']['Base_Date'] = stamp
    market = document['Calc']['MergeMarketData']['ExplicitMarketData']
    market['System Parameters']['Base_Date'] = stamp
    market['Bootstrapper Configuration'] = {'FXVolSurfaceParameters': {},
                                            'InterestRateCurveParameters': {}}
    return document


def open_book(path):
    """The live book at `path`, created blank first when nothing is there yet."""
    if not os.path.isfile(path):
        with open(path, 'w', encoding='utf-8', newline='') as handle:
            json.dump(blank_book(), handle, indent=2)
    return Book(path)


def mount_ui(application, directory):
    """Serve a built web UI at `/ui`, and say whether there was one to serve.

    The UI is a CLIENT of this service, optional to the core library, and its source lives
    outside the package - so the mount is a flag, never an import-time assumption. A release
    wheel carries the BUILT UI as package data (`derivus/_ui`, staged by the publish pipeline),
    which `main` offers as the default directory when `--ui` is not given; a source tree without
    a build simply serves no UI. `html=True` makes `/ui/` serve `index.html`; it does NOT fall
    back to it for an unknown subpath, so the SPA navigates by tab state and hash rather than by
    URL path - a router would ship deep links that 404 on reload.
    """
    import os
    from fastapi.staticfiles import StaticFiles

    if not os.path.isfile(os.path.join(directory, 'index.html')):
        return False
    application.mount('/ui', StaticFiles(directory=directory, html=True), name='ui')
    return True


def main():
    """`DV_Service` - serve the app on one uvicorn worker, which is all the executor's single
    compute thread and in-memory store can be shared across."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description='Serve the derivus verbs over HTTP.')
    parser.add_argument('-b', '--bind', type=str, help='host to bind to', default='127.0.0.1')
    parser.add_argument('-p', '--port', type=int, help='port to listen on', default=8000)
    parser.add_argument('-o', '--origin', action='append',
                        help='browser origin to allow; repeatable, defaults to any')
    parser.add_argument('-u', '--ui', type=str, default=None,
                        help='directory holding a built web UI to serve at /ui; defaults to the '
                             'build the wheel shipped, when there is one')
    parser.add_argument('-k', '--book', type=str, default=None,
                        help='job JSON file to serve live at /book - created blank if missing; '
                             'the file is the book of record. Defaults to book.json in DV_HOME '
                             '(~/.derivus); --no-book serves the verbs with no book')
    parser.add_argument('--no-book', action='store_true',
                        help='serve no live book - /book answers 404')
    parser.add_argument('-t', '--tick', type=float, nargs='?', const=TICK_SECONDS, default=None,
                        metavar='SECONDS',
                        help='refresh the book off this workstation\'s Bloomberg terminal every '
                             'SECONDS (default {:g} when the flag is given no value), through the '
                             'same queued job POST /book/bloomberg submits - so a tick serialises '
                             'with pricings and lands atomically. Routine only: an unprovisioned '
                             'DV_HOME refuses by name and keeps beating, since provisioning is '
                             'the interactive act'.format(TICK_SECONDS))
    args = parser.parse_args()

    if args.origin:
        ORIGINS[:] = args.origin
    if args.ui:
        if not mount_ui(app, args.ui):
            parser.error('--ui {} holds no index.html - build the UI first'.format(args.ui))
    else:
        # a wheel ships the built UI as package data; a source tree without a build serves none
        mount_ui(app, os.path.join(os.path.dirname(os.path.abspath(__file__)), '_ui'))
    if not args.no_book:
        # the user-data home names where a desk's own files live; a missing book starts blank
        path = args.book if args.book else os.path.join(dv_home(), 'book.json')
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        global BOOK
        BOOK = open_book(path)

    if args.tick is not None:
        if args.no_book:
            parser.error('--tick has nothing to tick into: it refreshes the live book, and '
                         '--no-book serves none')
        if args.tick <= 0:
            parser.error('--tick {:g} is not an interval'.format(args.tick))
        # a ticker that could never tick is a MISCONFIGURATION, so it is a usage error at startup
        # rather than a warning every beat from a thread nobody is reading
        from derivus_bloomberg.errors import BloombergUnavailable
        from derivus_bloomberg.session import blpapi_module
        try:
            blpapi_module()
        except BloombergUnavailable as error:
            parser.error('--tick needs a Bloomberg terminal on this workstation - {}'.format(error))
        Metronome(args.tick).start()

    uvicorn.run(app, host=args.bind, port=args.port)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
