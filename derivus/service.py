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
import os
import queue
import threading

from collections import namedtuple, OrderedDict
from copy import deepcopy
from itertools import count

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import Context, content_hash
from .schema import mapping
from ._version import __version__
from .config import CustomJsonEncoder, deal_at, remove_deal, sniff_indent, splice_deal

#: Cost class off `Calculation['Object']`: a light job jumps a heavy one among those still WAITING.
#: Anything unnamed is heavy, and `run_job` is the one that names it when it turns out not to run.
COST_CLASS = {'BaseValuation': 0, 'CreditMonteCarlo': 1, 'HedgeMonteCarlo': 1}
HEAVY = 1

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
        if self._cache[0] != stamp:
            with open(self.path) as handle:
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
                with open(temporary, 'w') as handle:
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
    """
    result = EXECUTOR.result(result_id)
    if result is None:
        raise HTTPException(404, 'Unknown result {}'.format(result_id))
    tables = result.get('tables')
    return result if tables is None else dict(
        result, tables={name: shape(table) for name, table in tables.items()})


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


@app.post('/book/deals', summary='Book, amend or delete one deal - validated, then written atomically')
def book_deals(request: dict):
    """`{action: 'add', deal, parent_reference?}`, `{action: 'amend', deal_path, fields}` or
    `{action: 'delete', deal_path}`. An amendment MERGES `fields` into the deal at `deal_path`.

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
        already_missing = set(load(document).validate()['factors'])
        if action == 'amend':
            deal = deal_at(document, request['deal_path'])['Instrument']['.Deal']
            deal.update(request['fields'])
            deal_path = request['deal_path']
        else:
            deal = request['deal']
            deal_path = splice_deal(document, deal, request.get('parent_reference'))
        verdict = load(document).validate()
        refused = list(verdict['deals'].get(deal.get('Reference'), []))
        refused += ['no market data for {}'.format(name)
                    for name in sorted(set(verdict['factors']) - already_missing)]
        if refused:
            return False, {'written': False, 'refused': refused, 'validate': verdict}
        return True, {'written': True, 'deal_path': deal_path, 'validate': verdict}

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


def mount_ui(application, directory):
    """Serve a built web UI at `/ui`, and say whether there was one to serve.

    The UI is a CLIENT of this service, optional to the core library, and lives outside the
    package - so the mount is a flag, never an import-time assumption. `html=True` makes `/ui/`
    serve `index.html`; it does NOT fall back to it for an unknown subpath, so the SPA navigates
    by tab state and hash rather than by URL path - a router would ship deep links that 404 on
    reload.
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
                        help='directory holding a built web UI to serve at /ui')
    parser.add_argument('-k', '--book', type=str, default=None,
                        help='job JSON file to serve live at /book - the file is the book of record')
    args = parser.parse_args()

    if args.origin:
        ORIGINS[:] = args.origin
    if args.ui and not mount_ui(app, args.ui):
        parser.error('--ui {} holds no index.html - build the UI first'.format(args.ui))
    if args.book:
        if not os.path.isfile(args.book):
            parser.error('--book {} does not exist'.format(args.book))
        global BOOK
        BOOK = Book(args.book)

    uvicorn.run(app, host=args.bind, port=args.port)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
