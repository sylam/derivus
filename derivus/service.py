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

`Context` is the in-process binding. The calculation verbs here own no logic of their own - each
builds a Context, calls one of its methods and serialises what comes back - while the book,
projection and queue verbs are this module's own layer over the same engine. A web SPA, a marimo
notebook and an Excel add-in are all clients of these endpoints, so nothing client-specific may
enter the surface: anything a client needs that the calculation verbs cannot answer is a missing
verb on `Context`, not an endpoint that reaches inside.

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
| `GET /book` | the live job document the service serves - `DV_HOME/book.json` unless `--book` names another - and the etag naming its state |
| `POST /book/deals` | book or delete one deal - validated BEFORE an atomic write, refusal writes nothing |
| `POST /book/price` | price the book, optionally with a candidate deal spliced in - a what-if, writes nothing |
| `POST /book/solve` | solve one field of a candidate deal to a target value - a root find over base valuations, writes nothing |
| `POST /book/market` | tick the book's market: quote blocks installed or value-updated, a values patch applied, the bootstrap run - one atomic write |
| `POST /book/bloomberg` | provision the security map, fetch the desk's FX vol surfaces off the terminal and tick the book |
| `POST /book/hn` | calibrate one pair's Heston-Nandi parameters off its built surface - on request, never on the tick |
| `POST /book/structure` | quote a named structure against the book - legs solved, the pending trade filed under its quote id |
| `POST /book/quote` | book a quote already given - the approval half, refused exactly as a booking is |
| `GET /book/risk` | the book's CONSOLIDATED risk - one greeks run over every counterparty at once, cached on what it reads |
| `POST /book/xva` | recalculate the XVA projection - one queued CMC per netting set, every set or the named ones |
| `GET /book/xva` | the XVA projection as it stands - the last run per netting set, joined with the book's own set list |

THE BLOTTER'S TWO DATA VIEWS are deliberately different kinds of thing. `/book/risk` is WHOLE-BOOK
and counterparty-blind: one base valuation with `Greeks: 'First'`, answered inline and cached on the
content of everything the run reads. `/book/xva` is PER NETTING SET and a cached projection, a
credit Monte Carlo being minutes of device time that must never ride a tick - `DV_HOME/xva.json`
holds the last run of each set and a desk asks for a recalc. The projection is a MOSAIC: each row
carries its own `as_of`, a partial recalc moves only the rows it names, and staleness is data rather
than a failure. `cva` and `fva` are columns of one run, both reductions of the same exposure cube.
There is no `kva`: no calculation in the engine computes one.

Every POST body is either a job document or `{"plan_id": ...}` naming one already prepared, and
nothing downstream can tell them apart - the plan cache holds a pristine parse and every read is a
deep copy, so both report the same `result_id`. A posted job is a job FILE, `Config.read_json`
taking `(text, name)` as readily as a path, so there is no second parser.

There is no sync/async split at the API level: `/execute` always answers with a `result_id` and a
status, and a base valuation is simply `done` by the first poll - one contract an Excel RTD cell and
a browser poll loop can both be written against. A finished run answers with the SHAPE of its tables
and never their cells. `fastapi` and `uvicorn` are the `service` extra, imported only here.

There is no auth and CORS is open by default: this is a TRUSTED-NETWORK deployment. Put it behind
something that terminates both, or narrow the origins with `DV_Service --origin`.
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

from . import Context, content_hash, solve_deal_field, spine
from .bootstrappers import HestonNandiModelParameters
from .schema import mapping
from ._version import __version__
from .config import (as_json, deal_at, remove_deal, sniff_indent, splice_deal, tables_of,
                     update_market_quote, walk_job_deals)
from .spine import replay

LOG = logging.getLogger(__name__)

#: Cost class off `Calculation['Object']`: a light job jumps a heavy one among those still WAITING.
#: Anything unnamed is heavy, and `run_job` is the one that names it when it turns out not to run.
COST_CLASS = {'BaseValuation': 0, 'CreditMonteCarlo': 1, 'HedgeMonteCarlo': 1}
HEAVY = 1

#: `--tick` with no value: a cadence a desk can watch a spot move on, without asking a terminal
#: for a whole surface more often than a surface moves.
TICK_SECONDS = 30.0

#: The metronome's backoff: this many failed ticks in a row stretch the interval by this factor
#: until one succeeds. A terminal that went away, a market that closed - keep beating, stop asking.
TICK_FAILURES = 3
TICK_STRETCH = 5

#: A quote prices on a LIVE spot when this workstation's terminal is up. That fetch rides in front
#: of every quote rather than on the cadence, so it gets a budget of its own: a terminal that is not
#: there must cost a salesperson nothing they can feel.
SPOT_TIMEOUT_MS = 2000

#: And the cost is paid once: a failure that reached the terminal is remembered process-wide for
#: this long, so consecutive quotes skip the attempt. A plain clock and no thread machinery - the
#: worst a race can do is one extra request.
SPOT_BACKOFF_SECONDS = 30.0

#: Why a quote fell back to the book's ticked spot, in ONE wording. A quote never fails for want of
#: a live one - the book's is a real market, ticked on the cadence.
NO_LIVE_SPOT = "priced on the book's ticked spot - {}"

#: `(monotonic stamp, note)` of the last live-spot attempt that REACHED the terminal and failed.
_SPOT_FAILURE = (0.0, None)

#: An SPA served from anywhere but the service itself cannot call it without this. Read when the
#: middleware stack is built - the first request, long after `main` has parsed `--origin`.
ORIGINS = ['*']

#: The ENVELOPE the declarations sit inside, which `/schema` cannot describe: market data under
#: `MergeMarketData.ExplicitMarketData` (or behind a `MarketDataFile` path), a deal as a `.Deal`
#: token in `Deals.Deals.Children[].Instrument`. A complete job that loads, prices and validates.
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

#: What the worker needs off the request thread: the id to file under, the Context to run, the replay
#: tuple that id was hashed from, and - under a configured spine home - the attestation LANE and the
#: job document it is checkable from. Both default to None, so an existing caller mints nothing.
Job = namedtuple('Job', 'result_id context replay lane evidence', defaults=(None, None))


def load(job):
    """A Context over a posted job, built by the decoder that reads a job file."""
    return Context().load_json((json.dumps(job), 'posted'))


#: What `/execute` does with a lane it was never given. A run is recorded IFF its output will be
#: cited by a fact, so silence is the safe reading. The two verbs that KNOW which lane they are in -
#: the what-if and the quote - name their own below rather than defaulting here.
DEFAULT_LANE = spine.CURIOSITY


def evidence_for(document, context):
    """The two objects a standing attestation is checkable from, as the bytes they are stored as:
    the job document the plan recompiles from, and the values vector the run read.

    Taken on the REQUEST thread, before anything runs, where the claim's `values_hash` was taken -
    the spine checks the vector against that claim rather than believing it.

    What is stored is the `Calc` envelope alone: `Patch`, `plan_id` and `lane` are how the
    submission arrived, not the job, and the values vector carries the whole market as patched.
    """
    job = {'Calc': document['Calc']} if 'Calc' in document else document
    return {'job': spine.canonical(job), 'values': spine.values_of(context)}


def attests(submitted):
    """Whether this job's completion owes the record an attestation.

    Asked BEFORE the result is canonicalised, because canonicalising one is not free: a credit
    Monte Carlo's exposure is a cube, and a run in a lane that mints nothing must not pay for it.
    """
    if submitted.lane != spine.STANDING or not spine.configured():
        return False
    return submitted.evidence is not None


def attest(submitted, result):
    """Append the standing lane's `run_completed` for a job whose numbers exist, answering the
    envelope. `result` is the canonical result bytes; `attests` has already said yes.

    Telemetry and curiosity mint nothing, so this is a no-op on every job carrying no standing lane.

    A refusal here FAILS THE RUN: a run whose attestation was refused has not acquired standing, and
    serving its numbers as though it had is the unbacked citation the rule exists to prevent.

    A standing job with no evidence attests somewhere else, and there is exactly one: the structure
    quote, recorded as `quote_filed` - it pins the same two hashes and carries what was solved, so a
    `run_completed` beside it would be two records of one act.
    """
    return spine.complete_run(submitted.replay, submitted.lane, submitted.evidence['job'],
                              submitted.evidence['values'], result)


def attested(envelope):
    """What a caller is told about an attestation: where it landed and in which lane. Not the
    whole envelope - the record's own position is what a citation needs, and the rest of it is a
    fold away."""
    return {'lsn': envelope['lsn'], 'lane': envelope['lane'],
            'event_hash': envelope['event_hash']}


def cost(calculation):
    """The queue's cost class for this job, and a crude size estimate beside it.

    The class is what the dispatcher orders by. The estimate is paths times grid points, where the
    last factor is a PROXY - the real grid needs the horizon the deal walk finds. A field a
    calculation does not carry counts as one, so a base valuation estimates 1.
    """
    grid = len(str(calculation.get('Time_Grid', '')).split()) or 1
    return {'class': COST_CLASS.get(calculation['Object'], HEAVY),
            'estimate': calculation.get('Batch_Size', 1) * calculation.get(
                'Simulation_Batches', 1) * grid,
            'basis': 'Batch_Size x Simulation_Batches x Time_Grid segments - an estimate'}


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

    What is cached is the PARSE - decoding a job document and building its Deals, Curves and
    DateLists - not the compile; `plan_id` is the content hash either way.

    The entry is pristine and every read a deep copy, because an execute patches the market and a
    describe calls `reset` on every instrument, so two executes off one plan cannot contaminate
    each other. Bounded and least-recently-used: a plan is a parse, cheap to redo, and never the
    record of anything - the replay tuple is.
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

    There is no cpu lane: a base valuation IS a Monte Carlo for an autocall or a TARF book, so every
    priced job takes the same road and device selection stays in the engine. What the queue orders
    is COST CLASS - a base valuation jumps a simulation among the jobs still waiting, within a class
    it is first in first out, and a running job is never preempted.

    A job is filed under the hash of its replay tuple, so submitting the same job twice is one
    execution and one result - dedupe and retry-idempotency are the same feature. A submission
    arriving while the first is queued or running coalesces onto it, the store being checked and
    written under the lock that enqueues.

    COALESCING DEDUPES NUMBERS AND NOT LANES: a standing submission landing on a run still in flight
    is PROMOTED onto it in `submit`, and `finish` attests the promotion when the numbers land. An
    arrival after the result is published finds `done` and attests on its own thread.

    A stored result holds the run's tables under `tables`, keyed by path - what the two result
    endpoints project, one serving their shapes and the other one table a page at a time.
    """

    def __init__(self):
        self.results = {}
        # result_id -> the STANDING submission that coalesced onto a run still in flight, held
        # until `finish` attests it
        self.standing = {}
        self.lock = threading.Lock()
        self.queue = queue.PriorityQueue()
        # arrival order breaks ties within a cost class, so the queue never compares two Jobs
        self.arrival = count()
        threading.Thread(target=self.work, daemon=True).start()

    def submit(self, job, cost):
        """File and enqueue the job unless its `result_id` is already known, and return the status
        the caller sees immediately - `queued`, or wherever an identical earlier submission got to.

        A standing submission coalescing onto a run still in flight is PROMOTED here, this being
        the only place that sees both: the job the worker dequeued carries the first caller's lane,
        which for a what-if mints nothing. Taken under the lock the worker publishes under, so this
        arm and `/execute`'s cannot both miss.

        An `error` result is not promoted onto: there are no numbers, and the promotion would sit
        here forever against a run that is never dequeued again.
        """
        with self.lock:
            if job.result_id not in self.results:
                self.results[job.result_id] = {'status': 'queued'}
                self.queue.put((cost, next(self.arrival), job))
            elif attests(job) and self.results[job.result_id]['status'] in ('queued', 'running'):
                # first standing declaration wins - one run, one attestation
                self.standing.setdefault(job.result_id, job)
            return self.results[job.result_id]['status']

    def result(self, result_id):
        with self.lock:
            return self.results.get(result_id)

    def note(self, result_id, **fields):
        """Add to a stored result from another thread. Answers the result as it now stands.

        The worker is still the only thread that WRITES one: this adds to a finished result, under
        the same lock, for the one caller that is the standing lane's attestation of numbers this
        store already held. A result nobody is holding is left alone rather than conjured.
        """
        with self.lock:
            stored = self.results.get(result_id)
            if stored is not None:
                stored.update(fields)
            return stored

    def finish(self, job, result, canonical_result):
        """Attest a finished run where the record is owed one, and publish it. ONE transition.

        Attesting and publishing are one step under one lock: a standing run whose attestation was
        refused has not acquired standing, so its numbers must not become readable first. That also
        leaves `submit` no window - a standing submission either gets in before this block and is
        attested here, or it finds a published `done` and attests on its own thread.

        WHICH SUBMISSION IS ATTESTED is the job this thread dequeued wherever that one owes an
        attestation, and the promoted one only where it does not: the evidence is a document rather
        than a hash, and the run's own submitter is the one whose document actually ran.

        `canonical_result` is a CALLABLE for the reason `attests` is asked before it. A refusal
        leaves `self.results` as it found it and the worker records the error. The lock order is
        this one then the spine's writer, never the reverse.
        """
        with self.lock:
            owed = self.standing.pop(job.result_id, None)
            if owed is None or attests(job):
                owed = job
            if attests(owed):
                result['attested'] = attested(attest(owed, canonical_result()))
            self.results[job.result_id] = result
            return result

    def work(self):
        """Run one job at a time, forever. A job that fails is a result like any other, so the
        thread survives it and the next job runs."""
        while True:
            _, _, job = self.queue.get()
            with self.lock:
                self.results[job.result_id] = {'status': 'running'}
            try:
                _, out = job.context.run_job()
                # Stats is a flat dict of timings and counts, not a table - through `tables_of` it
                # would flatten into fake table paths
                result = dict(status='done', tables=tables_of(as_json(out['Results'])),
                              stats=as_json(out.get('Stats', {})), **job.replay)
                # attested for whichever submission of this tuple declared standing, which is not
                # always the one this thread dequeued
                self.finish(job, result, lambda: spine.result_of(out))
            except Exception as error:
                with self.lock:
                    # a run that produced nothing attests nothing, and the promotion goes with it
                    self.standing.pop(job.result_id, None)
                    self.results[job.result_id] = {'status': 'error', 'error': str(error)}
            self.queue.task_done()


class Book:
    """One live job document on disk - the FILE is the source of truth.

    The state MCP, the SPA and Excel meet in: a booking lands as a file write and every client sees
    it on its next read. Reads check mtime and re-parse on change, so an external edit is picked up
    too; the etag is a hash of the text, so a client polls one small GET. Writes are atomic
    (write-temp-then-replace) in the file's own indent, and `mutate` holds one lock across
    read-edit-validate-write. Every read parses fresh, so an abandoned edit never reaches the cache.
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


#: The live book: `DV_HOME/book.json` by default, another file where `DV_Service --book` names one,
#: and None under `--no-book` - which is a 404 on every /book verb.
BOOK = None


app = FastAPI(title='derivus', version=__version__)
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=['*'], allow_headers=['*'])
EXECUTOR = ComputeExecutor()
PLANS = PlanCache()

#: Where a long job has got to, under the result id it will be filed as: `{done, total, note}`, and
#: `/results/{result_id}` merges it while the job is queued or running. The worker thread is the only
#: writer and drops its entry as the job leaves, so a reader tolerates absence.
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

    What makes a front end thin: panels, tables and enums are rendered from the declarations rather
    than restated, so a new field or deal type reaches the UI by being declared on the class.
    """
    return dict(as_json(mapping), engine_version=__version__)


@app.get('/schema/job', summary='The job envelope, as a skeleton that loads')
def schema_job():
    """The ENVELOPE the declarations sit inside - the one piece of contract `/schema` cannot state.

    `/schema` describes what goes in each block; this describes where the blocks go: market data
    under `MergeMarketData.ExplicitMarketData` (or behind a `MarketDataFile` path), a deal as a
    `.Deal` token inside `Deals.Deals.Children[].Instrument`, nested by each node's `Children`.
    Post it to `/validate` or `/execute` unchanged and it loads, prices and validates clean.
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
    """What the engine made of this job without running any of it - `cx.describe()` verbatim - plus
    the dispatcher's own reading under `cost`. Nothing here writes, so describing a job and then
    executing it reports exactly what executing it alone reports.
    """
    context = context_for(job)
    return dict(as_json(context.describe()),
                cost=cost(context.current_cfg.deals['Calculation']))


@app.post('/prepare', summary='Parse a job and name it by its plan')
def prepare(job: dict):
    """Parse the job, name it by its plan hash and keep the parse under that name.

    `/execute` then takes `{"plan_id": ..., "Patch": {...}}` in place of the document, which is what
    a client streaming market values wants: the plan is sent once and a tick is a delta.
    """
    context = load(job)
    return {'plan_id': PLANS.put(context), 'values_hash': context.values_hash(),
            'engine_version': __version__}


def lane_of(body, default=DEFAULT_LANE):
    """The attestation lane this submission declared, checked - or None where this box records
    nothing.

    Without a spine home the lane is ACCEPTED AND INERT, an unknown one included. Where a home is
    configured the lane is a decision the record will act on, so an unknown one refuses by name -
    `derivus_spine.verbs.check_lane` owns that refusal, the lanes being the record's vocabulary.
    """
    declared = body.get('lane', default)
    if not spine.configured():
        return None
    return spine.check_lane(declared)


@app.post('/execute', summary='Submit a job, and get the id its numbers will be filed under')
def execute(job: dict):
    """Submit a job - a document or a `plan_id` - optionally over a values `Patch`, a delta exactly
    as `patch_market` takes it, and optionally in a declared `lane`.

    The patch is applied BEFORE the hashes are taken, so `values_hash` describes what actually runs
    and two clients patching to the same market get one execution. The answer is always
    `{result_id, status}`, plus an `attested` block on the one path that can answer it here - a
    standing run whose numbers this box had already computed. Otherwise the worker attests at
    completion and the block arrives with the numbers at `/results/{result_id}`.

    `lane` is `telemetry | curiosity | standing` and answers one question: will a fact cite these
    numbers? Only standing appends anything; the default is curiosity. Without a spine home the
    parameter is accepted and inert.

    A STANDING run must post its DOCUMENT: an attestation carries the job the plan recompiles from
    and `Context.save_json` is not a complete round trip, so a `plan_id` refuses with the remedy.
    """
    context = context_for(job)
    context.patch_market(job.get('Patch', {}))
    stamp = replay(context)
    try:
        lane, evidence = lane_of(job), None
        if lane == spine.STANDING:
            if 'plan_id' in job:
                raise HTTPException(422, 'a standing run is attested from the job document the '
                                         'plan recompiles from, and a plan_id names a parse rather '
                                         'than a document - post the job itself, or run it in the '
                                         'curiosity lane, which mints nothing')
            evidence = evidence_for(job, context)
    except spine.SpineRefused as refused:
        raise HTTPException(422, str(refused))
    submitted = Job(content_hash(stamp), context, stamp, lane, evidence)
    calculation = context.current_cfg.deals['Calculation']
    answer = {'result_id': submitted.result_id,
              'status': EXECUTOR.submit(submitted, cost(calculation)['class'])}
    # a standing run whose numbers already exist still attests - the lane is about standing rather
    # than arithmetic. One that coalesced onto a queued or running run was promoted in `submit`
    if attests(submitted) and answer['status'] == 'done':
        stored = EXECUTOR.result(submitted.result_id)
        try:
            answer['attested'] = attested(attest(submitted, spine.result_stored(stored)))
        except spine.SpineRefused as refused:
            raise HTTPException(422, str(refused))
        EXECUTOR.note(submitted.result_id, attested=answer['attested'])
    return answer


@app.get('/results/{result_id}', summary='Status, the replay tuple, and the shape of each table')
def results(result_id: str):
    """`queued` / `running` while it waits, then the run's SUMMARY: the replay tuple its numbers
    reproduce from, and every table it produced named with its row count and column labels.

    No cells - a client reads the shapes here and fetches the one table it is showing from
    `/results/{result_id}/{table}`. A failed run carries the message and nothing else.

    A job that publishes its own progress carries it under `progress` while it waits or runs, so
    one poll loop serves a cell that wants a number and a screen that wants a bar.
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
        raise HTTPException(404, 'No book is being served - start DV_Service (it serves '
                                 'DV_HOME/book.json by default; --book names another file)')
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

    A structure IS its legs, so `/book/quote` books a container and everything under it as one
    trade. A composed deal arrives with the legs written INTO the container, the one placement
    `Config` does not walk, so lifting them here is what makes it bookable as it stands. The deal
    block is copied rather than edited: a quote file is the record of what was quoted.
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
    already lack (`already_missing`, taken before the change), so a book failing elsewhere cannot
    block a correct change. Both mutating actions of `/book/deals` and every approved quote end
    here, which makes a refusal one wording rather than three.
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
    closure `/book/deals` books through: baseline taken, node spliced in, verdict read off the
    whole document.

    `splice_deal` gives a container an empty `Children`, a hand-booking filling it one deal at a
    time; a quoted structure arrives with its legs composed, so they land with it as one booking,
    one atomic write, one verdict. `/book/quote` books through this same function rather than a
    second write path, so a structured deal is refused exactly as a hand-booked one is.
    """
    already_missing = set(load(document).validate()['factors'])
    node = booked_node(deal)
    deal_path = splice_deal(document, node['Instrument']['.Deal'], parent_reference)
    if node.get('Children'):
        deal_at(document, deal_path)['Children'] = node['Children']
    return deal_verdict(document, deal_references(node), deal_path, already_missing)


# ------------------------------------------------------------------------------------------------
# The book file's writers, under a spine. Every function below answers `{}` where no home is
# configured, which is what makes the edge bit-identical without one.

def instrument_of(node):
    """The canonical TERMS of a booked node - the deal block with its legs written back under it.
    One reading, used by the booking and the amendment alike, so an amended container is an
    amendment of the instrument that was booked rather than of a different spelling of it."""
    deal = dict(node['Instrument']['.Deal'])
    if node.get('Children'):
        deal['Children'] = node['Children']
    return deal


def enclosing_set(document, deal_path):
    """The `NettingCollateralSet` a deal sits inside, walking outward from its parent, or None.

    A CLIENT IS A NETTING SET - the counterparty and the CSA live on it - so this is where a fill's
    two reference fields come from. Walking outward rather than reading the immediate parent lets a
    trade booked inside a sub-container still name the set it belongs to.
    """
    positions = str(deal_path).split('/')
    for depth in range(len(positions) - 1, 0, -1):
        deal = deal_at(document, '/'.join(positions[:depth]))['Instrument']['.Deal']
        if deal.get('Object') == 'NettingCollateralSet':
            return deal
    return None


def book_name(document):
    """The book a fill is attributed to: the job document's own `Deals.Reference`, or None.

    Capability grants are (verb x book), so a desk scoped over one book must not reach another. A
    document naming none files a firm-level fact, which only a `*` grant reaches.
    """
    named = (document.get('Calc', {}).get('Deals', {}) or {}).get('Reference')
    return named if isinstance(named, str) and named else None


def spine_fill(document, deal_path, quantity, execution_reference, actor=None):
    """Append the `fill` for a booking, and answer what the record now says. `{}` with no home.

    THE EVENT GOES FIRST: called after the verdict and before `Book.mutate` writes the file, so a
    failed write leaves the record holding a trade the desk's cache does not - the correct way
    round, the file being the interim stand-in and the log what is true.

    Three things a fill carries have no default and refuse by name: the netting set and its
    counterparty, off the set the trade sits inside, and the execution reference, which is what
    makes a retry the same fact by construction.
    """
    if not spine.configured():
        return {}
    node = enclosing_set(document, deal_path)
    if node is None:
        raise spine.SpineRefused(
            'this booking sits at {} with no NettingCollateralSet above it, and a fill carries a '
            'counterparty and a netting set on the row: book it under the client\'s set - that is '
            'where the counterparty and the CSA are declared, and it is the unit the CVA is '
            'projected over'.format(deal_path))
    counterparty, _ = set_terms(node)
    if not counterparty:
        raise spine.SpineRefused(
            'the netting set {!r} names no counterparty in its Credit_Support_Amounts, and a fill '
            'carries one: declare the counterparty on the set, then book'.format(
                node.get('Reference')))
    if quantity is None:
        raise spine.SpineRefused(
            'this booking declares no quantity, and a fill is a SIGNED quantity of the instrument: '
            'post `quantity` with the sign the desk takes - position is a fold over these and is '
            'never written')
    if not execution_reference:
        raise spine.SpineRefused(
            'this booking declares no execution_reference, and a fill body must carry one: it is '
            'what makes a retry the same fact by construction and two legitimately identical clips '
            'two facts by construction - post the venue exec id or the ticket id')
    return {'recorded': spine.book(
        instrument_of(deal_at(document, deal_path)), quantity, counterparty,
        node.get('Reference'), execution_reference, actor_name=actor,
        book_name=book_name(document))}


def spine_amendment(document, deal_path, before, actor=None):
    """Append the `amendment` linking the terms that were to the terms that are. `{}` with no home.

    Economics are never edited, so this is a second row rather than a changed one, and both
    instruments are registered because both are cited - the old one dedups onto the address it
    already has.
    """
    if not spine.configured():
        return {}
    return {'recorded': spine.amend(
        before, instrument_of(deal_at(document, deal_path)), actor_name=actor,
        book_name=book_name(document))}


@app.post('/book/deals', summary='Book, amend or delete one deal - validated, then written atomically')
def book_deals(request: dict):
    """`{action: 'add', deal, parent_reference?}`, `{action: 'amend', deal_path, fields}` or
    `{action: 'delete', deal_path}`. An amendment MERGES `fields` into the deal at `deal_path`.

    `deal` is a deal block, or a whole node - a container with its `Children` - which is how a
    quoted structure books: the legs land with the container, in one write and under one verdict.

    The contract is VALIDATE-BEFORE-WRITE, one spelling for both mutating actions: the change lands
    on a copy, the whole document is validated, and the file is rewritten only if nothing is said
    against the CHANGED deal. A refusal is `{written: False, ...}` and touches nothing.

    UNDER A SPINE HOME the write is a pair and the event goes first: an `add` appends the `fill`
    and an `amend` the `amendment` before the file is rewritten, and the outcome carries the
    envelope under `recorded`. `quantity` and `execution_reference` become required, and the deal
    must sit under a `NettingCollateralSet` naming a counterparty; each of the three refuses by
    name. A `delete` records nothing: what ends a trade is an election, an expiry observation or a
    status transition, filed through `Context.apply_lifecycle`. Without a home this is inert.
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
            node = deal_at(document, request['deal_path'])
            before = instrument_of(node)
            node['Instrument']['.Deal'].update(request['fields'])
            written, outcome = deal_verdict(
                document, [node['Instrument']['.Deal'].get('Reference')], request['deal_path'],
                already_missing)
            if written:
                outcome = dict(outcome, **spine_amendment(
                    document, request['deal_path'], before, request.get('actor')))
            return written, outcome
        written, outcome = deal_edit(document, request['deal'], request.get('parent_reference'))
        if written:
            outcome = dict(outcome, **spine_fill(
                document, outcome['deal_path'], request.get('quantity'),
                request.get('execution_reference'), request.get('actor')))
        return written, outcome

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
    same what-if twice is one run.

    THE LANE IS CURIOSITY AND NOT A PARAMETER: nothing here will ever be cited by a fact, so it
    mints nothing whether or not a spine home is configured. A candidate that becomes a trade is
    quoted through `/book/structure` and booked through `/book/quote`."""
    document, _ = live_book().read()
    try:
        if request.get('deal') is not None:
            splice_deal(document, request['deal'], request.get('parent_reference'))
    except ValueError as error:
        raise HTTPException(422, str(error))
    document['Calc']['Calculation'].update(request.get('calculation_overrides', {}))
    context = load(document)
    stamp = replay(context)
    submitted = Job(content_hash(stamp), context, stamp, spine.CURIOSITY)
    calculation = context.current_cfg.deals['Calculation']
    return {'result_id': submitted.result_id,
            'status': EXECUTOR.submit(submitted, cost(calculation)['class'])}


#: The blotter's consolidated risk keyed by the content of everything the run reads - the
#: `structures.RISK_CACHE` discipline over the whole-book gradient. A standing book pays for ONE
#: greeks run and every later poll is a dict lookup; bounded, so a 30s tick leaks no vector per tick.
BOOK_RISK_CACHE = OrderedDict()
BOOK_RISK_LIMIT = 8


def as_of():
    """When a projection or a risk vector was computed, as one ISO stamp - what a blotter shows
    beside a number to say how old it is. Milliseconds, not seconds: two recalcs of one set can
    land inside the same second, and the stamp is the ordering key the mosaic is read by.
    """
    import datetime

    return datetime.datetime.now().isoformat(timespec='milliseconds')


def risk_etag(document):
    """What a consolidated risk run READS, hashed - the cache key, and the etag the answer carries.

    The whole `Deals` section, the whole `MergeMarketData` (a `MarketDataFile` path is market data
    too) and the `Calculation` block. Anything less serves yesterday's vector: a rolled `Base_Date`
    over an unmoved deal tree is a different risk.
    """
    return content_hash({'deals': document['Calc']['Deals'],
                         'market': document['Calc']['MergeMarketData'],
                         'calculation': document['Calc']['Calculation']})


def top_level_paths(document):
    """`{Reference: deal_path}` for the book's TOP-LEVEL deals - the trades a blotter shows a row
    each for. A reference that is not unique maps to None rather than to one of the nodes it could
    name, because the positional path is the identity and guessing it mis-labels a row."""
    found = {}
    for position, node in enumerate(document['Calc']['Deals']['Deals']['Children']):
        reference = node['Instrument']['.Deal'].get('Reference')
        found[reference] = None if reference in found else str(position)
    return found


def greek_rows(frame):
    """The gradient frame flattened, AGGREGATED across the whole book.

    `Greeks_First` is indexed by `(Rate, Tenor, Tenor2, Tenor3)` and carries one column per
    reporting node plus a `Value` column holding the factor's own level, so the consolidated risk
    is the sum across every column but that one.

    The index is padded to three coordinates whatever the factor's depth, and the depth is read
    back PER FACTOR rather than per row - a curve's front point is a tenor of 0.0 and would
    otherwise trim to nothing and read as a spot. So an all-zero factor carries no `tenor` key at
    all, and one with any non-zero coordinate carries the list out to its own last used one.
    """
    table = as_json(frame)['.DataFrame']
    columns = [position for position, name in enumerate(table['columns']) if name != 'Value']
    depth = {}
    for index in table['index']:
        used = [position for position, value in enumerate(index[1:], 1) if value]
        depth[index[0]] = max(depth.get(index[0], 0), used[-1] if used else 0)
    rows = []
    for index, data in zip(table['index'], table['data']):
        row = {'factor': str(index[0])}
        if depth[index[0]]:
            row['tenor'] = list(index[1:1 + depth[index[0]]])
        row['value'] = float(sum(data[position] or 0.0 for position in columns))
        rows.append(row)
    return rows


def consolidated_risk(document):
    """The consolidated view's whole computation: ONE base valuation with `Greeks: 'First'` over
    the book as it stands, in its own Context.

    One run, because the gradient is taken off the root netting set and the `mtm` frame's per-deal
    rows ride free in the same forward pass. Counterparties do not enter here; that is what makes
    this the CONSOLIDATED view and `/book/xva` the per-set one.

    `per_deal` is the frame's TOP-LEVEL rows, one per trade, so `mtm` is exactly their sum: a
    structure's legs and a set's members are inside the row their container reports, and adding
    them again would double the book. An empty book never reaches a run.
    """
    run = deepcopy(document)
    run['Calc']['Calculation'] = dict(run['Calc']['Calculation'],
                                      Object='BaseValuation', Greeks='First')
    answer = {'as_of': as_of(), 'currency': run['Calc']['Calculation'].get('Currency'),
              'mtm': 0.0, 'per_deal': [], 'greeks': []}
    if not run['Calc']['Deals']['Deals'].get('Children'):
        return answer

    _, out = load(run).run_job()
    table = as_json(out['Results']['mtm'])['.DataFrame']
    column = {name: position for position, name in enumerate(table['columns'])}
    paths = top_level_paths(document)
    # row 0 is the grand-total row the reporter inserts; every row naming IT as its parent is a
    # top-level deal, and anything deeper is a leg inside one of them
    root = table['data'][0][column['Reference']] if table['data'] else None
    for row in table['data'][1:]:
        if column.get('Parent') is None or row[column['Parent']] != root:
            continue
        reference = row[column['Reference']]
        value = float(row[column['Value']] or 0.0)
        answer['per_deal'].append({'reference': reference, 'deal_path': paths.get(reference),
                                   'value': value})
        answer['mtm'] += value
    gradient = out['Results'].get('Greeks_First')
    answer['greeks'] = [] if gradient is None else greek_rows(gradient)
    return answer


@app.get('/book/risk', summary="The book's consolidated risk - every counterparty at once")
def book_risk():
    """The consolidated view's feed: the book's mark and its whole-book gradient, in one answer.

    Counterparties do not matter here - the greeks are aggregated over the entire book and there is
    nothing to slice by; the per-set reading is `/book/xva`.

    Computed on a MISS and cached under `etag`, a hash of everything the run reads, so a blotter
    polls this on the same beat it polls `/book`. An empty book answers zeros without running
    anything; a book that will not price answers 422 naming the cause and caches nothing.

    `mtm` is the sum of `per_deal`, one row per top-level trade; `greeks` is `{factor, tenor?,
    value}` flattened off the gradient frame, trimmed at the last non-zero coordinate.
    """
    document, _ = live_book().read()
    etag = risk_etag(document)
    if etag not in BOOK_RISK_CACHE:
        try:
            answer = consolidated_risk(document)
        except Exception as error:
            raise HTTPException(422, 'the book will not price a consolidated risk run: {}'.format(
                error))
        if len(BOOK_RISK_CACHE) >= BOOK_RISK_LIMIT:
            BOOK_RISK_CACHE.popitem(last=False)
        BOOK_RISK_CACHE[etag] = answer
    BOOK_RISK_CACHE.move_to_end(etag)
    return dict(BOOK_RISK_CACHE[etag], etag=etag)


#: The desk PROJECTION's own file, beside the book in `DV_HOME`. A file rather than memory because
#: it is a desk file like the book: the blotter's XVA view is a READ of it, and it has to survive
#: the service restarting without the desk paying for a whole recalc to see its own numbers again.
XVA_FILE = 'xva.json'
XVA_LOCK = threading.Lock()

#: What a recalc runs at when the book's `Calculation` block states nothing of its own. THIS IS A
#: DESK PROJECTION, NOT A REGULATORY NUMBER: one batch of 1024 paths is the smallest count whose
#: exposure profile is not visibly noisy, and a book that states more is never overridden.
XVA_BATCH_SIZE = 1024
XVA_SIMULATION_BATCHES = 1

#: The CVA block a recalc switches on, in the shape the engine's own CVA fixtures configure it.
#: `Calculate` and `Counterparty` are decided per set; the rest is what a desk PROJECTION wants -
#: deterministic deflation and hazard rates, no gradient. A partial block KeyErrors inside the run.
XVA_CVA_BLOCK = {'Calculate': 'Yes', 'Counterparty': '', 'Bank': '',
                 'Deflate_Stochastically': 'No', 'Stochastic_Hazard_Rates': 'No',
                 'Gradient': 'No'}

#: The FVA block a recalc switches on, beside the CVA one and off the SAME run - the projection's
#: second column. `Counterparty` is the set's; the three curves are decided per set (see
#: `funding_curves`); the rest is the CVA block's own desk-projection posture. A partial block
#: KeyErrors in the run.
XVA_FVA_BLOCK = {'Calculate': 'Yes', 'Counterparty': '', 'Bank': '', 'Risk_Free_Curve': '',
                 'Funding_Cost_Interest_Curve': '', 'Funding_Benefit_Interest_Curve': '',
                 'Deflate_Stochastically': 'No', 'Stochastic_Funding': 'No', 'Gradient': 'No'}

#: The recalc in flight per netting set, as the POST filed it: `{Reference: result_id}`. The row in
#: `xva.json` names the run that PRODUCED it, never the one still running, so the view reads the
#: pending id here. Never cleaned: a settled id simply stops reading `queued`/`running`.
XVA_PENDING = {}


def xva_path():
    return os.path.join(dv_home(), XVA_FILE)


def read_projection():
    """The projection as it stands on disk - `{'sets': {Reference: row}}`, empty where no recalc
    has ever run. A fresh parse every time, the worker thread writing this file and the request
    thread reading it. A file that is there but not JSON refuses BY NAME: "no XVA at all" and "the
    file is broken" are different facts."""
    try:
        with open(xva_path(), encoding='utf-8') as handle:
            return json.load(handle)
    except FileNotFoundError:
        return {'sets': {}}
    except ValueError as error:
        raise HTTPException(422, '{} is not readable as an XVA projection: {}'.format(
            xva_path(), error))


def write_row(reference, row):
    """One set's row into the projection, atomically - the book writer's discipline, under one
    lock: read, replace that row, write a temp, `os.replace`.

    A MOSAIC on purpose. Only the named row moves, every other row keeps the `as_of` of the run
    that produced it, and a partial recalc is therefore a partial WRITE rather than a whole file
    rebuilt from whatever happened to be current. The lock is what lets the queue drain set after
    set without one row's write losing another's.
    """
    with XVA_LOCK:
        projection = read_projection()
        projection.setdefault('sets', {})[reference] = row
        text = json.dumps(projection, indent=2)
        os.makedirs(dv_home(), exist_ok=True)
        temporary = xva_path() + '.tmp'
        with open(temporary, 'w', encoding='utf-8', newline='') as handle:
            handle.write(text)
        os.replace(temporary, xva_path())
    return row


def netting_sets(document):
    """`[(deal_path, node)]` for every `NettingCollateralSet` the book carries, in book order -
    the XVA view's instruments. A set nested inside another container is still a set, so the whole
    tree is walked rather than the top level."""
    return [(deal_path, node)
            for deal_path, node in walk_job_deals(document['Calc']['Deals']['Deals']['Children'])
            if node['Instrument']['.Deal'].get('Object') == 'NettingCollateralSet']


def set_terms(deal):
    """`(counterparty, collateralized)` off a netting set's own CSA fields.

    `Credit_Support_Amounts.Counterparty` is the one the engine reads too - the `SurvivalProb`
    factor the CVA discounts by - so the recalc and the row cannot name two different
    counterparties. `Collateralized` is declared as 'True'/'False' TEXT and the string "False" is
    truthy in every browser, so the row carries the READING rather than the spelling.
    """
    support = deal.get('Credit_Support_Amounts') or {}
    return (support.get('Counterparty') or '',
            str(deal.get('Collateralized', 'False')).lower() == 'true')


def funding_curves(deal, risk_free):
    """`(cost, benefit)` funding curve names for a set's FVA, off the set's OWN declaration.

    `Funding_Rate` is the curve a `NettingCollateralSet` already names as what it funds at, so the
    projection reads the funding leg where the book states it rather than inventing a desk-wide
    spread nobody authored. A set that names none funds at the risk-free curve and FVA is then
    identically zero - FCA and FBA are both spreads OVER risk-free. That zero is a READING, not a
    placeholder: the remedy is to author `Funding_Rate` on the set.

    Cost and benefit are the same curve because a single `Funding_Rate` is all the set declares,
    and because `Config.calc_dependencies` registers the cost and risk-free curves as factor
    dependencies but not the benefit one, so a benefit curve nothing else pulls in fails the run.
    """
    funding = deal.get('Funding_Rate') or risk_free
    return funding, funding


def xva_document(document, node, counterparty):
    """A `Credit_Monte_Carlo` job over the book carrying ONE netting set's subtree, priced for BOTH
    adjustments the projection carries.

    The book's market data, calendars, bootstrappers and report currency travel WHOLE - the CVA
    reads the counterparty's `SurvivalProb` block off them - and four things are decided here: the
    deal tree becomes that set alone, the calculation becomes a CMC, and the CVA and FVA blocks are
    both switched on against the counterparty the SET names.

    ONE RUN, TWO NUMBERS. Both adjustments reduce the same exposure cube, so they cost one CMC and
    land on one `as_of` and one replay tuple - columns of a row rather than projections to
    reconcile.

    AND THE CUBE IS THE ONE A CVA-ONLY RUN SEES. `Funding_Valuation_Adjustment.Calculate` sets
    `CMC_State.scale_survival`, so a netting set under that flag reports its MTM already multiplied
    by the counterparty's survival probability; that factor is positive and deterministic per
    bucket, so the CVA integrand divides it back out and the `cva` column reads the same either
    way, 0 ULP on both of the gate's sets. FVA keeps the scaled cube it is defined on.

    The synthesized fields are a FLOOR: `setdefault` leaves a book stating its own path count,
    batching or deflation curve alone, and an authored adjustment block keeps every field but the
    counterparty. `Deflation_Interest_Rate` is synthesized because the declaration's default names
    a ZAR curve a book need not carry - the report currency's own is the one deflation any book
    that priced at all has, and the risk-free curve the funding spread is measured against.
    """
    run = deepcopy(document)
    run['Calc']['Deals']['Deals']['Children'] = [node]
    calculation = dict(run['Calc']['Calculation'], Object='CreditMonteCarlo')
    calculation.setdefault('Batch_Size', XVA_BATCH_SIZE)
    calculation.setdefault('Simulation_Batches', XVA_SIMULATION_BATCHES)
    calculation.setdefault('Deflation_Interest_Rate', calculation.get('Currency'))
    block = dict(XVA_CVA_BLOCK, **(calculation.get('Credit_Valuation_Adjustment') or {}))
    block.update(Calculate='Yes', Counterparty=counterparty)
    calculation['Credit_Valuation_Adjustment'] = block

    risk_free = calculation['Deflation_Interest_Rate']
    cost, benefit = funding_curves(node['Instrument']['.Deal'], risk_free)
    funding = dict(XVA_FVA_BLOCK, Risk_Free_Curve=risk_free,
                   Funding_Cost_Interest_Curve=cost, Funding_Benefit_Interest_Curve=benefit,
                   **(calculation.get('Funding_Valuation_Adjustment') or {}))
    funding.update(Calculate='Yes', Counterparty=counterparty)
    calculation['Funding_Valuation_Adjustment'] = funding
    run['Calc']['Calculation'] = calculation
    return run


class XvaJob:
    """One netting set's CVA and FVA as ONE unit of queued work, and the projection row it lands as.

    TWO COLUMNS, ONE RUN. Both adjustments reduce the same exposure cube, so one CMC produces both
    and the row carries them under one `as_of` and one replay tuple. There is no KVA column, not
    even a null one: nothing in the engine computes a capital charge, and a null would read as "not
    run yet" when the fact is "not implemented".

    PER-SET GRANULARITY IS THE POINT. A whole-book recalc queues one of these per set rather than
    one run over the lot, so the queue DRAINS between sets: a quote or a base valuation submitted
    mid-recalc jumps every set still waiting, and the projection fills in row by row.

    The row is written on completion EITHER WAY. A failed run - commonly a counterparty the market
    data carries no `SurvivalProb` block for - lands `status: 'failed'` with the engine's own
    wording, because a desk reads why a number is missing off the same file the numbers are in. The
    error still travels to the result store, so `/results/{result_id}` is never told otherwise.
    """

    def __init__(self, context, reference, terms, result_id, stamp):
        self.context, self.reference = context, reference
        self.counterparty, self.collateralized = terms
        self.result_id, self.stamp = result_id, stamp

    def row(self, **fields):
        """The set's row: who it is against, what was run, and what came back - the replay tuple
        included, so a number on the blotter can be reproduced from the row alone.

        Both adjustment columns are in the BASE dict as nulls, so a row is never filed missing one:
        a failure path states its error and nothing else, and every row the file carries has the
        same keys whatever happened to it."""
        return dict({'counterparty': self.counterparty, 'collateralized': self.collateralized,
                     'cva': None, 'fva': None,
                     'result_id': self.result_id, 'plan_hash': self.stamp['plan_hash'],
                     'values_hash': self.stamp['values_hash'], 'seed': self.stamp['seed'],
                     'as_of': as_of()}, **fields)

    @staticmethod
    def adjustments(results, stored=False):
        """The columns one run's results carry - `{'cva': ..., 'fva': ...}`.

        Takes the run's `Results` or a stored result's `tables`, the same mapping either side of
        the JSON. `stored` is where the leniency belongs: a result filed before this column existed
        carries no `fva` and is written UP rather than re-run, so a missing number there reads None.

        A FRESH run is strict. Both adjustments were asked for in the same job, so a missing `fva`
        table is a defect of this run rather than age, and filing a null a desk would read as 'no
        funding cost' must not happen quietly - it raises, and `run_job` files the row as failed.
        """
        fva = results.get('fva') if stored else results['fva']
        return {'cva': float(results['cva']), 'fva': None if fva is None else float(fva)}

    def landed(self, result):
        """The row a run the store ALREADY holds lands as.

        Content addressing makes an identical recalc one execution, so a set whose numbers are
        already filed is written up from them rather than paid for again - which also keeps a
        projection whose file was lost one request away from whole rather than one CMC.
        """
        if result['status'] != 'done':
            return self.row(status='failed', error=result.get('error'))
        return self.row(status='done', **self.adjustments(result['tables'], stored=True))

    def run_job(self):
        try:
            _, out = self.context.run_job()
            landed = self.adjustments(out['Results'])
        except Exception as error:
            write_row(self.reference, self.row(status='failed', error=str(error)))
            raise
        write_row(self.reference, self.row(status='done', **landed))
        return None, out


@app.post('/book/xva', summary='Recalculate the XVA projection - every netting set, or the named ones')
def book_xva(request: dict):
    """`{netting_sets: [Reference] | null}` - null is every `NettingCollateralSet` in the book.

    THE PROJECTION IS NOT REFRESHED ON THE TICK. A credit Monte Carlo is minutes of device time, so
    `xva.json` moves only when a desk asks, and this is the asking. One job per SET, at the CMC's
    own cost class, each writing its own row on completion - so a partial recalc moves the rows it
    names and the desk's quotes keep jumping the queue while the sets drain.

    Each job composes a `Credit_Monte_Carlo` over the book carrying THAT set's subtree, with both
    adjustment blocks on against the counterparty the set's own `Credit_Support_Amounts` names, so
    `cva` and `fva` share an `as_of` and a replay tuple. A reference that names no netting set
    refuses BY NAME and queues nothing at all.

    Answers `{queued: [{reference, result_id}]}`. Content addressing applies: the same set over an
    unmoved book is one run, and asking twice hands back the id of the first.
    """
    document, _ = live_book().read()
    found = {node['Instrument']['.Deal'].get('Reference'): node
             for _, node in netting_sets(document)}
    named = request.get('netting_sets')
    references = sorted(found) if named is None else list(named)
    unknown = [reference for reference in references if reference not in found]
    if unknown:
        raise HTTPException(422, 'the book carries no NettingCollateralSet called {} - its sets '
                                 'are {}'.format(', '.join(map(repr, sorted(set(unknown)))),
                                                 ', '.join(sorted(found)) or 'none'))

    queued = []
    for reference in references:
        node = found[reference]
        terms = set_terms(node['Instrument']['.Deal'])
        context = load(xva_document(document, node, terms[0]))
        stamp = replay(context)
        result_id = content_hash(stamp)
        run = XvaJob(context, reference, terms, result_id, stamp)
        status = EXECUTOR.submit(Job(result_id, run, stamp), HEAVY)
        XVA_PENDING[reference] = result_id
        # a queued or running job writes its own row when it lands; one that has already run writes
        # here only where the row is missing or names a different run, `as_of` meaning when the
        # number was COMPUTED
        if status not in ('queued', 'running'):
            standing = read_projection()['sets'].get(reference)
            if standing is None or standing.get('result_id') != result_id:
                write_row(reference, run.landed(EXECUTOR.result(result_id)))
        queued.append({'reference': reference, 'result_id': result_id})
    return {'queued': queued}


def projection_entry(row, **book):
    """One row of the XVA view: what the BOOK says the set is now, over what the last RUN said
    about it. Every field is always present - a never-run set reads `status: 'never run'` with
    nulls beside it - so a blotter renders one table rather than branching per row.

    A row filed before a column existed is still a row: `fva` is read like every other field, so a
    CVA-only row reads `done` with a null `fva`. The projection is a mosaic in TIME as well as
    across sets, and the remedy for a null column is the recalc an old `as_of` takes."""
    row = row or {}
    entry = dict({'reference': None, 'deal_path': None, 'counterparty': row.get('counterparty'),
                  'collateralized': row.get('collateralized'), 'note': None}, **book)
    entry.update(status=row.get('status', 'never run'), cva=row.get('cva'), fva=row.get('fva'),
                 as_of=row.get('as_of'), result_id=row.get('result_id'),
                 plan_hash=row.get('plan_hash'), values_hash=row.get('values_hash'),
                 seed=row.get('seed'), error=row.get('error'), recalc=None)
    pending = XVA_PENDING.get(entry['reference'])
    outcome = EXECUTOR.result(pending) if pending is not None else None
    if outcome is not None and outcome['status'] in ('queued', 'running'):
        entry['recalc'] = {'result_id': pending, 'status': outcome['status']}
    return entry


@app.get('/book/xva', summary='The XVA projection - the last run per netting set, as it stands')
def book_xva_view():
    """The projection, joined with the book's own set list - the blotter's second data view, and a
    READ: nothing here runs, and the numbers are as old as their rows say they are.

    Netting sets are the instruments. Each entry carries what the book says the set is NOW
    (`reference`, `deal_path`, `counterparty`, `collateralized`) over what the last run said
    (`cva`, `fva`, `as_of`, `status` and the replay tuple). A set with no row yet reads `status:
    'never run'`; a ROW whose set has since left the book carries a null `deal_path` and a `note`
    rather than being dropped. A recalc still queued or running rides under `recalc`.

    THE ADJUSTMENTS ARE COLUMNS OF ONE ROW, off one CMC per set, so they share an `as_of` and a
    replay tuple and can be added without asking whether they were priced on the same market. `fva`
    is the funding cost less the benefit over the exposure, both spreads over the risk-free curve,
    so a set declaring no `Funding_Rate` reads exactly 0.0. There is no `kva` column and no null
    one: nothing computes a capital charge, and a null would read as "not recalculated yet".

    STALENESS IS DATA. The rows carry different `as_of` stamps on purpose - that is what a partial
    recalc means - so the view never blocks, never runs a CMC, and never hides a number's age.
    """
    document, _ = live_book().read()
    rows = read_projection().get('sets') or {}
    entries, seen = [], set()
    for deal_path, node in netting_sets(document):
        deal = node['Instrument']['.Deal']
        reference = deal.get('Reference')
        counterparty, collateralized = set_terms(deal)
        seen.add(reference)
        entries.append(projection_entry(
            rows.get(reference), reference=reference, deal_path=deal_path,
            counterparty=counterparty, collateralized=collateralized))
    for reference in sorted(set(rows) - seen):
        entries.append(projection_entry(rows[reference], reference=reference, note=(
            'no NettingCollateralSet called {!r} is in the book any more - this row is the last '
            'run of a set that has since left it'.format(reference))))
    return {'as_of': as_of(), 'path': xva_path(), 'sets': entries}


class CapturedErrors(logging.Handler):
    """What the bootstrap has to say for itself: `Config.bootstrap` reports a family that could
    not run or wrote nothing at ERROR and carries on, so a market update captures that channel
    and refuses the write when anything landed on it - a book must never carry a market its own
    bootstrap complained about.

    IT CAPTURES ITS OWN THREAD AND NOTHING ELSE. `Config.bootstrap` publishes on the ROOT logger,
    which is every thread's - a queued `/book/price` logging a CRITICAL inside a tick's window
    turned a good tick into `written: False` with a foreign run's message as the reason.
    `record.thread` against the constructing thread's ident is the whole filter, so single-threaded
    behaviour is byte-identical."""

    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.messages = []
        #: the tick's own thread - a root handler hears the whole process
        self.thread = threading.get_ident()

    def emit(self, record):
        if record.thread == self.thread:
            self.messages.append(record.getMessage())


#: What a tick cannot do without, in the one wording both market verbs refuse in.
NO_BOOTSTRAPPER = ('the book declares no Bootstrapper Configuration - nothing can turn quotes '
                   'into price factors')

#: Why a quote is not a values patch HERE, though the engine takes one: `patch_market` composes a
#: moved quote with an EXECUTE that re-derives its curve, which the live book has no consumer for.
#: `quotes` is the path that CAN bootstrap, and does unless `bootstrap: 'No'` defers the solve.
QUOTE_NOT_A_PATCH = ('{} is a Market Prices block - post it under `quotes`, which value-updates it '
                     'and bootstraps in the same atomic write. A values patch has no consumer on a '
                     'live book - the engine takes one because an EXECUTE that rides re-derives its '
                     'curve from it - so it would leave the written price factors stale against '
                     'their own quotes with nothing to re-solve them')

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

    `patch` is the `Price Factors` half of `patch_market` here and nothing else: a quote IS a
    patchable value to the engine, but on the live book it moves through `quotes`, which
    bootstraps. See `QUOTE_NOT_A_PATCH`.
    """
    outcome = {'installed': [], 'updated': []}
    for name in sorted(quotes):
        outcome[update_market_quote(document, name, quotes[name])].append(name)
    wants_bootstrap = (bootstrap if bootstrap is not None else
                       ('Yes' if quotes else 'No')) == 'Yes'
    market = document['Calc']['MergeMarketData']['ExplicitMarketData']
    quoted = sorted(set(patch) & set(market.get('Market Prices', {})))
    if quoted:
        raise ValueError(QUOTE_NOT_A_PATCH.format(', '.join(quoted)))
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

    The practical tick path: a quote source posts `Market Prices` blocks, and an update may move
    only each point's `Quoted_Market_Value`, its two-way sides and its `Timestamp` - structure is a
    re-authoring, refused by name. `patch` is the values delta as `patch_market` takes it, less the
    `Market Prices` half. The bootstrap (default: run iff quotes arrived) turns the quotes into the
    price factors the pricers read, all in one atomic write - a bootstrap that reports an error
    writes NOTHING and hands the messages back.

    A `patch` naming a `Market Prices` block is refused with the remedy, the one place the book is
    stricter than the engine: see `QUOTE_NOT_A_PATCH`.
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

    `derivus_bloomberg` is imported INSIDE `run_job`: blpapi lives only on a terminal workstation
    and the service must serve every other verb on a machine that has never heard of it.

    The outcome is a book WRITE rather than tables, so it rides the run's Stats under `Bloomberg`,
    as a solve's coordinates ride `Solved`. Progress rides `PROGRESS` under the result id.

    `routine` is the metronome's mode: the interactive verb PROVISIONS a workstation that has no
    security map and a cadence must not, so a routine job checks for the map before opening a
    session at all - which keeps that refusal reachable with no terminal.
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
            try:
                written = self.book.mutate(
                    lambda document: market_edit(document, quotes, {}, 'Yes'))
            except (ValueError, KeyError) as error:
                # what `/book/market` turns into a 422, turned into the refusal a JOB lands as -
                # the metronome's failed-beat handling reads `refused` off these Stats, where
                # raising would reach it as an unhandled job error
                return self.outcome(started, written=False, refused=[str(error)])
            return self.outcome(started, provisioned=created, **written)
        finally:
            PROGRESS.pop(self.result_id, None)


def submit_bloomberg(scope, routine=False):
    """The terminal round trip as a queued job - the ONE seam `/book/bloomberg` and the metronome
    both ride.

    Same job class, same single-worker queue, same result store, so a posted tick and a beat of the
    cadence are indistinguishable downstream. `routine` is the only thing that separates them, and
    all it says is "do not provision".
    """
    live = live_book()
    document, etag = live.read()
    if not document['Calc']['MergeMarketData']['ExplicitMarketData'].get(
            'Bootstrapper Configuration'):
        raise HTTPException(422, NO_BOOTSTRAPPER)
    # a fetch is an ACT against the terminal rather than a function of the book, so the submission
    # clock names it: two ticks against an unmoved book are two trips, never one coalesced result
    result_id = content_hash({'book': etag, 'bloomberg': scope, 'at': time.perf_counter()})
    # the tick is TELEMETRY - a repaint superseded by the next one before anything could cite it -
    # and the lane is declared rather than left blank so the absence is a readable decision
    submitted = Job(result_id, BloombergJob(live, scope, result_id, routine), {}, spine.TELEMETRY)
    # light, or the book stops ticking for a whole-book recalc and then drains a burst of stale beats
    return {'result_id': result_id,
            'status': EXECUTOR.submit(submitted, COST_CLASS['BaseValuation'])}


@app.post('/book/bloomberg', summary="Fetch the desk's FX vol surfaces off Bloomberg and tick the book")
def book_bloomberg(request: dict):
    """`{pairs?: [PAIR], expiries?: [LABEL], pillars?: [DELTA]}` - the whole tick, terminal to
    book, as one queued job.

    `DV_Service --tick SECONDS` submits this same job on a cadence, so on a ticking service this
    verb is what FORCES a refresh between beats - and what provisions the workstation, since the
    metronome deliberately will not.

    The scope defaults to the desk's own: every `fx_vol` pair the security map carries, at the
    expiries it verified and the delta pillars `security_map.fx_vol_definition` defaults to. The
    map is provisioned first, then every requested surface is checked for staleness and fetched,
    and what came back is installed and bootstrapped through the seam `/book/market` uses, in ONE
    atomic write. A quote whose last print is late refuses the whole tick BY NAME and writes
    nothing: a dead series keeps answering with a plausible number.

    Answers `{result_id, status}` like `/execute`, a terminal round trip being minutes of work:
    `/results/{result_id}` carries `progress` while it runs and the outcome under `stats.Bloomberg`.
    The id names the ACT rather than the numbers, so two ticks are two fetches.
    """
    return submit_bloomberg(
        {key: request[key] for key in ('pairs', 'expiries', 'pillars') if key in request})


#: The `Bootstrapper Configuration` entry that turns a `HestonNandiModelPrices` block into the
#: parameters a TARF reads - borrowed for the run, never left standing (see `hn_edit`). Families run
#: in SORTED order, so `FXVolSurfaceParameters` rebuilds the surface before the fit reads it.
HN_FAMILY = 'HestonNandiModelParameters'


def hn_factor(block_name):
    """The price factor a `HestonNandiModelPrices.<name>` block writes. The family declares the two
    strings ARE different (`market_factor_type` against the class name) and no rule recovers one
    from the other, so the one place the name is composed is here."""
    return '{}.{}'.format(HN_FAMILY, block_name.split('.', 1)[1])


def hn_edit(document, pair):
    """The calibration as ONE edit closure over a wire document, for `Book.mutate`: the quote block
    authored off the book's own built surface, installed through the `/book/market` seam,
    bootstrapped, and the fitted factor landing with it in a single atomic write.

    A RE-CALIBRATION IS A RE-AUTHORING, so the block is dropped before it is installed rather than
    ticked over: its structure is a function of the surface - the delta-neutral straddle strike
    moves with the ATM vol, the 25 delta strikes with the whole smile - so re-emitting it after a
    tick legitimately moves the strikes, which `update_market_quote` refuses by name.

    THE FAMILY ENTRY IS BORROWED FOR THIS RUN AND HANDED BACK: every market tick is a bootstrap, so
    a book left declaring this family would refit on every tick. It is added if missing and removed
    once the fit has run; a book declaring the family itself keeps it.
    """
    market = document['Calc']['MergeMarketData']['ExplicitMarketData']
    if not market.get('Bootstrapper Configuration'):
        raise ValueError(NO_BOOTSTRAPPER)
    params = load(document).current_cfg.params
    name, block = HestonNandiModelParameters.fx_surface_block(
        pair, params['Price Factors'], params['System Parameters'],
        params['Price Factor Interpolation'])
    market.get('Market Prices', {}).pop(name, None)
    borrowed = HN_FAMILY not in market['Bootstrapper Configuration']
    market['Bootstrapper Configuration'].setdefault(HN_FAMILY, {})
    try:
        written, outcome = market_edit(document, {name: as_json(block)}, {}, 'Yes')
    except (ValueError, KeyError) as error:
        # what `/book/market` makes a 422 of, made into the refusal a queued job lands as - a
        # message under this run's own Stats, never an exception out of the worker
        written, outcome = False, {'written': False, 'refused': [str(error)]}
    finally:
        if borrowed:
            market['Bootstrapper Configuration'].pop(HN_FAMILY, None)
    factor = hn_factor(name)
    # the parameters are read back off the WRITE - a refused bootstrap leaves the section as it was,
    # and reporting the standing block would report the previous fit as this one's answer
    fitted = (market['Price Factors'].get(factor) or {}) if written else {}
    return written, dict(outcome, pair=pair, block=name, factor=factor,
                         quotes=len(block['instrument']['European_Options']),
                         source=block['instrument']['Quote_Source'],
                         parameters={key: value for key, value in fitted.items()
                                     if key != 'Property_Aliases'})


class HestonNandiJob:
    """One pair's Heston-Nandi calibration as ONE unit of queued work.

    The XVA mosaic's pattern, for its reason: the fit is a least squares over a Fourier inversion
    of a daily GARCH recursion and it takes MINUTES - 288 s for a ten-quote ladder reaching six
    months, past 21 minutes on one reaching a year - so it is queued at `HEAVY` and must not sit in
    front of a salesperson's quote.

    No second file and no projection: the fitted `HestonNandiModelParameters.<underlying>` block
    lands in the book's own `Price Factors`, which every read of the book already serves and an FX
    TARF resolves by naming convention. The outcome rides the run's `Stats` under `HestonNandi`.
    """

    def __init__(self, book, pair, result_id):
        self.book, self.pair, self.result_id = book, pair, result_id

    def run_job(self):
        started = time.perf_counter()
        # minutes behind one result id, so the worker says what it is doing
        PROGRESS[self.result_id] = {
            'done': 0, 'total': 1,
            'note': 'fitting {} against ten vega-weighted vols'.format(self.pair)}
        try:
            outcome = self.book.mutate(lambda document: hn_edit(document, self.pair))
        finally:
            PROGRESS.pop(self.result_id, None)
        return None, {'Results': {}, 'Stats': {'HestonNandi': dict(
            outcome, seconds=round(time.perf_counter() - started, 2))}}


@app.post('/book/hn', summary="Calibrate one pair's Heston-Nandi parameters off its built surface")
def book_hn(request: dict):
    """`{pair}` - fit the five Q-measure Heston-Nandi parameters for one FX pair against ten
    vega-weighted vols read off the surface the book already carries, and land them in it.

    ON REQUEST, NEVER ON THE TICK. The fit is minutes of work, so it is queued at the heavy cost
    class and a desk's quotes keep jumping it. A market tick moves the surface and leaves these
    parameters where they were, STRUCTURALLY, the family being borrowed into `Bootstrapper
    Configuration` for this run and handed back. Call it after a re-tick, before quoting the TARFs.

    The quote ladder is stated once, on `HestonNandiModelParameters.fx_surface_block`.

    Answers `{result_id, status}` like `/execute`; the outcome arrives under `stats.HestonNandi`.
    There is no GET side: the written factor IS the projection and `GET /book` serves it.

    The pair is REFUSED HERE, on the request thread, when the book carries no built surface for it.
    The job re-authors the block against the document it locks, so a book that moved between the
    check and the write is fitted as it is rather than as it was.
    """
    live = live_book()
    document, etag = live.read()
    pair = request.get('pair')
    if not pair:
        raise HTTPException(422, 'a calibration names the pair it fits, e.g. {"pair": "USD.ZAR"}')
    try:
        # the pre-flight IS the emitter, run on the read copy and thrown away: every refusal it
        # names is a fact about the book a desk must hear now rather than poll for
        params = load(document).current_cfg.params
        block_name, _ = HestonNandiModelParameters.fx_surface_block(
            pair, params['Price Factors'], params['System Parameters'],
            params['Price Factor Interpolation'])
    except (ValueError, KeyError) as error:
        raise HTTPException(422, str(error))
    # content addressed on the book it fits: the same calibration over an unmoved book is one
    # execution, and a tick moves the etag, so asking again after one genuinely refits
    result_id = content_hash({'book': etag, 'hn': pair})
    submitted = Job(result_id, HestonNandiJob(live, pair, result_id), {})
    return {'result_id': result_id, 'factor': hn_factor(block_name),
            'status': EXECUTOR.submit(submitted, HEAVY)}


class Metronome:
    """`DV_Service --tick SECONDS` - the background market ticker, as a thread that SUBMITS.

    It owns no fetching: every beat goes through `submit_bloomberg`, the seam `/book/bloomberg`
    rides, so a tick is the same job class on the same queue. What the thread adds is a clock and
    three disciplines.

    NEVER STACK. A terminal round trip can outlast an interval and two ticks never coalesce, so a
    beat that finds its predecessor `queued` or `running` is skipped - read off the executor's own
    store rather than off bookkeeping of the metronome's.

    NEVER PROVISION. An unprovisioned `DV_HOME` refuses by name and the cadence carries on:
    verifying a workstation's vocabulary stays the interactive verb's act.

    NEVER DIE. A failed tick is one warning line naming the cause, and the book is untouched by
    construction since a job that refuses writes nothing. Three failures in a row stretch the
    interval fivefold until one succeeds.
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
    # a solve is base valuations under the hood - light, so it jumps a draining recalc
    return {'result_id': submitted.result_id,
            'status': EXECUTOR.submit(submitted, COST_CLASS['BaseValuation'])}


def dv_home():
    """The desk's own user-data directory: `DV_HOME`, or `~/.derivus` when it is unset.

    Read on every call rather than captured at import, so a client that moves the setting moves
    both the book `main` opens and the quotes a running service files.
    """
    return os.path.expanduser(os.environ.get('DV_HOME', os.path.join('~', '.derivus')))


def quote_dir():
    """Where a pending trade waits for its approval: `DV_HOME/tmp`, created on first use.

    A quote is not a booking: `/book/structure` files the pending trade here under its `quote_id`
    and `/book/quote` books it from here, so the two desks meet in a FILE - and the file stays
    after the booking, being the audit trail of why the book carries what it carries.
    """
    return os.path.join(dv_home(), 'tmp')


def quote_stamp():
    """When a quote was GIVEN, as one ISO stamp in UTC - what the firm window is measured from.

    UTC and offset-aware, unlike `as_of`: a quote's age is arithmetic against a running clock, and
    a naive local stamp puts an hour into it twice a year. The `quote_id` cannot serve - it hashes
    a `perf_counter`, a monotonic tick with no epoch.
    """
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='milliseconds')


def quote_age(stamp):
    """How many seconds ago `stamp` was written, or None where there is nothing readable to
    measure - a pending file from before quotes were stamped, or a stamp that will not parse.
    None is not zero: an age nobody can establish is not an age inside the window."""
    import datetime

    if not stamp:
        return None
    try:
        given = datetime.datetime.fromisoformat(stamp)
    except (TypeError, ValueError):
        return None
    if given.tzinfo is None:
        given = given.replace(tzinfo=datetime.timezone.utc)
    return (datetime.datetime.now(datetime.timezone.utc) - given).total_seconds()


def check_pins(pending, document, quote_id):
    """The TWO-DIMENSIONAL firmness check on an approval: the verdict, or None where there is
    nothing to check.

    A quote pins the book's plan hash and its values hash, which are two clocks. The VALUES
    dimension asks whether the market moved or the pin aged; the PLAN dimension asks whether the
    book moved under the solve. They are disjoint by MEASUREMENT - a vol tick moves `values_hash`
    and leaves `plan_hash` bit-identical - so a ticking market is never read as a moved book.

    Neither supersedes `Quote Policy.firm_seconds`, which is the DESK'S mandate and a promise to a
    client. These two are the RECORD'S: whether the market and the book the price was computed
    against are still the ones a booking would land in. A quote has to pass all three.

    Both ages come off `quoted_at`, both pins having been taken at one instant; the signature keeps
    them apart because they are separate clocks.
    """
    pinned = pending.get('pinned')
    if not spine.configured() or not pinned:
        return None
    context = load(document)
    age = quote_age(pending.get('quoted_at'))
    return spine.check_firmness(
        pinned, {'plan_hash': context.plan_hash(), 'values_hash': context.values_hash()},
        {'values': age, 'plan': age}, quote_id=quote_id)


def solved_coordinates(legs):
    """What a quote SOLVED, as `leg.field -> number`.

    A quote reports its solve per leg as `{field: value}`, and the record flattens the pair into
    one name: the role says which leg was moved and the field says what about it, and a coordinate
    called `Strike_Price` on a three-leg structure names nothing.

    A leg that solved nothing contributes nothing, so a fully-specified quote files an empty
    object - a quote with no solved coordinate rather than a missing field.
    """
    found = {}
    for leg in legs:
        for field, value in (leg.get('solved') or {}).items():
            found['{}.{}'.format(leg.get('role'), field)] = value
    return found


def quote_quantity(pending):
    """The SIGNED quantity a booked quote lands as: the notional it was struck on, with the DESK's
    side on it.

    A fill carries a quantity and never a position, and the sign is which way the desk went. The
    quote is CLIENT paper and `/book/quote` books the mirror, so the desk's side is the opposite: a
    client who bought the structure leaves the desk short it. Read off the first leg carrying a
    side, a structure's legs being one trade.

    A quote with no notional or no sided leg answers None, which `spine_fill` turns into a named
    refusal rather than a guess.
    """
    quoted = pending.get('quote') or {}
    notional = (quoted.get('params') or {}).get('notional')
    sides = [leg.get('buy_sell') for leg in quoted.get('legs') or [] if leg.get('buy_sell')]
    if not isinstance(notional, (int, float)) or isinstance(notional, bool) or not sides:
        return None
    return float(notional) if sides[0] == 'Sell' else -float(notional)


def spot_failure():
    """The remembered live-spot failure while it is still believed, or None. Read before every
    attempt, so a dead terminal is learned once rather than costing every quote its timeout."""
    stamp, note = _SPOT_FAILURE
    return note if note is not None and time.monotonic() - stamp < SPOT_BACKOFF_SECONDS else None


def remember_spot_failure(note):
    """Remember a live-spot failure that REACHED the terminal, and only those: a missing map costs a
    stat and a dict lookup, and the home it names can change between two quotes."""
    global _SPOT_FAILURE
    _SPOT_FAILURE = (time.monotonic(), note)


def live_spot_crosses(base_currency, currencies, host=None, port=None):
    """`({PAIR: cross}, None)` off this workstation's terminal, or `(None, note)` naming why not.

    ONE request for every pair at once, through the `derivus_bloomberg` machinery a tick fetches a
    surface with: the security map says which verified ticker prices each currency against the
    book's own base. Imported HERE for the reason `BloombergJob` imports it inside `run_job`.

    An unprovisioned home, a missing blpapi, a pair the map never verified: all end as a named
    NOTE, never as an error and never as provisioning.

    `host`/`port` are the session's own endpoint arguments threaded through, so a workstation whose
    Desktop API is not on the default socket reaches this seam too.
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
        # its budget in the socket, where `timeout_ms` alone would never reach
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

    A spot is `bind='value'` data and `document` is the job's OWN copy, so the book FILE is never
    written by a quote. Only the spot moves - the vol surface and the curves stay the book's,
    because the cadence owns those and a delta-quoted FX surface is read at whatever spot stands.

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
    package. The sheet writer is reached the same way and is genuinely optional - `xlsxwriter` is
    the `quote` extra, and a desk without it gets the quote with `files['sheet_note']` naming the
    install rather than a refusal. The outcome rides the run's Stats under `Quote`.

    THE DOCUMENT TRAVELS WHOLE, which is what lets the quote read its own mandate: the book's
    `Quote Policy` block rides the copy the legs price against, so the risk-impact half is a
    property of the BOOK rather than of the service.

    THE SPOT IS LIVE. `patch_live_spot` puts this workstation's terminal spot onto THIS job's copy
    before the recipe runs; the surface and curves stay the book's, the cadence owning those. The
    book file is untouched either way and the outcome's `spot` block says which market was used.

    THE QUOTE IS FOR SOMEONE. `netting_set` names the client's `NettingCollateralSet` and travels
    into `structures.quote`, which checks it before pricing anything; the pending file carries it
    and `/book/quote` books the mirror under that node. `quoted_at` is written beside the outcome,
    the writer being the only thing that knows the clock the approval's window is measured against.

    A QUOTE PINS TWO HASHES under a spine home. They are the BOOK's, taken before the live spot
    lands on this copy, because the approval asks whether the market and book this trade would LAND
    against have moved; which spot the legs were struck on is answered under `spot`. The pair goes
    into the pending file under `pinned` and into the record as `quote_filed`, which also carries
    what was solved, the edge taken and any relayed client ask, in a sealed body a destroyed class
    key erases. With no home the pending file is byte for byte what it always was.
    """

    def __init__(self, document, structure, params, netting_set=None, request=None):
        self.document, self.structure, self.params = document, structure, params
        self.netting_set, self.request = netting_set, request

    def pinned(self):
        """The book's two hashes and the values vector behind them, or None with no home.

        Read at the TOP of the run, off the document as handed in, because `patch_live_spot` is
        about to move this copy's spot: pinning after it would pin a market that exists only inside
        this quote, and the dimension is supposed to answer for the book's.
        """
        if not spine.configured():
            return None
        context = load(self.document)
        return {'plan_hash': context.plan_hash(), 'values_hash': context.values_hash(),
                'values': spine.values_of(context)}

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

        # the book's own two hashes, taken before the live spot moves this copy's market
        pinned = self.pinned()
        # the live spot lands BEFORE anything is priced, so `engine_spot`, every solve bracket and
        # every leg - and the sheet written from this same document - read one market
        spot_source = patch_live_spot(self.document, self.params)
        outcome = structures.quote(self.document, self.structure, self.params, spot_source,
                                   self.netting_set)
        directory = quote_dir()
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, outcome['quote_id'] + '.json')
        outcome['files'] = dict(self.sheet(directory, outcome), quote=path)
        # one file IS the pending trade: what was quoted, the deal that books it, and WHEN - the
        # stamp is the writer's, since only the thing that files a quote knows when it was given
        record = {'quote': {name: value for name, value in outcome.items() if name != 'deal'},
                  'deal': outcome['deal'], 'quoted_at': quote_stamp()}
        if pinned is not None:
            # the EVENT first, then the pending file - the same order a booking writes in, and the
            # same reason: the record is what is true and the file is the desk's copy of it
            filed = spine.file_quote(
                outcome['quote_id'], self.structure, pinned['plan_hash'], pinned['values'],
                solved_coordinates(outcome['legs']), outcome['edge'],
                request=self.request, book_name=book_name(self.document))
            record['pinned'] = {'plan_hash': pinned['plan_hash'],
                                'values_hash': pinned['values_hash'], 'lsn': filed['lsn']}
            outcome['pinned'] = record['pinned']
        with open(path, 'w', encoding='utf-8', newline='') as handle:
            json.dump(as_json(record), handle, indent=2)
        return None, {'Results': {}, 'Stats': {'Quote': as_json(outcome)}}


@app.post('/book/structure', summary='Quote a named structure against the book')
def book_structure(request: dict):
    """`{structure: name, params: {...}, netting_set?}` - the sales verb: a structure named the way
    a desk names it, its parameters in MARKET terms, and back comes the whole quote with every leg
    priced and the solved ones solved.

    The recipe runs against the book's market data on an in-memory copy - the book file never
    moves, a quote not being a trade. The SPOT on that copy is this workstation's live one when the
    terminal is up and the book's last ticked one with a named reason when it is not; the surface
    and curves are always the book's, and the outcome's `spot` block says which. What it DOES write
    is the pending trade: `DV_HOME/tmp/<quote_id>.json` carries the quote, the composed deal and
    `quoted_at`, with a `.xlsx` sheet beside it when the `quote` extra is installed.

    `netting_set` is WHO the quote is for: the Reference of a `NettingCollateralSet` already in the
    book. Checked HERE against the book as it stands, and an unknown one refuses 422 naming the
    sets the book holds, so a salesperson finds out with the client still on the phone. Omitting it
    books the approval at the root.

    Answers `{result_id, status}` like `/execute`; the outcome arrives under `stats.Quote`. The id
    names the ACT rather than the numbers, so two identical asks are two quotes.

    THE LANE IS STANDING, the one verb here that is: a quote's solve is a run a fact is about to
    cite, and the fact is the quote itself. Under a spine home this files `quote_filed`, pinning
    the book's plan and values hashes beside the solved coordinates and the edge. `request` is
    optional and is what the CLIENT ASKED FOR, relayed - free text, filed in the sealed body.
    """
    from . import structures

    document, etag = live_book().read()
    structure = request.get('structure')
    if not structure:
        raise HTTPException(422, 'a quote names the structure it prices')
    params = request.get('params', {})
    netting_set = request.get('netting_set')
    # validated on THIS thread: a 422 a salesperson can read beats an `error` status to poll for
    try:
        structures.check_netting_set(document, netting_set)
    except ValueError as error:
        raise HTTPException(422, str(error))
    # a quote is an ACT, not a function of the book: asking twice is two quotes, both filed
    result_id = content_hash({'book': etag, 'structure': structure, 'params': params,
                              'netting_set': netting_set, 'at': time.perf_counter()})
    submitted = Job(result_id, StructureJob(document, structure, params, netting_set,
                                            request.get('request')), {}, spine.STANDING)
    # base valuations under the hood, so a salesperson's ask jumps every XVA set still waiting
    return {'result_id': result_id,
            'status': EXECUTOR.submit(submitted, COST_CLASS['BaseValuation'])}


@app.post('/book/quote', summary='Book a quote already given - the approval half of a structure')
def book_quote(request: dict):
    """`{quote_id}` - approve a quote and book it. The pending trade is read back from
    `DV_HOME/tmp/<quote_id>.json` and its deal goes through `deal_edit`, which is the same
    validate-before-write seam `/book/deals` books through: one atomic write, and a refusal that
    writes nothing and reads `{written: False, refused: [...]}` in the identical wording.

    The market may have moved since the quote was given, so the validation is against the book as
    it is NOW: an approval is a booking, and a booking is refused on what it would land in. The
    file is not deleted - what was quoted, at what market, is why the book carries what it carries.

    What books is the MIRROR of the pending deal, `structures.mirror` being the verb the
    risk-impact step also prices through, so the risk measured and the trade booked cannot disagree
    by a sign. The file keeps the client frame it was quoted in.

    WHERE it books is the quote's own `netting_set`, through `deal_edit`'s `parent_reference`, so a
    client's trade lands inside the subtree its CVA is projected over. A quote that named no set
    books at the root; a set that has been closed refuses in `splice_deal`'s own wording.

    THE MODEL BOOKS WITH THE TRADE. A quote may have priced its legs under a spot model the book's
    `Valuation Configuration` does not name, and that copy is thrown away when the quote is
    answered - so the pending file records the pin and it is merged HERE, inside the same edit
    closure the deal is spliced by. A pin whose parameters the book no longer carries REFUSES 422
    rather than booking a switch that would skip the deal at the next valuation.

    A QUOTE IS FIRM FOR A WINDOW. Where the BOOK declares a `Quote Policy`, `firm_seconds` is how
    long an approval may stand on the price given, and an older quote is refused 422 naming its
    age, the window and the remedy. No policy holds every quote approvable for ever. A pending file
    carrying no `quoted_at` is treated as AGED: an unknown age is not an age inside the window.

    AND UNDER A SPINE HOME, FIRM IN TWO MORE DIMENSIONS: a quote that pinned the book's plan and
    values hashes is checked against those standing NOW, each dimension separately, each refusal
    naming its own remedy - a ticked market wants a re-quote, a moved book wants the charge
    re-solved. An approval passes all three checks or none. It then appends the `fill` for the
    mirror BEFORE the file is written, under the quote id as its execution reference.
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
    # the window is read off the BOOK as it stands now, not off the quote: the mandate is the
    # desk's, the same way the validation is against the book as it is now
    document, _ = live.read()
    try:
        policy = structures.read_policy(document)
    except ValueError as error:
        raise HTTPException(422, str(error))
    if policy is not None:
        firm = policy['firm_seconds']
        age = quote_age(pending.get('quoted_at'))
        if age is None:
            raise HTTPException(422, 'quote {} carries no quoted_at - it was filed before quotes '
                                     'were stamped, so its age cannot be established, and an '
                                     'unknown age is not an age inside a window: {} holds a quote '
                                     'firm for {:.0f}s - re-quote: the market has had an unknown '
                                     'number of seconds to move'.format(
                                         quote_id, structures.QUOTE_POLICY, firm))
        if age > firm:
            # the age to a tenth: a zero-second window refused at 0.1s must name an age a desk can
            # tell from the window it broke
            raise HTTPException(422, 'quote {} was given {:.1f}s ago and {} holds a quote firm '
                                     'for {:.0f}s - re-quote: the market has had {:.1f} seconds '
                                     'to move'.format(quote_id, age, structures.QUOTE_POLICY,
                                                      firm, age))

    quoted = pending.get('quote') or {}
    parent, pinned = quoted.get('netting_set'), quoted.get('valuation_configuration')
    try:
        firmness = check_pins(pending, document, quote_id)
    except spine.SpineRefused as refused:
        raise HTTPException(422, str(refused))

    def approve(book):
        written, outcome = deal_edit(book, structures.mirror(pending['deal']), parent)
        # the model rides the SAME write as the deal - never a second one that could half-land
        if written and pinned:
            structures.pin_models(book, pending['deal'], pinned)
            outcome = dict(outcome, valuation_configuration=pinned)
        if written:
            # the event first, then the file - `spine_fill`'s law - with the QUOTE ID as the
            # execution reference that makes a retry the same fact
            outcome = dict(outcome, **spine_fill(
                book, outcome['deal_path'], quote_quantity(pending), quote_id,
                request.get('actor')))
            if firmness is not None:
                outcome = dict(outcome, firmness=firmness)
        return written, outcome

    try:
        return live.mutate(approve)
    except ValueError as error:
        raise HTTPException(422, str(error))


def blank_book():
    """A blank book: the job skeleton's envelope and market data with NO deals, dated today.

    Empty enough to mean nothing, furnished enough to work. The skeleton's USD factors stay so the
    first booking has market data to validate against (bare market data would refuse every deal on
    the missing-factor delta); the dates are stamped to today, unlike the skeleton's own fixed one,
    which is a gated contract; and the two bootstrapper families are declared so a fresh desk's
    first market tick finds something able to turn quotes into price factors."""
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

    The UI is a CLIENT of this service and its source lives outside the package, so the mount is a
    flag rather than an import-time assumption. A release wheel carries the built UI as package
    data (`derivus/_ui`), which `main` offers as the default; a source tree without a build serves
    none. `html=True` makes `/ui/` serve `index.html` but does NOT fall back to it for an unknown
    subpath, so the SPA navigates by tab state rather than URL path - a router would ship deep
    links that 404 on reload.
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
        # a ticker that could never tick is a misconfiguration: a usage error at startup rather
        # than a warning every beat from a thread nobody is reading
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
