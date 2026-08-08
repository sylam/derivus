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

`Context` is the in-process binding — `load_json`, `validate`, `market_patch` / `patch_market`,
`plan_hash` / `values_hash`, `run_job`. This module is the same verbs over HTTP, and it owns no
logic of its own: every endpoint builds a Context, calls one of those methods, and serialises what
comes back. A web SPA, a marimo notebook and an Excel add-in are all clients of the same four
endpoints, so nothing client-specific may enter the surface — and anything a client needs that the
verbs cannot answer is a missing verb on `Context`, not an endpoint that reaches inside.

| | |
| --- | --- |
| `GET /schema` | every declaration a front end renders from, plus the engine version |
| `POST /validate` | what would stop this job running, without running it |
| `POST /execute` | `{result_id, status}` — always, for every calculation |
| `GET /results/{result_id}` | the run's `Results` tables, stamped with the replay tuple |

A posted job is a job FILE: `Config.read_json` takes `(text, name)` as readily as a path, so the
same decoder builds the Curves, Timestamps and DateLists, and there is no second parser to keep in
step with the first.

There is no sync/async split at the API level. `/execute` always answers with a `result_id` and a
status, and a base valuation is simply `done` by the first poll — one contract an Excel RTD cell
and a browser poll loop can both be written against. `fastapi` and `uvicorn` are the `service`
extra and are imported only here, so `import derivus` never needs them.
"""

import json
import queue
import threading

from collections import namedtuple
from itertools import count

from fastapi import FastAPI, HTTPException

from . import Context, content_hash
from . import fields
from ._version import __version__
from .config import CustomJsonEncoder

#: Cost class off `Calculation['Object']`: a light job jumps a heavy one among those still WAITING.
#: Anything unnamed is heavy, and `run_job` is the one that names it when it turns out not to run.
COST_CLASS = {'BaseValuation': 0, 'CreditMonteCarlo': 1, 'HedgeMonteCarlo': 1}
HEAVY = 1

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
                result = dict(status='done', results=as_json(out['Results']), **job.replay)
            except Exception as error:
                result = {'status': 'error', 'error': str(error)}
            with self.lock:
                self.results[job.result_id] = result
            self.queue.task_done()


app = FastAPI(title='derivus', version=__version__)
EXECUTOR = ComputeExecutor()


@app.get('/schema')
def schema():
    """Every declaration in the engine, and the version that emitted them.

    This is what makes a front end thin: panels, tables and enums are rendered from `fields.mapping`
    rather than restated, so a field gaining a `bind`, a valid value or a new deal type reaches the
    UI by being declared on the class.
    """
    return dict(as_json(fields.mapping), engine_version=__version__)


@app.post('/validate')
def validate(job: dict):
    """What would stop this job running, verbatim from `cx.validate()` — the authoring messages of
    every deal in the book, and the price factors it names that the market data has no block for.
    Answered inline: it runs nothing, prices nothing and never reaches the queue."""
    return load(job).validate()


@app.post('/execute')
def execute(job: dict):
    """Submit a job, optionally over a values `Patch` — a delta, exactly what `patch_market` takes.

    The patch is applied BEFORE the hashes are taken, so `values_hash` describes what actually runs
    and two clients patching to the same market get one execution. The answer is always
    `{result_id, status}`: poll `/results/{result_id}` for the numbers, however cheap the job.
    """
    context = load(job)
    context.patch_market(job.get('Patch', {}))
    stamp = replay(context)
    submitted = Job(content_hash(stamp), context, stamp)
    cost = COST_CLASS.get(context.current_cfg.deals['Calculation']['Object'], HEAVY)
    return {'result_id': submitted.result_id, 'status': EXECUTOR.submit(submitted, cost)}


@app.get('/results/{result_id}')
def results(result_id: str):
    """`queued` / `running` while it waits, then the whole answer: the run's `Results` tables
    stamped with the replay tuple, or the message it failed with. Never a partial internal."""
    result = EXECUTOR.result(result_id)
    if result is None:
        raise HTTPException(404, 'Unknown result {}'.format(result_id))
    return result


def main():
    """`DV_Service` - serve the app on one uvicorn worker, which is all the executor's single
    compute thread and in-memory store can be shared across."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description='Serve the derivus verbs over HTTP.')
    parser.add_argument('-b', '--bind', type=str, help='host to bind to', default='127.0.0.1')
    parser.add_argument('-p', '--port', type=int, help='port to listen on', default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.bind, port=args.port)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
