"""The derivus verbs, from the client side — plain `requests` and nothing else.

This module imports neither `xlwings` nor `derivus`, and it must stay that way: it is the HTTP
binding for anything that is not the engine, so a marimo notebook, a plain script and the Excel
add-in are all the same client. The add-in happens to be the first one written against it.

It owns no logic either. Every method is one request to one endpoint, and the shapes it returns are
the service's own — `{result_id, status}` from a submit, a summary from a poll, a page from a table
fetch. A calculation that is cheap answers `done` on the first poll, so there is no sync path to
write separately.
"""
from __future__ import annotations

import json
from typing import Any

import requests

from .config import load_settings


def as_document(job: Any) -> dict[str, Any]:
    """A job as the service takes it. Excel hands JSON around as text and a script hands it around
    as a dict, and both are the same document."""
    return json.loads(job) if isinstance(job, str) else job


class ServiceClient:
    """A `DV_Service` at a URL.

    `session` is the transport seam: anything with `requests`-style `request(method, url, ...)`
    will do, which is how the gates drive the endpoints in process through fastapi's `TestClient`
    without a socket. A session handed in brings its own transport policy, so `timeout` applies only
    to the one made here. A status the service did not intend raises rather than returning a dict
    the caller has to inspect — a cell showing the message beats a cell showing nothing.
    """

    def __init__(self, base_url: str = '', session: Any = None, timeout: float = 60.0):
        self.base_url = (base_url or load_settings().service_url).rstrip('/')
        self.session = session if session is not None else requests.Session()
        self.transport = {} if session is not None else {'timeout': timeout}

    def call(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = self.session.request(
            method, self.base_url + path, **dict(self.transport, **kwargs))
        response.raise_for_status()
        return response.json()

    def schema(self) -> dict[str, Any]:
        """Every declaration the engine publishes, and the version that emitted them."""
        return self.call('GET', '/schema')

    def job_skeleton(self) -> dict[str, Any]:
        """A minimal job document that loads — the envelope `schema()` cannot describe."""
        return self.call('GET', '/schema/job')

    def validate(self, job: Any) -> dict[str, Any]:
        """What would stop this job running: the authoring messages and the factor want-list."""
        return self.call('POST', '/validate', json=as_document(job))

    def describe(self, job: Any) -> dict[str, Any]:
        """What the engine made of this job, and what the queue would make of it."""
        return self.call('POST', '/describe', json=as_document(job))

    def prepare(self, job: Any) -> dict[str, Any]:
        """Parse the job once and get back the `plan_id` naming it, for `submit` to patch against."""
        return self.call('POST', '/prepare', json=as_document(job))

    def submit(self, job: Any, patch: dict[str, Any] | None = None) -> dict[str, Any]:
        """Submit a job — a document, or `{'plan_id': ...}` naming one already prepared — optionally
        over a values patch. Answers `{result_id, status}` for every calculation, so what follows is
        always a `poll`."""
        document = dict(as_document(job))
        if patch:
            document['Patch'] = patch
        return self.call('POST', '/execute', json=document)

    def poll(self, result_id: str) -> dict[str, Any]:
        """Where the run got to, and once done the replay tuple and the shape of every table it
        produced. Never the cells: those come from `fetch_table`."""
        return self.call('GET', '/results/{}'.format(result_id))

    def fetch_table(self, result_id: str, table: str,
                    offset: int = 0, limit: int | None = None) -> dict[str, Any]:
        """One table of a finished run, a page at a time. `rows` and `columns` describe the whole
        table and `data` is the page; the default page is the rest of it."""
        paging: dict[str, Any] = {'offset': offset}
        if limit is not None:
            paging['limit'] = limit
        return self.call('GET', '/results/{}/{}'.format(result_id, table), params=paging)

    def book(self) -> dict[str, Any]:
        """The live job document the service serves, and the etag naming its current state."""
        return self.call('GET', '/book')

    def book_deal(self, deal: Any, parent_reference: str | None = None) -> dict[str, Any]:
        """Book one deal into the live book - validated first, written only if nothing is said
        against it. `{written: False, refused: [...]}` is an answer, not an error."""
        request: dict[str, Any] = {'action': 'add', 'deal': as_document(deal)}
        if parent_reference is not None:
            request['parent_reference'] = parent_reference
        return self.call('POST', '/book/deals', json=request)

    def amend_deal(self, deal_path: str, fields: Any) -> dict[str, Any]:
        """Merge `fields` into the deal at a positional path - validated first, written only if
        nothing is said against the amended deal."""
        return self.call('POST', '/book/deals', json={
            'action': 'amend', 'deal_path': deal_path, 'fields': as_document(fields)})

    def delete_deal(self, deal_path: str) -> dict[str, Any]:
        """Remove the deal at a positional path ('0/2/1') from the live book, subtree and all."""
        return self.call('POST', '/book/deals', json={'action': 'delete', 'deal_path': deal_path})

    def price_candidate(self, deal: Any = None, parent_reference: str | None = None,
                        calculation_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        """Price the live book, optionally with a candidate deal spliced in - a what-if that
        writes nothing. Answers `{result_id, status}` like `submit`."""
        request: dict[str, Any] = {}
        if deal is not None:
            request['deal'] = as_document(deal)
        if parent_reference is not None:
            request['parent_reference'] = parent_reference
        if calculation_overrides:
            request['calculation_overrides'] = calculation_overrides
        return self.call('POST', '/book/price', json=request)

    def update_market(self, quotes: Any = None, patch: dict[str, Any] | None = None,
                      bootstrap: str | None = None) -> dict[str, Any]:
        """Tick the live book's market: install/update quote blocks, patch values, run the
        bootstrap - one atomic write, refused whole if the bootstrap complains."""
        request: dict[str, Any] = {}
        if quotes is not None:
            request['quotes'] = as_document(quotes)
        if patch is not None:
            request['patch'] = patch
        if bootstrap is not None:
            request['bootstrap'] = bootstrap
        return self.call('POST', '/book/market', json=request)

    def solve_deal(self, deal: Any, field: str, target: float = 0.0,
                   bounds: list[float] | None = None, **options: Any) -> dict[str, Any]:
        """Solve one field of a candidate deal so its value lands on `target` - par, a margin, a
        strike to a premium. The answer's `stats.Solved` carries the solved value."""
        request: dict[str, Any] = {'deal': as_document(deal), 'field': field, 'target': target}
        if bounds is not None:
            request['bounds'] = bounds
        request.update(options)
        return self.call('POST', '/book/solve', json=request)
