"""The first real client of the service, gated against the service itself.

`excel_integration.service_client` is the HTTP binding everything that is not the engine goes
through — the Excel add-in today, a marimo notebook or a plain script next — so the two claims worth
holding are that it stands ALONE and that it owns no logic. Standing alone means importing neither
`xlwings` nor `derivus`: a client that needs the engine installed to talk to a service is not a
client. Owning no logic means every method is one request to one endpoint, which is asserted by
driving it against the real app rather than a mock.

`session` is the seam that makes that possible. fastapi's `TestClient` answers `request(method,
url, ...)` the way a `requests.Session` does, so handing one in drives the endpoints in process
with no socket, no port and no server thread. Nothing here needs Excel, and nothing here could:
`xlwings_udfs` cannot be imported without xlwings, which is why no logic lives in it.

The job document and its `Held` executor stub come from `test_service`, which is where they are
canonical — a second copy of either would be a fixture to keep in step rather than one to reuse.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ast
import json

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from derivus import service
from excel_integration import service_client
from excel_integration.service_client import ServiceClient, as_document

from test_service import AMOUNT, FACTORS, Held, RATE, SPOT, dump, job

CLIENT = ServiceClient(base_url='http://testserver', session=TestClient(service.app))


def test_the_client_imports_neither_the_engine_nor_excel():
    """The boundary that would rot silently: it costs nothing to reach for `derivus` here, and the
    add-in would keep working while every other client lost the right to exist. Read off the
    module's own import statements, because an import that is never executed is still a dependency.
    """
    with open(service_client.__file__) as source:
        tree = ast.parse(source.read())
    imported = {node.module.split('.')[0] for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module and node.level == 0}
    imported.update(name.name.split('.')[0] for node in ast.walk(tree)
                    if isinstance(node, ast.Import) for name in node.names)

    assert 'requests' in imported
    assert imported.isdisjoint({'derivus', 'xlwings'})


def test_the_base_url_is_the_argument_over_the_environment_over_the_default(monkeypatch):
    """Where the workbook finds the service. A session handed in routes wherever it likes, so this
    is the only place the URL is observable at all — and pointing a client at the wrong host is
    exactly the failure a gate driven through `TestClient` cannot see."""
    monkeypatch.setenv('RF_SERVICE_URL', 'http://configured:9000/')

    assert ServiceClient().base_url == 'http://configured:9000'
    assert ServiceClient(base_url='http://given:8080/').base_url == 'http://given:8080'
    monkeypatch.delenv('RF_SERVICE_URL')
    assert ServiceClient().base_url == 'http://127.0.0.1:8000'


def test_a_job_round_trips_submit_poll_and_fetch():
    """The whole client path, end to end: submit a document, poll until the service says done, and
    fetch the one table by name. The closed form is asserted at the end so the round trip cannot
    pass on an empty page."""
    submitted = CLIENT.submit(dump(job(Random_Seed=31)))
    service.EXECUTOR.queue.join()
    polled = CLIENT.poll(submitted['result_id'])
    table = CLIENT.fetch_table(submitted['result_id'], 'mtm')
    value = table['columns'].index('Value')

    assert submitted['status'] == 'queued'
    assert polled['status'] == 'done'
    assert polled['tables']['mtm']['rows'] == len(table['data'])
    assert table['data'][1][value] == pytest.approx(AMOUNT * SPOT * np.exp(-RATE * 2.0), rel=1e-9)


def test_a_document_travels_as_text_or_as_a_dict():
    """Excel hands JSON around as text and a script hands it around as a dict. Both are the same
    document, so both have to name the same run — otherwise the add-in and the notebook would be
    two clients after all."""
    document = job(Random_Seed=37)

    assert as_document(dump(document)) == json.loads(dump(document))
    assert CLIENT.submit(dump(document))['result_id'] == CLIENT.submit(
        as_document(dump(document)))['result_id']


def test_the_read_verbs_answer_the_service_verbatim():
    """Three requests that run nothing, over both a document the suite authors and the skeleton the
    service publishes. Asserted against what the endpoints themselves return, which is the claim
    that the client owns no logic of its own."""
    document = dump(job())

    assert CLIENT.schema()['Factor']['types']['FxRate']['Spot']['bind'] == 'value'
    assert CLIENT.validate(document) == {'deals': {}, 'factors': []}
    assert CLIENT.validate(CLIENT.job_skeleton()) == {'deals': {}, 'factors': []}
    assert CLIENT.describe(document)['deals'] == {'FixedCashflowDeal': 1}
    assert CLIENT.describe(document)['factors']['resolved'] == sorted(FACTORS)


def test_a_prepared_plan_is_submitted_by_name_and_patched():
    """What a streaming client does: send the plan once, then send ticks. The patched run has to
    reach a different number and a different id, and the unpatched one off the same plan has to
    land where the whole document landed."""
    document = dump(job(Random_Seed=41))
    plan = CLIENT.prepare(document)
    whole = CLIENT.submit(document)
    patched = CLIENT.submit({'plan_id': plan['plan_id']}, {'FxRate.ZAR': {'Spot': SPOT * 1.5}})
    by_name = CLIENT.submit({'plan_id': plan['plan_id']})
    service.EXECUTOR.queue.join()
    plain = CLIENT.fetch_table(whole['result_id'], 'mtm')
    value = plain['columns'].index('Value')

    assert plan['plan_id'] == CLIENT.poll(whole['result_id'])['plan_hash']
    assert by_name['result_id'] == whole['result_id']
    assert patched['result_id'] != whole['result_id']
    assert CLIENT.fetch_table(patched['result_id'], 'mtm')['data'][1][value] == pytest.approx(
        1.5 * plain['data'][1][value], rel=1e-9)


def test_a_table_is_fetched_a_page_at_a_time():
    """Paging is the reason the drill-down exists, so the client has to pass it through rather than
    quietly fetch the lot. Held through the executor to get a table long enough to page."""
    frame = pd.DataFrame({'v': np.arange(10.0)})
    service.EXECUTOR.submit(
        service.Job('paged', Held('paged', [], results={'mtm': frame}), {}), service.HEAVY)
    service.EXECUTOR.queue.join()

    assert CLIENT.fetch_table('paged', 'mtm')['data'] == [[float(i)] for i in range(10)]
    assert CLIENT.fetch_table('paged', 'mtm', offset=4, limit=3)['data'] == [[4.0], [5.0], [6.0]]
    assert CLIENT.fetch_table('paged', 'mtm', offset=9)['data'] == [[9.0]]
    assert CLIENT.fetch_table('paged', 'mtm', offset=99)['data'] == []


def test_a_name_the_service_does_not_hold_raises():
    """A 404 is an exception here rather than an empty answer, because a cell showing the message
    beats a cell showing nothing at all. `raise_for_status` is what makes it one."""
    with pytest.raises(httpx.HTTPStatusError):
        CLIENT.poll('nosuchresult')
    with pytest.raises(httpx.HTTPStatusError):
        CLIENT.fetch_table('nosuchresult', 'mtm')
