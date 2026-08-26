"""The MCP binding owns no logic, and these gates are what says so.

Every tool is a plain function the decorator registers, so the gates drive the FUNCTIONS against
the service in process (`configure(session=TestClient(...))` - the transport seam the Excel client
established) - no stdio, no subprocess, and the one async touch is reading the registry. What is
gated is the contract a model relies on: the import discipline (a thin client stays thin), the
registry carrying real docstrings, the schema tools being the declarations, a booking that prices,
a refusal that writes nothing and carries the engine's own messages, and the formatting round trip
that keeps the book diffable.
"""
import ast
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'mcp_integration'))

import numpy as np
import pytest
from fastapi.testclient import TestClient
from mcp.server.mcpserver.exceptions import ToolError

import derivus
import server as mcp_server
from derivus import service
from test_service import BINARY, BOOKED, RATE, SPOT, Held, dump, job

SERVER_FILE = mcp_server.__file__


@pytest.fixture(autouse=True)
def wired():
    """Every gate talks to the in-process service; the transport is torn back down after."""
    mcp_server.configure(base_url='http://testserver', session=TestClient(service.app))
    yield
    mcp_server.SERVICE = None


@pytest.fixture
def book(tmp_path):
    """A live book over a temp copy of the one-cashflow job, indent 2 (the formatting gate's
    baseline), taken down after the gate."""
    path = tmp_path / 'book.json'
    path.write_text(json.dumps(json.loads(dump(job())), indent=2))
    service.BOOK = service.Book(str(path))
    yield path
    service.BOOK = None


def test_the_mcp_server_imports_neither_the_engine_nor_the_add_in():
    """A thin client stays thin by construction: the whole point of the folder is that an MCP host
    can launch it without paying for (or depending on) torch and the engine. An import that never
    executes is still a dependency, so this reads the SOURCE, not the loaded module."""
    tree = ast.parse(open(SERVER_FILE).read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or '').split('.')[0])
    assert imported <= {'os', 'time', 'requests', 'mcp', 'mcp_types'}, imported
    assert imported.isdisjoint({'derivus', 'torch', 'pandas', 'numpy', 'excel_integration'})


def test_every_tool_is_registered_and_carries_its_contract():
    """The docstring IS the contract a model reads, so an empty one is an undocumented verb; and
    the read-only hints are what let a host run discovery without asking permission to write."""
    tools = {t.name: t for t in asyncio.run(mcp_server.MCP.list_tools())}
    expected = {'list_instrument_types', 'describe_instrument_type', 'describe_calculation_type',
                'describe_factor_type', 'job_skeleton', 'read_book', 'read_deal', 'book_deal',
                'delete_deal', 'price_candidate', 'execute_book', 'validate_book',
                'describe_book', 'poll_result', 'fetch_table', 'deal_values'}
    assert set(tools) == expected
    for name, tool in tools.items():
        assert tool.description and len(tool.description) > 60, f'{name} has no real contract'
    writers = {name for name, tool in tools.items()
               if not (tool.annotations and tool.annotations.read_only_hint)}
    assert writers == {'book_deal', 'delete_deal', 'price_candidate', 'execute_book'}


def test_the_schema_tools_are_the_declarations():
    """Non-vacuous both ways: the merge carries a known required field, and the containers answer
    is the store's - the thing C2 emitted precisely so this tool needs no engine."""
    listed = mcp_server.list_instrument_types()
    assert listed['groups'] == derivus.schema.mapping['Instrument']['groups']
    assert listed['containers'] == derivus.schema.mapping['Instrument']['containers']
    assert 'FXForwardDeal' in listed['types']

    binary = mcp_server.describe_instrument_type('EquityBinaryOption')
    assert 'Cash_Payoff' in binary['required']
    assert binary['fields']['Buy_Sell']['values'] == ['Buy', 'Sell']
    assert binary['accepts_children'] is False
    assert mcp_server.describe_instrument_type('StructuredDeal')['accepts_children'] is True

    with pytest.raises(ToolError, match='FXForwardDeal'):
        mcp_server.describe_instrument_type('fxforward')

    calc = mcp_server.describe_calculation_type('BaseValuation')
    assert 'Currency' in calc['fields']
    factor = mcp_server.describe_factor_type('InterestRate')
    assert factor['processes'], 'a curve with no process menu'
    assert mcp_server.job_skeleton()['Calc']['Calculation']['Object'] == 'BaseValuation'


def test_booking_a_deal_prices_it(book):
    """The whole flow a plain-language booking rides: read the book, book a deal, run the book,
    read the deal's value off the result - held to the closed form, so a booking that writes but
    does not price cannot pass."""
    assert [row['reference'] for row in mcp_server.read_book()['deals']] == ['CF1']

    outcome = mcp_server.book_deal(json.loads(dump(BOOKED)))
    assert outcome['written'] is True and outcome['deal_path'] == '1'
    assert mcp_server.read_deal('1')['deal']['Reference'] == 'CF2'

    run = mcp_server.execute_book()
    assert run['status'] == 'done' and run['waited'] is True
    values = mcp_server.deal_values(run['result_id'])
    assert values['CF2'] == pytest.approx(BOOKED['Amount'] * SPOT * np.exp(-RATE * 2.0), rel=1e-3)


def test_a_what_if_prices_without_writing(book):
    """The par-solve half: a candidate priced against the book with the file standing still -
    two of these at two amounts is the exact affine solve the booking docstring teaches."""
    before = book.read_bytes()
    run = mcp_server.price_candidate(deal=json.loads(dump(dict(BOOKED, Reference='TRIAL'))))
    assert run['status'] == 'done'
    assert mcp_server.deal_values(run['result_id'])['TRIAL'] == pytest.approx(
        BOOKED['Amount'] * SPOT * np.exp(-RATE * 2.0), rel=1e-3)
    assert book.read_bytes() == before


def test_a_rejected_booking_is_an_answer_that_wrote_nothing(book):
    """A refusal must reach the model as DATA - the engine's own messages, verbatim - because the
    model's next move is to fix exactly what they name. And it must not have touched the file."""
    before = book.read_bytes()
    outcome = mcp_server.book_deal(json.loads(dump(BINARY)))

    assert outcome['written'] is False
    assert 'Cash_Payoff is required' in outcome['refused']
    assert book.read_bytes() == before


def test_booking_then_deleting_is_byte_identical(book):
    """The book is a diffable file: through the MCP binding too, an undone booking leaves no
    trace, not even a reformat."""
    before = book.read_bytes()
    outcome = mcp_server.book_deal(json.loads(dump(BOOKED)))
    mcp_server.delete_deal(outcome['deal_path'])
    assert book.read_bytes() == before


def test_a_parent_that_takes_no_children_is_refused(book):
    """The refusal `containers` exists to make expressible without the engine: CF1 is a
    FixedCashflowDeal, and booking under it raises naming the type - the file untouched."""
    before = book.read_bytes()
    with pytest.raises(ToolError, match='FixedCashflowDeal'):
        mcp_server.book_deal(json.loads(dump(BOOKED)), parent_reference='CF1')
    with pytest.raises(ToolError, match='GHOST'):
        mcp_server.book_deal(json.loads(dump(BOOKED)), parent_reference='GHOST')
    assert book.read_bytes() == before


def test_the_service_being_down_names_dv_service(book):
    """'Connection refused' tells a model nothing actionable; the refusal names the service and
    how to start it."""
    class Down:
        def request(self, *args, **kwargs):
            raise __import__('requests').exceptions.ConnectionError('refused')

    mcp_server.configure(base_url='http://nowhere', session=Down())
    with pytest.raises(ToolError, match='DV_Service'):
        mcp_server.read_book()


def test_execute_hands_back_the_id_when_it_will_not_wait(book):
    """A zero wait is the escape hatch for a long simulation: the id and the way forward travel
    in `hint`, and `poll_result` finishes the story once the queue drains."""
    run = mcp_server.execute_book(wait_seconds=0.0)
    if run['status'] != 'done':  # the worker may still win the race on a tiny book
        assert 'poll_result' in run['hint']
    service.EXECUTOR.queue.join()
    settled = mcp_server.poll_result(run['result_id'])
    assert settled['status'] == 'done'
    assert 'mtm' in settled['tables']


def test_deal_values_refuses_a_result_with_no_mtm_frame():
    """A wrong projection is worse than a refusal: a result whose mtm is not the per-deal frame
    (or is absent) refuses instead of inventing numbers."""
    service.EXECUTOR.submit(
        service.Job('mcp-shape', Held('mcp-shape', [], results={'other': np.arange(3.0)}), {}),
        service.HEAVY)
    service.EXECUTOR.queue.join()
    with pytest.raises(ToolError):
        mcp_server.deal_values('mcp-shape')
