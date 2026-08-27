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

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from mcp.server.mcpserver.exceptions import ToolError

import derivus
from derivus_mcp import server as mcp_server
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
    path.write_text(json.dumps(json.loads(dump(job())), indent=2), newline='\n')
    service.BOOK = service.Book(str(path))
    yield path
    service.BOOK = None


def test_the_mcp_server_imports_neither_the_engine_nor_the_add_in():
    """A thin client stays thin by construction: the whole point of the package is that an MCP
    host can launch it without paying for (or depending on) torch and the engine. An import that
    never executes is still a dependency, so this reads the SOURCE, not the loaded module - and
    the package `__init__` too, since importing the server runs it."""
    imported = set()
    for source in (SERVER_FILE, os.path.join(os.path.dirname(SERVER_FILE), '__init__.py')):
        for node in ast.walk(ast.parse(open(source).read())):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or '').split('.')[0])
    assert imported <= {'asyncio', 'os', 'time', 'requests', 'mcp', 'mcp_types'}, imported
    assert imported.isdisjoint({'derivus', 'torch', 'pandas', 'numpy', 'excel_integration'})


def test_every_tool_is_registered_and_carries_its_contract():
    """The docstring IS the contract a model reads, so an empty one is an undocumented verb; and
    the read-only hints are what let a host run discovery without asking permission to write."""
    tools = {t.name: t for t in asyncio.run(mcp_server.MCP.list_tools())}
    expected = {'list_instrument_types', 'describe_instrument_type', 'describe_calculation_type',
                'describe_factor_type', 'job_skeleton', 'read_book', 'read_deal', 'book_deal',
                'amend_deal', 'delete_deal', 'price_candidate', 'solve_deal', 'execute_book',
                'validate_book', 'describe_book', 'poll_result', 'fetch_table', 'deal_values',
                'update_market_quotes', 'patch_market_values', 'tick_market_from_bloomberg'}
    assert set(tools) == expected
    for name, tool in tools.items():
        assert tool.description and len(tool.description) > 60, f'{name} has no real contract'
    writers = {name for name, tool in tools.items()
               if not (tool.annotations and tool.annotations.read_only_hint)}
    assert writers == {'book_deal', 'amend_deal', 'delete_deal', 'price_candidate', 'solve_deal',
                       'execute_book', 'update_market_quotes', 'patch_market_values',
                       'tick_market_from_bloomberg'}


def test_the_progress_tool_does_not_advertise_its_context():
    """`ctx` is the SDK's injection, not an argument: a host that saw it in the schema would try
    to fill it in, and the model would spend a field guessing at a transport object."""
    tools = {t.name: t for t in asyncio.run(mcp_server.MCP.list_tools())}
    schema = tools['tick_market_from_bloomberg'].input_schema
    assert set(schema['properties']) == {'pairs', 'expiries', 'pillars', 'wait_seconds'}


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


def test_solving_then_booking_a_structured_deal(book):
    """The structuring flow in one breath: solve the amount that marks the deal at the margin,
    get the deal back ready to book, book it, and the book marks it at the margin - the "3m par
    fx forward with a 200k sales margin" pattern, with the loop server-side and nothing large in
    the answer."""
    outcome = mcp_server.solve_deal(
        json.loads(dump(dict(BOOKED, Reference='SLV1'))), 'Amount', target=200_000.0)

    assert outcome['status'] == 'done'
    assert abs(outcome['solved']['residual']) <= 0.01
    assert outcome['solved_deal']['Amount'] == outcome['solved']['value']

    booked = mcp_server.book_deal(outcome['solved_deal'])
    run = mcp_server.execute_book()
    assert booked['written'] is True
    assert mcp_server.deal_values(run['result_id'])['SLV1'] == pytest.approx(200_000.0, abs=0.01)


def test_the_practical_loop_quotes_to_a_booked_structure(tmp_path):
    """The library's whole working day in four tool calls: a Bloomberg-normalized quote block
    ticks the market, the bootstrap writes the surface, `solve_deal` finds the strike that marks
    the option at the target premium, and the solved deal books - every step through the same
    tools a model drives, nothing large entering the conversation."""
    from test_service import FX_OPTION, fx_vol_quotes
    path = tmp_path / 'book.json'
    path.write_text(json.dumps(json.loads(dump(job(
        sections={'Bootstrapper Configuration': {'FXVolSurfaceParameters': {}}}))), indent=2), newline='\n')
    service.BOOK = service.Book(str(path))
    try:
        ticked = mcp_server.update_market_quotes(json.loads(dump(fx_vol_quotes())))
        assert ticked['written'] is True and 'FXVol.USD.ZAR' in ticked['new_factors']

        option = json.loads(dump(FX_OPTION))
        outcome = mcp_server.solve_deal(option, 'Strike_Price', target=500_000.0,
                                        bounds=[12.0, 30.0])
        assert outcome['status'] == 'done' and abs(outcome['solved']['residual']) <= 0.01

        booked = mcp_server.book_deal(outcome['solved_deal'])
        run = mcp_server.execute_book()
        assert booked['written'] is True
        assert mcp_server.deal_values(run['result_id'])['OPT1'] == pytest.approx(
            500_000.0, abs=0.01)

        # and a values tick moves the mark - the market is live, not a snapshot baked at load
        patched = mcp_server.patch_market_values({'FxRate.ZAR': {'Spot': SPOT * 1.02}})
        moved = mcp_server.execute_book()
        assert patched['written'] is True
        assert mcp_server.deal_values(moved['result_id'])['OPT1'] > 550_000.0
    finally:
        service.BOOK = None


def test_a_rejected_booking_is_an_answer_that_wrote_nothing(book):
    """A refusal must reach the model as DATA - the engine's own messages, verbatim - because the
    model's next move is to fix exactly what they name. And it must not have touched the file."""
    before = book.read_bytes()
    outcome = mcp_server.book_deal(json.loads(dump(BINARY)))

    assert outcome['written'] is False
    assert 'Cash_Payoff is required' in outcome['refused']
    assert book.read_bytes() == before


def test_an_amendment_changes_the_value_it_names(book):
    """The 'change a value' flow in plain language: amend the amount, see the deal carry it, see
    the book mark it - and an amendment that breaks the deal is an answer, not a write."""
    outcome = mcp_server.amend_deal('0', {'Amount': 500_000.0})
    assert outcome['written'] is True
    assert mcp_server.read_deal('0')['deal']['Amount'] == 500_000.0

    run = mcp_server.execute_book()
    assert mcp_server.deal_values(run['result_id'])['CF1'] == pytest.approx(
        500_000.0 * SPOT * np.exp(-RATE * 2.0), rel=1e-3)

    before = book.read_bytes()
    refused = mcp_server.amend_deal('0', {'Discount_Rate': 'GBP'})
    assert refused['written'] is False and refused['refused']
    assert 'validate' not in refused
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


class Counting:
    """A transport that records every path it carries - how a gate proves a tool REFUSED without
    fetching, rather than fetched and then refused."""

    def __init__(self, inner):
        self.inner, self.paths = inner, []

    def request(self, method, url, **kwargs):
        self.paths.append(url)
        return self.inner.request(method, url, **kwargs)


def held_result(result_id, results):
    service.EXECUTOR.submit(service.Job(result_id, Held(result_id, [], results=results), {}),
                            service.HEAVY)
    service.EXECUTOR.queue.join()


def test_a_run_comes_back_as_shapes_never_cells(book):
    """The minimal-context rule: the model learns the run happened - identity, stats, one line per
    table - and never holds a table's columns or cells unless it asks for a page."""
    run = mcp_server.execute_book()
    assert set(run) <= {'result_id', 'status', 'plan_hash', 'values_hash', 'seed',
                        'stats', 'tables', 'waited', 'error'}
    for name, shape in run['tables'].items():
        assert isinstance(shape, str) and 'rows x' in shape, (name, shape)


def test_fetch_table_is_capped_and_a_cube_is_refused():
    """A page is at most 200 rows however much is asked for, and a table wider than 60 columns -
    a simulation cube - is refused BY NAME, pointed at the web UI, with nothing fetched."""
    long = pd.DataFrame({'a': np.arange(500.0), 'b': np.arange(500.0)})
    wide = pd.DataFrame(np.zeros((3, 100)), columns=[str(c) for c in range(100)])
    held_result('mcp-caps', {'long': long, 'wide': wide})

    page = mcp_server.fetch_table('mcp-caps', 'long', limit=10_000)
    assert len(page['data']) == 200 and page['rows'] == 500

    counting = Counting(TestClient(service.app))
    mcp_server.configure(base_url='http://testserver', session=counting)
    with pytest.raises(ToolError, match='web UI'):
        mcp_server.fetch_table('mcp-caps', 'wide')
    assert not any(path.endswith('/wide') for path in counting.paths)


def test_deal_values_refuses_a_cube_without_fetching():
    """A Monte Carlo's mtm is dates x scenarios; `deal_values` reads its SHAPE and refuses before
    a single cell travels - the recorded transport is the proof."""
    cube = pd.DataFrame(np.zeros((5, 80)), columns=[str(c) for c in range(80)])
    held_result('mcp-cube', {'mtm': cube})

    counting = Counting(TestClient(service.app))
    mcp_server.configure(base_url='http://testserver', session=counting)
    with pytest.raises(ToolError, match='base valuation'):
        mcp_server.deal_values('mcp-cube')
    assert not any('/mtm' in path for path in counting.paths)


def test_a_booking_answer_is_the_booking_not_the_book(book):
    """The booking outcome carries what happened to THIS deal; the rest of the book's troubles
    arrive as counts with a pointer, never as the whole verdict."""
    clean = mcp_server.book_deal(json.loads(dump(BOOKED)))
    assert 'validate' not in clean and 'book_issues' not in clean
    assert clean['written'] is True

    rejected = mcp_server.book_deal(json.loads(dump(BINARY)))
    assert 'validate' not in rejected
    assert rejected['written'] is False and rejected['refused']
    assert rejected['book_issues']['deal_messages'] == 1
    assert 'validate_book' in rejected['book_issues']['hint']


class Terminal:
    """A transport that answers the bloomberg submit and then reads a scripted `/results`
    sequence - the tool's own contract with no terminal, no service and no socket (the `Down`
    precedent). The last poll stands once the script runs out."""

    class Reply:
        def __init__(self, payload):
            self.status_code, self.payload, self.text = 200, payload, ''

        def json(self):
            return self.payload

    def __init__(self, *polls):
        self.polls, self.requests = list(polls), []

    def request(self, method, url, **kwargs):
        self.requests.append((method, url, kwargs.get('json')))
        if url.endswith('/book/bloomberg'):
            return self.Reply({'result_id': 'bbg-1', 'status': 'queued'})
        return self.Reply(self.polls.pop(0) if len(self.polls) > 1 else self.polls[0])


class Watching:
    """The host's progress channel minus the host: what the gate reads is what the tool told it."""

    def __init__(self):
        self.reported = []

    async def report_progress(self, done, total, note=None):
        self.reported.append((done, total, note))


def test_a_bloomberg_tick_returns_the_finished_payload():
    """A run that is already done comes back verbatim - a provisioning answer is what installed
    and what was refused, not a shape summary - and the None arguments never reach the wire."""
    finished = {'result_id': 'bbg-1', 'status': 'done', 'installed': ['FXVol.USD.ZAR'],
                'updated': [], 'verified': 42}
    terminal = Terminal(finished)
    mcp_server.configure(base_url='http://testserver', session=terminal)

    answer = asyncio.run(mcp_server.tick_market_from_bloomberg(pairs=['USDZAR']))

    assert answer == finished
    assert terminal.requests[0] == ('POST', 'http://testserver/book/bloomberg',
                                    {'pairs': ['USDZAR']})


def test_a_bloomberg_tick_that_will_not_wait_hands_back_the_id():
    """The same escape hatch `execute_book` has: past the wait, the id and the way forward travel
    in `hint` - the provisioning is on the service, not in this call."""
    mcp_server.configure(base_url='http://testserver',
                         session=Terminal({'result_id': 'bbg-1', 'status': 'running'}))

    answer = asyncio.run(mcp_server.tick_market_from_bloomberg(wait_seconds=0.0))

    assert answer['result_id'] == 'bbg-1' and answer['status'] == 'running'
    assert 'poll_result' in answer['hint']


def test_provisioning_reports_its_progress_while_it_runs(monkeypatch):
    """The five-minute first use only survives because of these notifications: a client resets
    its timeout on each one, and the user watching sees which candidate the terminal is on. So
    every poll that carries a progress dict must reach the context, `note` included."""
    delays = []

    async def instant(seconds, *rest):  # the gate is about the loop, not its patience
        delays.append(seconds)

    monkeypatch.setattr(mcp_server.asyncio, 'sleep', instant)
    terminal = Terminal(
        {'status': 'queued', 'progress': {'done': 0, 'total': 3, 'note': 'copying the seed'}},
        {'status': 'running', 'progress': {'done': 2, 'total': 3, 'note': 'verifying USDZAR'}},
        {'status': 'done', 'installed': ['FXVol.USD.ZAR']})
    mcp_server.configure(base_url='http://testserver', session=terminal)
    watching = Watching()

    answer = asyncio.run(mcp_server.tick_market_from_bloomberg(ctx=watching))

    assert answer == {'status': 'done', 'installed': ['FXVol.USD.ZAR']}
    assert watching.reported == [(0, 3, 'copying the seed'), (2, 3, 'verifying USDZAR')]
    assert delays == [2, 2], 'a poll between notifications, not a spin'
