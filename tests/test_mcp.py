"""The MCP binding owns no logic, and these gates are what says so.

Every tool is a plain function the decorator registers, so the gates drive the FUNCTIONS against
the service in process (`configure(session=TestClient(...))`) - no stdio, no subprocess, and a tool
that waits is awaited the way a host awaits it. Gated: the import discipline (a thin client stays
thin), the registry carrying real docstrings, the schema tools being the declarations, a booking
that prices, a refusal that writes nothing and carries the engine's own messages, the formatting
round trip that keeps the book diffable, and the notifications a waiting tool keeps its host on.
"""
import ast
import asyncio
import inspect
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from mcp.server.mcpserver.exceptions import ResourceError, ToolError

import derivus
from derivus_mcp import server as mcp_server
from derivus import service
from test_service import AMOUNT, BINARY, BOOKED, RATE, SPOT, Held, dump, job

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
    """An MCP host launches this package without paying for torch and the engine. An import that
    never executes is still a dependency, so read the SOURCE, not the loaded module - and the
    package `__init__` too, since importing the server runs it."""
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
                'describe_factor_type', 'describe_configuration', 'job_skeleton', 'desk_status',
                'read_book', 'read_deal', 'book_deal', 'amend_deal', 'delete_deal',
                'price_candidate', 'solve_deal', 'execute_book', 'validate_book', 'describe_book',
                'poll_result', 'fetch_table', 'deal_values', 'configure_book',
                'describe_curve', 'configure_curve', 'set_base_date', 'update_market_quotes',
                'patch_market_values', 'describe_securities', 'configure_securities',
                'verify_securities', 'book_dependencies', 'setup_market',
                'tick_market_from_bloomberg', 'describe_structure', 'solve_structure',
                'book_quote', 'approve_quote', 'reject_quote', 'calibrate_spot_model',
                'book_risk_summary', 'xva_view', 'recalc_xva', 'book_reconcile', 'book_diary',
                'close_check', 'book_activity', 'book_markets', 'declare_market', 'declare_close',
                'export_settlements', 'file_status', 'describe_calculations',
                'configure_calculation', 'run_calculation'}
    assert set(tools) == expected
    for name, tool in tools.items():
        assert tool.description and len(tool.description) > 60, f'{name} has no real contract'
    writers = {name for name, tool in tools.items()
               if not (tool.annotations and tool.annotations.read_only_hint)}
    assert writers == {'book_deal', 'amend_deal', 'delete_deal', 'price_candidate', 'solve_deal',
                       'execute_book', 'update_market_quotes', 'patch_market_values',
                       'tick_market_from_bloomberg', 'solve_structure', 'book_quote',
                       'approve_quote', 'reject_quote', 'declare_market', 'declare_close',
                       'export_settlements', 'file_status',
                       'recalc_xva', 'calibrate_spot_model', 'configure_book', 'configure_curve',
                       'set_base_date', 'configure_securities', 'verify_securities',
                       'setup_market', 'configure_calculation', 'run_calculation'}


def read_resource(uri):
    """One of the server's documents the way a HOST opens it - by URI through the registry, never
    by calling the function behind it. Text comes back as text and the sheet as bytes."""
    return [chunk.content for chunk in asyncio.run(mcp_server.MCP.read_resource(uri))][0]


def test_the_instructions_a_host_shows_are_the_desks_orientation():
    """A desktop host reads the server's instructions once and shows them to the MODEL before it
    calls anything, so they have to be the desk's orientation - where to start, what a deal is
    written in, and the one axis a model gets wrong - rather than this module's maintenance notes.

    Killing mutation: the module docstring installed again, which opens by telling the reader
    which packages this file is allowed to import and never names the strike axis.
    """
    instructions = mcp_server.MCP.instructions

    assert instructions == mcp_server.INSTRUCTIONS != mcp_server.__doc__
    assert 'RF_SERVICE_URL' not in instructions, "the maintainer's docstring, not the desk's"
    for said in ('START WITH desk_status', 'solve_structure', '{".Timestamp": "YYYY-MM-DD"}',
                 '{".Percent": 2.5}', 'Strike_Price is on the ENGINE axis', '1/17.50',
                 '{written: false, refused: [...]}', 'book_diary', 'close_check',
                 'QUOTING IS NOT BOOKING', 'book_quote is the ACCEPTANCE', 'approve_quote',
                 'declare_close', 'export_settlements', 'file_status'):
        assert said in instructions, said


def test_the_host_is_offered_three_walks_and_four_documents():
    """A host renders prompts as commands and resources as documents to open, so both are contract
    the same way a tool schema is: exactly the three walks a desk runs, each taking the arguments
    it needs filled, and the four documents - the book, one schema store, a pending quote, and the
    sheet that goes to the client.

    Killing mutations: a prompt argument dropped or made optional, which offers a host a command
    it cannot fill; the sheet declared `application/json`, which hands a host a zip to render as
    text.
    """
    prompts = {prompt.name: [(argument.name, argument.required)
                             for argument in (prompt.arguments or [])]
               for prompt in asyncio.run(mcp_server.MCP.list_prompts())}

    assert prompts == {
        'quote_a_structure': [('structure', True), ('pair', True), ('notional', True),
                              ('notional_currency', True), ('expiry', True), ('client', False)],
        'import_a_legacy_book': [('counterparty', True)],
        'morning_desk_check': []}
    assert [(str(found.uri), found.mime_type)
            for found in asyncio.run(mcp_server.MCP.list_resources())] == [
        ('derivus://book', 'application/json')]
    assert [(found.uri_template, found.mime_type)
            for found in asyncio.run(mcp_server.MCP.list_resource_templates())] == [
        ('derivus://schema/{store}', 'application/json'),
        ('derivus://quote/{quote_id}', 'application/json'),
        ('derivus://quote/{quote_id}/sheet', mcp_server.SHEET_MIME)]


def test_desk_status_orients_a_model_in_one_call(book, tmp_path, monkeypatch):
    """The call the instructions say to start with, against the in-process service: one answer
    carrying everything the next verb needs - the day the book is valued as of, what it holds,
    every curve with the knots it solved on and the latest print its rows carry, and whether this
    desk can fetch a market at all. The tool is one `service().call`; the composition is the
    verb's.

    `DV_HOME` is the gate's own tmp and `blpapi` is made absent, so a workstation that happens to
    carry a terminal and a provisioned map reads the same as a sandboxed one. No session is opened
    either way.

    Killing mutation: `snapped` read off the book's base date rather than off the rows, which a
    curve set up on a later print then reads as the day the book stands at.
    """
    from derivus_bloomberg import session
    from derivus_bloomberg.errors import BloombergUnavailable
    from test_service import CURVE_ROWS

    def absent():
        raise BloombergUnavailable('no blpapi on this workstation')

    monkeypatch.setenv('DV_HOME', str(tmp_path / 'home'))
    monkeypatch.setattr(session, 'blpapi_module', absent)
    bare = mcp_server.desk_status()

    assert set(bare) == {'etag', 'base_date', 'base_currency', 'calculation', 'deals',
                         'netting_sets', 'curves', 'surfaces', 'models', 'xva', 'spine',
                         'terminal'}
    assert bare['spine'] is None, 'a box that records nothing has no position to report'
    assert bare['base_date'] == '2024-06-28' and bare['base_currency'] == 'USD'
    assert {key: bare['calculation'][key] for key in ('Object', 'Currency', 'paths')} == {
        'Object': 'BaseValuation', 'Currency': 'USD', 'paths': 1}
    assert 'MCMC_Simulations is 1' in bare['calculation']['paths_note']
    assert (bare['deals'], bare['netting_sets'], bare['curves']) == (1, [], [])
    assert bare['terminal'] == {'present': False, 'ticking': None, 'provisioned': False}

    mcp_server.configure_curve('ZAR', 'ZAR', CURVE_ROWS)
    curves = mcp_server.desk_status()['curves']

    assert curves == [{'curve': 'ZAR', 'currency': 'ZAR', 'snapped': '2024-06-28',
                       'interpolation': 'Linear',
                       'knots': [row['tenor'] for row in CURVE_ROWS], 'held_out': []}]


def test_the_diary_reaches_a_model_as_rows_it_can_act_on(book, tmp_path, monkeypatch):
    """The tool is one `service().call` over the book the service already serves, so what a model
    reads is the verb's own rows - the leg, the day, the amount where the compile determines one,
    and `null` where it does not.

    Killing mutation: the tool composing or rounding the rows, which puts a second reading of the
    schedule between the model and the engine's own.
    """
    monkeypatch.setenv('DV_HOME', str(tmp_path / 'home'))
    answer = mcp_server.book_diary()

    assert set(answer) >= {'as_of', 'etag', 'result_id', 'rows'}
    assert answer['rows'], 'the one-cashflow book announces nothing'
    assert {row['kind'] for row in answer['rows']} <= {'payment', 'fixing', 'expiry', 'barrier'}
    for row in answer['rows']:
        assert set(row) == {'key', 'instrument', 'leg', 'schedule_index', 'kind', 'due_date',
                            'currency', 'amount', 'determined', 'notional', 'index', 'source',
                            'observed', 'needs', 'reason', 'state'}
        assert row['amount'] is None or row['determined'] is True

    day = min(row['due_date'] for row in answer['rows'])
    assert mcp_server.book_diary(due_before=day)['rows'] == [
        row for row in answer['rows'] if row['due_date'] <= day]


def test_the_record_reads_say_so_on_a_box_that_records_nothing(book, tmp_path, monkeypatch):
    """The four reads that need a RECORD, and a deployment that keeps none answers a 404 the tool
    turns into the service's own sentence rather than an empty answer a model would read as
    agreement, or as a record holding nothing.

    Killing mutation: any of them answering `{}` where no home is configured, which tells a model
    the file and the record agree on a box that has no record.
    """
    monkeypatch.setenv('DV_HOME', str(tmp_path / 'home'))
    for tool, arguments in ((mcp_server.book_reconcile, {}),
                            (mcp_server.close_check, {'date': '2030-01-01'}),
                            (mcp_server.book_activity, {}), (mcp_server.book_markets, {})):
        with pytest.raises(ToolError) as refusal:
            tool(**arguments)
        assert 'DV_SPINE_HOME' in str(refusal.value) and '404' in str(refusal.value)


def test_the_record_reads_are_read_only_and_say_what_they_read(book):
    """A host runs discovery without asking permission to write, so every record read carries the
    read-only hint; and each docstring is the model's whole contract for a verb it cannot try out
    on a box with no record.

    Killing mutation: any of them annotated as a writer, which makes a host prompt before a model
    may ask what the book owes.
    """
    tools = {tool.name: tool for tool in asyncio.run(mcp_server.MCP.list_tools())}
    for name, said in (('book_diary', 'determined'), ('close_check', 'legal'),
                       ('book_reconcile', 'record'), ('book_activity', 'strip'),
                       ('book_markets', 'close')):
        assert tools[name].annotations.read_only_hint is True, name
        assert said in tools[name].description, name
        assert name in mcp_server.INSTRUCTIONS, name


def test_the_strip_and_the_markets_reach_a_model_as_the_record_answers_them(book, tmp_path,
                                                                            monkeypatch):
    """Each tool is one `service().call`, so what a model reads is the fold's own rows: the strip
    newest last with the head to page from, and the close standing per market with the LSN of the
    one it restated.

    Killing mutation: either tool composing an answer of its own - a `since` it keeps, or the rows
    reversed - which puts a second reading of the record between the model and the log.
    """
    from derivus_spine import SpineLog, init_home

    actor, home = 'subject-desk-one', tmp_path / 'spine'
    init_home(home, actor)
    monkeypatch.setenv('DV_SPINE_HOME', str(home))
    monkeypatch.setenv('DV_HOME', str(tmp_path / 'home'))
    log = SpineLog(home)
    try:
        genesis = log.head()[0]
        vector = log.store.put(b'{"EURUSD":1.0851}')
        log.append('market_declared', {'name': 'official', 'values_hash': vector}, actor=actor)
        first = log.append('official_close_declared',
                           {'market': 'official', 'values_hash': vector}, actor=actor)['lsn']
        log.append('official_close_declared',
                   {'market': 'official', 'values_hash': log.store.put(b'{"EURUSD":1.0857}')},
                   actor=actor)
    finally:
        log.close()

    strip = mcp_server.book_activity()
    assert [row['lsn'] for row in strip['rows']] == list(range(1, first + 2))
    assert strip['lsn'] == first + 1
    assert strip['rows'][-1]['summary'] == 'the official close was declared'
    assert mcp_server.book_activity(since=genesis)['rows'] == strip['rows'][genesis:]
    assert mcp_server.book_activity(limit=1)['rows'] == strip['rows'][-1:]

    markets = mcp_server.book_markets()
    assert markets['lsn'] == strip['lsn']
    assert [close['supersedes_lsn'] for close in markets['closes']] == [first]
    assert [name['name'] for name in markets['names']] == ['official']
    assert markets['snapshots'] == []


def test_the_mark_the_close_the_file_and_the_settlement_reach_a_model_as_four_verbs(book, tmp_path,
                                                                                    monkeypatch):
    """The record's four WRITES a model can reach: the mark, the close behind `close_check`'s own
    verdict, the settlement file struck on the market the desk DESIGNATED for the export, and the
    back office saying a row was PAID. Each tool is one `service().call`, so what a model gets is
    the verb's own answer and a refusal is the record's own sentence.

    The last of the four closes the loop the first three leave open: a close over that day is
    ILLEGAL while the payment stands, a file instructs it, `file_status` says it moved against the
    row's own derived key, and then the next file no longer carries it and the day is legal - which
    is the whole of what a settlement is for.

    Killing mutations: `declare_close` sending a market or a date the caller did not name, which
    closes a board nobody asked about or on a day nobody chose; `export_settlements` passing a
    market through, which would let a model strike a settlement file on any board it can name; and
    `file_status` filing under a reference rather than the row's key, which settles nothing the
    export can see.
    """
    from derivus_spine import SpineLog, init_home, policy

    actor, home = 'subject-desk-one', tmp_path / 'spine'
    init_home(home, actor)
    monkeypatch.setenv('DV_SPINE_HOME', str(home))
    monkeypatch.setenv('DV_SPINE_ACTOR', actor)
    monkeypatch.setenv('DV_HOME', str(tmp_path / 'home'))
    log = SpineLog(home)
    try:
        policy.declare(log, actor, policy.TIERS_POLICY,
                       {'tiers': [{'name': 'desk', 'four_eyes': True}],
                        'designations': {'settlement_export': 'official'}})
    finally:
        log.close()

    with pytest.raises(ToolError) as unmarked:
        mcp_server.export_settlements(due_before='2099-01-01')
    assert 'official' in str(unmarked.value), 'the export resolved a board nothing stands under'

    marked = mcp_server.declare_market('official')
    closed = mcp_server.declare_close()
    assert closed['market'] == 'official' and closed['date'] == '2024-06-28'
    assert closed['values_hash'] == marked['values_hash'] and closed['supersedes_lsn'] is None
    assert [row['market'] for row in mcp_server.book_markets()['closes']] == ['official']

    exported = mcp_server.export_settlements(due_before='2099-01-01')
    assert exported['market'] == {'name': 'official', 'values_hash': closed['values_hash'],
                                  'lsn': closed['recorded']['lsn']}, 'the close did not move it'
    assert exported['count'] == len(exported['rows']) == 1
    assert exported['totals'] == {'ZAR': AMOUNT}

    with pytest.raises(ToolError) as illegal:
        mcp_server.declare_close(date='2099-01-01')
    assert 'not legal' in str(illegal.value), 'a close passed over a payment nobody had made'

    settled = mcp_server.file_status(exported['rows'][0]['key'], 'settled', actor=actor)
    assert settled['recorded']['lsn'] > closed['recorded']['lsn']
    assert mcp_server.file_status(exported['rows'][0]['key'], 'settled', actor=actor)[
        'recorded']['lsn'] == settled['recorded']['lsn'], 'saying it twice is two facts'
    assert mcp_server.export_settlements(due_before='2099-01-01')['count'] == 0, \
        'the next file instructs the payment the desk has already made'
    assert mcp_server.close_check('2099-01-01')['legal'] is True, \
        'the close still waits on a payment the back office said had moved'


def test_the_fx_strike_axis_is_published_on_the_field_a_model_fills_in():
    """A model books off `describe_instrument_type`, so the axis belongs ON the declaration: the
    field a desk gets wrong is the one that has to say which way round it is and where market
    terms are taken instead. The engine reads no description, so this moves no number.

    Killing mutation: the description left to default to the field name, which is what a model
    then reads as 'the strike' and fills in with 17.50 - and the equity barrier beside it, which
    is NOT on this axis, is what says the sweep did not describe every strike in the file.
    """
    strike = mcp_server.describe_instrument_type('FXOptionDeal')['fields']['Strike_Price']
    barrier = mcp_server.describe_instrument_type('FXBarrierOption')['fields']['Barrier_Price']
    equity = mcp_server.describe_instrument_type('EquityBarrierOption')['fields']['Barrier_Price']

    assert 'reporting currency per unit of Underlying_Currency' in strike['description']
    assert '1/17.50' in strike['description'] and 'solve_structure' in strike['description']
    assert barrier['description'].startswith('Barrier price on the ENGINE axis')
    assert equity['description'] == 'Barrier Price', 'an equity barrier is not on the FX axis'


#: The tools that sit on a run and therefore have to speak while they sit.
WAITING = ('price_candidate', 'execute_book', 'solve_deal', 'solve_structure',
           'calibrate_spot_model', 'recalc_xva', 'tick_market_from_bloomberg',
           'verify_securities', 'setup_market')


def test_no_tool_advertises_the_context_the_sdk_injects():
    """`ctx` is the SDK's injection, not an argument: a host that saw it in the schema would try
    to fill it in, and the model would spend a field guessing at a transport object. Every waiting
    tool takes one now, so the whole registry is read rather than the one that had it first.

    Killing mutation: annotate any tool's `ctx` as something other than `Context` - a `dict`, say -
    and the SDK stops injecting it and advertises it instead.
    """
    tools = {t.name: t for t in asyncio.run(mcp_server.MCP.list_tools())}
    for name, tool in tools.items():
        assert 'ctx' not in tool.input_schema['properties'], name
    assert set(tools['tick_market_from_bloomberg'].input_schema['properties']) == {
        'pairs', 'expiries', 'pillars', 'wait_seconds'}


def test_every_waiting_tool_is_async_and_takes_the_context():
    """A desktop host cuts a tool call that stays quiet for about a minute, so a tool that sits on
    a run has to be able to speak while it sits: `async`, with the injected context last.

    Killing mutation: make any one of them `def` again, or drop its `ctx`, and it fails by name.
    """
    for name in WAITING:
        tool = getattr(mcp_server, name)
        assert asyncio.iscoroutinefunction(tool), name
        assert list(inspect.signature(tool).parameters)[-1] == 'ctx', name


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


def test_the_structure_store_is_the_quoting_menu():
    """The whole menu comes off `/schema` with nothing composed here: every structure with its
    sales names, then one opened up - parameters, the VARIATIONS it is dealt as with their own
    legs, recipe. A name that is not a structure refuses with the close match, matching on the
    vernacular too.

    The LIST is a projection of the store and carries enough of it to ask with: the parameter that
    SELECTS a variation is that variation's own, so a menu publishing the shared fields alone hides
    `floor` and `cap` until the entry is opened. And it says whether a direction has to be stated -
    computed from the declarations, so a strip, whose two forms name one level, reads 'required'
    where a forward extra reads 'optional'.
    """
    listed = mcp_server.describe_structure()
    vernaculars = {entry['name']: entry['vernacular'] for entry in listed['structures']}
    menu = {entry['name']: entry for entry in listed['structures']}

    assert [entry['name'] for entry in listed['structures']] == [
        'Accumulator', 'ForwardExtra', 'Seagull', 'Straddle', 'Strangle',
        'TargetRedemptionForward', 'ZeroCostCollar']
    assert listed['count'] == 7 and all(vernaculars.values())
    assert 'collar' in vernaculars['ZeroCostCollar']

    assert menu['ForwardExtra']['variations'] == {'floor': ['floor'], 'cap': ['cap']}
    assert menu['Seagull']['variations'] == {'floor': ['floor', 'lower_floor'],
                                             'cap': ['cap', 'upper_cap']}
    assert menu['Accumulator']['variations'] == {'buy': [], 'sell': []}
    assert menu['Straddle']['variations'] == {} and menu['Straddle']['direction'] is None
    assert menu['ForwardExtra']['direction'] == 'optional', 'the level selects a forward extra'
    assert menu['Accumulator']['direction'] == 'required', 'nothing else can select a strip'
    assert 'floor' not in menu['ForwardExtra']['parameters'], (
        'a level is the variation\'s own, not a parameter every form of it shares')

    collar = mcp_server.describe_structure('ZeroCostCollar')
    assert collar['structure'] == 'ZeroCostCollar' and collar['vernacular']
    assert collar['fields'] and collar['recipe'] and set(collar['variations']) == {'floor', 'cap'}
    assert 'protection' in str(collar['variations']), 'the leg the cap is solved against is unnamed'
    assert 'FXOptionDeal' in str(collar['variations'])
    assert collar['variations']['floor']['buys'] == 'quote', 'a client given a floor sells the base'

    with pytest.raises(ToolError, match='ZeroCostCollar'):
        mcp_server.describe_structure('collar')


def test_booking_a_deal_prices_it(book):
    """The whole flow a plain-language booking rides: read the book, book a deal, run the book,
    read the deal's value off the result - held to the closed form, so a booking that writes but
    does not price cannot pass."""
    assert [row['reference'] for row in mcp_server.read_book()['deals']] == ['CF1']

    outcome = mcp_server.book_deal(json.loads(dump(BOOKED)))
    assert outcome['written'] is True and outcome['deal_path'] == '1'
    assert mcp_server.read_deal('1')['deal']['Reference'] == 'CF2'

    run = asyncio.run(mcp_server.execute_book())
    assert run['status'] == 'done' and run['waited'] is True
    values = mcp_server.deal_values(run['result_id'])
    assert values['CF2'] == pytest.approx(BOOKED['Amount'] * SPOT * np.exp(-RATE * 2.0), rel=1e-3)


def test_a_what_if_prices_without_writing(book):
    """The par-solve half: a candidate priced against the book with the file standing still -
    two of these at two amounts is the exact affine solve the booking docstring teaches."""
    before = book.read_bytes()
    run = asyncio.run(mcp_server.price_candidate(
        deal=json.loads(dump(dict(BOOKED, Reference='TRIAL')))))
    assert run['status'] == 'done'
    assert mcp_server.deal_values(run['result_id'])['TRIAL'] == pytest.approx(
        BOOKED['Amount'] * SPOT * np.exp(-RATE * 2.0), rel=1e-3)
    assert book.read_bytes() == before


def test_solving_then_booking_a_structured_deal(book):
    """The structuring flow: solve the amount that marks the deal at the margin, get the deal back
    ready to book, book it, and the book marks it there - the loop server-side."""
    outcome = asyncio.run(mcp_server.solve_deal(
        json.loads(dump(dict(BOOKED, Reference='SLV1'))), 'Amount', target=200_000.0))

    assert outcome['status'] == 'done'
    assert abs(outcome['solved']['residual']) <= 0.01
    assert outcome['solved_deal']['Amount'] == outcome['solved']['value']

    booked = mcp_server.book_deal(outcome['solved_deal'])
    run = asyncio.run(mcp_server.execute_book())
    assert booked['written'] is True
    assert mcp_server.deal_values(run['result_id'])['SLV1'] == pytest.approx(200_000.0, abs=0.01)


def test_a_margin_target_is_money_and_the_deal_records_what_was_charged(book):
    """A sales margin is an amount in a currency, not a number in whatever the book reports in.
    This book reports DOLLARS and carries a rand rate, so `{'amount': 50000, 'currency': 'ZAR'}`
    crosses at the book's own spot and the deal is solved to mark there - the desk's own side of
    the ticket, at PLUS the margin. What comes back records the charge as agreed, and books with
    it: the field is on every deal, a margin being a property of the ticket."""
    outcome = asyncio.run(mcp_server.solve_deal(
        json.loads(dump(dict(BOOKED, Reference='SLV2'))), 'Amount',
        target={'amount': 50_000.0, 'currency': 'ZAR'}))

    assert outcome['status'] == 'done'
    assert outcome['solved']['margin'] == {'amount': 50_000.0, 'currency': 'ZAR',
                                           'pricing_currency': 'USD', 'value': 50_000.0 * SPOT}
    assert outcome['solved_deal']['Sales_Margin'] == 50_000.0
    assert outcome['solved_deal']['Sales_Margin_Currency'] == 'ZAR'

    assert mcp_server.book_deal(outcome['solved_deal'])['written'] is True
    run = asyncio.run(mcp_server.execute_book())
    assert mcp_server.deal_values(run['result_id'])['SLV2'] == pytest.approx(
        50_000.0 * SPOT, abs=0.01)


def test_the_practical_loop_quotes_to_a_booked_structure(tmp_path):
    """Four tool calls: a Bloomberg-normalized quote block ticks the market, the bootstrap writes
    the surface, `solve_deal` finds the strike marking the option at the target premium, and the
    solved deal books."""
    from test_service import FX_OPTION, fx_vol_quotes
    path = tmp_path / 'book.json'
    path.write_text(json.dumps(json.loads(dump(job(
        sections={'Bootstrapper Configuration': {'FXVolSurfaceParameters': {}}}))), indent=2), newline='\n')
    service.BOOK = service.Book(str(path))
    try:
        ticked = mcp_server.update_market_quotes(json.loads(dump(fx_vol_quotes())))
        assert ticked['written'] is True and 'FXVol.USD.ZAR' in ticked['new_factors']

        option = json.loads(dump(FX_OPTION))
        outcome = asyncio.run(mcp_server.solve_deal(option, 'Strike_Price', target=500_000.0,
                                                    bounds=[12.0, 30.0]))
        assert outcome['status'] == 'done' and abs(outcome['solved']['residual']) <= 0.01

        booked = mcp_server.book_deal(outcome['solved_deal'])
        run = asyncio.run(mcp_server.execute_book())
        assert booked['written'] is True
        assert mcp_server.deal_values(run['result_id'])['OPT1'] == pytest.approx(
            500_000.0, abs=0.01)

        # and a values tick moves the mark - the market is live, not a snapshot baked at load
        patched = mcp_server.patch_market_values({'FxRate.ZAR': {'Spot': SPOT * 1.02}})
        moved = asyncio.run(mcp_server.execute_book())
        assert patched['written'] is True
        assert mcp_server.deal_values(moved['result_id'])['OPT1'] > 550_000.0
    finally:
        service.BOOK = None


def test_the_quoting_day_runs_from_a_structure_name_to_a_booked_collar(tmp_path, monkeypatch):
    """The structures layer end to end: the market ticks off a quote block; `solve_structure`
    quotes a zero-cost collar against the live book, both legs priced and the cap solved to a zero
    net, off the structure's own declaration; the quote and its sheet land in `DV_HOME/tmp` as one
    pending trade; `book_quote` approves it; the book marks the collar, legs and all.

    `DV_HOME` is set for real - where a pending trade waits is the contract under test."""
    from test_service import fx_vol_quotes
    monkeypatch.setenv('DV_HOME', str(tmp_path / 'home'))
    path = tmp_path / 'book.json'
    path.write_text(json.dumps(json.loads(dump(job(
        sections={'Bootstrapper Configuration': {'FXVolSurfaceParameters': {}}}))), indent=2),
        newline='\n')
    service.BOOK = service.Book(str(path))
    try:
        ticked = mcp_server.update_market_quotes(json.loads(dump(fx_vol_quotes())))
        assert ticked['written'] is True and 'FXVol.USD.ZAR' in ticked['new_factors']

        # the level a client names is the VARIATION's own parameter, and stating it is what
        # selects the form - a floor is the exporter's collar
        menu = mcp_server.describe_structure('ZeroCostCollar')
        declared = dict(menu['fields'], **menu['variations']['floor']['fields'])
        assert set(declared) >= {'pair', 'expiry', 'notional', 'notional_currency', 'floor'}
        # a structure may take its expiry as a tenor or as a date - the declaration says which
        expiry = ({'.Timestamp': '2025-06-28'}
                  if declared['expiry']['widget'] == 'DatePicker' else '1Y')
        # this book's FxRate.ZAR carries 18.5 USD per ZAR, so the pair's MARKET quote is its
        # reciprocal: the desk asks in market terms and the runner does the inversion, which is
        # the convention under test. A floor 5% out of the money.
        asked = {'pair': 'USDZAR', 'expiry': expiry, 'notional': 1_000_000.0,
                 'notional_currency': 'USD', 'floor': 1.0 / (SPOT * 0.95)}
        quote = asyncio.run(mcp_server.solve_structure('ZeroCostCollar', asked))

        assert quote['structure'] == 'ZeroCostCollar' and quote['quote_id']
        assert len(quote['legs']) == 2 and 'protection' in {leg['role'] for leg in quote['legs']}
        assert [leg for leg in quote['legs'] if leg['solved']], 'nothing was solved'
        # zero cost is the CONTRACT - and two worthless legs would satisfy it vacuously
        assert quote['net'] == pytest.approx(0.0, abs=1.0)
        assert min(abs(leg['premium']) for leg in quote['legs']) > 1.0

        # the same collar with the desk's charge in it: a margin is an amount in the currency it
        # was agreed in, and the client's paper is worth MINUS it
        charged = asyncio.run(mcp_server.solve_structure(
            'ZeroCostCollar', asked, margin={'amount': 500.0, 'currency': 'ZAR'}))
        assert charged['margin']['value'] == pytest.approx(500.0 * SPOT, rel=1e-12)
        assert charged['net'] == pytest.approx(-500.0 * SPOT, abs=1.0)
        assert charged['deal']['Sales_Margin'] == 500.0

        pending = quote['files']['quote']
        assert os.path.dirname(pending) == str(tmp_path / 'home' / 'tmp')
        assert quote['files']['sheet'] or quote['files']['sheet_note']
        with open(pending, encoding='utf-8') as handle:
            assert json.load(handle)['deal'], 'the pending file is not the trade it stands for'

        # what a host is handed is a URI it can open, not a path on the service's disk
        assert quote['resources'] == {
            'quote': 'derivus://quote/{}'.format(quote['quote_id']),
            'sheet': 'derivus://quote/{}/sheet'.format(quote['quote_id'])}
        assert json.loads(read_resource(quote['resources']['quote']))['deal'] == quote['deal']
        if quote['files']['sheet']:
            assert read_resource(quote['resources']['sheet'])[:2] == b'PK'
        else:
            with pytest.raises(ResourceError, match='derivus'):
                read_resource(quote['resources']['sheet'])

        booked = mcp_server.book_quote(quote['quote_id'])
        assert booked['written'] is True and 'validate' not in booked
        node = mcp_server.read_deal(booked['deal_path'])
        assert len(node['children']) == 2, 'the structure booked without its legs'

        run = asyncio.run(mcp_server.execute_book())
        assert mcp_server.deal_values(run['result_id'])[
            node['deal']['Reference']] == pytest.approx(quote['net'], abs=1.0)
        assert os.path.isfile(pending), 'the pending trade is an audit trail, not a scratch file'

        with pytest.raises(ToolError, match='tmp'):
            mcp_server.book_quote('nothing-was-ever-quoted-under-this')
    finally:
        service.BOOK = None


def test_a_recorded_desk_books_and_accepts_through_the_binding(tmp_path, monkeypatch):
    """THE WARNING CLOSES: a desk that keeps a record books through these tools. `book_deal` carries
    the signed quantity, the execution reference and the seat the service demands under a home, and
    without them it refuses in the SERVICE's own words rather than in a paraphrase here.

    And the two-act desk tier runs through the binding exactly as it does through the service: the
    acceptance is filed, the booking waits on a second seat, `approve_quote` under another one
    satisfies it and the retry books. Nothing is monkeypatched - a real home under tmp, a real
    policy declared through the record's own verb, the fault injected as DATA.
    """
    from derivus_spine import SpineLog, init_home, policy

    from test_service import fx_vol_quotes
    home = tmp_path / 'spine'
    init_home(home, 'subject-desk-one')
    monkeypatch.setenv('DV_SPINE_HOME', str(home))
    monkeypatch.setenv('DV_SPINE_ACTOR', 'subject-desk-one')
    monkeypatch.setenv('DV_HOME', str(tmp_path / 'home'))

    client = {'Object': 'NettingCollateralSet', 'Reference': 'CLIENT_A', 'Netted': 'True',
              'Collateralized': 'False', 'Agreement_Currency': 'USD', 'Balance_Currency': 'USD',
              'Funding_Rate': 'USD', 'Liquidation_Period': 0.0, 'Settlement_Period': 0.0,
              'Credit_Support_Amounts': {'Counterparty': 'CPTY_A'}}
    document = json.loads(dump(job(
        sections={'Bootstrapper Configuration': {'FXVolSurfaceParameters': {}}})))
    document['Calc']['Deals']['Deals']['Children'].append(
        {'Instrument': {'.Deal': client}, 'Children': []})
    path = tmp_path / 'book.json'
    path.write_text(json.dumps(document, indent=2), newline='\n')
    service.BOOK = service.Book(str(path))
    try:
        assert mcp_server.update_market_quotes(json.loads(dump(fx_vol_quotes())))['written']
        deal = {'Object': 'FixedCashflowDeal', 'Reference': 'CF-RECORDED', 'Currency': 'ZAR',
                'Discount_Rate': 'ZAR', 'Amount': 250_000.0,
                'Payment_Date': {'.Timestamp': '2026-06-28'}}

        with pytest.raises(ToolError, match='quantity'):
            mcp_server.book_deal(deal, parent_reference='CLIENT_A')
        with pytest.raises(ToolError, match='execution_reference'):
            mcp_server.book_deal(deal, parent_reference='CLIENT_A', quantity=-250_000.0)

        booked = mcp_server.book_deal(deal, parent_reference='CLIENT_A', quantity=-250_000.0,
                                      execution_reference='EXEC-11', actor='subject-desk-one')
        assert booked['written'] is True and booked['recorded']['lsn']

        log = SpineLog(home)
        try:
            policy.declare(log, 'subject-desk-one', policy.TIERS_POLICY,
                           {'tiers': [{'name': 'desk', 'four_eyes': True}]})
        finally:
            log.close()

        quote = asyncio.run(mcp_server.solve_structure('ZeroCostCollar', {
            'pair': 'USDZAR', 'expiry': '1Y', 'notional': 1_000_000.0,
            'notional_currency': 'USD', 'floor': 1.0 / (SPOT * 0.95)},
            netting_set='CLIENT_A', actor='subject-desk-one'))
        waiting = mcp_server.book_quote(quote['quote_id'], actor='subject-desk-one')
        assert waiting['written'] is False and waiting['tier']['name'] == 'desk'
        assert 'no verdict is filed' in waiting['waits_on']

        signed = mcp_server.approve_quote(quote['quote_id'], 'subject-desk-two')
        assert signed['ticket'] == waiting['accepted']['ticket']
        accepted = mcp_server.book_quote(quote['quote_id'], actor='subject-desk-one')
        assert accepted['written'] is True and accepted['tier']['approval_lsn'] == signed[
            'recorded']['lsn']
        assert accepted['accepted'] == waiting['accepted'], 'a retry minted a second acceptance'
    finally:
        service.BOOK = None


def test_the_configuration_is_declared_and_one_dial_is_set(tmp_path):
    """The two configuration tools end to end, through the in-process service: the store read as
    the declarations with the menu it NAMES resolved beside it, one dial set on the live book, and
    a dial that will not build refused by name with the file untouched. The entry is asked for by
    the factor the family writes and lands under the class name the book spells it by."""
    from test_service import configured_book, surface_nodes

    path = tmp_path / 'book.json'
    configured_book(path, {'FXVolSurfaceParameters': {}})
    try:
        declared = mcp_server.describe_configuration()
        entry = declared['sections']['Bootstrapper Configuration']['types']['FXVol']
        assert entry['aliases'] == ['FXVolSurfaceParameters']
        assert entry['fields']['Grid_Tolerance']['value'] == 1e-4, 'not the declared default'
        assert 'HermiteRT' in declared['interpolations']['InterestRate']

        outcome = mcp_server.configure_book(
            'Bootstrapper Configuration', 'FXVol', {'Grid_Tolerance': 0.5})
        assert outcome['written'] is True and outcome['rewrote'] == ['FXVol.USD.ZAR']
        assert json.loads(path.read_text())['Calc']['MergeMarketData']['ExplicitMarketData'][
            'Bootstrapper Configuration'] == {
                'FXVolSurfaceParameters': {'Prices': 'FXVol', 'Grid_Tolerance': 0.5}}
        assert surface_nodes(path)

        before = path.read_bytes()
        with pytest.raises(ToolError, match='Sigma_L_Bounds'):
            mcp_server.configure_book('Bootstrapper Configuration', 'LogVar2FJModelParameters',
                                      {'Sigma_L_Bounds': '2.0,0.3'})
        assert path.read_bytes() == before
    finally:
        service.BOOK = None


def test_a_curve_is_described_set_up_and_read_back(book):
    """The two curve tools end to end through the in-process service, which is the whole move the
    ruling asks for: a desk reads what it could set up, states the benchmark rows, and the book
    solves the curve off them.

    `describe_curve` with nothing named carries the SEEDED menu - the curves this workstation's
    seed declares, each with its conventions and its tenor/security rows - and after the set-up it
    carries the book's own block read back as the definition it is, in the same row shape
    `configure_curve` takes. Neither tool owns any of that: both are one call each.

    `interpolation` ROUND TRIPS as the curve's own rule, which is the one thing the two tools carry
    that lives in the section rather than on the block. Killing mutation: the tool dropping it, so
    a model naming a scheme gets the default back and nothing says so.
    """
    from test_service import CURVE_ROWS

    tools = {tool.name: tool for tool in asyncio.run(mcp_server.MCP.list_tools())}
    assert set(tools['describe_curve'].input_schema['properties']) == {'curve'}
    assert set(tools['configure_curve'].input_schema['properties']) == {
        'curve', 'currency', 'rows', 'discount_rate', 'conventions', 'interpolation'}

    menu = mcp_server.describe_curve()
    assert menu['curves'] == {} and 'ZAR' in menu['seeded']
    assert menu['seeded']['ZAR']['conventions']['front'] == 'fixings/3M'
    assert {'tenor': '3M', 'security': 'JIBA3M Index'} in menu['seeded']['ZAR']['rows']

    outcome = mcp_server.configure_curve('ZAR', 'ZAR', CURVE_ROWS)
    described = mcp_server.describe_curve('ZAR')
    definition = described['curves']['InterestRatePrices.ZAR']

    assert outcome['written'] is True and outcome['rewrote'] == ['InterestRate.ZAR']
    assert outcome['knots'] == [row['tenor'] for row in CURVE_ROWS]
    assert 'seeded' not in described, 'a named curve is not the menu'
    # the book states no `Price Factor Interpolation`, so the readout says what the curve gets
    assert definition['currency'] == 'ZAR' and definition['interpolation'] == 'Linear'
    assert definition['interpolation_source'] == 'default'
    assert definition['conventions']['fixed_frequency'] == '3M'
    assert definition['rows'] == [dict(row, use='Yes') for row in CURVE_ROWS]

    mcp_server.configure_curve('ZAR', 'ZAR', CURVE_ROWS, interpolation='HermiteRT')
    ruled = mcp_server.describe_curve('ZAR')['curves']['InterestRatePrices.ZAR']
    assert (ruled['interpolation'], ruled['interpolation_source']) == ('HermiteRT', 'curve')

    with pytest.raises(ToolError, match='Cubic'):
        mcp_server.configure_curve('ZAR', 'ZAR', CURVE_ROWS, interpolation='Cubic')
    with pytest.raises(ToolError, match='3Q'):
        mcp_server.configure_curve('ZAR', 'ZAR', [dict(row, tenor='3Q') if row['tenor'] == '5Y'
                                                  else row for row in CURVE_ROWS])


def test_the_date_is_set_by_name_and_the_curve_follows_it(book):
    """A model that can book and quote can also say WHEN. One string, both of the book's dates, and
    every curve block re-rolled onto the day it names - the whole "value this as of" move, with no
    document editing and no second verb to remember.

    Killing mutations: the tool posting the date under any other key (the verb answers 422 naming
    `base_date`); the answer not carrying the date it wrote, which is the only thing a model can
    read the move back off.
    """
    from test_service import CURVE_ROWS

    tools = {tool.name: tool for tool in asyncio.run(mcp_server.MCP.list_tools())}
    assert set(tools['set_base_date'].input_schema['properties']) == {'base_date'}
    mcp_server.configure_curve('ZAR', 'ZAR', CURVE_ROWS)
    before = mcp_server.describe_curve('ZAR')

    outcome = mcp_server.set_base_date('2024-09-30')
    described = mcp_server.describe_curve('ZAR')
    rows = described['curves']['InterestRatePrices.ZAR']['rows']

    assert outcome['written'] is True and outcome['base_date'] == '2024-09-30'
    assert outcome['reauthored'] == ['InterestRatePrices.ZAR'] and outcome['held_out'] == []
    assert described['base_date'] == '2024-09-30' and before['base_date'] == '2024-06-28'
    assert rows == before['curves']['InterestRatePrices.ZAR']['rows'], 'a re-roll moved a quote'
    assert json.loads(book.read_text())['Calc']['Calculation']['Base_Date'] == {
        '.Timestamp': '2024-09-30'}

    with pytest.raises(ToolError, match='no date'):
        mcp_server.set_base_date('the thirtieth')


def test_the_ticker_vocabulary_and_its_evidence_are_one_read(book, tmp_path, monkeypatch):
    """The three vocabulary tools against the in-process service, which is the whole move an IPV
    reader needs: read what this desk could quote, say what it quotes, and read every knot back with
    the print behind it. `DV_HOME` is the gate's own tmp, so the seed is the packaged questionnaire
    and the map is this gate's, never the workstation's; no terminal is reached either way.

    Killing mutations: `describe_securities` answering the seed alone, so a knot's evidence needs
    the map file opened beside it; `configure_securities` posting the entry anywhere but under its
    block and key, which the read then answers unchanged.
    """
    from test_service import CURVE_ROWS

    monkeypatch.setenv('DV_HOME', str(tmp_path / 'home'))
    tools = {tool.name: tool for tool in asyncio.run(mcp_server.MCP.list_tools())}
    bare = mcp_server.describe_securities('rates')

    assert set(tools['describe_securities'].input_schema['properties']) == {'block'}
    assert set(tools['configure_securities'].input_schema['properties']) == {
        'block', 'key', 'entry'}
    assert set(tools['verify_securities'].input_schema['properties']) == {
        'block', 'key', 'securities', 'wait_seconds'}
    assert bare['provisioned'] is False and set(bare['seed']) == {'rates'}
    assert bare['used'] == [] and bare['seed']['rates']['ZAR']['prefix'] == 'SASW'

    mcp_server.configure_curve('ZAR', 'ZAR', CURVE_ROWS)
    written = mcp_server.configure_securities('fx_vol', 'pairs', ['USDZAR'])
    answer = mcp_server.describe_securities()
    used = {row['security']: row for row in answer['used']}

    assert written['written'] is True and written['key'] == 'pairs'
    assert 'USDZARV3M BGN Curncy' in written['candidates']
    assert not [name for name in written['candidates'] if name.startswith('EURZAR')]
    assert answer['seed']['fx_vol']['pairs'] == ['USDZAR']
    assert [row['tenor'] for row in answer['used']] == [row['tenor'] for row in CURVE_ROWS]
    assert used['SASW1 BGN Curncy']['quote'] == 7.62 and used['SASW1 BGN Curncy']['use'] == 'Yes'
    assert used['JIBA3M Index']['evidence'] == {'verdict': 'unmapped'}, 'an unverified knot'

    with pytest.raises(ToolError, match='vocabulary block'):
        mcp_server.configure_securities('curves', 'ZAR', {})


def test_a_market_is_set_up_from_what_a_trade_needs(book, tmp_path, monkeypatch):
    """THE TWO MARKET-BUILDING TOOLS ARE THIN. `book_dependencies` is the walk verbatim - the
    CANDIDATE's own factors, each missing one carrying the seed entry that would supply it - and
    `setup_market` is one POST that waits, carrying only the argument it was given.

    A workstation with no terminal refuses at the service, and the model reads that refusal rather
    than a crash: no session is opened here or anywhere.

    WHAT A MODEL RECEIVES IS THE JOB'S OWN OUTCOME - `stats.Setup` with the id and the status
    beside it - held here against a REAL `/results` envelope rather than a flat dict the service
    never emits, which is the only way the shape can be gated at all.

    Killing mutations: the walk composed here out of `validate_book` plus `describe_securities`,
    which answers the BOOK's want-list rather than the candidate's; `setup_market` sending every
    argument it declares, which names a `deal_path` beside a `pair` and the service refuses; the
    raw envelope handed back, which carries `tables` and no `result_id`.
    """
    from derivus_bloomberg import session as bloomberg
    from derivus_bloomberg.errors import BloombergUnavailable
    from test_service import CURVE_ROWS, XCCY

    def absent():
        raise BloombergUnavailable('no blpapi on this workstation')

    monkeypatch.setenv('DV_HOME', str(tmp_path / 'home'))
    monkeypatch.setattr(bloomberg, 'blpapi_module', absent)
    tools = {tool.name: tool for tool in asyncio.run(mcp_server.MCP.list_tools())}
    walked = mcp_server.book_dependencies(deal=json.loads(dump(XCCY)))
    rows = {row['factor']: row for row in walked['factors']}

    assert set(tools['book_dependencies'].input_schema['properties']) == {
        'deal_path', 'deal', 'parent_reference'}
    assert set(tools['setup_market'].input_schema['properties']) == {
        'pair', 'deal', 'parent_reference', 'deal_path', 'wait_seconds'}
    assert rows['InterestRate.EUR']['supply']['key'] == 'EUR'
    assert rows['FxRate.JPY']['supply'] == {'block': 'fx_spot', 'key': 'USDJPY',
                                            'securities': 1, 'verified': 0}
    assert 'InterestRate.ZAR' not in rows, 'the walk answered for the book, not the candidate'
    assert mcp_server.book_dependencies()['factors'], 'the whole book is a walk too'

    with pytest.raises(ToolError, match='Bootstrapper Configuration'):
        asyncio.run(mcp_server.setup_market(pair='EURZAR'))

    mcp_server.configure_curve('ZAR', 'ZAR', CURVE_ROWS)
    with pytest.raises(ToolError, match='blpapi'):
        asyncio.run(mcp_server.setup_market(pair='EURZAR'))

    # the envelope a finished queued job really answers with: the outcome under its own Stats key,
    # beside the tables and the status `/results` always carries
    landed = {'written': True, 'installed': ['FXVol.EUR.ZAR'], 'refused': [],
              'not_supplied': [], 'check': [], 'seconds': 1.2}
    terminal = Terminal({'status': 'done', 'tables': {}, 'stats': {'Setup': landed}})
    mcp_server.configure(base_url='http://testserver', session=terminal)
    answer = asyncio.run(mcp_server.setup_market(pair='EURZAR'))

    assert answer == dict(landed, result_id='bbg-1', status='done')
    assert 'tables' not in answer, 'the result envelope reached the model instead of the outcome'
    assert terminal.requests[0] == ('POST', 'http://testserver/book/setup', {'pair': 'EURZAR'})

    # past the wait the answer is the pointer it always was, the set-up carrying on service-side
    mcp_server.configure(base_url='http://testserver',
                         session=Terminal({'status': 'running'}))
    waiting = asyncio.run(mcp_server.setup_market(pair='EURZAR', wait_seconds=0.0))

    assert waiting['status'] == 'running' and 'poll_result' in waiting['hint']


def test_a_rejected_booking_is_an_answer_that_wrote_nothing(book):
    """A refusal must reach the model as DATA - the engine's own messages, verbatim - because the
    model's next move is to fix exactly what they name. And it must not have touched the file."""
    before = book.read_bytes()
    outcome = mcp_server.book_deal(json.loads(dump(BINARY)))

    assert outcome['written'] is False
    assert 'Cash_Payoff is required' in outcome['refused']
    assert book.read_bytes() == before


def test_the_store_names_what_must_be_stated_and_a_missing_term_refuses_by_name(book):
    """WHAT A HOST ASKS is what must be STATED, so `required` is the terms - the fields whose
    declared default is a blank panel's value - and `convention` marks every field whose default IS
    what leaving it out means. A model booking the `required` list alone books a live swap; drop
    one of them and the binding answers the engine's own sentence with the file untouched.

    KILLING MUTATION: `required` published as `default is REQUIRED` alone, which is what it was.
    `SwapInterestDeal` then lists NOTHING - not its currency, not its dates, not its rate - and a
    model reading the store books an empty block that prices at whatever the panel would show.
    """
    swap = mcp_server.describe_instrument_type('SwapInterestDeal')
    assert sorted(swap['required']) == ['Currency', 'Effective_Date', 'Maturity_Date', 'Object',
                                        'Pay_Rate_Type', 'Principal', 'Swap_Rate']
    assert swap['fields']['Pay_Timing']['convention'] is True
    assert 'convention' not in swap['fields']['Swap_Rate']
    assert 'required' not in swap['fields']['Pay_Timing']

    before = book.read_bytes()
    terms = {'Object': 'SwapInterestDeal', 'Reference': 'SW1', 'Currency': 'ZAR',
             'Discount_Rate': 'ZAR', 'Interest_Rate': 'ZAR',
             'Effective_Date': {'.Timestamp': '2024-06-28'},
             'Maturity_Date': {'.Timestamp': '2026-06-28'}, 'Pay_Rate_Type': 'Fixed',
             'Swap_Rate': 8.0, 'Principal': 1_000_000.0}
    # a host with nothing to say for a field sends `null` as readily as it omits the key, and
    # `Pay_Rate_Type: null` is the expensive one - the pricer's `== 'Fixed'` takes the else branch
    # and the swap becomes receive-fixed, the same notional with the other sign
    for key in ('Swap_Rate', 'Pay_Rate_Type', 'Principal'):
        for said in ({k: v for k, v in terms.items() if k != key}, dict(terms, **{key: None})):
            refused = mcp_server.book_deal(said)
            assert refused['written'] is False, (key, refused)
            assert '{} is not stated'.format(key) in refused['refused'], refused['refused']
            assert book.read_bytes() == before
    assert mcp_server.book_deal(terms)['written'] is True


def test_what_a_model_actually_mistypes_comes_back_as_data(book):
    """The four a model sends unprompted - a misspelt type, no type at all, an amount as text and a
    date as a bare string - reach it as `{written: false, refused: [...]}` with the file untouched,
    which is the shape whose next move is to fix what it names.

    KILLING MUTATION: all four WRITTEN, `book_issues: {deal_messages: 1}`; the bare date a tool
    error reading `DV_Service answered 500 for POST /book/deals: Internal Server Error`.
    """
    before = book.read_bytes()
    refused = {
        'typo': mcp_server.book_deal(json.loads(dump(dict(BOOKED, Object='FixedCashflwDeal')))),
        'nameless': mcp_server.book_deal({'Reference': 'CF3', 'Amount': 1.0}),
        'text amount': mcp_server.book_deal(json.loads(dump(dict(BOOKED, Amount='1e6')))),
        'bare date': mcp_server.book_deal(
            dict(json.loads(dump(BOOKED)), Payment_Date='2027-01-15'))}

    assert not any(outcome['written'] for outcome in refused.values())
    assert all(outcome['refused'] for outcome in refused.values())
    assert "Object is 'FixedCashflwDeal'" in refused['typo']['refused'][0]
    assert refused['text amount']['refused'] == ["Amount must be a number, not '1e6'"]
    assert '.Timestamp' in refused['bare date']['refused'][0]
    assert book.read_bytes() == before


def test_an_amendment_changes_the_value_it_names(book):
    """The 'change a value' flow in plain language: amend the amount, see the deal carry it, see
    the book mark it - and an amendment that breaks the deal is an answer, not a write."""
    outcome = mcp_server.amend_deal('0', {'Amount': 500_000.0})
    assert outcome['written'] is True
    assert mcp_server.read_deal('0')['deal']['Amount'] == 500_000.0

    run = asyncio.run(mcp_server.execute_book())
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
    run = asyncio.run(mcp_server.execute_book(wait_seconds=0.0))
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
    run = asyncio.run(mcp_server.execute_book())
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
    """A transport that answers a queued submit - the tick's or the set-up's - and then reads a
    scripted `/results` sequence: the tool's own contract with no terminal, no service and no
    socket (the `Down` precedent). The last poll stands once the script runs out."""

    class Reply:
        def __init__(self, payload):
            self.status_code, self.payload, self.text = 200, payload, ''

        def json(self):
            return self.payload

    def __init__(self, *polls):
        self.polls, self.requests = list(polls), []

    def request(self, method, url, **kwargs):
        self.requests.append((method, url, kwargs.get('json')))
        if url.endswith('/book/bloomberg') or url.endswith('/book/setup'):
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
    """A client resets its timeout on each notification, which is what carries the five-minute
    first use - so every poll reaches the context: how long the wait has run, the wait it was
    given, and the run's status carrying the job's own note where it publishes one.

    Killing mutation: report the status alone and both notes collapse to `queued` and `running`,
    leaving a user watching a provisioning with nothing to watch.
    """
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
    assert [(total, note) for _, total, note in watching.reported] == [
        (360.0, 'queued - copying the seed'), (360.0, 'running - verifying USDZAR')]
    assert delays == [0.25, 0.25], 'a poll between notifications, not a spin'


def test_a_waiting_tool_notifies_the_host_the_whole_time_it_waits(book):
    """The cadence, against the real queue: the one worker is held for three seconds by a job of
    the store's own, so the candidate really is queued while the tool waits - and a host that cuts
    a call quiet for a minute hears from it about once a second. Nothing is patched.

    Killing mutation: drop the `report_progress` call from `_await_result` and the context records
    nothing; leave it inside an `if progress` and a run publishing no note goes silent too.
    """
    holding = threading.Event()
    service.EXECUTOR.submit(service.Job('mcp-busy', Held('mcp-busy', [], hold=holding), {}),
                            service.HEAVY)
    threading.Timer(3.0, holding.set).start()
    watching = Watching()

    started = time.monotonic()
    # a candidate nothing else prices, so this really queues rather than reading a stored run
    run = asyncio.run(mcp_server.price_candidate(
        deal=json.loads(dump(dict(BOOKED, Reference='BUSY', Amount=3_141_592.0))), ctx=watching))
    waited = time.monotonic() - started

    assert run['status'] == 'done' and waited > 3.0
    assert watching.reported, 'a wait a host hears nothing from is a wait it cuts'
    seconds = [done for done, _, _ in watching.reported]
    assert max(b - a for a, b in zip([0.0] + seconds, seconds + [waited])) <= 5.0
    said = {(total, note) for _, total, note in watching.reported}
    assert (120.0, 'queued') in said and said <= {(120.0, 'queued'), (120.0, 'running')}
