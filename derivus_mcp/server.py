"""The derivus verbs as MCP tools, over HTTP - a client of `DV_Service` and nothing else.

This module imports `requests` and `mcp` and must stay that way (a gate reads its imports): it is
a client of the same endpoints the web UI and the Excel add-in are clients of, so it can never
reach into the engine to answer something the verbs cannot - and it lives outside the `derivus`
package because importing any of it pulls the whole engine (torch included) into a process that
only wants to talk HTTP. Anything a tool needs that an endpoint cannot answer is a missing verb on
the service, never code here.

The tools work the service's LIVE BOOK (`DV_Service --book <job file>`): the file on disk is the
book of record, a booking validates before it writes, and every other client - the web UI's etag
poll, Excel - sees a booking on its next read. Market data moves on the engine's own terms: quote
blocks tick through `update_market_quotes` (values only - structure is a re-authoring, refused by
name - with the bootstrap judging the whole write), spots and vols through `patch_market_values`
(the `bind='value'` seam; a structural key is refused by the engine's rule), and the dials a family
fits under through `configure_book`, which re-bootstraps what they change. What never moves from
here is structure: a new factor, a moved pillar, a changed convention is authoring, not ticking.

Answers are SUMMARIES AND POINTERS, never payloads: the model needs to know a deal booked or a
calculation ran, not to hold a simulation cube in its context. A run comes back as its replay
tuple, its stats and each table's SHAPE; cells come one capped page at a time through
`fetch_table`, a table too wide to read as text is refused by name (that is what the web UI is
for), and `deal_values` checks a result's shape before it fetches anything.

THE BLOTTER'S TWO DATA VIEWS are `book_risk_summary` and `xva_view`, and they answer two different
questions on purpose. RISK is whole-book and COUNTERPARTY-BLIND: one base valuation with first-order
Greeks over everything the desk holds, cached on the book's own content, so it refreshes with the
book and costs nothing to ask again. XVA is PER NETTING SET and a CACHED PROJECTION: a credit Monte
Carlo takes minutes, so it never rides a tick - `xva_view` reads the last run of each set off the
desk's own file, each row carrying its own `as_of`, and `recalc_xva` is the only thing that moves
them. Staleness there is data rather than a failure.

A structure is DECLARED, never composed here: `describe_structure` reads the desk's own
vocabulary - the sales names, the legs, the recipe - off that same `/schema`, `solve_structure`
runs the recipe server-side and files the pending trade under its quote id, and `book_quote` is the
approval that makes it a trade. What a zero-cost collar IS therefore never depends on which model
is driving.

Run: `DV_MCP` (stdio; `python -m derivus_mcp.server` from a source tree), with `RF_SERVICE_URL`
naming the service (default http://127.0.0.1:8000 - the same variable the Excel add-in reads).
"""
import asyncio
import os
import time

import requests
from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ResourceError, ToolError
from mcp_types import ToolAnnotations

#: What a host shows the MODEL before it calls anything - the desk's orientation, not this
#: module's maintenance notes. A host reads it once per session, so it says what the desk IS,
#: where to start, the wire forms, the one axis a model gets wrong, and what a refusal means.
INSTRUCTIONS = """\
derivus desk - one trading book with its market, priced by the derivus engine and served by
DV_Service. You are a dealer's assistant on this desk: everything you do goes through these
tools, the book on disk is the record, and every write is validated before it lands.

START WITH desk_status: the book's date and currency, its curves and surfaces with when each
was snapped, the calibrated models, the netting sets and the last XVA per set, and whether a
terminal is present. A sandboxed desk has no terminal - the snapped market is what prices, and
set_base_date is how the book is valued as of another day.

A DEAL IS BOOKED BY STATING WHAT MUST BE STATED - the terms: what it is, in what currency, on
what dates, at what strike and for how much. Everything else is a CONVENTION the declaration
already says, and leaving it out means exactly that default. describe_instrument_type lists the
two, and a term left unsaid - the key absent, or sent as null, which says the same nothing - is
refused by name rather than priced at a placeholder.

A WORKING DAY: read_book to see what is held; describe_instrument_type before booking a type
you have not booked; book_deal for a plain instrument; for FX options solve_structure and then
book_quote, which take market terms and handle the axis; book_risk_summary for the mark and
its gradient; execute_book or price_candidate for a what-if; solve_deal for a par amount or a
strike to a target; recalc_xva ONLY on request, it is minutes, and xva_view for what stands.

THE WIRE FORMS a deal is written in: dates {".Timestamp": "YYYY-MM-DD"}, periods
{".DateOffset": "3M"}, percentages {".Percent": 2.5}, numbers as numbers; a curve or a surface
is named by the Price Factors block it must match, which desk_status lists. An FX option's
Strike_Price is on the ENGINE axis - the book's reporting currency per unit of
Underlying_Currency, so a USDZAR strike of 17.50 on a USD book is 1/17.50 - which is why FX
options are quoted through solve_structure, where strikes are market terms. Structures are
declared: describe_structure lists them with the parameters they take; never compose a collar
or a seagull from legs by hand.

REFUSALS ARE ANSWERS: {written: false, refused: [...]} names what to fix - fix that one thing
and post again. Deals are addressed by deal_path, which is positional, so pass the reference
you read at a path whenever you amend or delete. A legacy book imports as one
NettingCollateralSet carrying its deals as Children, through book_deal; one bad deal refuses
the whole batch and names it.

THE MARKET: update_market_quotes and patch_market_values move values; configure_curve,
configure_book and set_base_date change structure and re-solve; tick_market_from_bloomberg
needs a terminal on the service's own workstation, and describe_securities reads the ticker
vocabulary behind it with the print every curve knot was solved from. A booking refused for
market data the book lacks is answered by book_dependencies - what that trade needs and which
seed entry would supply it - and cured by setup_market, which discovers only what is unknown and
installs the surface, spots and curves as editable defaults in one write. Bootstrapping dials,
Bloomberg ticker codes and curve set-ups are normally configured once in the web UI - ask
before changing them here.

THE RECORD, where this desk keeps one: book_diary is everything the book owes or is owed with
the fact each row waits on, close_check says whether a close on a day is legal and names what
is outstanding, and book_reconcile says where the book file and the record disagree."""

MCP = MCPServer('derivus', instructions=INSTRUCTIONS)
READ_ONLY = ToolAnnotations(read_only_hint=True)

SERVICE = None


class Service:
    """A `DV_Service` at a URL, `call` raising `ToolError` for an unreachable service or a 4xx/5xx.

    `session` is the transport seam - anything with a requests-style `request(method, url, ...)` -
    which is how a caller drives the tools in process without a socket. It duplicates
    `excel_integration`'s client because the import gate holds this module to `requests` + `mcp`.
    """

    def __init__(self, base_url=None, session=None, timeout=120.0):
        self.base_url = (base_url if base_url is not None
                         else os.getenv('RF_SERVICE_URL', 'http://127.0.0.1:8000')).rstrip('/')
        self.session = session if session is not None else requests.Session()
        self.transport = {} if session is not None else {'timeout': timeout}

    def request(self, method, path, **kwargs):
        try:
            response = self.session.request(method, self.base_url + path,
                                            **dict(self.transport, **kwargs))
        except requests.RequestException as error:
            raise ToolError(
                'DV_Service is not reachable at {}: {}. Start it with `DV_Service --book '
                '<job file>`, or point RF_SERVICE_URL at a running one.'.format(
                    self.base_url, error))
        if response.status_code >= 400:
            raise ToolError('DV_Service answered {} for {} {}: {}'.format(
                response.status_code, method, path, response.text[:500]))
        return response

    def call(self, method, path, **kwargs):
        return self.request(method, path, **kwargs).json()


def configure(base_url=None, session=None):
    """Bind the service this process talks to and return it. `main` calls it from the environment;
    a caller driving the tools in process passes its own `session`."""
    global SERVICE
    SERVICE = Service(base_url, session)
    return SERVICE


def service():
    return SERVICE if SERVICE is not None else configure()


#: The caps that keep a result out of the model's context: a page is at most this many rows, and a
#: table wider than this is a scenario cube the model should never hold as text.
MAX_PAGE_ROWS = 200
MAX_TABLE_COLUMNS = 60

#: How many gradient rows a risk summary carries - the biggest by absolute size. The whole vector
#: is reached through `execute_book({"Greeks": "First"})` and `fetch_table`.
MAX_GREEK_ROWS = 15


#: What a quote sheet IS on the wire, so a host offers it as the spreadsheet a client is sent.
#: The service names it too - the import gate is what keeps this module from reading it off there.
SHEET_MIME = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'


def _resource(path, binary=False):
    """A service GET for a RESOURCE: the same call, with the refusal wearing the exception a
    resource refuses by, so a host reads the service's own words rather than a generic crash."""
    try:
        answer = service().request('GET', path)
    except ToolError as error:
        raise ResourceError(str(error))
    return answer.content if binary else answer.json()


def _raw_result(result_id):
    return service().call('GET', '/results/{}'.format(result_id))


def _summary(raw, result_id):
    """A run trimmed to what the model should hold: identity, stats, and each table's shape as one
    line. Never a column list and never a cell - those stay behind `fetch_table`."""
    trimmed = {'result_id': result_id, 'status': raw.get('status')}
    for key in ('plan_hash', 'values_hash', 'seed', 'stats', 'error'):
        if key in raw:
            trimmed[key] = raw[key]
    if 'tables' in raw:
        trimmed['tables'] = {
            name: '{} rows x {} columns'.format(shape['rows'], len(shape['columns']) or 1)
            for name, shape in raw['tables'].items()}
    return trimmed


async def _await_result(result_id, wait_seconds, ctx=None, summarise=True,
                        hint=', and fetch_table once it is done'):
    """Poll `/results/{id}` until it settles or `wait_seconds` runs out, saying on every poll how
    long it has waited, so one tool call returns the answer rather than a model burning a turn per
    poll - and a host never hears silence.

    A desktop host cuts a tool call that stays quiet for about a minute and resets that clock on
    every notification, so a run measured in minutes only finishes if the wait speaks: `done` is
    the seconds waited, `total` the wait asked for, and the note is the run's status with whatever
    the job itself is publishing. On timeout the id and the way forward travel in `hint`, so a long
    run stays reachable.
    """
    started = time.monotonic()
    while True:
        # Every HTTP call is blocking, so it goes to a thread and the event loop stays free to put
        # the progress notifications on the wire.
        raw = await asyncio.to_thread(_raw_result, result_id)
        if raw.get('status') not in ('queued', 'running'):
            return dict(_summary(raw, result_id), waited=True) if summarise else raw
        waited = time.monotonic() - started
        if ctx is not None:
            note = (raw.get('progress') or {}).get('note')
            await ctx.report_progress(round(waited, 1), wait_seconds,
                                      ' - '.join(filter(None, (raw['status'], note))))
        if waited >= wait_seconds:
            return {'result_id': result_id, 'status': raw['status'],
                    'hint': 'still {} - call poll_result({!r}) to check again{}'.format(
                        raw['status'], result_id, hint)}
        # a quick run should not pay a second of latency, a long one should not spin
        await asyncio.sleep(1.0 if waited >= 2.0 else 0.25)


def _booking(outcome):
    """A booking outcome trimmed to what happened to this deal, plus a count of anything else
    outstanding in the book. The whole verdict is `validate_book`'s to serve."""
    verdict = outcome.pop('validate', None) or {}
    issues = {name: count for name, count in
              (('deal_messages', len(verdict.get('deals', {}))),
               ('missing_factors', len(verdict.get('factors', [])))) if count}
    if issues:
        outcome['book_issues'] = dict(issues, hint='validate_book lists them')
    return outcome


def _walk(children, path=''):
    for position, node in enumerate(children):
        deal_path = '{}/{}'.format(path, position) if path else str(position)
        yield deal_path, node
        yield from _walk(node.get('Children', []), deal_path)


# --------------------------------------------------------------------------- discovery


@MCP.tool(annotations=READ_ONLY)
def list_instrument_types() -> dict:
    """Every deal type the engine can price, before booking anything unfamiliar.

    `groups` is the create menu (a human-oriented grouping by asset class), `types` is the flat
    list of bookable type names, and `containers` names the types that can HOLD other deals - a
    structured deal over its legs, a netting set over a book. Only a container may be named as
    `parent_reference` in `book_deal`. Type names are exact class names (`FXForwardDeal`,
    `QEDI_CustomAutoCallSwap`), not descriptions.
    """
    instrument = service().call('GET', '/schema')['Instrument']
    return {'groups': instrument['groups'], 'containers': instrument['containers'],
            'types': sorted(instrument['types']), 'count': len(instrument['types'])}


@MCP.tool(annotations=READ_ONLY)
def describe_instrument_type(deal_type: str) -> dict:
    """Every field one deal type takes, as the engine declares it - call this before booking a
    type you have not booked before.

    `fields` is keyed by the JSON key you write in the deal. Each entry says what the field is
    (`description`), what it defaults to (`value`), whether you must STATE it (`required` - also
    summarised in the top-level `required` list), whether leaving it out MEANS that default
    (`convention`), and for a choice exactly which strings are valid (`values` - a field with
    `values` accepts nothing else). State every `required` field: the rest are conventions and an
    omitted one is read as its `value`, while a required one left out - or sent as `null` - is
    refused by name, because its default is what a blank panel shows rather than a term anybody
    meant. Dates are
    `{".Timestamp": "YYYY-MM-DD"}`, percentages `{".Percent": 2.5}` (already in percent), rate
    curves are named by a string that must match a `Price Factors` block. `accepts_children` says
    whether this type can hold other deals.

    `deal_type` is one of the names `list_instrument_types` returns, spelled exactly.
    """
    schema = service().call('GET', '/schema')['Instrument']
    sections = schema['types'].get(deal_type)
    if sections is None:
        close = [t for t in schema['types'] if deal_type.lower() in t.lower()]
        raise ToolError('{!r} is not a deal type. {}'.format(
            deal_type, 'Close matches: {}'.format(', '.join(close)) if close
            else 'Call list_instrument_types for the full list.'))
    fields = {}
    for section in sections:
        fields.update(schema['sections'][section])
    return {'deal_type': deal_type, 'sections': sections,
            'accepts_children': deal_type in schema['containers'], 'fields': fields,
            'required': [key for key, meta in fields.items() if meta.get('required')]}


@MCP.tool(annotations=READ_ONLY)
def describe_structure(name: str = None) -> dict:
    """The structures this desk quotes - a collar, a strangle, a seagull - and exactly what each
    one asks for. Call this before `solve_structure`: a structure is DECLARED (its legs, and the
    order they are priced and solved in), so what you supply are PARAMETERS, never deal fields.

    With no `name`: every structure with its `vernacular` - the sales names a desk actually says
    ("zero-cost collar, range forward, cylinder") - which is how a plain-language ask finds the
    right one, the `parameters` every form of it shares, its `variations` with the extra parameter
    each of those takes, and `direction`: 'required' where only `buy_currency`/`sell_currency` can
    say which variation is meant, 'optional' where naming the level says it, null where there is
    one form.

    With a `name` (an exact structure name from that
    list): `fields`, the parameters as declared - what each is, what it defaults to, whether it is
    required, and for a choice exactly which strings are valid; `recipe`, the steps in order - what
    is priced, and which leg is solved to what; and then either `legs`, what the structure books
    and which parameter each leg reads, or `variations`.

    VARIATIONS are the ways one structure is dealt - a forward extra FLOORS the pair for an
    exporter and CAPS it for an importer - and each carries the side of the pair its client buys,
    the extra parameters only it takes (a `floor` or a `cap`) and its own legs. Supply the level
    the client named, or `buy_currency` / `sell_currency`, or both: the runner selects the one
    variation consistent with what you state and refuses rather than guessing, and the quote says
    which it dealt. A field carrying `selector` is one of those two directions: `required` means
    the variations name the same level and nothing else can tell them apart, `optional` means the
    level says it on its own.

    STRIKES ARRIVE IN MARKET TERMS. A strike parameter is quoted the way the pair trades - a
    USDZAR strike is 15.50, never its reciprocal - and the runner puts it on the engine's axis
    itself. That inversion is the one thing not to do by hand here.

    `solve_structure` then runs the recipe and answers with the composed deal, each leg's premium,
    the strikes it solved and where the pending files landed; `book_quote` is the approval that
    turns that quote into a trade.
    """
    store = service().call('GET', '/schema').get('Structure')
    if store is None:
        raise ToolError('this DV_Service publishes no Structure store - it is older than the '
                        'structures vocabulary. Upgrade the service, or compose the legs yourself '
                        'with solve_deal.')
    types = store['types']
    if name is None:
        # a projection of the store and nothing else: the level that selects a variation is that
        # variation's OWN parameter, so the menu carries those beside the shared ones
        return {'structures': [{'name': key, 'vernacular': declared['vernacular'],
                                'parameters': sorted(declared['fields']),
                                'variations': {word: sorted(form['fields'])
                                               for word, form in
                                               (declared.get('variations') or {}).items()},
                                'direction': next(
                                    (meta['selector'] for meta in declared['fields'].values()
                                     if meta.get('selector')), None)}
                               for key, declared in sorted(types.items())],
                'count': len(types)}
    declared = types.get(name)
    if declared is None:
        # A sales name is how a model spells it, so the vernacular is searched beside the names.
        close = [key for key, entry in sorted(types.items())
                 if name.lower() in key.lower() or name.lower() in entry['vernacular'].lower()]
        raise ToolError('{!r} is not a structure. {}'.format(
            name, 'Close matches: {}'.format(', '.join(close)) if close
            else 'Call describe_structure() for the list with the sales names.'))
    return dict({'structure': name}, **declared)


@MCP.tool(annotations=READ_ONLY)
def describe_calculation_type(calc_type: str) -> dict:
    """Every field one calculation type takes (`BaseValuation`, `CreditMonteCarlo`,
    `HedgeMonteCarlo`) - what `calculation_overrides` in `execute_book` / `price_candidate` may
    override."""
    types = service().call('GET', '/schema')['Calculation']['types']
    if calc_type not in types:
        raise ToolError('{!r} is not a calculation type - one of: {}'.format(
            calc_type, ', '.join(sorted(types))))
    fields = types[calc_type]
    return {'calc_type': calc_type, 'fields': fields,
            'required': [key for key, meta in fields.items() if meta.get('required')]}


@MCP.tool(annotations=READ_ONLY)
def describe_factor_type(factor_type: str) -> dict:
    """Every field one price-factor type carries (`InterestRate`, `FxRate`, `VolatilityGrid`, ...)
    plus the stochastic processes that can drive it and the interpolations it accepts - for
    READING the book's market data; nothing here edits it."""
    schema = service().call('GET', '/schema')
    if factor_type not in schema['Factor']['types']:
        raise ToolError('{!r} is not a factor type - one of: {}'.format(
            factor_type, ', '.join(sorted(schema['Factor']['types']))))
    return {'factor_type': factor_type, 'fields': schema['Factor']['types'][factor_type],
            'processes': schema['Process_factor_map'].get(factor_type, []),
            'interpolations': schema['Interpolation_factor_map'].get(factor_type, [])}


@MCP.tool(annotations=READ_ONLY)
def describe_configuration() -> dict:
    """Every dial the book's bootstrap can be set with, per section - what `configure_book` writes.

    `Bootstrapper Configuration` holds one entry per price family, keyed by the price factor that
    family WRITES (`InterestRate`, `FXVol`, `LogVar2FJModelParameters`, ...) with the class name an
    older book spells it by beside it as an alias; the entry's dials are the boxes a fit is solved
    in, the seeds it starts from and the budgets it stops on, each with the default it stands at
    where the book states nothing. `Price Factor Interpolation` sets one method per routed curve
    type, and `interpolations` is that menu. Quote ladders and instrument tables are NOT dials -
    they belong to the quote block and move through `update_market_quotes`."""
    schema = service().call('GET', '/schema')
    return {'sections': schema['Configuration'],
            'interpolations': schema['Interpolation_factor_map']}


@MCP.tool(annotations=READ_ONLY)
def job_skeleton() -> dict:
    """A complete minimal job document that loads and prices - the reference for the ENVELOPE
    shape (where market data, deals and the calculation sit), which the field declarations alone
    cannot tell you."""
    return service().call('GET', '/schema/job')


# --------------------------------------------------------------------------- the live book


@MCP.tool(annotations=READ_ONLY)
def desk_status() -> dict:
    """START HERE. What this desk is set up with, in one call - what every other verb has to work
    with, and how old each piece of it is.

    The book's `base_date` (the day it is valued as of), `base_currency` and `calculation`; how
    many `deals` it holds and the `netting_sets` a client's trade can be booked under; `curves`,
    one per bootstrapped block, with the knots it solved on, the `interpolation` it is built under,
    the latest print its rows carry (`snapped`) and any benchmark `held_out`; `surfaces` with their
    own quote stamps; `models`, the spot models calibrated onto this book; `xva`, the last
    projection per set with its `as_of`.

    `terminal` is what this desk CAN do: `present` says whether `tick_market_from_bloomberg` has a
    terminal to ask - a sandboxed desk reads False and prices on the market the book last snapped
    - `ticking` the cadence the service refreshes itself on, and `provisioned` whether the desk's
    ticker vocabulary has been verified against a terminal at all.

    Names come back exactly as the book spells them, so a curve or surface named here is the
    string a deal's `Discount_Rate` or `FX_Volatility` must match. Numbers are elsewhere:
    `book_risk_summary` for the mark, `xva_view` for the projection, `describe_curve` for the rows.
    """
    return service().call('GET', '/book/status')


@MCP.tool(annotations=READ_ONLY)
def read_book() -> dict:
    """The live book, summarised one row per deal: `deal_path` (the positional identity every
    verb uses - references are NOT unique), type, reference, currency, whether it is ignored, and
    how many children it holds. Also the calculation the book runs and what market data it
    carries. `read_deal` fetches any one deal in full."""
    live = service().call('GET', '/book')
    calc = live['document']['Calc']
    calculation = calc['Calculation']
    market = calc.get('MergeMarketData', {})
    factors = sorted(market.get('ExplicitMarketData', {}).get('Price Factors', {}))
    deals = [{'deal_path': deal_path,
              'object': node['Instrument']['.Deal'].get('Object'),
              'reference': node['Instrument']['.Deal'].get('Reference'),
              'currency': node['Instrument']['.Deal'].get('Currency'),
              'ignored': node.get('Ignore') == 'True',
              'children': len(node.get('Children', []))}
             for deal_path, node in _walk(calc['Deals']['Deals']['Children'])]
    # Field names and capped factor names: the vocabulary the model books against, never a payload.
    return {'path': live['path'], 'etag': live['etag'],
            'reference': calc['Deals'].get('Reference'),
            'calculation': {
                'Object': calculation.get('Object'), 'Currency': calculation.get('Currency'),
                'Base_Date': calculation.get('Base_Date'),
                'other_fields': sorted(set(calculation) - {'Object', 'Currency', 'Base_Date'})},
            'market_data': {'file': market.get('MarketDataFile', ''),
                            'factor_count': len(factors),
                            'factors': factors[:80] + (
                                ['... and {} more'.format(len(factors) - 80)]
                                if len(factors) > 80 else [])},
            'deals': deals, 'count': len(deals)}


@MCP.tool(annotations=READ_ONLY)
def read_deal(deal_path: str) -> dict:
    """One deal of the live book, every field verbatim, plus the `deal_path` of each child."""
    live = service().call('GET', '/book')
    for path, node in _walk(live['document']['Calc']['Deals']['Deals']['Children']):
        if path == deal_path:
            return {'deal_path': path, 'deal': node['Instrument']['.Deal'],
                    'ignored': node.get('Ignore') == 'True',
                    'children': ['{}/{}'.format(path, i)
                                 for i in range(len(node.get('Children', [])))]}
    raise ToolError('no deal at path {!r} - read_book lists the paths'.format(deal_path))


@MCP.tool()
def book_deal(deal: dict, parent_reference: str | None = None) -> dict:
    """Book one deal into the live book. VALIDATED FIRST: the service splices it into a copy,
    validates the whole document, and only writes the file if nothing is said against this deal -
    its own authoring rules, or market data the book does not carry. A refusal comes back as
    `{written: false, refused: [messages]}` - read the messages, fix the deal, book again; it is
    an answer, not an error.

    `deal` is a flat field dict: `Object` (a name from `list_instrument_types`), `Reference`
    (your trade id), and every field `describe_instrument_type` marks `required` - the terms.
    A convention you leave out is read as its declared value; a term you leave out, or send as
    `null`, is refused by name - send the number, not a placeholder for one. `parent_reference`
    books it INSIDE a container deal (a structure, a netting set).

    To book AT PAR or at a target margin, solve before you book: a linear payoff's value is affine
    in its amount, so `price_candidate` twice at two trial amounts gives the exact amount that
    lands the value on the target - then book that.

    The engine's FX convention is REPORTING units per one unit of the currency (`FxRate.ZAR`
    carries USD per ZAR), and an FX option's `Strike_Price` lives on that same axis - so a desk's
    'USDZAR call', the option paid when ZAR weakens, is authored as a PUT on ZAR with
    `Underlying_Currency` ZAR. Quote an FX option through `solve_structure` instead and the strike
    stays in market terms: the runner crosses the axis and `book_quote` books what it composed.

    On success the answer carries the new `deal_path`, and every other client (the web UI, Excel)
    sees the deal on its next read. The answer is about THIS booking; anything else outstanding in
    the book arrives as counts under `book_issues`, with `validate_book` for the detail.
    """
    request = {'action': 'add', 'deal': deal}
    if parent_reference is not None:
        request['parent_reference'] = parent_reference
    return _booking(service().call('POST', '/book/deals', json=request))


@MCP.tool()
def amend_deal(deal_path: str, fields: dict, reference: str | None = None) -> dict:
    """Change one or more fields of a booked deal - "make the notional 3m", "move settlement a
    week". `fields` MERGES into the deal at `deal_path` (from `read_book`); every other field
    stands. The same validate-before-write contract as `book_deal`: a refusal comes back as
    `{written: false, refused: [messages]}` with the file untouched - read the messages, fix,
    amend again. Values wear their wire form: dates `{".Timestamp": "YYYY-MM-DD"}`, percentages
    `{".Percent": 2.5}`, plain numbers as numbers. `reference` names the deal you read at that
    path: another host's booking moves every position, and a path that no longer holds it refuses
    rather than amending whoever sits there now."""
    return _booking(service().call('POST', '/book/deals', json=dict(
        {'action': 'amend', 'deal_path': deal_path, 'fields': fields},
        **({} if reference is None else {'reference': reference}))))


@MCP.tool()
def delete_deal(deal_path: str, reference: str | None = None) -> dict:
    """Remove the deal at `deal_path` from the live book, its children with it. The write is
    atomic and every other client sees it on its next read. `reference` names the deal you read at
    that path, so a path another host's booking has moved refuses rather than deleting whoever
    sits there now - always pass it when other hosts share the book."""
    return service().call('POST', '/book/deals', json=dict(
        {'action': 'delete', 'deal_path': deal_path},
        **({} if reference is None else {'reference': reference})))


# --------------------------------------------------------------------------- pricing


@MCP.tool()
async def price_candidate(deal: dict | None = None, parent_reference: str | None = None,
                          calculation_overrides: dict | None = None,
                          wait_seconds: float = 120.0, ctx: Context = None) -> dict:
    """Price the book PLUS a candidate deal without booking anything - the what-if verb, and the
    solving half of a par booking: price a trial amount, price a second, solve the affine
    relation for the amount that lands the value on your target, then `book_deal` the answer.

    The candidate joins an in-memory copy only; the book file never moves. `calculation_overrides`
    merges into the calculation block: `{"Greeks": "First"}` for the AAD delta vector,
    `{"Greeks": "All"}` for the full second-order block - `Greeks_Second` is the cross-gamma
    matrix a trading read of an options book needs (spot gamma on the diagonal, vanna against the
    surface's own nodes beside it), one backward pass however many factors. Waits up to
    `wait_seconds` for the run; the answer carries `tables` and `stats`, and `deal_values`
    projects per-deal values from it. Content-addressed: the same what-if twice is one run.
    """
    request = {}
    if deal is not None:
        request['deal'] = deal
    if parent_reference is not None:
        request['parent_reference'] = parent_reference
    if calculation_overrides:
        request['calculation_overrides'] = calculation_overrides
    submitted = await asyncio.to_thread(service().call, 'POST', '/book/price', json=request)
    return await _await_result(submitted['result_id'], wait_seconds, ctx)


@MCP.tool()
def update_market_quotes(quotes: dict, bootstrap: bool = True) -> dict:
    """Tick the live book's market with quote blocks - `{name: block}` exactly as a quote source
    emits them (`derivus_bloomberg.to_market_prices_block` for an FX vol surface; any `Market
    Prices` family). An existing block may move ONLY its quoted values and timestamps - a changed
    pillar, expiry or convention is refused by name, because structure is a re-authoring, never a
    tick. With `bootstrap` (the default) the engine turns the quotes into the price factors the
    pricers read, and the book file gains everything in one atomic write; a bootstrap that
    complains refuses the WHOLE write and hands its messages back as `refused`. After a
    successful tick, `solve_deal` and `price_candidate` price against the fresh market."""
    return service().call('POST', '/book/market', json={
        'quotes': quotes, 'bootstrap': 'Yes' if bootstrap else 'No'})


@MCP.tool()
def configure_book(section: str, entry: str, fields: dict) -> dict:
    """Set the dials the live book bootstraps its market with - `fields` MERGED into one entry
    `describe_configuration` declares, everything else in it standing.

    `section` is `Bootstrapper Configuration` (`entry` the price factor a family writes, or the
    class name the book spells it by) or `Price Factor Interpolation` (`entry` a routed factor
    type, `fields` `{"method": ...}` for what every factor of it is built with or `{"id": "<curve
    name>", "method": ...}` for that one curve's own rule, a blank method clearing it - though a
    curve's rule is ordinarily set by `configure_curve`). A book states each hyperparameter ONCE
    here and every quote block of that family is read over it, so this is where a fit's box, seed
    or budget moves - never inside a quote.

    The change is checked by building what reads it, so a malformed dial is refused by name before
    a quote is read; the whole market is then re-bootstrapped in the same atomic write, and a
    bootstrap that complains refuses everything and hands its messages back as `refused` with the
    file untouched. On success the answer names the entry written, the dials it now carries, and
    the price factors the re-bootstrap rewrote."""
    return service().call('POST', '/book/configure',
                          json={'section': section, 'entry': entry, 'fields': fields})


@MCP.tool(annotations=READ_ONLY)
def describe_curve(curve: str = None) -> dict:
    """The book's interest-rate curves as DEFINITIONS - what each one is bootstrapped from, and
    what a desk could set up.

    `curves` is one entry per `InterestRatePrices` block the book carries: its currency, the curve
    it discounts on, the interpolation the solved factor carries with `interpolation_source` saying
    whether a rule names this curve (`curve`) or it takes what every curve takes (`default`), the
    conventions its benchmarks were authored under (the calendar, the settlement lag, both legs'
    frequency and day count,
    whether the swap rows compound overnight, and any near-end scheme), and the rows themselves -
    tenor, the security each quote came off, the number, and whether the curve uses it. `base_date`
    is the day every block on the book is authored on, and a block's `snapped` is the latest print
    its own rows carry, so a curve dated past its quotes says so.

    With no `curve` named the answer also carries `seeded`: the curve entries this workstation's
    seed declares, each with its conventions and the tenor/security rows it could be set up with.
    That is the menu - pick the rows a desk quotes, put them through `configure_curve`, and the
    book solves the curve from them."""
    return service().call('GET', '/book/curve',
                          params={'curve': curve} if curve is not None else None)


@MCP.tool()
def configure_curve(curve: str, currency: str, rows: list, discount_rate: str = None,
                    conventions: dict = None, interpolation: str = None) -> dict:
    """Set a curve up in the live book from its BENCHMARK INSTRUMENTS, and solve it.

    `rows` are the benchmarks, `[{"tenor": "3M", "security": "JIBA3M Index", "quote": 7.41}, ...]`,
    and the TENOR says what each instrument is: `ON` and the curve's declared front are deposits,
    `1Mx4M` is a FRA, `6M1M` a swap starting in six months, anything else a spot swap ending at
    that tenor. A row states its own `quote` in percent, or a `security` this workstation's
    terminal prices it off; `use: "No"` holds a benchmark out without deleting it. An unquoted row
    is refused by name - a benchmark with no number identifies no knot.

    `conventions` completes what the seed's entry for this curve does not state - `calendar`,
    `spot_days`, `compounding` (`OIS` for an overnight benchmark), `fixed_frequency`,
    `float_frequency`, `fixed_day_count`, `float_day_count`, `front_day_count`, `curve_day_count`,
    `near_interpolation` and `near_tenor`. `describe_curve` shows what the seed already declares;
    a curve nothing seeds needs the whole set. `discount_rate` names the curve the quotes discount
    on, blank being the self-discounting single curve.

    `interpolation` is THIS CURVE'S OWN SCHEME - `HermiteRT`, `Hermite`, `LinearRT` or `Linear` -
    set as a rule in the book's `Price Factor Interpolation` in the same write, before the solve,
    so the curve is solved under what it will be read under. Blank clears the rule and the curve
    takes what every curve takes; leaving it out leaves the rule as it stands, so re-stating a
    curve's rows never changes its scheme.

    The block is AUTHORED, not ticked: it is re-installed whole and the market re-bootstrapped in
    one atomic write, so a bootstrap that complains writes nothing and names what it complained
    about. The answer names the date the block was authored on, the block, the knots it solved on
    and the price factors that moved. A row priced off the terminal carries the print's own clock,
    and where that is later than the day the book stands at the book ROLLS ONTO IT - both its dates
    - and every other curve is re-authored there too. Afterwards `tick_market_from_bloomberg` keeps
    these rows valued off the terminal."""
    return service().call('POST', '/book/curve', json=dict(
        conventions or {}, curve=curve, currency=currency, rows=rows,
        **({} if discount_rate is None else {'discount_rate': discount_rate}),
        **({} if interpolation is None else {'interpolation': interpolation})))


@MCP.tool()
def set_base_date(base_date: str) -> dict:
    """Set the live book's CALCULATION DATE - the day everything it holds is valued as of.

    `base_date` is an ISO day, `"2026-09-19"`. The book carries the date twice - the day its
    curves' benchmarks roll off and the day the pricers run on - and this moves both together.
    Every interest-rate curve block is re-authored on the new day from its own rows and conventions
    (the same benchmarks on new dates, the quotes exactly as they stand, no terminal asked) and the
    whole market is re-bootstrapped in one atomic write, so a bootstrap that complains writes
    nothing and hands its messages back. The answer names the date, the blocks re-authored and the
    price factors the re-solve rewrote; a block too old to carry its conventions is named in
    `held_out` and left standing.

    `tick_market_from_bloomberg` and `configure_curve` already roll the date FORWARD onto the day
    their quotes were snapped. This is the verb that puts it anywhere - a back-valuation included -
    so call it to value the book as of a day of someone's choosing, never to catch up with a
    tick."""
    return service().call('POST', '/book/date', json={'base_date': base_date})


@MCP.tool()
def patch_market_values(patch: dict) -> dict:
    """Move market VALUES in the live book - `{factor: {field: value}}`, e.g.
    `{"FxRate.ZAR": {"Spot": 19.25}}`. Only value-bound fields move (spots, rate columns, vols);
    a structural key - anything that would change the plan - is refused by the engine's own rule.
    A `Market Prices` block is refused here too, naming `update_market_quotes` as the remedy: a
    quote moves the book through the path that bootstraps, never through a values patch. One atomic
    write; every client sees it on its next read."""
    return service().call('POST', '/book/market', json={'patch': patch})


@MCP.tool()
async def tick_market_from_bloomberg(pairs: list = None, expiries: list = None,
                                     pillars: list = None, wait_seconds: float = 360.0,
                                     ctx: Context = None) -> dict:
    """Tick the live book's market off THIS workstation's Bloomberg terminal - the whole "get me
    today's market" move, and on a fresh machine the call that PROVISIONS the desk.

    IT COVERS THE CURVES TOO: every `InterestRatePrices` block the book carries has its rows
    re-priced off the securities they name and re-solved, beside the FX surfaces. A row whose
    print the terminal refuses is held out by name in `held_out` rather than refusing the tick,
    and `configure_curve` is what puts it back. THE SNAP SETS THE DATE: where the prints came back
    later than the day the book stands at, the book rolls onto the latest of them and every curve
    is re-authored there, so a book ticked with today's quotes is dated today. It never rolls back;
    `set_base_date` is what values the book as of any other day.

    WHEN THE SERVICE RUNS WITH `--tick`, THE MARKET REFRESHES ITSELF on a cadence, through this
    same job - so call this verb only to FORCE a refresh between beats, or to provision on first
    use, which the cadence deliberately never does. On a ticking service a book that already
    carries today's surfaces does not need this call; on an unprovisioned one, nothing else will
    make the cadence start working.

    First use does the setup before it fetches anything: it creates `DV_HOME` (`~/.derivus`
    unless the variable says otherwise), copies the packaged seed - the ticker vocabulary the
    desk owns and edits - into it, then asks this terminal about every candidate the seed spells
    and keeps only the ones it answers for (what the security IS, whether it prices, when it last
    printed). That verification is the few minutes: it walks candidate by candidate reporting
    progress as it goes, which is both what keeps this call alive and what the user watches. Then
    it fetches the surfaces and installs them through the same quote-block tick
    `update_market_quotes` rides, with the bootstrap judging the whole write. Later calls skip
    the provisioning and just fetch.

    `pairs` narrows the currency pairs to fetch (`["USDZAR"]`); `expiries` narrows the surface's
    expiry column; `pillars` the delta pillars (`[0.25]`, the default - `[0.1, 0.25]` for wider
    wings the map verified). Omit all three for the desk's own scope - every pair the map
    carries, at the expiries it verified. The finished run's outcome rides `stats.Bloomberg`: what installed,
    what updated, whether the map had to be provisioned, or the refusal messages.

    A refusal is an answer and it NAMES what stopped it: no terminal answering on this machine,
    a candidate whose last print is stale (a retired benchmark keeps quoting a plausible price -
    that is the trap the verification exists for), a seed spelling the terminal does not know, or
    a bootstrap complaint that refuses the whole write. Each names the ticker or the file, so the
    next move is to fix that one thing and call again.
    """
    request = {key: value for key, value in (('pairs', pairs), ('expiries', expiries),
                                             ('pillars', pillars)) if value is not None}
    submitted = await asyncio.to_thread(
        service().call, 'POST', '/book/bloomberg', json=request)
    # a provisioning answer is what installed and what was refused, so this one is not summarised
    return await _await_result(submitted['result_id'], wait_seconds, ctx, summarise=False,
                               hint='; the provisioning carries on service-side either way')


@MCP.tool(annotations=READ_ONLY)
def describe_securities(block: str = None) -> dict:
    """The desk's TICKER VOCABULARY and the terminal evidence behind it - what each curve knot was
    actually priced off, in one read.

    `seed` is what this desk could quote, block by block: `rates` keyed by curve, each entry naming
    the `prefix` its swap strip is spelled from, the NAME fragment it is checked against (`expect`),
    which `weeks`/`months`/`years` it carries, its `overnight` fixing and its `conventions`;
    `swaption` the same per currency; `fx_vol` the `pairs`, `expiries`, delta `pillars` and the
    `leverage_prior` per pair; `fx_spot` the `pairs`. `map` is what a TERMINAL answered: every
    candidate it verified, each carrying the `name` it answered with, its `last_update` and when it
    was `verified`, and a `rejected` ledger saying why a candidate did not make it (`invalid`,
    `mismatch`, `unpriced`, `dead`). The seed is a desk's claim; the map is evidence, and only a
    terminal writes one.

    `used` IS WHERE A KNOT'S PRINT IS READ - the IPV join: every curve row the book carries with
    its tenor, security, quote and the print's own timestamp, and that security's evidence beside
    it, or the verdict that rejected it, or `unmapped` where the map has never heard of it. `block`
    narrows the read to one of `fx_vol`, `fx_spot`, `rates`, `swaption`."""
    return service().call('GET', '/book/securities',
                          params={'block': block} if block is not None else None)


@MCP.tool()
def configure_securities(block: str, key: str, entry: dict | list | None = None) -> dict:
    """Set one entry of the desk's own ticker vocabulary - the seed `describe_securities` reads
    back, in the same shape.

    `block` is `rates`, `swaption`, `fx_vol` or `fx_spot`; `key` is what that block is keyed by -
    the curve for `rates` (`"ZAR-ZARONIA"`), the currency for `swaption`, and for the two FX blocks
    the field itself (`pairs`, `expiries`, `pillars`, `leverage_prior`). `entry` is what stands
    under it and NULL REMOVES IT. A `rates` entry takes `prefix`, `expect` (the fragment the
    terminal's own NAME must carry), `currency`, `source`, `weeks`/`months`/`long_months` (true or
    the labels wanted), `years`, `overnight`, `fixings`, `fras`, `forwards` and `conventions`; an
    `fx_vol` entry is one of `pairs` (`["USDZAR", ...]`), `expiries` (`{"3M": 0.2493, ...}`),
    `pillars` (`[0.1, 0.25]`) and `leverage_prior` (`{"USDZAR": -0.4}`).

    The merged seed is validated by SPELLING every candidate it now names, so a malformed entry is
    refused before anything is written and the answer carries the tickers that block now spells.
    THIS CHANGES NO NUMBER: a seed names candidates, and `verify_securities` is what asks the
    terminal about them and writes the map a curve is then set up off."""
    return service().call('POST', '/book/securities',
                          json={'block': block, 'key': key, 'entry': entry})


@MCP.tool()
async def verify_securities(block: str = None, key: str = None, securities: list = None,
                            wait_seconds: float = 600.0, ctx: Context = None) -> dict:
    """Re-verify the desk's ticker vocabulary against THIS workstation's terminal, and write what
    it answered into the security map.

    Every entry the map already carries in scope is re-probed and any DRIFT named by its path -
    renamed, unpriced, gone stale, gone entirely - which is the check a dead benchmark needs: a
    retired series keeps quoting a plausible price and only its update date says so. Every
    candidate the seed spells that the map has never heard of is probed once and lands under
    `added` with its verdict. Nothing already verified is re-asked, so a seed that just gained a
    curve costs the terminal that curve's names alone.

    Scope it with `block`, a `key` inside it (a curve, a currency or a pair) or `securities` to
    re-ask about by name - the last re-verifies those entries and grows nothing, answering a ticker
    the map does not carry under `unknown`. With nothing named the WHOLE vocabulary is re-verified,
    which is minutes of terminal time. A workstation with no terminal refuses by name: a sandboxed
    desk reads its map through `describe_securities` and never verifies.

    Call it after `configure_securities` adds an entry, and when a curve row reads `unmapped` or
    carries a verdict in `used`. Past `wait_seconds` the id and the way forward travel in `hint`
    and the verification carries on service-side."""
    request = {name: value for name, value in (('block', block), ('key', key),
                                               ('securities', securities)) if value is not None}
    submitted = await asyncio.to_thread(
        service().call, 'POST', '/book/securities/verify', json=request)
    # a verification's answer is the drift and the verdicts, so this one is not summarised
    return await _await_result(submitted['result_id'], wait_seconds, ctx, summarise=False,
                               hint='; the verification carries on service-side either way')


@MCP.tool(annotations=READ_ONLY)
def book_dependencies(deal_path: str = None, deal: dict = None,
                      parent_reference: str = None) -> dict:
    """What a trade, a portfolio or the whole book NEEDS from the market - and which of this
    desk's seed entries would supply whatever is missing. Call this the moment a booking, a
    what-if or a solve is refused for market data.

    With a `deal` it walks that CANDIDATE, spliced the way `price_candidate` splices one, so the
    answer is what THIS trade needs rather than what the book happens to lack; `parent_reference`
    puts it under a container the way a booking would. With a `deal_path` it walks the deals under
    one node of the live book - a netting set is a portfolio - and with neither, the whole book.
    Nothing is priced and nothing is written.

    Every factor comes back with a `status`. A `missing` one carries `supply`: the vocabulary
    `block` and `key` this desk seeds it under, how many `securities` the seed spells for it and
    how many a terminal has `verified`, plus `conventions` for a curve. `supply` null with a `note`
    means this desk's vocabulary spells nothing for that factor - an equity or a commodity - and
    the market for it is authored by hand.

    A `deal` the booking verb would refuse is refused HERE in its words rather than walked, so a
    misspelt type is never answered "nothing is missing". `setup_market` acts on this answer."""
    if deal is None:
        return service().call('GET', '/book/dependencies',
                              params={'deal_path': deal_path} if deal_path is not None else None)
    return service().call('POST', '/book/dependencies', json=dict(
        {'deal': deal},
        **({} if parent_reference is None else {'parent_reference': parent_reference})))


@MCP.tool()
async def setup_market(pair: str = None, deal: dict = None, parent_reference: str = None,
                       deal_path: str = None, wait_seconds: float = 600.0,
                       ctx: Context = None) -> dict:
    """Build the market a trade needs, off THIS workstation's terminal - the cure for a booking
    refused for missing market data, in one call rather than five.

    Name ONE of: a `pair` (`"EURZAR"`) for that surface, both legs' spots and the curve each
    discounts on; a `deal` (with `parent_reference` where it would be booked under a container) for
    everything that candidate reaches; a `deal_path` for one node of the live book. None of them
    builds everything the book lacks. Nothing missing writes nothing and asks no terminal.

    It discovers only what is UNKNOWN - the names the security map has never heard of for the
    entries that would supply the want, and its rejected ledger asked again; the map is evidence
    and is written as it is gathered, BEFORE the book. Then every wanted print is checked for
    freshness first, one late or dead security refusing the whole trip by name with nothing
    written; then the surface, each new currency's spot crossed onto the engine's axis and each new
    curve's seeded benchmarks are fetched and installed in ONE atomic write the bootstrap judges.

    NOTHING LANDS THAT WOULD DEPEND ON A BLOCK THE BOOK WILL NOT CARRY. A curve nothing can
    supply - no seed entry, no conventions, a strip the terminal leaves too short to solve - is a
    `not_supplied` row carrying the reason, and a NEW CURRENCY IS INSTALLED AS A PAIR OR NOT AT
    ALL: its curve values its own benchmarks in the book's base, so spot and curve are held with
    each other, and a surface with whichever leg is not coming, each saying what it waits on. What
    is left still lands, so a deal wanting two currencies where one is dead gets the other
    complete.

    READ THREE FACTS OFF THE ANSWER. `written` says whether the book moved, and is true iff
    something landed. `installed` is exactly the factors that were written. `refused` is what
    refused the WRITE - a late print, the bootstrap's own words - and is only ever non-empty when
    `written` is false; `not_supplied` is `{factor, reason}` for every want that could not be
    filled, reported beside a write that still lands everything it could.

    WHERE THE PRINTS MOVE THE BOOK'S DATE the market it already carries is re-priced in the same
    trip and lands in the same write, so the book never holds two days of quotes under one date;
    `check` names every standing block that moved.

    What lands is ORDINARY blocks - `configure_curve` and `update_market_quotes` edit them from
    then on, and the web UI's Curves and Securities screens show them. `check` names what a trader
    should look at there: a curve set up on the conventions this build ships, benchmarks the screen
    held out, a surface the book's base currency is neither leg of. Bootstrapping dials and ticker
    codes are still configured once in the web UI - this verb changes neither.

    Refused by name where this workstation has no terminal, where the book declares no bootstrapper
    configuration, where the pair is one this desk's vocabulary does not spell (which
    `configure_securities` adds), and where the `deal` is one `book_deal` would itself refuse. Past
    `wait_seconds` the id travels in `hint` and the set-up carries on service-side."""
    request = {name: value for name, value in (('pair', pair), ('deal', deal),
                                               ('parent_reference', parent_reference),
                                               ('deal_path', deal_path)) if value is not None}
    submitted = await asyncio.to_thread(service().call, 'POST', '/book/setup', json=request)
    answer = await _await_result(submitted['result_id'], wait_seconds, ctx, summarise=False,
                                 hint='; the set-up carries on service-side either way')
    # the job's own outcome is the answer, not the result envelope it rides in; a run that has not
    # settled carries no outcome and is handed back as the pointer it is
    outcome = (answer.get('stats') or {}).get('Setup')
    return answer if outcome is None else dict(
        outcome, result_id=submitted['result_id'], status=answer.get('status'))


@MCP.tool()
async def calibrate_spot_model(pair: str, family: str = '', wait_seconds: float = 1800.0,
                               ctx: Context = None) -> dict:
    """Fit one FX pair's spot-model parameters to the vol surface the book already carries, and
    land them in the book - the model a TARF or an accumulator prices on when its `SpotModel` is
    that family. `family` blank takes the one the book's own runner pins, which is what a desk
    wants: calibrating one model and pricing under another is the one mistake this verb prevents.

    WHEN TO CALL IT: after a market re-tick and BEFORE quoting TARFs or accumulators on that pair.
    A tick moves the surface and NEVER refits these parameters - they are a calibration, not a
    quote, and this verb is the only thing that moves them, by construction rather than by
    convention. Skip it and the TARF prices on yesterday's dynamics against today's spot. Call it
    once per pair per re-tick, not per quote.

    THE SPEC IS THE FAMILY'S AND IT IS NOT A PARAMETER OF THIS CALL: the parameters are fitted
    against vega-weighted implied vols read off the pair's BUILT surface, on the ladder that family
    declares - ATM pillars, which is what identifies the variance level and its term structure,
    plus the delta wings, which is what identifies the skew and the wings' width. Weight is the
    Black vega off the same surface, normalised. NOTHING PAST 1Y: TARFs and accumulators are
    sub-year products, and a parameter fitted to the 2Y smile is borrowed against products nobody
    quotes. An expiry the surface does not carry is moved to the NEAREST QUOTED one AT OR UNDER a
    year and the installed block says so - never interpolated silently, and never snapped onto a
    pillar past the cap.

    TEN RUNGS ARE NOT TEN QUOTES, and a thin surface REFUSES here rather than fitting. Every rung
    the surface does not carry lands on a contract another rung already named, so a two-pillar
    surface collapses the ladder onto four distinct contracts and four do not identify five
    parameters. Below the family's own minimum the verb refuses by name, saying which pillars the
    surface carries - the remedy is to quote the pair at more expiries.

    `pair` is the surface's own name (`"USD.ZAR"`). The answer is the run's outcome under
    `stats.SpotModel`: the family and parameters fitted, the block installed, the factor written,
    the quotes' provenance and the fit's wall time. THIS IS THE EXPENSIVE ONE - measured in the
    tens of minutes, at the same cost class as an XVA recalculation, so quotes and valuations keep
    jumping the queue while it runs. `poll_result` follows it if the wait runs out; the fit carries
    on service-side either way.

    There is nothing further to read: the written `<family>ModelParameters.<currency>` factor in
    the book's `Price Factors` IS the result, so `read_book` serves it like any other market data.
    A pair the book carries no built surface for refuses BY NAME - tick the market first.
    """
    submitted = await asyncio.to_thread(service().call, 'POST', '/book/model',
                                        json={'pair': pair, 'family': family})
    outcome = await _await_result(submitted['result_id'], wait_seconds, ctx)
    return dict(outcome, factor=submitted['factor'])


@MCP.tool()
async def solve_deal(deal: dict, field: str, target: float | dict = 0.0,
                     bounds: list | None = None, calculation_overrides: dict | None = None,
                     wait_seconds: float = 300.0, ctx: Context = None) -> dict:
    """Solve ONE field of a candidate deal so the deal's own value lands on `target`, and get the
    deal back READY TO BOOK - the structuring tool. A par forward: solve the amount to target 0.
    A sales margin: target the margin. A zero-cost collar: fix one strike, solve the other to
    target 0. A strike to a premium: solve `Strike_Price` with `bounds` around spot.

    `target` is a number in the book's reporting currency, or MONEY - `{'amount': 50000.0,
    'currency': 'ZAR'}` - which is how a sales margin is actually agreed: an amount in a currency,
    crossed to the reporting one at the book's own spots, refusing by name a currency the book
    carries no rate for. This deal is the one the desk BOOKS, so a margin target marks it at plus
    the margin, and `solved_deal` comes back carrying `Sales_Margin` and `Sales_Margin_Currency` -
    the record of what was charged, which prices nothing and books with the trade.

    Prefer this over hand-iterating `price_candidate`: the root find runs server-side against the
    book's market data (brentq inside `bounds`, else a secant from the field's current value -
    exact in two pricings for an amount) and nothing large enters the conversation. The answer
    carries `solved` (the field's value, the pricing count, the residual) and `solved_deal` - the
    deal with the field set, which `book_deal` books as-is.

    The engine's FX convention is REPORTING units per one unit of the currency (`FxRate.ZAR`
    carries USD per ZAR), and an FX option's `Strike_Price` lives on that same axis - so a desk's
    'USDZAR call', the option paid when ZAR weakens, is a PUT on ZAR with `Underlying_Currency`
    ZAR, and the strike solved here is quoted on that axis too.
    """
    request = {'deal': deal, 'field': field, 'target': target}
    if bounds is not None:
        request['bounds'] = bounds
    if calculation_overrides:
        request['calculation_overrides'] = calculation_overrides
    submitted = await asyncio.to_thread(service().call, 'POST', '/book/solve', json=request)
    outcome = await _await_result(submitted['result_id'], wait_seconds, ctx)
    solved = outcome.get('stats', {}).pop('Solved', None) if 'stats' in outcome else None
    if solved is not None:
        outcome['solved'] = solved
        outcome['solved_deal'] = dict(deal, **{field: solved['value']})
        if solved.get('margin'):
            outcome['solved_deal'].update(
                Sales_Margin=solved['margin']['amount'],
                Sales_Margin_Currency=solved['margin']['currency'])
    return outcome


@MCP.tool()
async def solve_structure(structure: str, params: dict, netting_set: str | None = None,
                          margin: dict | None = None, wait_seconds: float = 120.0,
                          ctx: Context = None) -> dict:
    """Quote a whole structure against the live book - the collar, strangle and seagull verb, and
    the one to reach for instead of composing legs by hand: the structure declares its own legs,
    their conventions and the order they solve in, so the finance does not depend on this
    conversation getting it right.

    `structure` is a name from `describe_structure`, and `params` fills the parameters IT declares
    - nothing else. STRIKES ARE MARKET TERMS (a USDZAR strike is 15.50); the runner puts them on
    the engine's axis. The recipe runs server-side against the book's market data: each leg priced
    alone, each solved leg found by the same root find `solve_deal` rides.

    `netting_set` is WHO the quote is for, and it is worth naming on any quote for a real client. A
    CLIENT IS A NETTING SET: the counterparty and the CSA are declared on the
    `NettingCollateralSet` node, `recalc_xva` projects a CVA per set, and booking the trade UNDER
    that node is the only thing that puts it inside the subtree the projection prices - a trade
    booked at the root has no counterparty and no CVA. Pass the set's Reference (see `xva_view` or
    `read_book` for the ones the book holds); an unknown one refuses HERE, naming the sets the book
    holds, rather than at the approval when the client already has the sheet. Left out, the
    approval books at the root exactly as before.

    `margin` is the SALES MARGIN, and it is how a desk actually quotes: `{'amount': 50000.0,
    'currency': 'ZAR'}` - an amount in the currency it was agreed in, which need not be a currency
    of the pair. It crosses to the book's pricing currency at the book's own spots and is charged
    where the two-way's own charge is: on the coordinate the recipe solves, so the cap of a collar
    comes in and the strike of a strip moves against the client by exactly that much, or on the
    PREMIUM where the recipe solves nothing and the client simply pays more. `charged_on` says
    which. A currency the book carries no rate for refuses by name rather than being crossed at a
    rate somebody guessed.

    The answer IS the quote - `quote_id`, the params as read, one row per leg (role, deal type,
    buy/sell, the strike in MARKET terms, the premium, what was solved) and the `net`: zero for a
    zero-cost structure, MINUS the margin where one was charged, since every premium here is in
    the client's sign and the client's paper is worth minus what they paid for it. The `margin`
    block says what was charged and what it converted to; the booked mirror is the bank's side of
    it, marking at plus the margin. `deal` rides with it, the composed structured deal ready to
    book, carrying `Sales_Margin` and `Sales_Margin_Currency` as the record of what was agreed.

    A quote prices on the LIVE spot when this workstation's terminal is up, and on the book's last
    ticked one - with the reason named - when it is not; the outcome's `spot` block says which was
    used (`value_market`, the pair as quoted, with `source` and `note`).

    Where the book's vol quotes carry a two-way, every leg is still priced at the MID and what the
    market charges for the spread is levied where the margin is - on the coordinate the recipe
    solves, or on the PREMIUM where it solves nothing, `charged_on` naming which. Each leg's vega is
    read per quoted pillar and charged that pillar's own half, so `spread_charge` is what that leg
    cost the client, `spread` is the pillar rows behind it and `spread_source` says whether the
    vega came off the surface or off a lognormal reading of a leg priced under a fitted model.
    `net_mid` is the legs at mid - what the trade will be worth on the book once booked - and the
    desk's `edge` is the charge, never negative. A pillar's `half` is always the MARKET's own
    half-spread; where a `Quote Policy` tightens, the scale is said once under `risk.scale` and the
    money carries it. With no two-way in the book nothing is charged, `net_mid` equals `net`, and
    `spread_note` says so; a leg no vega reaches carries a null `spread_charge` with a note, never
    a zero, and so does `risk.charge_full` where no leg could be read.

The `risk` block says what the trade does to the BOOK and what that was worth to the client. Where
the book declares a `Quote Policy`, the candidate is measured against the book with and without it
in the vol quotes a desk trades - `buckets` carries `dV/d(ATM)`, `dV/d(RR)` and `dV/d(BF)` per
pillar `before` and `after`, with that pillar's own `half_spread` - and a trade that NETS THE BOOK
DOWN is quoted tighter by `participation` of the hedge cost it saves: `charge_full` is the full
two-way, `charge_effective` what was actually charged, and `scale` the ratio the whole quote was
re-solved at. A risk-adding trade is quoted at the full spread and never wider - the market's own
spread is the ceiling - and no quote is pushed through the mid. `scale` is null where the feature
never ran, and `note` says why (no policy declared, no two-way, or a bucket past its limit, named).
The BOOK IS NOT TOUCHED. What is written is the pending trade:
    `DV_HOME/tmp/<quote_id>.json` holds the quote and its deal, with `<quote_id>.xlsx` - the sheet
    that goes to the client - beside it when the sheet writer is installed; `files` names both, and
    `files.sheet_note` names the install when there is no sheet. A missing sheet writer never
    refuses a quote.

    Then `book_quote(quote_id)` is the approval that makes it a trade. Two identical asks are two
    quotes, each with its own id and its own files - a quote is an act, not a lookup. A quote is
    also FIRM ONLY FOR A WINDOW where the book declares one (`Quote Policy.firm_seconds`, ten
    minutes by default): approve it while it is fresh, or re-quote.
    """
    submitted = await asyncio.to_thread(
        service().call, 'POST', '/book/structure',
        json={'structure': structure, 'params': params,
              'netting_set': netting_set, 'margin': margin})
    outcome = await _await_result(submitted['result_id'], wait_seconds, ctx)
    quote = outcome.get('stats', {}).get('Quote')
    if quote is not None:
        return dict(quote, resources={
            'quote': 'derivus://quote/{}'.format(quote['quote_id']),
            'sheet': 'derivus://quote/{}/sheet'.format(quote['quote_id'])})
    # No quote: the run summary is the answer, and its poll pointer follows up on the stats rather
    # than on a table.
    if 'hint' in outcome:
        outcome['hint'] = ('still {} - call poll_result({!r}) to check again; the quote lands '
                           'under stats.Quote, and the pending trade is filed the moment it '
                           'does'.format(outcome['status'], submitted['result_id']))
    return outcome


@MCP.tool()
def book_quote(quote_id: str) -> dict:
    """Approve a quote and book it - the second half of `solve_structure`, and the only thing that
    turns a quote into a trade.

    `quote_id` is the one the quote carries. The service reads the pending trade back from
    `DV_HOME/tmp/<quote_id>.json` and books the MIRROR of its deal - the quote's legs carry the
    CLIENT's side, and the book holds the bank's position, so every booked leg lands on the
    opposite side from the one quoted - through the SAME validate-before-write seam
    `book_deal` uses: validated against the book as it is NOW - the market may have moved since
    the quote was given - written atomically, and refused as `{written: false, refused:
    [messages]}` with the file untouched. A refusal is an answer; an id with no file behind it is
    a tool error naming the directory it looked in.

    WHERE it books is the quote's own `netting_set`: the mirror lands UNDER that
    `NettingCollateralSet` node, which is what makes `recalc_xva` see the trade - the client's CVA
    is projected over that subtree and a trade booked at the root is outside it. A quote that named
    no set books at the root, as it always did.

    A QUOTE IS FIRM FOR A WINDOW. Where the book declares a `Quote Policy`, its `firm_seconds` is
    how long an approval may stand on the price that was given; past it this is a tool error naming
    the age, the window and the remedy, and NOTHING is written - re-quote with `solve_structure`
    and approve that. A book declaring no policy holds a quote approvable indefinitely.

    The pending file is NOT deleted. What was quoted, at what market, when, under what id, is the
    audit trail of why the book carries what it carries - and the sheet the client saw stands
    beside it.
    """
    # An approval books through `deal_edit`, so its answer is a booking's and takes a booking's trim.
    return _booking(service().call('POST', '/book/quote', json={'quote_id': quote_id}))


@MCP.tool()
async def execute_book(calculation_overrides: dict | None = None,
                       wait_seconds: float = 120.0, ctx: Context = None) -> dict:
    """Run the book's own calculation as it stands - `price_candidate` with no candidate. Waits
    up to `wait_seconds`; on timeout the answer's `hint` says how to pick the run up later."""
    return await price_candidate(calculation_overrides=calculation_overrides,
                                 wait_seconds=wait_seconds, ctx=ctx)


@MCP.tool(annotations=READ_ONLY)
def validate_book() -> dict:
    """What would stop the live book running, without running it: authoring messages per deal
    reference, and the price factors named by deals that the market data has no block for."""
    live = service().call('GET', '/book')
    return service().call('POST', '/validate', json=live['document'])


@MCP.tool(annotations=READ_ONLY)
def describe_book() -> dict:
    """What the engine makes of the live book without pricing it: deals counted by type, the
    factor universe (resolved and missing), the calculation as loaded, and a crude cost read."""
    live = service().call('GET', '/book')
    return service().call('POST', '/describe', json=live['document'])


@MCP.tool(annotations=READ_ONLY)
def book_risk_summary() -> dict:
    """The desk's CONSOLIDATED risk: what the whole book is worth and what it is exposed to, in
    one read - the question after every booking and every tick.

    COUNTERPARTIES DO NOT MATTER HERE. This is one base valuation with first-order Greeks over the
    book as it stands, aggregated across everything the desk holds, so there is nothing to slice by
    counterparty and no netting set enters it. The per-counterparty number is XVA, which is a
    different calculation for a different reason - see `xva_view`.

    Answers the mark (`mtm` in the book's report currency), `as_of`, the `etag` the service cached
    it under, how many deals it covers, and the LARGEST gradient rows by absolute size - `factor`
    (the price factor), `tenor` (its coordinates, absent for a spot) and `value` (the derivative in
    report currency per unit of that factor). Never the whole per-deal table: `deal_values` on an
    `execute_book` run serves that, and the web blotter renders the lot.

    Cheap and cached on the book's own content, so asking again after nothing moved costs nothing;
    a booking or a market tick moves the etag and the numbers follow.
    """
    risk = service().call('GET', '/book/risk')
    greeks = sorted(risk['greeks'], key=lambda row: -abs(row['value']))
    return {'as_of': risk['as_of'], 'etag': risk['etag'], 'currency': risk['currency'],
            'mtm': risk['mtm'], 'deals': len(risk['per_deal']),
            'greeks': greeks[:MAX_GREEK_ROWS], 'greek_rows': len(greeks),
            'hint': 'the per-deal values are behind execute_book + deal_values; XVA is xva_view'}


@MCP.tool(annotations=READ_ONLY)
def xva_view() -> dict:
    """The XVA projection: one row per netting set, as the LAST recalculation left it.

    A CACHED PROJECTION, not a live number, and deliberately so - a credit Monte Carlo takes
    minutes, so it never rides a market tick. Every row carries its own `as_of`, and rows are as
    old as their last recalc: STALENESS IS DATA here, not a failure. `recalc_xva` is what moves
    them.

    Netting sets are the instruments. Each row says what the book holds now (`reference`,
    `deal_path`, `counterparty`, `collateralized`) over what the last run said (`cva`, `as_of`,
    `status` and the replay tuple - `result_id`, `plan_hash`, `values_hash`, `seed`). `status` is
    `done`, `failed` (with the engine's own wording in `error` - a counterparty with no survival
    curve lands here) or `never run`. A set the book no longer holds is still reported, with a
    `note` saying so and no `deal_path`; a recalc still in flight rides under `recalc`.
    """
    return service().call('GET', '/book/xva')


@MCP.tool()
async def recalc_xva(netting_sets: list | None = None, wait_seconds: float = 600.0,
                     ctx: Context = None) -> dict:
    """Recalculate the XVA projection - every netting set, or only the ones named.

    THIS IS THE EXPENSIVE ONE. Each set is a credit Monte Carlo over that set's own subtree of the
    book, minutes of device time apiece, which is exactly why the blotter reads a cached projection
    instead of running one per tick. Ask for it when the market or the book has genuinely moved, or
    when a desk wants today's number - and prefer naming the sets you care about
    (`netting_sets=["NS_ACME"]`) over recalculating everything.

    One job is queued PER SET, at the heavy cost class, so quotes and valuations keep jumping the
    queue and the projection fills in row by row - a partial recalc writes only the rows it names
    and leaves every other row's `as_of` exactly where it was. A reference that names no netting
    set refuses BY NAME and queues nothing at all, so a typo never half-runs a book.

    Waits up to `wait_seconds` for the LAST set queued and answers pointers - the queued
    `{reference, result_id}` pairs and where that last run got to. Read the numbers with `xva_view`
    once it is done; `poll_result` follows any one set.
    """
    submitted = await asyncio.to_thread(service().call, 'POST', '/book/xva',
                                        json={'netting_sets': netting_sets})
    queued = submitted['queued']
    if not queued:
        return dict(submitted, hint='the book carries no netting sets - there is no XVA to run')
    # Freshly queued sets drain in order through one worker, so the last settling usually means
    # every one has - but a cached set answers 'done' at once, so xva_view's per-row status rules.
    last = await _await_result(queued[-1]['result_id'], wait_seconds, ctx)
    return {'queued': queued, 'last': last,
            'hint': 'xva_view reads the rows and each row carries its own status; poll_result '
                    'follows any one set by its result_id'}


@MCP.tool(annotations=READ_ONLY)
def book_reconcile() -> dict:
    """Where the book file and the book of record disagree - the record read AT ITS HEAD.

    A READING, never a refusal. The file is the desk's working copy and the record is what is true,
    so this names three things: a trade the record holds that the file has lost
    (`in_record_not_in_file`), a deal the file holds that nobody booked (`in_file_not_in_record`),
    and an instrument the two count a different number of clips of (`quantity_mismatch`) - the file
    carrying terms and never a signed quantity, which lives only in the record.

    The record is folded AT ITS HEAD, so a booking whose file write never landed is exactly what
    shows up here. `events_behind` counts every event since the file was written and
    `positions_behind` the fills and amendments among them; an empty answer with both at 0 is a
    desk whose copy is exactly the record. 404 on a box that records nothing.
    """
    return service().call('GET', '/book/reconcile')


@MCP.tool(annotations=READ_ONLY)
def book_diary(due_before: str | None = None) -> dict:
    """Everything the book OWES or is owed: every payment, fixing and expiry its deals carry.

    The compile's own schedule, so the diary and the pricer cannot disagree about a payment. A row
    names its deal's instrument, the leg it sits on, its due date, its currency, its notional, and
    its amount where the compile determines one - one PAYMENT per leg and pay day, spelled by the
    pricer's own line, and a floating coupon reads `amount: null` and `determined: false` until its
    resets fix, which is not the same as owing nothing. `state` is `due`, `observed`, `settled` or
    `expired`, `reason` says why the record cannot answer a row, and `key` is what a settlement
    fact names the row by.

    `due_before='YYYY-MM-DD'` trims it to what falls due by a day. Cached on the book's own content
    and computed on the compute queue, so asking again after nothing moved costs nothing.
    """
    return service().call('GET', '/book/diary',
                          params={} if due_before is None else {'due_before': due_before})


@MCP.tool(annotations=READ_ONLY)
def close_check(date: str) -> dict:
    """Whether a close on `date` is legal, and what it is waiting on - the catch-up rule as a read.

    A close is legal when every diary entry due on or before that day has its fact. Outstanding is
    exactly three things: a fixing no declared source has printed, a payment no settlement was
    filed against, and an expiry whose terms leave a choice nobody has elected. `legal: false`
    comes with the rows - each one names the deal, the leg and the day - so what has to happen
    before the close is a list rather than a verdict.

    THIS DECLARES NOTHING. Declaring the close is a separate act; this says whether the record is
    ready for one. 404 on a box that records nothing.
    """
    return service().call('GET', '/book/close/check', params={'date': date})


@MCP.tool(annotations=READ_ONLY)
def poll_result(result_id: str) -> dict:
    """Where a run got to: `queued`/`running`, or `done` with the replay tuple, the run's stats
    and each table's SHAPE as one line (fetch cells with `fetch_table`), or `error` with the
    message. Never the cells themselves."""
    return _summary(_raw_result(result_id), result_id)


@MCP.tool(annotations=READ_ONLY)
def fetch_table(result_id: str, table: str, offset: int = 0, limit: int = 50) -> dict:
    """One table of a finished run, paged and CAPPED: at most 200 rows per call, and a table wider
    than 60 columns - a scenario cube - is refused by name, because it belongs in the web UI and
    not in a context window. Table names come from the result summary; a grouped table is a path
    (`cashflows/USD`)."""
    shape = _raw_result(result_id).get('tables', {}).get(table)
    if shape is None:
        raise ToolError('result {} has no table {!r} - poll_result lists them'.format(
            result_id, table))
    if len(shape['columns']) > MAX_TABLE_COLUMNS:
        raise ToolError('{} is {} columns wide - a simulation cube, not a table to read as text. '
                        'View it in the web UI, or fetch a summary table instead.'.format(
                            table, len(shape['columns'])))
    return service().call('GET', '/results/{}/{}?offset={}&limit={}'.format(
        result_id, table, offset, min(int(limit), MAX_PAGE_ROWS)))


@MCP.tool(annotations=READ_ONLY)
def deal_values(result_id: str) -> dict:
    """`{reference: value}` off a finished base valuation's `mtm` table - the question after every
    booking and every what-if. The SHAPE is checked before anything is fetched: a Monte Carlo's
    mtm is a scenario cube and is refused unfetched. Row 0's `Total` is the whole book."""
    raw = _raw_result(result_id)
    shape = raw.get('tables', {}).get('mtm')
    if shape is None or 'Reference' not in shape['columns'] or 'Value' not in shape['columns']:
        raise ToolError('this result carries no per-deal mtm frame - it is not a base valuation. '
                        'Its tables: {}'.format(', '.join(raw.get('tables', {})) or 'none'))
    if shape['rows'] > 500:
        raise ToolError('the mtm frame holds {} deals - too many to hold as text; page it with '
                        'fetch_table instead'.format(shape['rows']))
    page = service().call('GET', '/results/{}/mtm?offset=0&limit={}'.format(
        result_id, shape['rows']))
    reference, value = page['columns'].index('Reference'), page['columns'].index('Value')
    return {row[reference] if row[reference] else 'Total': row[value] for row in page['data']}


# --------------------------------------------------------------------------- prompts


@MCP.prompt()
def quote_a_structure(structure: str, pair: str, notional: str, notional_currency: str,
                      expiry: str, client: str = None) -> str:
    """Quote one declared structure for a client - its own parameters, its legs at market terms,
    booked only on the user's word."""
    return (
        '1. describe_structure({0!r}) - the parameters it declares and the legs it books.\n'
        '2. solve_structure({0!r}, params) filling ONLY those parameters: pair {1}, notional {2} '
        '{3}, expiry {4}, strikes in MARKET terms{5}.\n'
        '3. Report every leg - role, buy/sell, strike, premium - then the net, the net_mid the '
        'book will mark it at, and the edge between them.\n'
        '4. book_quote(quote_id) ONLY on the user\'s word. A quote is firm for a window; past '
        'it, re-quote.'.format(structure, pair, notional, notional_currency, expiry,
                               ', netting_set {!r}'.format(client) if client else ''))


@MCP.prompt()
def import_a_legacy_book(counterparty: str) -> str:
    """Load a counterparty's existing trades into the book as one netting set, and mark them."""
    return (
        '1. Take the deals the user supplies in derivus form and wrap them as ONE node: '
        '{{"Object": "NettingCollateralSet", "Reference": "{0}", "Netted": "True", '
        '"Collateralized": "False", "Children": [{{"Instrument": {{".Deal": deal}}}}, ...]}}.\n'
        '2. book_deal(that node) - one call. One bad deal refuses the WHOLE batch and names it: '
        'fix that child and post the batch again.\n'
        '3. execute_book, then book_risk_summary for the mark and the gradient.\n'
        '4. recalc_xva(["{0}"]) only if the user asks for it - it is minutes.'.format(
            counterparty))


@MCP.prompt()
def morning_desk_check() -> str:
    """Open the desk for the day: what it is set up with, today's market, the mark, and what is
    stale."""
    return (
        '1. desk_status - the book\'s date, its curves and surfaces with when each was snapped, '
        'and whether a terminal is present.\n'
        '2. Where terminal.present: tick_market_from_bloomberg for today\'s market. Where it is '
        'not, say what date the market was snapped and price on it.\n'
        '3. book_risk_summary - the mark and the biggest gradient rows.\n'
        '4. xva_view - name every set whose as_of is older than today; recalc_xva is minutes, so '
        'ask before running it.\n'
        '5. Where desk_status carries a spine block: book_reconcile - it folds the record and '
        'names any trade the file and the record disagree about, which is the one check the '
        'status read deliberately does not pay for.\n'
        '6. set_base_date ONLY on the user\'s word - a tick already rolls the book onto the day '
        'its prints came from.')


# --------------------------------------------------------------------------- resources


#: The stores `/schema` publishes, which is the menu `derivus://schema/{store}` serves and the
#: refusal names.
SCHEMA_STORES = ('Instrument', 'Structure', 'Calculation', 'Factor', 'Process', 'MarketPrices',
                 'Configuration')


@MCP.resource('derivus://book', mime_type='application/json')
def book_document() -> dict:
    """The live book as the job document itself - every deal and all its market data, verbatim.
    `read_book` is the summary a model should normally hold; this is the file."""
    return _resource('/book')['document']


@MCP.resource('derivus://schema/{store}', mime_type='application/json')
def schema_store(store: str) -> dict:
    """One store of the engine's declarations, whole - what every describe_* tool reads one entry
    out of, for a host that wants the vocabulary in front of it."""
    if store not in SCHEMA_STORES:
        raise ResourceError('{!r} is not a schema store - one of: {}'.format(
            store, ', '.join(SCHEMA_STORES)))
    return _resource('/schema')[store]


@MCP.resource('derivus://quote/{quote_id}', mime_type='application/json')
def quote_pending(quote_id: str) -> dict:
    """The pending trade a quote filed: what was quoted, the deal that books it, and when. It
    stands after the approval too, being the audit trail of why the book carries what it does."""
    return _resource('/book/quote/{}'.format(quote_id))


@MCP.resource('derivus://quote/{quote_id}/sheet', mime_type=SHEET_MIME)
def quote_sheet(quote_id: str) -> bytes:
    """The quote sheet itself, as the spreadsheet a client is sent - so a host hands over the file
    rather than a path on the service's own disk. A quote given where the sheet writer was not
    installed refuses by name, and is still approvable."""
    return _resource('/book/quote/{}/sheet'.format(quote_id), binary=True)


def main():
    """Serve the tools over stdio, for an MCP host that launches this as a subprocess."""
    configure()
    MCP.run(transport='stdio')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
