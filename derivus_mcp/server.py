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
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations

MCP = MCPServer('derivus', instructions=__doc__)
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

    def call(self, method, path, **kwargs):
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
        return response.json()


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
    (`description`), what it defaults to (`value`), whether you must supply it (`required` - also
    summarised in the top-level `required` list), and for a choice exactly which strings are valid
    (`values` - a field with `values` accepts nothing else). Dates are `{".Timestamp":
    "YYYY-MM-DD"}`, percentages `{".Percent": 2.5}` (already in percent), rate curves are named by
    a string that must match a `Price Factors` block. `accepts_children` says whether this type
    can hold other deals.

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
    right one, and the parameters each takes. With a `name` (an exact structure name from that
    list): `fields`, the parameters as declared - what each is, what it defaults to, whether it is
    required, and for a choice exactly which strings are valid; `legs`, what the structure books
    and which parameter each leg reads; and `recipe`, the steps in order - what is priced, and
    which leg is solved to what.

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
        return {'structures': [{'name': key, 'vernacular': declared['vernacular'],
                                'parameters': sorted(declared['fields'])}
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
    (your trade id), and the fields `describe_instrument_type` declares. `parent_reference` books
    it INSIDE a container deal (a structure, a netting set).

    To book AT PAR or at a target margin, solve before you book: a linear payoff's value is affine
    in its amount, so `price_candidate` twice at two trial amounts gives the exact amount that
    lands the value on the target - then book that.

    The engine's FX convention is REPORTING units per one unit of the currency (`FxRate.ZAR`
    carries USD per ZAR), and an FX option's `Strike_Price` lives on that same axis - so a desk's
    'USDZAR call', the option paid when ZAR weakens, is authored as a PUT on ZAR with
    `Underlying_Currency` ZAR.

    On success the answer carries the new `deal_path`, and every other client (the web UI, Excel)
    sees the deal on its next read. The answer is about THIS booking; anything else outstanding in
    the book arrives as counts under `book_issues`, with `validate_book` for the detail.
    """
    request = {'action': 'add', 'deal': deal}
    if parent_reference is not None:
        request['parent_reference'] = parent_reference
    return _booking(service().call('POST', '/book/deals', json=request))


@MCP.tool()
def amend_deal(deal_path: str, fields: dict) -> dict:
    """Change one or more fields of a booked deal - "make the notional 3m", "move settlement a
    week". `fields` MERGES into the deal at `deal_path` (from `read_book`); every other field
    stands. The same validate-before-write contract as `book_deal`: a refusal comes back as
    `{written: false, refused: [messages]}` with the file untouched - read the messages, fix,
    amend again. Values wear their wire form: dates `{".Timestamp": "YYYY-MM-DD"}`, percentages
    `{".Percent": 2.5}`, plain numbers as numbers."""
    return _booking(service().call('POST', '/book/deals', json={
        'action': 'amend', 'deal_path': deal_path, 'fields': fields}))


@MCP.tool()
def delete_deal(deal_path: str) -> dict:
    """Remove the deal at `deal_path` from the live book, its children with it. The write is
    atomic and every other client sees it on its next read."""
    return service().call('POST', '/book/deals',
                          json={'action': 'delete', 'deal_path': deal_path})


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
    class name the book spells it by) or `Price Factor Interpolation` (`entry` `modeldefaults`,
    `fields` one method per routed curve type). A book states each hyperparameter ONCE here and
    every quote block of that family is read over it, so this is where a fit's box, seed or budget
    moves - never inside a quote.

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
    it discounts on, the interpolation the solved factor carries, the conventions its benchmarks
    were authored under (the calendar, the settlement lag, both legs' frequency and day count,
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
                    conventions: dict = None) -> dict:
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

    The block is AUTHORED, not ticked: it is re-installed whole and the market re-bootstrapped in
    one atomic write, so a bootstrap that complains writes nothing and names what it complained
    about. The answer names the date the block was authored on, the block, the knots it solved on
    and the price factors that moved. A row priced off the terminal carries the print's own clock,
    and where that is later than the day the book stands at the book ROLLS ONTO IT - both its dates
    - and every other curve is re-authored there too. Afterwards `tick_market_from_bloomberg` keeps
    these rows valued off the terminal."""
    return service().call('POST', '/book/curve', json=dict(
        conventions or {}, curve=curve, currency=currency, rows=rows,
        **({} if discount_rate is None else {'discount_rate': discount_rate})))


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
    by moving the coordinate the recipe already solves, so the cap of a collar comes in and the
    strike of a strip moves against the client by exactly that much. A currency the book carries no
    rate for refuses by name rather than being crossed at a rate somebody guessed, and so does a
    structure whose recipe SOLVES nothing - a strangle is quoted at the client's own two strikes,
    so there is no coordinate to charge on.

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

    Where the book's vol quotes carry a two-way, the legs are priced on the sides of it a desk
    would deal - each leg's `vol_spread` is the signed vol shift it took, in the surface's own
    units - and `net_mid` is the same legs marked at MID, which is what the trade will be worth on
    the book once booked. Read in the client's sign convention like every premium here, so the
    desk's edge on a zero-cost structure is `net` less `net_mid`. With no two-way in the book
    every shift is zero, `net_mid` equals `net`, and `spread_note` says so.

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
        return quote
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


def main():
    """Serve the tools over stdio, for an MCP host that launches this as a subprocess."""
    configure()
    MCP.run(transport='stdio')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
