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
(the `bind='value'` seam; a structural key is refused by the engine's rule). What never moves
from here is structure: a new factor, a moved pillar, a changed convention is authoring, not
ticking.

Answers are SUMMARIES AND POINTERS, never payloads: the model needs to know a deal booked or a
calculation ran, not to hold a simulation cube in its context. A run comes back as its replay
tuple, its stats and each table's SHAPE; cells come one capped page at a time through
`fetch_table`, a table too wide to read as text is refused by name (that is what the web UI is
for), and `deal_values` checks a result's shape before it fetches anything.

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
    """A `DV_Service` at a URL. `session` is the transport seam - anything with a requests-style
    `request(method, url, ...)`, which is how the gates drive the tools in process through
    fastapi's TestClient without a socket. Deliberately a second copy of what `excel_integration`
    has: the import gate holds this module to `requests` + `mcp`, so it can never import the
    add-in's client (which is not packaged anyway)."""

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
    """Bind the service this process talks to. `main` calls it from the environment; a gate calls
    it with a TestClient."""
    global SERVICE
    SERVICE = Service(base_url, session)
    return SERVICE


def service():
    return SERVICE if SERVICE is not None else configure()


#: The caps that keep a result out of the model's context: a page is at most this many rows, and a
#: table wider than this is a scenario cube the model should never hold as text.
MAX_PAGE_ROWS = 200
MAX_TABLE_COLUMNS = 60


def _raw_result(result_id):
    return service().call('GET', '/results/{}'.format(result_id))


def _summary(raw, result_id):
    """A run as the model should hold it: identity, stats, and each table's SHAPE as one line -
    never a column list, never a cell. The cells stay behind `fetch_table`."""
    trimmed = {'result_id': result_id, 'status': raw.get('status')}
    for key in ('plan_hash', 'values_hash', 'seed', 'stats', 'error'):
        if key in raw:
            trimmed[key] = raw[key]
    if 'tables' in raw:
        trimmed['tables'] = {
            name: '{} rows x {} columns'.format(shape['rows'], len(shape['columns']) or 1)
            for name, shape in raw['tables'].items()}
    return trimmed


def _await_result(result_id, wait_seconds):
    """Poll `/results/{id}` until it settles or the wait runs out - one tool call that returns the
    answer beats a model burning a turn per poll. On timeout the id and the way forward travel in
    `hint`, so a long simulation stays reachable."""
    deadline = time.monotonic() + wait_seconds
    interval, stepped_up = 0.25, time.monotonic() + 2.0
    while True:
        summary = _raw_result(result_id)
        if summary.get('status') not in ('queued', 'running'):
            return dict(_summary(summary, result_id), waited=True)
        if time.monotonic() >= deadline:
            return {'result_id': result_id, 'status': summary['status'],
                    'hint': 'still {} - call poll_result({!r}) to check again, and fetch_table '
                            'once it is done'.format(summary['status'], result_id)}
        if time.monotonic() >= stepped_up:
            interval = 1.0
        time.sleep(interval)


def _booking(outcome):
    """A booking outcome as the model should hold it: what happened to THIS deal, and a COUNT of
    anything else outstanding in the book - never the whole verdict, which `validate_book` serves
    to whoever asks for it."""
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
        # a sales name is how a model spells it - so the vernacular is searched beside the names
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
    # summaries and pointers: the calculation's headline plus its field NAMES, and the factor
    # names capped - the model needs the vocabulary it books against, never a payload
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
def price_candidate(deal: dict | None = None, parent_reference: str | None = None,
                    calculation_overrides: dict | None = None,
                    wait_seconds: float = 120.0) -> dict:
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
    submitted = service().call('POST', '/book/price', json=request)
    return _await_result(submitted['result_id'], wait_seconds)


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
def patch_market_values(patch: dict) -> dict:
    """Move market VALUES in the live book - `{factor: {field: value}}`, e.g.
    `{"FxRate.ZAR": {"Spot": 19.25}}`. Only value-bound fields move (spots, rate columns, vols);
    a structural key - anything that would change the plan - is refused by the engine's own rule.
    One atomic write; every client sees it on its next read."""
    return service().call('POST', '/book/market', json={'patch': patch})


@MCP.tool()
async def tick_market_from_bloomberg(pairs: list = None, expiries: list = None,
                                     pillars: list = None, wait_seconds: float = 360.0,
                                     ctx: Context = None) -> dict:
    """Tick the live book's FX market off THIS workstation's Bloomberg terminal - the whole "get
    me today's market" move, and on a fresh machine the call that PROVISIONS the desk.

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
    result_id = submitted['result_id']
    deadline = time.monotonic() + wait_seconds
    while True:
        # every HTTP call is blocking, so it goes to a thread - the event loop stays free to put
        # the progress notifications on the wire
        raw = await asyncio.to_thread(_raw_result, result_id)
        if raw.get('status') not in ('queued', 'running'):
            return raw
        progress = raw.get('progress')
        if progress and ctx is not None:
            # a client resets its timeout on progress: this is what buys a five-minute first use
            await ctx.report_progress(progress['done'], progress['total'], progress.get('note'))
        if time.monotonic() >= deadline:
            return {'result_id': result_id, 'status': raw['status'],
                    'hint': 'still {} - call poll_result({!r}) to check again; the provisioning '
                            'carries on service-side either way'.format(raw['status'], result_id)}
        await asyncio.sleep(2)


@MCP.tool()
def solve_deal(deal: dict, field: str, target: float = 0.0, bounds: list | None = None,
               calculation_overrides: dict | None = None, wait_seconds: float = 300.0) -> dict:
    """Solve ONE field of a candidate deal so the deal's own value lands on `target`, and get the
    deal back READY TO BOOK - the structuring tool. A par forward: solve the amount to target 0.
    A sales margin: target the margin. A zero-cost collar: fix one strike, solve the other to
    target 0. A strike to a premium: solve `Strike_Price` with `bounds` around spot.

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
    submitted = service().call('POST', '/book/solve', json=request)
    outcome = _await_result(submitted['result_id'], wait_seconds)
    solved = outcome.get('stats', {}).pop('Solved', None) if 'stats' in outcome else None
    if solved is not None:
        outcome['solved'] = solved
        outcome['solved_deal'] = dict(deal, **{field: solved['value']})
    return outcome


@MCP.tool()
def solve_structure(structure: str, params: dict, wait_seconds: float = 120.0) -> dict:
    """Quote a whole structure against the live book - the collar, strangle and seagull verb, and
    the one to reach for instead of composing legs by hand: the structure declares its own legs,
    their conventions and the order they solve in, so the finance does not depend on this
    conversation getting it right.

    `structure` is a name from `describe_structure`, and `params` fills the parameters IT declares
    - nothing else. STRIKES ARE MARKET TERMS (a USDZAR strike is 15.50); the runner puts them on
    the engine's axis. The recipe runs server-side against the book's market data: each leg priced
    alone, each solved leg found by the same root find `solve_deal` rides.

    The answer IS the quote - `quote_id`, the params as read, one row per leg (role, deal type,
    buy/sell, the strike in MARKET terms, the premium, what was solved) and the `net`: zero for a
    zero-cost structure, the margin otherwise. `deal` rides with it, the composed structured deal
    ready to book.

    Where the book's vol quotes carry a two-way, the legs are priced on the sides of it a desk
    would deal - each leg's `vol_spread` is the signed vol shift it took, in the surface's own
    units - and `net_mid` is the same legs marked at MID, which is what the trade will be worth on
    the book once booked. Read in the client's sign convention like every premium here, so the
    desk's edge on a zero-cost structure is `net` less `net_mid`. With no two-way in the book
    every shift is zero, `net_mid` equals `net`, and `spread_note` says so. The BOOK IS NOT TOUCHED. What is written is the pending trade:
    `DV_HOME/tmp/<quote_id>.json` holds the quote and its deal, with `<quote_id>.xlsx` - the sheet
    that goes to the client - beside it when the sheet writer is installed; `files` names both, and
    `files.sheet_note` names the install when there is no sheet. A missing sheet writer never
    refuses a quote.

    Then `book_quote(quote_id)` is the approval that makes it a trade. Two identical asks are two
    quotes, each with its own id and its own files - a quote is an act, not a lookup.
    """
    submitted = service().call('POST', '/book/structure',
                               json={'structure': structure, 'params': params})
    outcome = _await_result(submitted['result_id'], wait_seconds)
    quote = outcome.get('stats', {}).get('Quote')
    if quote is not None:
        return quote
    # no quote: the run summary IS the answer - an error naming itself, or the poll pointer, whose
    # follow-up here is the stats rather than a table
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
    `DV_HOME/tmp/<quote_id>.json` and books its deal through the SAME validate-before-write seam
    `book_deal` uses: validated against the book as it is NOW - the market may have moved since
    the quote was given - written atomically, and refused as `{written: false, refused:
    [messages]}` with the file untouched. A refusal is an answer; an id with no file behind it is
    a tool error naming the directory it looked in.

    The pending file is NOT deleted. What was quoted, at what market, under what id, is the audit
    trail of why the book carries what it carries - and the sheet the client saw stands beside it.
    """
    # an approval books through `deal_edit`, so its answer is a booking's - and gets a booking's
    # trim: what happened to THIS deal, the rest of the book's troubles as counts
    return _booking(service().call('POST', '/book/quote', json={'quote_id': quote_id}))


@MCP.tool()
def execute_book(calculation_overrides: dict | None = None,
                 wait_seconds: float = 120.0) -> dict:
    """Run the book's own calculation as it stands - `price_candidate` with no candidate. Waits
    up to `wait_seconds`; on timeout the answer's `hint` says how to pick the run up later."""
    return price_candidate(calculation_overrides=calculation_overrides,
                           wait_seconds=wait_seconds)


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
