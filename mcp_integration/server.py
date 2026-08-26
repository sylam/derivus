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

Run: `python mcp_integration/server.py` (stdio), with `RF_SERVICE_URL` naming the service
(default http://127.0.0.1:8000 - the same variable the Excel add-in reads).
"""
import os
import time

import requests
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations

MCP = MCPServer('derivus', instructions=__doc__)
READ_ONLY = ToolAnnotations(read_only_hint=True)

SERVICE = None


class Service:
    """A `DV_Service` at a URL. `session` is the transport seam - anything with a requests-style
    `request(method, url, ...)`, which is how the gates drive the tools in process through
    fastapi's TestClient without a socket. Deliberately a second copy of what `excel_integration`
    has: the add-in is not an installed package, so nothing here can import it."""

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
    lands the value on the target - then book that. On success the answer carries the new
    `deal_path`, and every other client (the web UI, Excel) sees the deal on its next read. The
    answer is about THIS booking; anything else outstanding in the book arrives as counts under
    `book_issues`, with `validate_book` for the detail.
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
    merges into the calculation block (e.g. `{"Greeks": "First"}`). Waits up to `wait_seconds`
    for the run; the answer carries `tables` and `stats`, and `deal_values` projects per-deal
    values from it. Content-addressed: the same what-if twice is one run.
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
