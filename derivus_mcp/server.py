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
`fetch_table`, a table too wide to read as text is refused by name (the service serves it whole as
a file), and `deal_values` checks a result's shape before it fetches anything.

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
#: This part is true of every desk; `RECORD` is what a desk that keeps a record adds to it.
INSTRUCTIONS = """\
derivus desk - one trading book with its market, priced by the derivus engine and served by
DV_Service. You are this desk's assistant: everything you do goes through these tools, the book
on disk is what they read and write, and every write is validated before it lands.

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
its gradient; price_candidate for a what-if, or the book as it stands; solve_deal for a par
amount or a strike to a target; book_diary for every payment, fixing and expiry the book owes or
is owed;
xva_increment for what one trade would add to its client's CVA, FVA and peak exposure;
recalc_xva ONLY on request, it is minutes, and xva_view for what stands.
The desk keeps its own NAMED CALCULATIONS - a credit Monte Carlo at another path count, a PFE
grid, a base valuation with Greeks: describe_calculations lists them, configure_calculation saves
one, run_calculation prices the live book (or one subtree) under it. They are this workstation's:
the book's own calculation never moves.

QUOTING IS NOT BOOKING. solve_structure gives a price and books nothing - quote as often as
the client asks. book_quote is the ACCEPTANCE: call it when the client TAKES the price, and it
is what books the trade.

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
you read at a path whenever you amend or delete.

THE MARKET: update_market_quotes and patch_market_values move values; configure_curve,
configure_book and set_base_date change structure and re-solve; tick_market_from_bloomberg
needs a terminal on the service's own workstation, and describe_securities reads the ticker
vocabulary behind it with the print every curve knot was solved from. A booking refused for
market data the book lacks is answered by book_dependencies - what that trade needs and which
seed entry would supply it - and cured by setup_market, which discovers only what is unknown and
installs the surface, spots and curves as editable defaults in one write; declare_survival_curve
writes a new client's credit curve from the spread the CVA desk states. Bootstrapping dials,
Bloomberg ticker codes and curve set-ups are normally configured once in the web UI - ask
before changing them here."""

#: What a desk that keeps a record adds to `INSTRUCTIONS`: every tool it names and that part
#: does not is one the service answers its 404 for on a desk that keeps none.
RECORD = """\
THIS DESK KEEPS A RECORD, and the rest is what that adds.

AN ACCEPTANCE IS RECORDED: book_quote files the quote it books, and between the quote and the
acceptance the market moves, which is reported and never refused. EVERY BOOKING IS A TICKET:
where the desk's policy wants a second seat the trade books all the same and reads PENDING -
the answer carries a tier and waits_on - and that seat signs it with
approve_ticket(actor, ticket), or quote_id in place of the ticket for an accepted quote;
reject_ticket files the other verdict, the trade standing rejected until somebody closes it.
Never book it a second time. worklist names what waits on somebody: tickets to sign, payments
due, fills to confirm, collateral calls, closes to mark, rejected trades.
{written: false, refused: [...]} WITH `accepted` beside the two keys is a different answer: the
client's price is already on the record and the desk's own policy will not book it, so say
which bound it broke and do not re-quote. A legacy book - trades the book file holds and nobody
booked - is taken on as the import_a_legacy_book prompt walks: book_reconcile lists each under
in_file_not_in_record, and book_deal(deal_path, reference, quantity=1,
execution_reference=its reference) books it where it sits, moving no number.

THE RECORD: book_diary's rows carry the fact each waits on, close_check says whether a close on
a day is legal and names what
is outstanding, book_reconcile says where the book file and the record disagree, book_activity
is one line per event with the head to page from, and book_markets the official close standing
per market - its latest by day - with the close of its day each superseded; book_files lists the
market data files and calendars the record declares by their content, each beside the file the
book names that holds it now, and declare_file declares one by its bytes, the job still naming it
by path. DECLARING IS THE OTHER
HALF: declare_close puts the day's close on the record over the values the book is carrying, once
close_check says the day is legal - declare_market is that same act under any other name -
export_settlements strikes the settlement file for a day, on the market the desk designated for it
and on no other, and
file_status says a payment settled or a confirmation matched, against the row's own key, which is
what close_check then stops waiting on - with the amount, asset, kind and settlement reference where
money moved, including collateral and margin under an agreement; file_lifecycle prints the fixing,
or files the election, a close waits on; and book_cash reads the movements
and their balances back; collateral_calls reads what each agreement's CSA calls on a marked close,
settled with file_status as collateral against the agreement. THE P&L: once the day's close is
declared, mark_book values one unit of every instrument on it, and book_pnl reads what the book
made between two marked days or since the last marks. THE PAPER the book trades under is
declared too, by the seat
that keeps the legal documents: declare_legal_entity and declare_agreement, the terms a netting set
and never a balance, and describe_agreements to read them - a booking then names its agreement, the
first one under it bringing its netting set, and its portfolio, a path under the book or a node
declared on it (declare_portfolio, describe_portfolios), and its quantity is the position change in
units of the deal, 1 booking it as written; book_positions reads what stands, each where it sits
with the status its tickets read, and worklist what waits on the seat asking. The record's reads
take the seat reading as actor: a seat that acts at one node of the book reads that node's rows,
and describe_seat says which verbs it holds and at which nodes. THE OFFICIAL CALCULATION is
declared as well, by the admin seat: declare_official_calculation states what the desk's credit
run runs at - its paths, its batches, its seed, the two funding curves - and
describe_official_calculation reads what stands; a named calculation stays a seat's own. CREDIT
LIMITS are the admin seat's too: declare_credit_limits states what each legal entity's peak
exposure is held against and describe_credit_limits reads what stands, the usage against each
riding on the XVA view and on a trade's credit charge; a limit blocks nothing."""


def orientation(recording):
    """The desk's orientation in the mode its service is in: the part every desk reads and, where
    the desk keeps a record, what the record adds."""
    return INSTRUCTIONS + '\n\n' + RECORD if recording else INSTRUCTIONS


MCP = MCPServer('derivus', instructions=orientation(True))
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
        # where the service checks who is asking, the ID token every call carries
        if os.getenv('DV_SPINE_TOKEN'):
            self.transport['headers'] = {'Authorization': 'Bearer ' + os.getenv('DV_SPINE_TOKEN')}

    def request(self, method, path, **kwargs):
        """The service's answer, or `ToolError` carrying its refusal whole - its `code`, then its
        sentence - and, where it took the request and never answered, that a write may still land."""
        try:
            response = self.session.request(method, self.base_url + path,
                                            **dict(self.transport, **kwargs))
        except requests.ReadTimeout as error:
            raise ToolError('DV_Service took the request and did not answer in time ({}): what it '
                            'asked may still land, so read what it changes before sending it '
                            'again.'.format(error))
        except requests.RequestException as error:
            raise ToolError(
                'DV_Service is not reachable at {}: {}. Start it with `DV_Service --book '
                '<job file>`, or point RF_SERVICE_URL at a running one.'.format(
                    self.base_url, error))
        if response.status_code >= 400:
            try:
                said = '{code}: {detail}'.format(**response.json())
            except (ValueError, KeyError, TypeError):
                said = 'DV_Service answered {}: {}'.format(response.status_code, response.text)
            raise ToolError(said)
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
#: is reached through `price_candidate(calculation_overrides={"Greeks": "First"})` and
#: `fetch_table`.
MAX_GREEK_ROWS = 15

#: How many price factor names a book read lists, and how many deals' values one read answers:
#: past them a list is a payload, paged through `fetch_table`.
MAX_FACTOR_NAMES, MAX_DEAL_ROWS = 80, 500


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


def _summary(raw, result_id, stat=None):
    """A run trimmed to what the model should hold. Where `stat` names the job's own outcome and
    the run filed it under its stats, that outcome with the run's `result_id` and `status`
    beside it; else identity - the replay tuple, the engine version in it - stats, and each
    table's shape as one line. Never a column list and never a cell - those stay behind
    `fetch_table`."""
    outcome = (raw.get('stats') or {}).get(stat) if stat else None
    if outcome is not None:
        return dict(outcome, result_id=result_id, status=raw.get('status'))
    trimmed = {'result_id': result_id, 'status': raw.get('status')}
    for key in ('plan_hash', 'values_hash', 'engine_version', 'seed', 'quasi_seed', 'stats',
                'error'):
        if key in raw:
            trimmed[key] = raw[key]
    if 'tables' in raw:
        trimmed['tables'] = {
            name: '{} rows x {} columns'.format(shape['rows'], len(shape['columns']) or 1)
            for name, shape in raw['tables'].items()}
    return trimmed


async def _await_result(result_id, wait_seconds, ctx=None, stat=None):
    """Poll `/results/{id}` until it settles or `wait_seconds` runs out, saying on every poll how
    long it has waited, so one tool call returns the answer rather than a model burning a turn per
    poll - and a host never hears silence. A settled run answers as `poll_result(result_id, stat)`
    does (`_summary`).

    A desktop host cuts a tool call that stays quiet for about a minute and resets that clock on
    every notification, so a run measured in minutes only finishes if the wait speaks: `done` is
    the seconds waited, `total` the wait asked for, and the note is the run's status with whatever
    the job itself is publishing. On timeout the id and that `poll_result` call travel in `hint`,
    so a long run stays reachable.
    """
    started = time.monotonic()
    while True:
        # Every HTTP call is blocking, so it goes to a thread and the event loop stays free to put
        # the progress notifications on the wire.
        raw = await asyncio.to_thread(_raw_result, result_id)
        if raw.get('status') not in ('queued', 'running'):
            return _summary(raw, result_id, stat)
        waited = time.monotonic() - started
        if ctx is not None:
            note = (raw.get('progress') or {}).get('note')
            await ctx.report_progress(round(waited, 1), wait_seconds,
                                      ' - '.join(filter(None, (raw['status'], note))))
        if waited >= wait_seconds:
            return {'result_id': result_id, 'status': raw['status'],
                    'hint': 'still {} - call poll_result({}) to check again; the run carries on '
                            'service-side either way'.format(raw['status'], ', '.join(
                                repr(part) for part in (result_id, stat) if part))}
        # a quick run should not pay a second of latency, a long one should not spin
        await asyncio.sleep(1.0 if waited >= 2.0 else 0.25)


def _booking(outcome):
    """A booking outcome trimmed to what happened to this deal, plus a count of anything else
    outstanding in the book. The whole verdict is `validate_book`'s to serve."""
    verdict = outcome.pop('validate', None) or {}
    issues = {name: count for name, count in
              (('deal_messages', len(verdict.get('deals', {}))),
               ('missing_factors', len(verdict.get('factors', []))),
               ('calculation_messages', len(verdict.get('calculation', [])))) if count}
    if issues:
        outcome['book_issues'] = dict(issues, hint='outstanding on the book before this booking, '
                                                   'apart from what `refused` names: validate_book '
                                                   'lists them')
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
    `QEDI_CustomAutoCallSwap`), not descriptions; `vernacular` gives each the plain names a desk
    says for it, so "an NDF" or "a cap" is looked up there.
    """
    instrument = service().call('GET', '/schema')['Instrument']
    return {'groups': instrument['groups'], 'containers': instrument['containers'],
            'types': sorted(instrument['types']), 'vernacular': instrument['vernacular'],
            'count': len(instrument['types'])}


def _unknown(name, kind, vernacular, listing):
    """`ToolError` for `name`, which is not `kind`: the names in `vernacular` - `{name: its sales
    names, comma separated}` - whose name or one of whose sales names holds it, those a sales name
    says whole first, else `listing`. A sales name is how a model spells it."""
    asked = name.strip().lower()
    said = {key: [word.strip().lower() for word in words.split(',')]
            for key, words in sorted(vernacular.items())}
    close = sorted((key for key, words in said.items()
                    if asked in key.lower() or any(asked in word for word in words)),
                   key=lambda key: asked not in said[key])
    raise ToolError('{!r} is not {}. {}'.format(
        name, kind, 'Close matches: {}'.format(', '.join(close)) if close else listing))


@MCP.tool(annotations=READ_ONLY)
def describe_instrument_type(name: str) -> dict:
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

    `name` is one of the names `list_instrument_types` returns, spelled exactly; a name that is
    not one is answered with the types whose name or `vernacular` holds it, those it names whole
    first.
    """
    schema = service().call('GET', '/schema')['Instrument']
    sections = schema['types'].get(name)
    if sections is None:
        _unknown(name, 'a deal type', schema['vernacular'],
                 'Call list_instrument_types for the full list.')
    fields = {}
    for section in sections:
        fields.update(schema['sections'][section])
    return {'deal_type': name, 'vernacular': schema['vernacular'][name],
            'sections': sections, 'accepts_children': name in schema['containers'],
            'fields': fields,
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

    `solve_structure` then runs the recipe and answers with the composed deal, each leg's premium,
    the strikes it solved and where the pending files landed; `book_quote` is the approval that
    turns that quote into a trade.
    """
    types = service().call('GET', '/schema')['Structure']['types']
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
    if name not in types:
        _unknown(name, 'a structure', {key: entry['vernacular'] for key, entry in types.items()},
                 'Call describe_structure() for the list with the sales names.')
    return dict({'structure': name}, **types[name])


@MCP.tool(annotations=READ_ONLY)
def describe_calculation_type(name: str) -> dict:
    """Every field one calculation type takes, any type the schema declares and a name it does not
    refused with the list - what `calculation_overrides` in `price_candidate` may override."""
    types = service().call('GET', '/schema')['Calculation']['types']
    if name not in types:
        raise ToolError('{!r} is not a calculation type - one of: {}'.format(
            name, ', '.join(sorted(types))))
    fields = types[name]
    return {'calc_type': name, 'fields': fields,
            'required': [key for key, meta in fields.items() if meta.get('required')]}


@MCP.tool(annotations=READ_ONLY)
def describe_factor_type(name: str) -> dict:
    """Every field one price-factor type carries (`InterestRate`, `FxRate`, `FXVol`, ...)
    plus the stochastic processes that can drive it and the interpolations it accepts - for
    READING the book's market data; nothing here edits it."""
    schema = service().call('GET', '/schema')
    if name not in schema['Factor']['types']:
        raise ToolError('{!r} is not a factor type - one of: {}'.format(
            name, ', '.join(sorted(schema['Factor']['types']))))
    return {'factor_type': name, 'fields': schema['Factor']['types'][name],
            'processes': schema['Process_factor_map'].get(name, []),
            'interpolations': schema['Interpolation_factor_map'].get(name, [])}


@MCP.tool(annotations=READ_ONLY)
def describe_configuration() -> dict:
    """Every dial the book's bootstrap can be set with, per section - what `configure_book` writes.

    `Bootstrapper Configuration` holds one entry per price family, keyed by the family, the price
    factor it WRITES (`InterestRate`, `FXVol`, `LogVar2FJModelParameters`, ...); the entry's dials
    are the boxes a fit is solved in, the seeds it starts from and the budgets it stops on, each
    with the default it stands at where the book states nothing. `Price Factor Interpolation` sets
    one method per routed curve type, and `interpolations` is that menu. Quote ladders and
    instrument tables are NOT dials - they belong to the quote block and move through
    `update_market_quotes`."""
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

    The book's `base_date` (the day its MARKET is dated, what its curves roll off), `base_currency`
    and `calculation`, whose `Base_Date` is the day the book is VALUED as of - the two days differ
    wherever the book states them apart, and `set_base_date` moves both; how
    many `deals` it holds - a structure one with its legs, a netting set none - and the
    `netting_sets` a client's trade can be booked under; `curves`,
    every interest-rate curve the market holds, with the knots it solved on, the `interpolation` it
    is built under, the latest print its rows carry (`snapped`) and any benchmark `held_out` - one
    no quote block solved has no knots and no print; `surfaces` with their
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
    # what the book PRICES on, its market data file and its explicit block merged
    factors = sorted(service().call('GET', '/book/market')['market']['Price Factors'])
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
                            'factors': factors[:MAX_FACTOR_NAMES] + (
                                ['... and {} more'.format(len(factors) - MAX_FACTOR_NAMES)]
                                if len(factors) > MAX_FACTOR_NAMES else [])},
            'deals': deals}


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


def _stated(**fields):
    """The optional fields a caller actually stated, as the body to post. A key left out is a key
    the endpoint never sees, which is what keeps a null from meaning a number."""
    return {name: value for name, value in fields.items() if value is not None}


@MCP.tool()
def book_deal(deal: dict | None = None, parent_reference: str | None = None,
              quantity: float | None = None, execution_reference: str | None = None,
              actor: str | None = None, agreement: str | None = None,
              portfolio: str | None = None, price: float | None = None,
              price_currency: str | None = None, deal_path: str | None = None,
              reference: str | None = None, value_date: str | None = None) -> dict:
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

    WHERE THE DESK KEEPS A RECORD, three more are required and refuse by name without them:
    `quantity` is the position change in units of the deal - 1 books it as written, -1 closes it,
    -0.5 unwinds half, the book pricing every position at its net through the fields its type marks
    `sized` (`describe_instrument_type`) - `execution_reference`
    is the venue exec id or ticket id that makes a booking sent again the one it was - answered
    `booked` where it stands, and refused under other fields - and the deal must sit under a
    `NettingCollateralSet` naming a counterparty.
    `agreement` names an agreement `describe_agreements` lists - the set it sits under must be the
    set of that name - `portfolio` the path where the position sits, its top node the book's own
    name (`<book>/Rates/EM`, the book itself if left out), and `price` what it was done at - the
    consideration per unit as written, signed from the desk's side, in `price_currency` where it
    was agreed in another currency than the book reports in: the service crosses it at the
    booking's own board, so never convert it by hand. `value_date` (`YYYY-MM-DD`) is the day that
    consideration settles - `book_diary` then announces it as a `premium` row due that day, and
    one booked without it is due on no day; stated without a price it is refused. `actor` is the
    seat the fact is filed under. A desk that keeps no record ignores all of them.

    EVERY BOOKING IS A TICKET, which the answer carries: where the desk's tiers policy caps sizes
    or tenors, the size is read off the deal as booked at its `quantity` - nothing you state - a
    strip at every fixing still to come, and a type declaring none carries no size, so a cap passes
    it to the next tier; the answer's `tier` names the tier it fell in and, as `fell_past`, the
    bound each earlier tier failed it on; a tier wanting a second seat books it PENDING with
    `waits_on`, which another seat clears with `approve_ticket`; a ticket no tier admits books
    nothing and names every bound it broke. An amendment is a ticket too, routed the same way, as
    is `book_quote`'s acceptance. An `agreement` whose
    netting set the book does not carry yet is brought by its first booking, out of the terms legal
    declared.

    A deal the book ALREADY HOLDS that nobody booked - each `book_reconcile` names under
    `in_file_not_in_record`, a book written before its record or landed by another system - is
    booked at its `deal_path` in place of `deal`, `reference` the deal you read there: the record
    takes the fill and the deal stays as the file holds it. Its take-on is `quantity` 1 with its
    own reference as the `execution_reference`, which moves no number; closing it later is a
    booking of -1.

    On success the answer carries the new `deal_path`, and every other client (the web UI, Excel)
    sees the deal on its next read. The answer is about THIS booking; anything else outstanding in
    the book arrives as counts under `book_issues`, with `validate_book` for the detail.
    """
    return _booking(service().call('POST', '/book/deals', json=dict(
        {'action': 'add', 'deal': deal} if deal_path is None else {
            'action': 'fill', 'deal_path': deal_path},
        **_stated(parent_reference=parent_reference, quantity=quantity,
                  execution_reference=execution_reference, actor=actor, agreement=agreement,
                  portfolio=portfolio, price=price, price_currency=price_currency,
                  reference=reference, value_date=value_date))))


@MCP.tool()
def amend_deal(deal_path: str, fields: dict, reference: str | None = None,
               actor: str | None = None) -> dict:
    """Change one or more fields of a booked deal - "make the notional 3m", "move settlement a
    week". `fields` MERGES into the deal at `deal_path` (from `read_book`); every other field
    stands. The same validate-before-write contract as `book_deal`: a refusal comes back as
    `{written: false, refused: [messages]}` with the file untouched - read the messages, fix,
    amend again. Values wear their wire form: dates `{".Timestamp": "YYYY-MM-DD"}`, percentages
    `{".Percent": 2.5}`, plain numbers as numbers. `reference` names the deal you read at that
    path: another host's booking moves every position, and a path that no longer holds it refuses
    rather than amending whoever sits there now. `actor` is the seat the amendment is filed under
    where the desk keeps a record. A RESTRIKE IS A TICKET: where the desk's tiers policy is in
    force it is routed as `book_deal` is, on the size of the terms it amends to at every unit of
    the position it moves, and under four eyes the position reads PENDING until another seat
    signs (`approve_ticket`)."""
    return _booking(service().call('POST', '/book/deals', json=dict(
        {'action': 'amend', 'deal_path': deal_path, 'fields': fields},
        **_stated(reference=reference, actor=actor))))


@MCP.tool()
def delete_deal(deal_path: str, reference: str | None = None,
                actor: str | None = None) -> dict:
    """Remove the deal at `deal_path` from the live book, its children with it. The write is
    atomic and every other client sees it on its next read. `reference` names the deal you read at
    that path, so a path another host's booking has moved refuses rather than deleting whoever
    sits there now - always pass it when other hosts share the book. On a recording desk, demands
    `book` over the book (`actor`)."""
    return service().call('POST', '/book/deals', json=dict(
        {'action': 'delete', 'deal_path': deal_path},
        **_stated(reference=reference, actor=actor)))


# --------------------------------------------------------------------------- pricing


@MCP.tool()
async def price_candidate(deal: dict | None = None, parent_reference: str | None = None,
                          quote_id: str | None = None, calculation_overrides: dict | None = None,
                          actor: str | None = None, wait_seconds: float = 120.0,
                          ctx: Context = None) -> dict:
    """Price the book PLUS a candidate deal without booking anything - the what-if verb.

    The candidate is `deal` under `parent_reference`, or the `quote_id` `solve_structure` answered,
    priced as accepting that quote would book it; with neither, the book's own calculation runs
    as the book stands. It joins an in-memory copy only; the book file never moves.
    `calculation_overrides` merges into the calculation block: `{"Greeks": "First"}`
    for the AAD delta vector, `{"Greeks": "All"}` for the full second-order block - `Greeks_Second` is the cross-gamma
    matrix a trading read of an options book needs (spot gamma on the diagonal, vanna against the
    surface's own nodes beside it), one backward pass however many factors. Waits up to
    `wait_seconds` for the run; the answer carries `tables` and `stats`, and `deal_values`
    projects per-deal values from it. Content-addressed: the same what-if twice is one run.
    `actor` is the seat reading.
    """
    request = _stated(deal=deal, parent_reference=parent_reference, quote_id=quote_id,
                      calculation_overrides=calculation_overrides or None, actor=actor)
    submitted = await asyncio.to_thread(service().call, 'POST', '/book/price', json=request)
    return await _await_result(submitted['result_id'], wait_seconds, ctx)


@MCP.tool()
async def xva_increment(deal: dict | None = None, parent_reference: str | None = None,
                        quote_id: str | None = None, deal_path: str | None = None,
                        reference: str | None = None, calculation_overrides: dict | None = None,
                        actor: str | None = None, wait_seconds: float = 600.0,
                        ctx: Context = None) -> dict:
    """What a trade does to its client's credit charge: how it moves the CVA, the FVA and the peak
    exposure of its netting set - asked before anyone books it, or of a trade the book holds.

    BEFORE IT BOOKS, name the trade as `price_candidate` takes one - `deal`, with
    `parent_reference` the client's netting set or a node inside it - or pass the `quote_id`
    `solve_structure` answered, which prices what accepting that quote would book. ONCE IT IS
    BOOKED, pass its `deal_path` from `read_book` and nothing else names it: the set is priced as
    it stands and without the trade, so the answer is what that trade adds - a booked quote's id
    refuses, naming that path. `reference` names the deal you read at the path, so a path the
    book has moved under refuses. The set is run twice on the SAME simulated paths,
    with the trade and with the trade beside its mirror, a pair worth nothing on every path, so
    the difference is the trade's and not the set's Monte Carlo noise; asked of the projection's
    last row instead, that noise can be as large as the trade. The paths remove the set's noise;
    the trade's own Monte Carlo error falls only with more paths. Each of `cva`, `fva`, `EE` and every `PFE_` column comes back `with`, `without` and
    `increment`, each side of a column its peak and the day it falls, `{value, day}`, in the
    `currency` named; `paths` says what both
    runs drew. Nothing is booked or recorded. A trade under no netting set, or one the booking
    would refuse, is refused by name; a set whose counterparty has no credit curve fails in the
    engine's words, as `recalc_xva` does. `calculation_overrides` merges into the run's credit
    Monte Carlo (`{"Object": "CreditMonteCarlo", "Random_Seed": 7}`). Waits up to `wait_seconds`.
    Where the client, or an entity above it, has a credit limit, `limits` holds each against the
    trade: `limit`, `usage` with the trade and `usage_without` it - the other sets' peaks off the
    XVA view's rows, this set's off these two runs - `headroom` after the trade, null where a set
    counted under `unrun` has no finished run, `breached` where usage with the trade is over the
    limit and `breached_without` where it already was without it. Say so plainly when the trade
    is what takes it over; the answer blocks nothing. `not_comparable`, where set, is read out in
    place of a usage.
    """
    request = _stated(deal=deal, parent_reference=parent_reference, quote_id=quote_id,
                      deal_path=deal_path, reference=reference,
                      calculation_overrides=calculation_overrides or None, actor=actor)
    submitted = await asyncio.to_thread(service().call, 'POST', '/book/price',
                                        json=dict(request, xva_increment=True))
    return await _await_result(submitted['result_id'], wait_seconds, ctx, stat='Increment')


@MCP.tool(annotations=READ_ONLY)
def describe_calculations() -> dict:
    """The desk's own NAMED CALCULATIONS - `{calculations: {name: block}}`, each block a
    `Calculation` as the book states one: `Object` names a type the schema declares and every other
    key is one of that type's declared fields, `describe_calculation_type` listing them. A field a
    block does not state is the book's own when it runs - the date and the currency among them.
    Kept on this workstation, never in the book and never on the record."""
    return service().call('GET', '/calculations')


@MCP.tool()
def configure_calculation(name: str, calculation: dict | None = None) -> dict:
    """Save a named calculation, or remove it with `calculation` null.

    `calculation` is a `Calculation` block: `{"Object": "CreditMonteCarlo", "MCMC_Simulations":
    8192, "Time_Grid": "0d 1m(1m) 2y(3m)", "Credit_Valuation_Adjustment": {"Calculate": "Yes",
    "Counterparty": "<SurvivalProb name>", ...}}` - the type, then only the fields you want to
    differ from the book's own calculation. A credit Monte Carlo deflates on the base currency's
    own curve, which is no setting of it. A
    container such as `Credit_Valuation_Adjustment` is stated whole. Judged against the type's declarations before anything is written: an unknown
    type, an undeclared field or a value outside a field's menu comes back `{written: false,
    refused: [...]}` naming it, the file untouched. The book and the record never move."""
    return service().call('POST', '/calculations', json={'name': name, 'calculation': calculation})


@MCP.tool()
async def run_calculation(name: str, deal_path: str | None = None, actor: str | None = None,
                          wait_seconds: float = 120.0, ctx: Context = None) -> dict:
    """Price the live book under one named calculation - the whole book, or the subtree at
    `deal_path` (a netting set, a structure, one trade). The block is merged over the book's own
    calculation exactly as `price_candidate` merges `calculation_overrides`, so asking for the same
    dials either way is ONE run. Nothing is booked and nothing is recorded. Waits up to
    `wait_seconds`; the answer carries `tables` and `stats` as `price_candidate`'s does, and a
    credit Monte Carlo is minutes - past the wait the answer is the result id to poll."""
    request = dict({'name': name}, **_stated(deal_path=deal_path, actor=actor))
    submitted = await asyncio.to_thread(service().call, 'POST', '/calculations/run', json=request)
    return await _await_result(submitted['result_id'], wait_seconds, ctx)


@MCP.tool()
def update_market_quotes(quotes: dict, bootstrap: bool = True, actor: str | None = None) -> dict:
    """Tick the live book's market with quote blocks - `{name: block}` exactly as a quote source
    emits them (`derivus_bloomberg.to_market_prices_block` for an FX vol surface; any `Market
    Prices` family). An existing block may move ONLY its quoted values and timestamps - a changed
    pillar, expiry or convention is refused by name, because structure is a re-authoring, never a
    tick. With `bootstrap` (the default) the engine turns the quotes into the price factors the
    pricers read, and the book file gains everything in one atomic write; a bootstrap that
    complains refuses the WHOLE write and hands its messages back as `refused`. After a
    successful tick, `solve_deal` and `price_candidate` price against the fresh market. `actor` is
    the seat writing it - on a recorded desk one holding `mark`, as for every verb that moves the
    market."""
    return service().call('POST', '/book/market', json=dict({
        'quotes': quotes, 'bootstrap': 'Yes' if bootstrap else 'No'}, **_stated(actor=actor)))


@MCP.tool()
def configure_book(section: str, entry: str, fields: dict, actor: str | None = None) -> dict:
    """Set the dials the live book bootstraps its market with - `fields` MERGED into one entry
    `describe_configuration` declares, everything else in it standing.

    `section` is `Bootstrapper Configuration` (`entry` the family, the price factor it writes) or
    `Price Factor Interpolation` (`entry` a routed factor type, `fields` `{"method": ...}` for what
    every factor of it is built with or `{"id": "<curve name>", "method": ...}` for that one curve's
    own rule, a blank method clearing it - though a curve's rule is ordinarily set by
    `configure_curve`). A book states each hyperparameter ONCE here and every quote block of that
    family is read over it, so this is where a fit's box, seed or budget moves - never inside a
    quote.

    The change is checked by building what reads it, so a malformed dial is refused by name before
    a quote is read; the whole market is then re-bootstrapped in the same atomic write, and a
    bootstrap that complains refuses everything and hands its messages back as `refused` with the
    file untouched. On success the answer names the entry written, the dials it now carries, and
    the price factors the re-bootstrap rewrote. `actor` is the seat writing it."""
    return service().call('POST', '/book/configure', json=dict(
        {'section': section, 'entry': entry, 'fields': fields}, **_stated(actor=actor)))


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
    return service().call('GET', '/book/curve', params=_stated(curve=curve))


@MCP.tool()
def configure_curve(curve: str, currency: str, rows: list, discount_rate: str = None,
                    conventions: dict = None, interpolation: str = None,
                    actor: str | None = None) -> dict:
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
    these rows valued off the terminal. `actor` is the seat writing it."""
    return service().call('POST', '/book/curve', json=dict(
        conventions or {}, curve=curve, currency=currency, rows=rows,
        **_stated(discount_rate=discount_rate, interpolation=interpolation, actor=actor)))


@MCP.tool()
def set_base_date(base_date: str, actor: str | None = None) -> dict:
    """Set the live book's CALCULATION DATE - the day everything it holds is valued as of.

    `base_date` is an ISO day, `"2026-09-19"`. The book carries the date twice - the day its
    curves' benchmarks roll off and the day the pricers run on - and this moves both together.
    Every interest-rate curve block is re-authored on the new day from its own rows and conventions
    (the same benchmarks on new dates, the quotes exactly as they stand, no terminal asked) and,
    where the book declares a bootstrapper, the whole market is re-bootstrapped in one atomic write,
    so a bootstrap that complains writes nothing and hands its messages back. The answer names the
    date, the blocks re-authored and the price factors the re-solve rewrote; a block too old to
    carry its conventions is named in `held_out` and left standing.

    `tick_market_from_bloomberg` and `configure_curve` already roll the date FORWARD onto the day
    their quotes were snapped. This is the verb that puts it anywhere - a back-valuation included -
    so call it to value the book as of a day of someone's choosing, never to catch up with a
    tick. `actor` is the seat writing it."""
    return service().call('POST', '/book/date',
                          json=dict({'base_date': base_date}, **_stated(actor=actor)))


@MCP.tool()
def patch_market_values(patch: dict, actor: str | None = None) -> dict:
    """Move market VALUES in the live book - `{factor: {field: value}}`, e.g.
    `{"FxRate.ZAR": {"Spot": 0.05195}}`, each on its field's own axis as `describe_factor_type`
    states it - that spot is USDZAR at 19.25 on a USD book. Only value-bound fields move (spots,
    rate columns, vols); a structural key - anything that would change the plan - is refused by
    the engine's own rule.
    A `Market Prices` block is refused here too, naming `update_market_quotes` as the remedy: a
    quote moves the book through the path that bootstraps, never through a values patch. One atomic
    write; every client sees it on its next read. `actor` is the seat writing it."""
    return service().call('POST', '/book/market',
                          json=dict({'patch': patch}, **_stated(actor=actor)))


@MCP.tool()
def retire_market_overrides(paths: list[list[str]], actor: str | None = None) -> dict:
    """Put the market data FILE's entry back where the live book's own block states one over it -
    "take the spot back to the file's". Each path runs from the section down to the entry,
    `[["Price Factors", "FxRate.ZAR"]]`, or `["Price Factor Interpolation", ".ModelParams",
    "modelfilters", "InterestRate"]` for one type's curve rules; the entry leaves the block, a
    section with its last entry, so the market reads the file's again. Where the book declares a
    bootstrapper the WHOLE market is re-bootstrapped in the same atomic write, and a bootstrap that
    complains writes nothing and hands its messages back as `refused`. A path the block does not
    state refuses by name, the service's 422 naming the key it lacks. Retiring the block's system
    date, `["System Parameters", "Base_Date"]` or its section whole, is `set_base_date` back to the
    file's day first: both the book's dates move and every curve block is re-authored there.
    `actor` is the seat writing it."""
    return service().call('POST', '/book/market',
                          json=dict({'retire': paths}, **_stated(actor=actor)))


@MCP.tool()
async def tick_market_from_bloomberg(pairs: list = None, expiries: list = None,
                                     pillars: list = None, actor: str | None = None,
                                     wait_seconds: float = 360.0, ctx: Context = None) -> dict:
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
    carries, at the expiries it verified. The answer is the finished run's outcome: what installed,
    what updated, whether the map had to be provisioned, or the refusal messages.

    A refusal is an answer and it NAMES what stopped it: no terminal answering on this machine,
    a candidate whose last print is stale (a retired benchmark keeps quoting a plausible price -
    that is the trap the verification exists for), a seed spelling the terminal does not know, or
    a bootstrap complaint that refuses the whole write. Each names the ticker or the file, so the
    next move is to fix that one thing and call again. `actor` is the seat the run is queued under.
    """
    request = _stated(pairs=pairs, expiries=expiries, pillars=pillars, actor=actor)
    submitted = await asyncio.to_thread(
        service().call, 'POST', '/book/bloomberg', json=request)
    return await _await_result(submitted['result_id'], wait_seconds, ctx, stat='Bloomberg')


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
    return service().call('GET', '/book/securities', params=_stated(block=block))


@MCP.tool()
def configure_securities(block: str, key: str, entry: dict | list | None = None,
                         actor: str | None = None) -> dict:
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
    terminal about them and writes the map a curve is then set up off. `actor` is the seat writing
    it."""
    return service().call('POST', '/book/securities', json=dict(
        {'block': block, 'key': key, 'entry': entry}, **_stated(actor=actor)))


@MCP.tool()
async def verify_securities(block: str = None, key: str = None, securities: list = None,
                            actor: str | None = None, wait_seconds: float = 600.0,
                            ctx: Context = None) -> dict:
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
    and the verification carries on service-side. `actor` is the seat the run is queued under."""
    request = _stated(block=block, key=key, securities=securities, actor=actor)
    submitted = await asyncio.to_thread(
        service().call, 'POST', '/book/securities/verify', json=request)
    return await _await_result(submitted['result_id'], wait_seconds, ctx, stat='Securities')


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
    how many a terminal has `verified`, plus `conventions` for a curve, and a `note` saying who
    supplies it. `supply` null means this desk's vocabulary spells nothing for that factor - an
    equity or a commodity - and the market for it is authored by hand, or written as the note says
    (`declare_survival_curve` writes a counterparty's `SurvivalProb`).

    A `deal` the booking verb would refuse is refused HERE in its words rather than walked, so a
    misspelt type is never answered "nothing is missing". `setup_market` acts on this answer."""
    if deal is None:
        return service().call('GET', '/book/dependencies', params=_stated(deal_path=deal_path))
    return service().call('POST', '/book/dependencies', json=dict(
        {'deal': deal}, **_stated(parent_reference=parent_reference)))


@MCP.tool()
async def setup_market(pair: str = None, deal: dict = None, parent_reference: str = None,
                       deal_path: str = None, actor: str | None = None,
                       wait_seconds: float = 600.0, ctx: Context = None) -> dict:
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
    `wait_seconds` the id travels in `hint` and the set-up carries on service-side. `actor` is the
    seat the run is queued under."""
    request = _stated(pair=pair, deal=deal, parent_reference=parent_reference,
                      deal_path=deal_path, actor=actor)
    submitted = await asyncio.to_thread(service().call, 'POST', '/book/setup', json=request)
    return await _await_result(submitted['result_id'], wait_seconds, ctx, stat='Setup')


@MCP.tool()
async def calibrate_spot_model(pair: str, family: str = '', actor: str | None = None,
                               wait_seconds: float = 1800.0, ctx: Context = None) -> dict:
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

    `pair` is the surface's own name (`"USD.ZAR"`). The answer is the run's outcome: the family
    and parameters fitted, the block installed, the factor written, the quotes' provenance and the
    fit's wall time. THIS IS THE EXPENSIVE ONE - measured in the
    tens of minutes, at the same cost class as an XVA recalculation, so quotes and valuations keep
    jumping the queue while it runs. `poll_result` follows it if the wait runs out; the fit carries
    on service-side either way.

    There is nothing further to read: the written `<family>ModelParameters.<currency>` factor in
    the book's `Price Factors` IS the result, so `read_book` serves it like any other market data.
    A pair the book carries no built surface for refuses BY NAME - tick the market first. `actor` is
    the seat the run is queued under.
    """
    submitted = await asyncio.to_thread(service().call, 'POST', '/book/model', json=dict(
        {'pair': pair, 'family': family}, **_stated(actor=actor)))
    outcome = await _await_result(submitted['result_id'], wait_seconds, ctx, stat='SpotModel')
    return dict(outcome, factor=submitted['factor'])


@MCP.tool()
async def solve_deal(deal: dict, field: str, target: float | dict = 0.0,
                     bounds: list | None = None, calculation_overrides: dict | None = None,
                     actor: str | None = None, wait_seconds: float = 300.0,
                     ctx: Context = None) -> dict:
    """Solve ONE field of a candidate deal so the deal's own value lands on `target`, and get the
    deal back READY TO BOOK - the structuring tool. A par forward: solve the amount to target 0.
    A sales margin: target the margin. A zero-cost collar: fix one strike, solve the other to
    target 0. A strike to a premium: solve `Strike_Price` with `bounds` around spot.

    `target` is a number in the book's reporting currency, or MONEY - `{'amount': 50000.0,
    'currency': 'ZAR'}` - which is how a sales margin is actually agreed: an amount in a currency,
    crossed to the reporting one at the book's own spots, refusing by name a currency the book
    carries no rate for. This deal is the one the desk BOOKS, so a margin target marks it at plus
    the margin, and the solved deal comes back carrying `Sales_Margin` and `Sales_Margin_Currency`
    - the record of what was charged, which prices nothing and books with the trade.

    Prefer this over hand-iterating `price_candidate`: the root find runs server-side against the
    book's market data (brentq inside `bounds`, else a secant from the field's current value -
    exact in two pricings for an amount) and nothing large enters the conversation. The answer
    carries the field's `value`, the pricing count (`evaluations`), the `residual` and `deal` - the
    deal with the field set, which `book_deal` books as-is. `actor` is the seat reading.
    """
    request = dict({'deal': deal, 'field': field, 'target': target}, **_stated(
        bounds=bounds, calculation_overrides=calculation_overrides or None, actor=actor))
    submitted = await asyncio.to_thread(service().call, 'POST', '/book/solve', json=request)
    return await _await_result(submitted['result_id'], wait_seconds, ctx, stat='Solved')


@MCP.tool()
async def solve_structure(structure: str, params: dict, netting_set: str | None = None,
                          margin: dict | None = None, actor: str | None = None,
                          portfolio: str | None = None, wait_seconds: float = 120.0,
                          ctx: Context = None) -> dict:
    """Quote a whole structure against the live book - the collar, strangle and seagull verb, and
    the one to reach for instead of composing legs by hand: the structure declares its own legs,
    their conventions and the order they solve in, so the finance does not depend on this
    conversation getting it right.

    `structure` is a name from `describe_structure`, and `params` fills the parameters IT declares
    - nothing else. The recipe runs server-side against the book's market data: each leg priced
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
    the client's sign and the client's paper is worth minus what they paid for it. Every money in
    it is in `currency`, the book's reporting currency. The `margin`
    block says what was charged and what it converted to; the booked mirror is the bank's side of
    it, marking at plus the margin. `deal` rides with it, the composed structured deal ready to
    book, carrying `Sales_Margin` and `Sales_Margin_Currency` as the record of what was agreed.

    A quote prices on the LIVE spot when this workstation's terminal is up, and on the book's last
    ticked one - with the reason named - when it is not; the outcome's `spot` block says which was
    used (`value_market`, the pair as quoted, with `source` and `note`), and `forward` is the
    pair's outright to the delivery, quoted the same way.

    Where the book's vol quotes carry a two-way, every leg is still priced at the MID and what the
    market charges for the spread is levied where the margin is - on the coordinate the recipe
    solves, or on the PREMIUM where it solves nothing, `charged_on` naming which. Each leg's vega is
    read per quoted pillar and charged that pillar's own half, so `spread_charge` is what that leg
    cost the client, `spread` is the pillar rows behind it - those a vega reaches; the quote
    resource keeps every quoted pillar - and `spread_source` says whether the vega came off the
    surface or off a lognormal reading of a leg priced under a fitted model.
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
    `files.sheet_note` says what the desk lacks when there is no sheet. A missing sheet writer never
    refuses a quote.

    Then `book_quote(quote_id)` is the ACCEPTANCE that makes it a trade - call it when the client
    takes the price and not before. NOTHING IS RECORDED HERE: a desk quotes many times a day and
    the record holds the one that comes back, so the quotes nobody accepts simply expire in the
    service's own tmp. Two identical asks are two quotes, each with its own id and its own files -
    a quote is an act, not a lookup. A quote is also FIRM ONLY FOR A WINDOW where the book declares
    one (`Quote Policy.firm_seconds`, ten minutes by default): accept it while it is fresh, or
    re-quote. `actor` is the seat the quote is attributed to where the desk keeps a record, and
    `portfolio` the node the trade books into once accepted - a path under the book, the book's
    own where none is named - stated with the price.
    """
    submitted = await asyncio.to_thread(
        service().call, 'POST', '/book/structure',
        json=dict({'structure': structure, 'params': params}, **_stated(
            netting_set=netting_set, margin=margin, actor=actor, portfolio=portfolio)))
    quote = await _await_result(submitted['result_id'], wait_seconds, ctx, stat='Quote')
    if 'legs' not in quote:
        return quote
    # a leg's spread is one row per quoted pillar, most of them at no vega: the answer keeps the
    # rows a vega reaches and the quote resource every one
    legs = [dict(leg, spread=leg['spread'] and [row for row in leg['spread'] if row['vega']])
            for leg in quote['legs']]
    return dict(quote, legs=legs, resources={
        'quote': 'derivus://quote/{}'.format(quote['quote_id']),
        'sheet': 'derivus://quote/{}/sheet'.format(quote['quote_id'])})


@MCP.tool()
def book_quote(quote_id: str, actor: str | None = None) -> dict:
    """THE ACCEPTANCE: the client took the price, so record the quote and book the trade. The
    second half of `solve_structure`, and the only thing that turns a quote into a trade - call it
    on the client's word and never to see what would happen.

    `quote_id` is the one the quote carries. The service reads the pending trade back from
    `DV_HOME/tmp/<quote_id>.json` and books the MIRROR of its deal - the quote's legs carry the
    CLIENT's side, and the book holds the bank's position, so every booked leg lands on the
    opposite side from the one quoted - through the SAME validate-before-write seam
    `book_deal` uses: validated against the book as it is NOW, written atomically, and refused as
    `{written: false, refused: [messages]}` with the file untouched. A refusal is an answer; an id
    with no file behind it is a tool error saying the desk holds no quote under it.

    WHERE it books is the quote's own `netting_set`: the mirror lands UNDER that
    `NettingCollateralSet` node, which is what makes `recalc_xva` see the trade - the client's CVA
    is projected over that subtree and a trade booked at the root is outside it. A quote that named
    no set books at the root, as it always did.

    A QUOTE IS FIRM FOR A WINDOW. Where the book declares a `Quote Policy`, its `firm_seconds` is
    how long an acceptance may stand on the price that was given; past it this is a tool error
    naming the age, the window and the remedy, and NOTHING is written - re-quote with
    `solve_structure` and accept that. A book declaring no policy holds a quote bookable
    indefinitely.

    WHERE THE DESK KEEPS A RECORD the answer says more. THE MARKET MOVING IS NEWS, NOT A REFUSAL:
    `market` reports the values the quote was struck on, the ones standing now and whether they
    moved. Inside the firm window a booking elsewhere is news too: `firmness.plan` reports the plan
    the quote was priced on and the one it lands in. What does refuse is a board that was already
    stale when the price was given, and the book having moved under the solve where the book
    declares no `Quote Policy`. And where a `tiers` policy is in force the ticket is routed:
    the answer carries `accepted` - the acceptance is filed either way - and `tier`, and under a
    tier wanting a second seat the trade books PENDING with `waits_on` saying why: have that seat
    call `approve_ticket(actor, quote_id=quote_id)` - never accept again. `refused` beside
    `accepted` is a ticket no tier admits. `actor` is the seat the acceptance and the fill are
    filed under; the trade books into the `portfolio` the quote stated.

    TWO REFUSALS WEAR ONE SHAPE and `accepted` is the tell. `{written: false, refused: [...]}` with
    no `accepted` is the book declining the deal - read the messages, fix, book again. The same two
    keys WITH `accepted` is the client's price already on the record and the desk's own policy
    declining to book it: do not re-quote, say which tier refused and what it caps. And
    `{written: false, booked: {...}}` means this quote is already on the book - nothing to do.

    The pending file is NOT deleted. What was quoted, at what market, when, under what id, is the
    audit trail of why the book carries what it carries - and the sheet the client saw stands
    beside it.
    """
    # An acceptance books through `deal_edit`, so its answer is a booking's and takes a booking's
    # trim.
    return _booking(service().call('POST', '/book/quote', json=dict(
        {'quote_id': quote_id}, **_stated(actor=actor))))


@MCP.tool(annotations=READ_ONLY)
def list_quotes(actor: str | None = None) -> dict:
    """Every quote this desk has given, newest first, one row each - enough to pick one without
    opening it: `quote_id`, the `structure` and the `variation` it was dealt as, the client's
    `netting_set`, the `net` in its `currency`, when it was given (`quoted_at`) and by whom
    (`quoted_by`), the `portfolio` it books at, `accepted` and `booked` null while it is only
    quoted, `booked.deal_path` where the trade landed once it is on the book and
    `booked.reference` the trade's own reference, which still finds it after a node ahead of it is
    deleted moves the path - and where it stands as of the read's `as_of`: `stands` is `booked`,
    `accepted`, `firm` until `firm_until`, `lapsed` past it (`book_quote` refuses it: re-quote),
    `quoted` where the book declares no window (`firm_seconds` null), or `refused` where its
    `Quote Policy` will not read, `policy_refused` saying why. A quote whose file will not read is
    its `quote_id` and `unreadable`, the sentence why, alone. `actor` is the seat reading, which is
    shown a client's quote as it is shown a position, where it holds a grant at the node the quote
    books under, and every quote it gave itself."""
    return service().call('GET', '/book/quotes', params=_stated(actor=actor))


@MCP.tool()
def approve_ticket(actor: str, ticket: str | None = None, quote_id: str | None = None) -> dict:
    """Sign the ticket a booking carries - the second pair of eyes a tier under four eyes wants -
    named by its `ticket`, or for an accepted quote by its `quote_id`, the signature being over the
    ticket that acceptance minted and so reaching that quote and no other.

    Call it when a booking came back PENDING, `tier` under four eyes with `waits_on`: the trade is
    already booked, and this is what clears it. `actor` is the seat that signs and is REQUIRED: the
    booker's own signature is filed and clears nothing, and a seat the desk has not scoped to
    approve where the ticket books is refused in the record's own words. The verdict is filed where
    the booking sits, read off the record. Answers `{recorded: {lsn}, ticket, portfolio, status}`,
    `status` what the trade reads now; signing twice is one fact.
    """
    return service().call('POST', '/book/approve', json=dict(
        _stated(ticket=ticket, quote_id=quote_id), actor=actor))


@MCP.tool()
def reject_ticket(reason: str, actor: str, ticket: str | None = None,
                  quote_id: str | None = None) -> dict:
    """Refuse the ticket a booking carries, named as `approve_ticket` names it, with the reason on
    the row.

    A verdict is never withdrawn, so what stands is what was filed LAST: a rejection after an
    approval is what the record says, and a later approval moves it back. The trade happened, so a
    rejected one STANDS until somebody closes it, and `worklist` names it. `reason` is required.
    Answers `{recorded: {lsn}, ticket, portfolio, status}`.
    """
    return service().call('POST', '/book/reject', json=dict(
        _stated(ticket=ticket, quote_id=quote_id), reason=reason, actor=actor))


@MCP.tool(annotations=READ_ONLY)
def validate_book(actor: str | None = None) -> dict:
    """What would stop the live book running, without running it: authoring messages per deal
    reference, the price factors named by deals that the market data has no block for, and what
    the book's `Calculation` block's own declarations say against it. `actor` is the seat asking,
    which the queue admits as it admits a what-if."""
    live = service().call('GET', '/book')
    return service().call('POST', '/validate', json=dict(live['document'], **_stated(actor=actor)))


@MCP.tool(annotations=READ_ONLY)
def describe_book(actor: str | None = None) -> dict:
    """What the engine makes of the live book without pricing it: deals counted by type, the
    factor universe (resolved and missing), the calculation as loaded, and a crude cost read.
    `actor` is the seat asking, which the queue admits as it admits a what-if."""
    live = service().call('GET', '/book')
    return service().call('POST', '/describe', json=dict(live['document'], **_stated(actor=actor)))


@MCP.tool(annotations=READ_ONLY)
def book_risk_summary(actor: str | None = None) -> dict:
    """The desk's CONSOLIDATED risk: what the whole book is worth and what it is exposed to, in
    one read - the question after every booking and every tick.

    COUNTERPARTIES DO NOT MATTER HERE. This is one base valuation with first-order Greeks over the
    book as it stands, aggregated across everything the desk holds, so there is nothing to slice by
    counterparty and no netting set enters it. The per-counterparty number is XVA, which is a
    different calculation for a different reason - see `xva_view`.

    Answers the mark (`mtm` in the book's report currency), `as_of`, the `etag` the service cached
    it under, and the LARGEST rows by absolute size, twice - how many deals the book holds being
    `desk_status`'s to say. `quotes` is
    the risk a desk hedges in - `block` (the quote block), `quote` (the benchmark or smile quote)
    and `value`, the derivative per unit of the quote AS QUOTED (a par rate in percent reads per
    1%), taken through the calibration that builds the factor, `quoted` naming the factors those
    rows stand for and `quote_note` why a book's quotes did not reach its risk where they did not.
    `greeks` is every factor's own row - `factor`, `tenor` (absent for a spot) and `value` per
    unit of that factor - where a spot, and anything no quote builds, is read. Every row also
    carries `level`, the factor or quote it is taken at, and `risk`, the same derivative per the
    move `per` names - a spot per 1% of itself, a rate per bp, a vol per vol point. Never the whole
    per-deal table: `deal_values` on a `price_candidate` run serves that, and the web blotter
    renders the lot.

    Cheap and cached on the book's own content, so asking again after nothing moved costs nothing;
    a booking or a market tick moves the etag and the numbers follow. `actor` is the seat reading.
    """
    risk = service().call('GET', '/book/risk', params=_stated(actor=actor))

    def size(row):
        # largest per the move each row is read per, the per-unit derivative where it reads none
        return -abs(row['value'] if row.get('risk') is None else row['risk'])

    greeks, quotes = sorted(risk['greeks'], key=size), sorted(risk['quotes'], key=size)
    return {'as_of': risk['as_of'], 'etag': risk['etag'], 'currency': risk['currency'],
            'mtm': risk['mtm'], 'quotes': quotes[:MAX_GREEK_ROWS], 'quote_rows': len(quotes),
            'quoted': risk['quoted'], 'quote_note': risk['quote_note'],
            'greeks': greeks[:MAX_GREEK_ROWS], 'greek_rows': len(greeks),
            'hint': 'the per-deal values are behind price_candidate + deal_values; XVA is '
                    'xva_view'}


@MCP.tool(annotations=READ_ONLY)
def xva_view(reference: str | None = None) -> dict:
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
    `note` saying so and no `deal_path`; a recalc still in flight rides under `recalc`. `currency`
    is the one the row's run reported in, null on a row filed before it was kept. `peaks` are the
    run's largest `EE` and `PFE_<percentile>`, each `{value, day}`; a set's whole exposure profile
    is not in this list. Name one set as `reference` and the answer is that set's row alone with
    its `profile`, `{columns, index, data}` by day, a set whose last run filed none refusing by
    name. Beside the rows, `funding` names the two curves a recalc asked now would fund its FVA
    at, `cost` and `benefit`, null for one the calculation does not state - the official
    calculation's where the desk declares one (`describe_official_calculation`), else the book's.
    Where it carries `refused`, the official calculation cannot be read and a recalc would refuse
    in those words: say so, the rows still standing as their runs left them. `limits` is one entry
    per legal entity a credit limit is declared for (`describe_credit_limits`): `limit`, the
    `usage` the rows' peak `PFE` add up to over the entity's sets and those of every entity under
    it, and the `headroom` left, in the limit's currency. `sets` is how many it sums and `unrun`
    how many have no finished run: where any has none the usage is a floor and `headroom` is null,
    and where all have none `usage` and `breached` are null too, so say which sets want a recalc
    rather than quote a headroom. `breached` is usage above the
    limit; nothing is blocked. Where `not_comparable` is set, read its sentence out and state no
    usage.
    """
    if reference is None:
        return service().call('GET', '/book/xva')
    profile = service().call('GET', '/book/xva/{}/profile'.format(
        requests.utils.quote(reference, safe='')))['profile']
    return dict(next(row for row in service().call('GET', '/book/xva')['sets']
                     if row['reference'] == reference), profile=profile)


@MCP.tool()
async def recalc_xva(netting_sets: list | None = None, actor: str | None = None,
                     wait_seconds: float = 600.0, ctx: Context = None) -> dict:
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
    once it is done; `poll_result` follows any one set. `actor` is the seat the runs are queued
    under.
    """
    submitted = await asyncio.to_thread(service().call, 'POST', '/book/xva', json=dict(
        {'netting_sets': netting_sets}, **_stated(actor=actor)))
    queued = submitted['queued']
    if not queued:
        return submitted
    # Freshly queued sets drain in order through one worker, so the last settling usually means
    # every one has - but a cached set answers 'done' at once, so xva_view's per-row status rules.
    last = await _await_result(queued[-1]['result_id'], wait_seconds, ctx)
    return {'queued': queued, 'last': last,
            'hint': 'xva_view reads the rows and each row carries its own status; poll_result '
                    'follows any one set by its result_id'}


@MCP.tool(annotations=READ_ONLY)
def book_reconcile(actor: str | None = None) -> dict:
    """Where the book file and the book of record disagree - the record read AT ITS HEAD.

    A READING, never a refusal. The file is the desk's working copy and the record is what is true,
    and they meet where the book prices - a position is its terms under its agreement's set, priced
    at the record's net, so an unwind is no disagreement. This names two things, each row by its
    `netting_set`, the trade's `reference` and the `client`: a trade the record holds that the file
    has lost, which nothing prices (`in_record_not_in_file`) - book it again under its execution
    reference to put it back - and a deal the file holds that nobody booked under that set, which
    prices as written (`in_file_not_in_record`, with its `deal_path`) - book it or delete it.

    The record is folded AT ITS HEAD, so a booking whose file write never landed is exactly what
    shows up here. `events_behind` counts every event since the file was written and
    `positions_behind` the fills and amendments among them; an empty answer with both at 0 is a
    desk whose copy is exactly the record; `terms_mismatch` names a client's netting set whose
    paper is not what legal declared. `actor` is the seat reading. 404 on a box that records
    nothing.
    """
    return service().call('GET', '/book/reconcile', params=_stated(actor=actor))


@MCP.tool(annotations=READ_ONLY)
def book_diary(due_before: str | None = None, actor: str | None = None) -> dict:
    """Everything the book OWES or is owed: every payment, fixing and expiry its deals carry, and
    a `premium` row for each booking whose price moves money, due on its `value_date`.

    The compile's own schedule, so the diary and the pricer cannot disagree about a payment. A row
    names its deal's instrument, `reference` and `deal_path` - the path is the identity, since
    references are not unique - the leg it sits on, its due date, its currency, its notional, and
    its amount where the compile determines one - one PAYMENT per leg and pay day, spelled by the
    pricer's own line, and a floating coupon reads `amount: null` and `determined: false` until its
    resets fix, which is not the same as owing nothing. `state` is `due`, `observed`, `settled` or
    `expired`, `reason` says why the record cannot answer a row, and `key` is what a settlement
    fact names the row by.

    `due_before='YYYY-MM-DD'` trims it to what falls due by a day. Cached on the book's own content
    and computed on the compute queue, so asking again after nothing moved costs nothing. `actor`
    is the seat reading.
    """
    return service().call('GET', '/book/diary',
                          params=_stated(due_before=due_before, actor=actor))


@MCP.tool(annotations=READ_ONLY)
def close_check(date: str) -> dict:
    """Whether a close on `date` is legal, and what it is waiting on - the catch-up rule as a read.

    A close is legal when every diary entry due on or before that day has its fact. `legal: false`
    comes with the `outstanding` rows - each one names the deal, the leg, the day and its `kind` -
    so what has to happen before the close is a list rather than a verdict.

    THIS DECLARES NOTHING. Declaring the close is a separate act; this says whether the record is
    ready for one. 404 on a box that records nothing.
    """
    return service().call('GET', '/book/close/check', params={'date': date})


@MCP.tool(annotations=READ_ONLY)
def book_activity(since: int | None = None, limit: int | None = None,
                  actor: str | None = None, deal_path: str | None = None,
                  agreement: str | None = None, entity: str | None = None) -> dict:
    """The record's own strip: one line per event - where it sits, when it was recorded and when
    it is TRUE, who said it, and the sentence for what it was: `summary` names the trade and
    client, agreement or entity it is about, `by` is who did it (the seat refused on a refusal,
    the tier on an automatic signature), and `deal_path`, `agreement` and `entity` are what it
    names, null where it names none; on a restrike `changed` is each field it moved, `{field, was,
    became}`.

    Newest last, beside the `lsn` to ask again from. A PAGE WALKS FORWARD: with `since` it is the
    OLDEST `limit` rows after that position and the `lsn` is the last row delivered, so asking
    again with the one you were given reaches every event in turn - that is how a whole record is
    read without holding it at once; with no `since` it is the NEWEST `limit` rows and the `lsn`
    is the head, `limit` left out being the service's own page. The strip opens no body, so it reads EVERY type - including one this hub has no
    sentence for, which renders its own name rather than dropping out of the sequence. `actor` is
    the seat reading. `deal_path` reads one trade's rows alone, through its restrikes - the path
    `book_positions` gives - and `agreement` or `entity` the rows that name that agreement or that
    legal entity, a client's history, paged the same way. 404 on a box that records nothing.
    """
    return service().call('GET', '/book/activity', params=_stated(
        since=since, limit=limit, actor=actor, deal_path=deal_path, agreement=agreement,
        entity=entity))


@MCP.tool(annotations=READ_ONLY)
def book_markets() -> dict:
    """The record's markets: the official close standing per market, the names a values vector was
    declared under, and the snapshots registered against a book.

    A market stands on its latest close by day, and a close carries its `date` and the LSN of the
    close of its day it RESTATED - a close is superseded by a new one rather than corrected in
    place - so what a market was marked at before is still readable, and the
    `values_hash` on a row is the address of the vector itself. 404 on a box that records nothing.
    """
    return service().call('GET', '/book/markets')


@MCP.tool()
def declare_market(name: str, actor: str | None = None) -> dict:
    """Point a market NAME at the values the book is carrying right now - the desk's mark.

    Officialness is a property of the name: every values vector is stored the same way, and
    `official` is the name the firm's own board carries. A `private/<subject>/<name>` board is one
    seat's own and the subject in the name must be the seat declaring it. A seat the desk has not
    scoped to mark is refused in the record's own words. Answers `{recorded: {lsn}, name,
    values_hash}`, and re-declaring the same numbers under one name is one fact.
    """
    return service().call('POST', '/book/markets',
                          json=dict({'name': name}, **_stated(actor=actor)))


@MCP.tool(annotations=READ_ONLY)
def book_files() -> dict:
    """The market data files and calendars the record declares by their content: under each name
    the declaration standing - its `kind`, the `blob` its bytes are stored under, the declaration
    it replaced (`supersedes`) - and `matches`, which file the book names (`MarketDataFile` or
    `CalendDataFile`) holds that content now, null where neither does. A job names its files by
    path and a run says the content it read. 404 on a box that records nothing.
    """
    return service().call('GET', '/book/files')


@MCP.tool()
def declare_file(name: str, kind: str, path: str, actor: str | None = None) -> dict:
    """Declare a market data file or a calendar a job names, by the bytes it holds now: `name` the
    desk's name for it, `kind` `market` or `calendar`, and `path` where it sits on the service's
    own box. The record keeps the bytes under their sha256, the digest a run of a job naming the
    file reports reading, and the latest declaration under a name stands; declaring the standing
    content again is one fact. The job still names the file by path. A seat the desk has not scoped
    to mark is refused in the record's own words. Answers `{recorded: {lsn}, name, blob}`.
    """
    return service().call('POST', '/book/files', json=dict(
        {'name': name, 'kind': kind, 'path': path}, **_stated(actor=actor)))


@MCP.tool()
def declare_close(market: str | None = None, date: str | None = None,
                  actor: str | None = None) -> dict:
    """Declare the official close - the day the desk agrees what the book was worth.

    `market` defaults to the one the desk designates for its P&L, and to `official` where it
    designates none, and `date` to the book's own base date. IT RUNS BEHIND
    `close_check`: a day that check calls illegal is refused here naming what it waits on, and
    nothing is recorded - so call `close_check` first and file the missing fixings, settlements and
    elections. A `private/<subject>/...` board belongs to that subject alone and a close on it is
    refused for anyone else, exactly as a mark is. A second close for one day SUPERSEDES that day's
    first rather than correcting it - never a later day's - and the answer names the position it
    stands over:
    `{recorded: {lsn}, market, date, values_hash, supersedes_lsn}`.
    """
    return service().call('POST', '/book/close',
                          json=_stated(market=market, date=date, actor=actor))


@MCP.tool()
def file_status(subject: str, status: str, actor: str | None = None, amount: float | None = None,
                asset: str | None = None, kind: str | None = None, reference: str | None = None,
                value_date: str | None = None, agreement: str | None = None) -> dict:
    """Say that a payment settled, a confirmation matched, or collateral moved - the back office's
    half of the record, each act its own verb: `settled` with no money or with a `payment`, a
    `premium` or a `fee` is the `settle` seat's, `confirmed` the `confirm` seat's, and
    `collateral` or `margin` moved the `collateral` seat's.

    `subject` is the row's own `key`, which `book_diary` carries on every row and
    `export_settlements` carries on every row it instructs: it is a 64-hex address, so paste the
    one the row gives you rather than a reference or a date. `status` is the desk's own word, and
    `settled` is the one `close_check` reads - a close does not pass over a payment nobody said had
    moved.

    WHERE MONEY MOVED, say how much: `amount` signed from the bank's side (received positive, paid
    or posted negative), the `asset` it is an amount of (a currency, or a security for collateral),
    its `kind` - `payment` against a diary row's key, `premium` against a premium row's, `fee`
    against an instrument, `collateral` or `margin` against an agreement the record declares - the
    settlement system's own `reference` and the `value_date` (`YYYY-MM-DD`) it settled on, all
    under `settled`; and for a payment the `agreement` it was paid under, the one its row carries,
    required where the row's instrument sits under more than one. Filed again under one reference,
    a movement restates the one it corrects; `book_cash` reads them back.

    A key the book no longer announces a row for is still RECORDED: the record holds what it was
    told and the reading is a fold, so a late settlement against a paid-away row is a fact rather
    than a refusal. `actor` is the seat it is filed under. Answers `{recorded: {lsn}, subject,
    status}` and what moved, and saying one thing twice is one fact.
    """
    return service().call('POST', '/book/transition', json=dict(
        {'subject': subject, 'status': status},
        **_stated(actor=actor, amount=amount, asset=asset, kind=kind, reference=reference,
                  value_date=value_date, agreement=agreement)))


@MCP.tool()
def file_lifecycle(event_type: str, body: dict, actor: str | None = None,
                   effective_time: str | None = None) -> dict:
    """Print a fixing, or file a holder's choice or an agent's ruling - the fact `close_check`
    waits on for a fixing row, or for an expiry that leaves a choice.

    `event_type` `fixing_observed` takes `body` `{index, date, source, value}`: the `index` and the
    `date` (`YYYY-MM-DD`) the `book_diary` fixing row names, a `source` the desk's fixings policy
    orders for that index, and the printed `value`. `election` takes `{instrument, choice}` and
    `determination` `{subject, ruling}`. A knock, an expiry or an accrual follows from these and is
    refused by name, as is a price printed at zero or below. `effective_time`
    (`YYYY-MM-DDTHH:MM:SS.ffffffZ`) is when the fact is true, which orders a restated print. `actor`
    is the seat it is filed under - a print is the firm's and wants `mark` over the whole firm, an
    election `book` at the node its position sits at, which the record reads and no caller states.
    Answers `{recorded: {lsn}, event_type}`, and saying one thing twice is one fact.
    """
    return service().call('POST', '/book/lifecycle', json=dict(
        {'event_type': event_type, 'body': body},
        **_stated(actor=actor, effective_time=effective_time)))


@MCP.tool(annotations=READ_ONLY)
def book_cash(date: str | None = None, actor: str | None = None) -> dict:
    """The money the record's settlements moved: every movement under the settlement system's
    reference - its kind, what it settled, the asset, the signed amount and the value date - and
    the balances they sum to per kind, subject and asset, so collateral held under an agreement is
    one row. `date` (`YYYY-MM-DD`) reads the movements settled on or before it. A restated movement
    stands in place of the one it corrects. `actor` is the seat reading. 404 on a box that
    records nothing.
    """
    return service().call('GET', '/book/cash', params=_stated(date=date, actor=actor))


@MCP.tool(annotations=READ_ONLY)
def collateral_calls(date: str | None = None, actor: str | None = None) -> dict:
    """What each agreement's CSA calls on a marked close (`date`, `YYYY-MM-DD`, the book's own day
    where none is named), in the agreement's currency: the exposure - the positions under it at
    that close as the netting set's own recursion reads them - the support its terms require, the
    collateral and the margin held per asset as of that day, kept apart, and the `call`, signed
    from the bank's side (received positive), its `direction` `call` where the bank receives and
    `post` where it posts, none where the difference clears neither minimum transfer; a call nobody
    can work out is null with `unknown` naming why. Settle one with `file_status` - kind
    `collateral`, the agreement as subject, the amount the call names - and `worklist` drops it
    once filed, whatever its value date. Refused on a day with no marks. `actor` is the seat
    reading. 404 on a box that records nothing.
    """
    return service().call('GET', '/book/collateral', params=_stated(date=date, actor=actor))


@MCP.tool()
async def mark_book(actor: str | None = None, wait_seconds: float = 120.0,
                    ctx: Context = None) -> dict:
    """Mark the book at the close: one unit of every instrument it holds or traded since its last
    marks, valued on the close standing under the market the desk designated for `pnl`, and the
    run attested - which is what every later P&L reads that close from. Call it once the day's close
    is declared. Refused where no market is designated for `pnl`, where it stands on no close or on
    a close for another day than the book's, and for a day behind the last one marked - marks run
    forward. Waits up to `wait_seconds` for the run, so the P&L reads the marks once this answers:
    the run's replay tuple, stats and table shapes beside the `market`, `close_lsn` and the count
    of `instruments` marked.
    """
    submitted = await asyncio.to_thread(service().call, 'POST', '/book/marks',
                                        json=_stated(actor=actor))
    outcome = await _await_result(submitted['result_id'], wait_seconds, ctx)
    return dict(outcome, **{key: submitted[key] for key in ('market', 'close_lsn', 'instruments')})


@MCP.tool(annotations=READ_ONLY)
def book_pnl(start: str | None = None, end: str | None = None, portfolio: str | None = None,
             agreement: str | None = None, client: str | None = None,
             explain: bool = False, actor: str | None = None) -> dict:
    """The desk's P&L between the marks of two days (`YYYY-MM-DD`), or from the last marks to now
    where `end` is not named: per position and in total, the values at both ends - each close being
    the end of its day, so less what the day paid - the premiums the fills paid, the payments and
    fees, `existing` (what the positions held moved) and `trading` (what the window's fills earned
    against the end), and the P&L split realised - at average cost, the cash included - and
    unrealised. Narrowed by a `portfolio` path, an `agreement` or a `client` entity with everything
    grouped under it. A figure nobody can know - a fill with no price, a payment nothing determined
    or settled - is named under `unknown`, a total over it is null and `complete` is false; a
    settlement that moved another amount than the payment it settles determines is booked at the
    determined amount and listed under `breaks`.
    `explain` adds why the held positions moved - the carry of the start's book to the end's day,
    the market move per risk factor off the start's sensitivities, and the residual - at the price
    of three more valuations. `actor` is the seat reading, every total summed over what it sees.
    """
    return service().call('GET', '/book/pnl', params=_stated(
        start=start, end=end, portfolio=portfolio, agreement=agreement, client=client,
        explain=explain or None, actor=actor))


@MCP.tool(annotations=READ_ONLY)
def describe_agreements() -> dict:
    """The paper the book trades under: every legal entity the record declares, with the parent it
    is grouped under, and every agreement with the entity it is with, the document's kind, the
    netting-set terms its positions compile into and whether a run collateralises them
    (`collateralized`) - what the `document` seat declares with
    `declare_legal_entity` and `declare_agreement` - and `settlement_state`, the netting set's
    fields and collateral columns no agreement's terms may state. 404 on a box that records
    nothing.
    """
    return dict(service().call('GET', '/book/entities'),
                **service().call('GET', '/book/agreements'))


@MCP.tool(annotations=READ_ONLY)
def book_positions(actor: str | None = None) -> dict:
    """The book's positions as the record holds them, one row per instrument under an agreement and
    a portfolio: the counterparty, the NET quantity in units of the deal - the fills summed, 1 being
    the deal as written - the clips behind it, `booked` the fill that booked it first (who, where on
    the record, when; a restrike keeps it), and `deal_paths` where the book file holds it, the
    path `read_deal` and `amend_deal` take. The portfolio is a path whose top node is the book; the
    client is the counterparty, whose name and parent `describe_agreements` gives. A position closed
    to zero stands at a quantity of 0 where it sat, and one whose deal has expired stands until the
    day it settles - `expired` and `settles` say when - and rolls off on that day. `status` is what
    its tickets read - `rejected`, `pending`, `approved` or `unticketed` - and `pending` the
    quantity awaiting a second seat; a pending trade prices and settles all the same. Each of its
    `tickets` names the `tier` it fell in, `fell_past` why it fell there - the sentence of each
    check an earlier tier failed on - and, while pending, `waits_on`, why it waits. `actor` is the
    seat reading. 404 on a box that records nothing.
    """
    return service().call('GET', '/book/positions', params=_stated(actor=actor))


@MCP.tool(annotations=READ_ONLY)
def worklist(actor: str | None = None) -> dict:
    """What waits on the seat asking, read off what stands - nothing here is filed, so a row leaves
    the moment its fact lands. Six lists, each what one verb acts on where `actor` holds it, each
    row `{kind, what, key, lsn, since}` with `key` what the clearing fact is filed against:
    `pending` tickets this seat may sign (`approve_ticket`) - never its own under four eyes, which
    with every ticket it is shown and may not sign are `elsewhere`, each with `may` saying why;
    `payments` due by the book's day nobody settled, each with its `amount`, `currency` and the
    `agreement` it is due under where it is one, which a settle seat clears with `file_status`
    `settled`, kind `payment`;
    `unconfirmed` fills, which a confirm seat clears with `file_status` `confirmed` against the
    key; the collateral `calls` the day's marks make, keyed by agreement with the `amount` and
    `currency` called, which a collateral seat clears with `file_status` `settled`, kind
    `collateral` or `margin`, the agreement as subject; `unmarked` closes (`mark_book`, for a mark
    seat); and `rejected` trades still standing, which a book seat closes out. `counts` says how
    many each list holds; each list answers the newest 200. A ticket's row names its trade -
    `reference`, `deal_path`, `client`, `agreement`, `booker` - and a pending one its `tier`,
    `fell_past` why it fell there, the `price` and `currency` it was booked at and `waits_on`, why
    it waits. `seat` is the seat the
    lists wait on, null where none was named and they are everything waiting on anyone. 404 on a
    box that records nothing.
    """
    return service().call('GET', '/book/worklist', params=_stated(actor=actor))


@MCP.tool(annotations=READ_ONLY)
def describe_seat(actor: str | None = None) -> dict:
    """What the seat asking may do on this desk, before it tries: `seat`, the seat its acts are
    filed under, and `verbs` - for each verb a grant names (`book`, `approve`, `mark`, `settle`,
    `confirm`, `collateral`, `document`, `validate`, `admin`) the nodes of the book's tree it
    holds it at, a node reaching every path under it; null where it holds it everywhere, empty
    where nowhere - and `demands`, the verb each act the record files asks under the map in force
    (a `fill` `book`, an `approval` `approve`, a `confirmed` transition `confirm`, unless the
    desk's capabilities document moves one). A request naming no seat where grants are in force is
    refused, as its acts would be. 404 on a box that records nothing."""
    return service().call('GET', '/book/seat', params=_stated(actor=actor))


@MCP.tool(annotations=READ_ONLY)
def describe_portfolios() -> dict:
    """The book's declared tree: every node the record declares, with the seat that declared it and
    where. Once any node of a book stands, a booking sits in the book itself or a declared node.
    404 on a box that records nothing."""
    return service().call('GET', '/book/portfolios')


@MCP.tool()
def declare_portfolio(path: str, actor: str | None = None) -> dict:
    """Declare a node of the book's tree - `<book>/FX/Options` - judged at its parent: a seat with
    `admin` there declares it, and the book itself is the firm's to declare. Answers
    `{recorded: {lsn}, path}`; declaring one path twice is one fact."""
    return service().call('POST', '/book/portfolios',
                          json=dict({'path': path}, **_stated(actor=actor)))


@MCP.tool()
def declare_legal_entity(entity: str, name: str, parent: str | None = None,
                         actor: str | None = None) -> dict:
    """Declare a legal entity the book may trade with - the credit unit a client is - as the
    `document` seat, which files an `entity_declared`.

    `entity` is its id as the legal system knows it, `name` what a reader is shown, `parent` the
    entity it is grouped under, if any. A second declaration under one id restates it, and a name
    identifies one entity: one standing under another id is refused, naming that id. Answers
    `{recorded: {lsn}, entity}`.
    """
    return service().call('POST', '/book/entities', json=dict(
        {'entity': entity, 'name': name}, **_stated(parent=parent, actor=actor)))


@MCP.tool()
def declare_agreement(agreement: str, entity: str, kind: str, terms: dict,
                      actor: str | None = None) -> dict:
    """Declare an agreement with a declared legal entity as the `document` seat, which files an
    `agreement_declared`: its id, the document's `kind` as you label it (an ISDA with a CSA, a
    GMRA, a long-form confirmation), and its `terms` - a `NettingCollateralSet` block
    (`describe_instrument_type`), which every position booked under the agreement is compiled into.

    The terms carry the paper and nothing else: no Children - a position is a fill booked against
    the agreement - and no balance or collateral amount, which arrive as settlements. The set's
    counterparty is `entity`, written into `Credit_Support_Amounts.Counterparty`, so leave it out;
    terms naming another are refused. The engine judges them before anything is recorded and a
    refusal names what to fix. A second declaration under one id restates the agreement, and an id
    belongs to one entity: one standing under another entity is refused, naming that entity.
    Answers `{recorded: {lsn}, agreement, terms}`, `terms` the address they are stored at.
    """
    return service().call('POST', '/book/agreements', json=dict(
        {'agreement': agreement, 'entity': entity, 'kind': kind, 'terms': terms},
        **_stated(actor=actor)))


@MCP.tool(annotations=READ_ONLY)
def describe_official_calculation() -> dict:
    """What the desk's OFFICIAL runs run at, as the record declares it: `calculations`, per
    calculation type the block of that type's fields its official run runs at - `CreditMonteCarlo`
    is the one type that has one - with the `blob` the document is stored under and the `lsn` of
    its declaration, each null where none is declared. A named calculation (`describe_calculations`)
    and a what-if's overrides are a seat's own and never official. 404 on a box that records
    nothing.
    """
    return service().call('GET', '/book/calculations')


@MCP.tool()
def declare_official_calculation(calculations: dict, actor: str | None = None) -> dict:
    """Declare what the desk's official runs run at as the `admin` seat, which files a
    `policy_declared`: `calculations` is `{"CreditMonteCarlo": {...}}`, a block of that type's own
    fields (`describe_calculation_type`) - `Batch_Size` and `Simulation_Batches` for the paths,
    `Random_Seed`, and `Funding_Valuation_Adjustment` with its `Funding_Cost_Interest_Curve` and
    `Funding_Benefit_Interest_Curve` - and no `Object`, the type being the key. A container such as
    `Funding_Valuation_Adjustment` is stated whole.

    Never choose these numbers or curves yourself: they are the desk's policy, so declare what you
    are told. The engine judges the block before anything is recorded and a refusal names the field
    to fix; a curve the book does not carry is refused by the run that reads it. From then on
    `recalc_xva`, `xva_increment` and a credit `run_calculation` read the block over the book's own
    calculation, a seat's own dials going over it. The last declaration stands, and `{}` declares
    that nothing is stated. Answers `{recorded: {lsn}, blob}`.
    """
    return service().call('POST', '/book/calculations', json=dict(
        {'calculations': calculations}, **_stated(actor=actor)))


@MCP.tool(annotations=READ_ONLY)
def describe_credit_limits() -> dict:
    """The credit limit each counterparty is held to, as the record declares it: `limits`, per
    legal entity `{amount, currency}`, with the `blob` the document is stored under and the `lsn`
    of its declaration, each null where none is declared. What stands against a limit - usage,
    headroom, a breach - is `xva_view`'s `limits`. 404 on a box that records nothing.
    """
    return service().call('GET', '/book/limits')


@MCP.tool()
def declare_credit_limits(limits: dict, actor: str | None = None) -> dict:
    """Declare the desk's counterparty credit limits as the `admin` seat, which files a
    `policy_declared`: `limits` is `{"<entity id>": {"amount": 5000000, "currency": "USD"}}`, an
    amount above zero per declared legal entity (`describe_agreements` lists them), in the
    currency the book's credit run reports in - a limit in another is never converted and reads
    not comparable. A limit on a parent entity is drawn on by every entity under it.

    Never choose a limit yourself: it is the desk's credit policy, so declare what you are told.
    The document stands WHOLE: state every limit that is to stand, one left out being withdrawn,
    and `{}` withdrawing all. An entity the record does not declare is refused by name with
    nothing recorded. Nothing is blocked by a limit: `xva_view` reports usage against it and
    `xva_increment` whether a trade would breach it. Answers `{recorded: {lsn}, blob}`.
    """
    return service().call('POST', '/book/limits', json=dict(
        {'limits': limits}, **_stated(actor=actor)))


@MCP.tool()
def declare_survival_curve(counterparty: str, spread: float | list[dict],
                           recovery: float | None = None, actor: str | None = None) -> dict:
    """Write a counterparty's survival curve - `SurvivalProb.<counterparty>`, what its netting
    set's CVA discounts by - from the credit spread the CVA desk states: `spread` in basis points,
    a number for a flat curve or rows `[{tenor, spread}]` (`tenor` like `5Y`) for a term structure,
    and `recovery` the share recovered on default, the factor's convention where unstated
    (`describe_factor_type('SurvivalProb')`), which the answer names. Each spread is read as the
    hazard spread / (1 - recovery), the last one's line run on to fifty years, so a flat spread is a
    flat hazard to fifty years. Never look a spread up or guess one: it is the desk's number.
    On a recorded desk the counterparty is a declared legal entity and the seat holds `mark`.
    Answers `{written, factor, recovery, curve, horizon}`, `horizon` the years the last line runs
    on to; `recalc_xva` then prices the set.
    """
    return service().call('POST', '/book/survival', json=dict(
        {'counterparty': counterparty, 'spread': spread}, **_stated(recovery=recovery, actor=actor)))


@MCP.tool()
def export_settlements(due_before: str, actor: str | None = None) -> dict:
    """The settlement file for one day: every payment and premium due by `due_before` that nobody
    has settled, each row with its key, `kind` (`payment` or `premium`), deal, leg, currency, amount
    and the `agreement` it is due under where it is one, and the totals by currency - each row a
    settle seat clears with `file_status` `settled` under the row's own kind, naming that agreement.

    `due_before` is required - a settlement file is struck FOR a day, and one carrying every future
    payment would instruct the whole book.

    IT NAMES NO MARKET, and cannot: which board a settlement file is struck on is the one the desk
    DESIGNATED for the export, and a desk that designated none is refused with the declaration that
    fixes it. A row whose amount is not yet determined - a floating coupon before its resets fix -
    and a row naming no currency refuse BY NAME, because instructing a payment of zero is a wrong
    payment rather than a missing one. The answer carries `values_hash` (the board the file was
    struck on), `market {name, values_hash, lsn}` (the name it was resolved under and where that
    name stands), `due_before`, `totals`, `rows` and `count`.
    """
    return service().call('POST', '/book/settlements',
                          json=dict({'due_before': due_before}, **_stated(actor=actor)))


@MCP.tool(annotations=READ_ONLY)
def poll_result(result_id: str, stat: str | None = None) -> dict:
    """Where a run got to: `queued`/`running`, or `done` with the replay tuple (`plan_hash`,
    `values_hash`, `engine_version`, `seed`, `quasi_seed`), the run's stats and each table's SHAPE
    as one line (fetch cells with `fetch_table`), or `error` with the message. Never the cells
    themselves. `stat` names the outcome a tool answers with - the `hint` of a tool that stopped
    waiting passes it on - and a run that filed one answers that outcome, with `result_id` and
    `status` beside it, as the tool would have."""
    return _summary(_raw_result(result_id), result_id, stat)


@MCP.tool(annotations=READ_ONLY)
def fetch_table(result_id: str, table: str, offset: int = 0, limit: int = 50) -> dict:
    """One table of a finished run, paged and CAPPED: at most 200 rows per call, and a table wider
    than 60 columns - a scenario cube, a wide book's `Greeks_Second` - is refused by name with
    nothing fetched, the refusal saying the desk downloads it whole. Table names come
    from the result summary; a grouped table is a path (`cashflows/USD`)."""
    shape = _raw_result(result_id).get('tables', {}).get(table)
    if shape is None:
        raise ToolError('result {} has no table {!r} - poll_result lists them'.format(
            result_id, table))
    if len(shape['columns']) > MAX_TABLE_COLUMNS:
        raise ToolError('{} is {} columns wide, more than the {} read here as text, so nothing was '
                        'fetched: the desk downloads it whole as a CSV file, and a narrower table '
                        'of the run reads here - poll_result lists them.'.format(
                            table, len(shape['columns']), MAX_TABLE_COLUMNS))
    return service().call('GET', '/results/{}/{}?offset={}&limit={}'.format(
        result_id, table, offset, min(int(limit), MAX_PAGE_ROWS)))


@MCP.tool(annotations=READ_ONLY)
def deal_values(result_id: str) -> dict:
    """`{total, deals}` off a finished base valuation's `mtm` table - the question after every
    booking and every what-if: `total` the whole book, the table's root row, and `deals` each node
    under it as `{reference, value}` in book order, a reference being no deal's identity. The SHAPE
    is checked before anything is fetched: a Monte Carlo's mtm is a scenario cube and is refused
    unfetched. A run that failed answers its own error."""
    raw = _raw_result(result_id)
    if raw.get('status') == 'error':
        raise ToolError('run {} failed: {}'.format(result_id, raw.get('error')))
    shape = raw.get('tables', {}).get('mtm')
    if shape is None or 'Reference' not in shape['columns'] or 'Value' not in shape['columns']:
        raise ToolError('this result carries no per-deal mtm frame - it is not a base valuation. '
                        'Its tables: {}'.format(', '.join(raw.get('tables', {})) or 'none'))
    if shape['rows'] > MAX_DEAL_ROWS:
        raise ToolError('the mtm frame holds {} deals - too many to hold as text; page it with '
                        'fetch_table instead'.format(shape['rows']))
    page = service().call('GET', '/results/{}/mtm?offset=0&limit={}'.format(
        result_id, shape['rows']))
    reference, value = page['columns'].index('Reference'), page['columns'].index('Value')
    root, *nodes = page['data']
    return {'total': root[value],
            'deals': [{'reference': row[reference], 'value': row[value]} for row in nodes]}


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
        '4. book_quote(quote_id) ONLY on the user\'s word that the CLIENT ACCEPTED - that is what '
        'records the quote and books the trade. Nothing is recorded before it. A quote is firm '
        'for a window; past it, re-quote.\n'
        '5. If the answer carries a tier with waits_on, the trade is booked PENDING and the '
        'desk\'s policy wants another seat: have it call approve_ticket(actor, quote_id) - never '
        'accept again. With `refused` BESIDE `accepted` no tier admits the ticket - say which '
        'bound it broke, and do not re-quote. A moved market is reported under `market` and is '
        'never a refusal.'.format(
            structure, pair, notional, notional_currency, expiry,
            ', netting_set {!r}'.format(client) if client else ''))


@MCP.prompt()
def import_a_legacy_book(counterparty: str) -> str:
    """Take a counterparty's trades the book file holds and nobody booked onto the record, and
    mark them."""
    return (
        '1. book_reconcile - every deal the book file holds that nobody booked is listed under '
        'in_file_not_in_record with its deal_path and reference; take those under the netting '
        'set "{0}" (read_book says which set a path sits in).\n'
        '2. For each: book_deal(deal_path=its deal_path, reference=its reference, quantity=1, '
        'execution_reference=its reference, price=what it was done at where the user says). The '
        'record takes the fill and the file only its pin, so no number moves; a tier with '
        'waits_on books it PENDING for another seat.\n'
        '3. book_reconcile again: none of them is left under in_file_not_in_record.\n'
        '4. price_candidate, then book_risk_summary for the mark and the gradient. Closing one '
        'later is a booking of -1: book_deal(its terms, parent_reference="{0}", quantity=-1, a new '
        'execution_reference).\n'
        '5. recalc_xva(["{0}"]) only if the user asks for it - it is minutes.'.format(
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


@MCP.resource('derivus://book', mime_type='application/json')
def book_document() -> dict:
    """The live book as the job document itself - every deal and all its market data, verbatim.
    `read_book` is the summary a model should normally hold; this is the file."""
    return _resource('/book')['document']


@MCP.resource('derivus://schema/{store}', mime_type='application/json')
def schema_store(store: str) -> dict:
    """One store of the engine's declarations, whole - what every describe_* tool reads one entry
    out of, for a host that wants the vocabulary in front of it - any `/schema` publishes."""
    stores = {key: value for key, value in _resource('/schema').items() if isinstance(value, dict)}
    if store not in stores:
        raise ResourceError('{!r} is not a schema store - one of: {}'.format(
            store, ', '.join(stores)))
    return stores[store]


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


def orient():
    """Install the orientation of the mode the service is in, read off its status, whose `spine` is
    null on a desk that keeps no record. A service that does not answer leaves what stands."""
    try:
        recording = service().call('GET', '/book/status')['spine'] is not None
    except ToolError:
        return
    # the SDK reads this when a session opens and offers no setter for it
    MCP._lowlevel_server.instructions = orientation(recording)


def main():
    """Serve the tools over stdio, for an MCP host that launches this as a subprocess."""
    configure()
    orient()
    MCP.run(transport='stdio')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
