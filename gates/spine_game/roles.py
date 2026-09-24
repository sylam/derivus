########################################################################
# Copyright (C)  Shuaib Osman (vretiel@gmail.com)
# This file is part of Derivus.
#
# Derivus is free for noncommercial use under the terms of the PolyForm
# Noncommercial License 1.0.0. You should have received a copy of the license
# along with Derivus. If not, see
# <https://polyformproject.org/licenses/noncommercial/1.0.0>.
#
# Derivus is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
########################################################################

"""The desk: seven seats, what the document scopes each of them for, and the day they play.

A ROLE IS A CALLABLE over the table, which is the whole of the interface: the scripted players
below are functions, and a host driving `DV_MCP` against the same hub plays the same day by handing
one of its own in. They reach the record the way a model does - the binding's tool FUNCTIONS over
one `Service` bound to a hub on an ephemeral port - so nothing here knows a verb the desk does not
have, and the seat is the `actor` a call names rather than a connection of its own.

One act has no binding verb and says so: a STANDING run is not something a model may declare at
all - `/execute` has no tool, by the binding's own shape - so financial control posts the close's
own valuation over the transport directly, and the script names it. Everything else a seat does
here is a tool a host has.

The table writes the SCRIPT as it goes - who asked what, in which lane, and what came back - because
the record answers what HAPPENED and the oracle needs what was ASKED. The suite's own fixtures are
imported where they are used rather than at the top, so this module loads for its seats and its
grants without the engine behind them.
"""
import asyncio
import json
import time
from pathlib import Path

from derivus import spine as seam
from derivus_mcp import server as binding
from derivus_spine.capability import ANY_BOOK, CAPABILITIES_POLICY, canonical_document
from derivus_spine.verbs import CURIOSITY, STANDING
from derivus_spine.vocabulary import ADMIN, APPROVE, BOOK, FIRM_CLASS, MARK, VALIDATE

#: The seats at this desk. Pseudonymous references, as every subject in the record is.
SALES = 'subject-sales'
TRADER = 'subject-trader'
RISK = 'subject-risk'
CONFIRMATIONS = 'subject-confirmations'
SETTLEMENTS = 'subject-settlements'
CONTROL = 'subject-control'
AUDIT = 'subject-audit'
#: Who is not at it. Named here so red and the oracle mean one subject by it.
STRANGER = 'subject-nobody'

#: The book the desk's own document names, which is the scope every desk grant is over. Control is
#: scoped over `*` instead: the close, the mark and a standing attestation are firm-level facts.
DESK_BOOK = 'service'

#: seat -> the `(verb, scope)` grants it holds. THE SIX IMPLY NOTHING ABOUT EACH OTHER, so a seat
#: that prices says `validate` and a seat that books says both, and every DESK grant is over the
#: desk's own book - a ticket is the plan this book would have, so one `approve` grant answers both
#: the tier's automatic signature and the second seat's own. Control is scoped over `*` because the
#: mark, the close and a standing attestation are firm-level facts.
GRANTS = {
    SALES: ((VALIDATE, DESK_BOOK),),
    TRADER: ((BOOK, DESK_BOOK), (VALIDATE, DESK_BOOK), (APPROVE, DESK_BOOK)),
    RISK: ((APPROVE, DESK_BOOK), (VALIDATE, DESK_BOOK)),
    CONFIRMATIONS: ((BOOK, DESK_BOOK), (VALIDATE, DESK_BOOK)),
    SETTLEMENTS: ((BOOK, DESK_BOOK), (VALIDATE, DESK_BOOK)),
    AUDIT: ((VALIDATE, DESK_BOOK),),
    CONTROL: ((ADMIN, ANY_BOOK), (MARK, ANY_BOOK), (BOOK, ANY_BOOK), (VALIDATE, ANY_BOOK)),
}

#: The workflow the day runs under: one tier, wanting a human other than the booker, and the board
#: the settlement export is struck on.
TIERS = {'tiers': [{'name': 'desk', 'four_eyes': True}],
         'designations': {'settlement_export': 'official'}}

#: The position the desk opens the day already holding, booked through the hub rather than written
#: into the file: a legacy trade reaches this record the way every other does, there being no import
#: verb, and the book and the record then agree from act one.
CARRIED = {'Object': 'FixedCashflowDeal', 'Reference': 'CF1', 'Currency': 'ZAR',
           'Discount_Rate': 'ZAR', 'Calendars': None,
           'Payment_Date': {'.Timestamp': '2026-06-28'}}

#: The board the desk marks and closes on.
OFFICIAL = 'official'
#: The structure sales quotes, and the client it is quoted for.
STRUCTURE = 'ZeroCostCollar'
CLIENT = 'CLIENT_A'

#: The day the desk closes on - the book's own base date, spelled once so the check and the
#: declaration ask about one day.
CLOSE_DATE = '2024-06-28'
#: What a settlement file struck for everything the book owes asks for.
FAR_FUTURE = '2099-01-01'
#: The states the two back-office seats move a row to. `settled` is the diary's own word for a
#: payment that has been made; `confirmed` is a state the diary reads as no state at all.
SETTLED, CONFIRMED = 'settled', 'confirmed'
#: What `/book/reconcile` can say the file and the record disagree about - the reading the desk's
#: own control against a second writer is, and what audit writes into the script.
DIVERGENCES = ('in_record_not_in_file', 'in_file_not_in_record', 'quantity_mismatch')
#: How long a seat waits on the hub's compute before it says so rather than hanging the day.
WAIT_SECONDS = 600.0


def document():
    """The capabilities document the day runs under, canonical.

    Every seat's READ row is there too: a class key is wrapped to whoever the document admits, and
    the blob read a replica pulls with is held to the same rows.
    """
    return canonical_document({
        'grants': sorted(({'subject': seat, 'verb': verb, 'book': scope}
                          for seat, held in GRANTS.items() for verb, scope in held),
                         key=lambda grant: (grant['subject'], grant['verb'])),
        'read': [{'subject': seat, 'class': FIRM_CLASS} for seat in sorted(GRANTS)]})


class Table:
    """The desk mid-day: the hub every seat books through, and the script it writes as it goes.

    ONE TRANSPORT for every seat, bound once - what makes a seat a seat here is the `actor` a call
    names and never a connection of its own, which is also the posture the page states: `actor` is
    attribution, and the honest control is that the hub is bound to localhost.
    """

    def __init__(self, url, home):
        self.url, self.home, self.acts = url, Path(home), []
        #: what one role leaves for the next: the quote ids struck, the rows already settled
        self.standing = {}
        binding.configure(base_url=url)

    def did(self, seat, act, **marks):
        """Record one act in the script and answer it.

        `marks` is what the oracle reads: `lane` and `replay` for a run, `denied` for an append the
        writer refused, `refused` for one turned away short of it.
        """
        self.acts.append(dict({'seat': seat, 'act': act}, **marks))
        return self.acts[-1]

    def hub(self, method, path, **kwargs):
        """One call on the hub over the binding's own transport - the two acts no tool exists
        for."""
        return binding.service().call(method, path, **kwargs)

    def file(self, seat, event_type, body, effective_time=None):
        """Append one fact through the hub's own writer, under `seat`.

        What this reaches for is a PRINT - `apply_lifecycle`'s own type, which the faults below
        file with a truth-time of their own and no endpoint takes. The writer still adjudicates the
        seat, which is what makes it the hub's append rather than a second writer's.
        """
        with seam.writing() as log:
            return log.append(event_type, body, actor=seat, book=DESK_BOOK,
                              effective_time=effective_time)

    def read(self, fold):
        """`fold(log)` over the hub's home, on a handle this closes. Reading never claims a home,
        so this runs beside a booking rather than behind it."""
        return seam.folded(fold)

    def settled(self, result_id):
        """The run at `result_id` once it stops moving - its whole answer, replay tuple included.

        The RAW result rather than the binding's summary, which trims the engine version out: what
        an attestation is filed under is the four coordinates, and three of them is not a tuple.
        """
        deadline = time.monotonic() + WAIT_SECONDS
        while time.monotonic() < deadline:
            ran = self.hub('GET', '/results/{}'.format(result_id))
            if ran['status'] not in ('queued', 'running'):
                return ran
            time.sleep(0.25)
        raise AssertionError('{} was still running after {:.0f}s'.format(result_id, WAIT_SECONDS))

    def script(self, day):
        """The script as it goes on disk beside the run: what was asked, by whom, and the answer."""
        return {'day': day, 'hub': self.url, 'home': str(self.home), 'acts': self.acts}


def open_the_desk(table):
    """Financial control: put the market on the book and point the official name at it.

    The mark is the book's own values under a name, so the board the day quotes on and the board
    the settlement file is struck on are one number.
    """
    from test_service import dump, fx_vol_quotes

    binding.update_market_quotes(json.loads(dump(fx_vol_quotes())))
    marked = binding.declare_market(OFFICIAL, actor=CONTROL)
    table.did(CONTROL, 'declare_market official', recorded=marked['recorded']['lsn'])


def carry_the_position(table):
    """The trader books the trade the desk opened the day holding.

    A legacy position reaches this record through the booking verb like every other - there is no
    import verb - and booking it here is what makes the day's own `/book/reconcile` reading MEAN
    something: three empty lists over a book whose every deal the record saw.
    """
    from test_service import AMOUNT

    booked = binding.book_deal(dict(CARRIED, Amount=AMOUNT), parent_reference=CLIENT,
                               quantity=AMOUNT, execution_reference='EXEC-CARRIED', actor=TRADER)
    table.did(TRADER, 'book_deal (the position carried in)', written=booked['written'],
              recorded=booked['recorded']['lsn'])


def strike_the_settlement_file(table):
    """Settlements: strike the file for what the book owes, then file what was paid.

    The export names no market - the board is the one the tiers policy designates for it - and the
    settlement is filed against the row's own DERIVED KEY, which is what invariant seven holds the
    diary against.
    """
    exported = binding.export_settlements(due_before=FAR_FUTURE, actor=SETTLEMENTS)
    table.standing['settled'] = [row['key'] for row in exported['rows'] if row['key']]
    table.did(SETTLEMENTS, 'export_settlements', lane=CURIOSITY, rows=exported['count'])
    for key in table.standing['settled']:
        landed = binding.file_status(key, SETTLED, actor=SETTLEMENTS)
        table.did(SETTLEMENTS, 'file_status settled', recorded=landed['recorded']['lsn'])


def quote(table, seat=SALES, named='the client'):
    """Quote the day's structure under `seat`, which files NOTHING, and answer the quote id.

    A desk quotes many times a day and the record holds the one that comes back, so this runs in the
    curiosity lane and the head does not move on it. The one place a structure is solved here, so
    the day and the adversary quote the same thing.
    """
    from test_service import SPOT

    struck = asyncio.run(binding.solve_structure(
        STRUCTURE, {'pair': 'USDZAR', 'expiry': '1Y', 'notional': 1_000_000.0,
                    'notional_currency': 'USD', 'floor': 1.0 / (SPOT * 0.95)},
        netting_set=CLIENT, actor=seat))
    table.did(seat, 'solve_structure {} for {}'.format(STRUCTURE, named), lane=CURIOSITY,
              quote_id=struck['quote_id'])
    return struck['quote_id']


def quote_the_client(table):
    """Sales: the day's quote, for the client the desk is working."""
    table.standing['quote'] = quote(table)


def accept_the_price(table):
    """The trader: the client took the price, so record the quote and book the trade.

    Under a workflow the acceptance is filed and the booking WAITS, which is a normal answer - the
    client's price is a fact whatever the desk's own policy then says.
    """
    waiting = binding.book_quote(table.standing['quote'], actor=TRADER)
    table.did(TRADER, 'book_quote', accepted=waiting['accepted']['lsn'], written=waiting['written'],
              refused=None if waiting['written'] else
              'the {} tier holds the booking: {}'.format(waiting['tier']['name'],
                                                         waiting['waits_on']))


def sign_the_ticket(table):
    """The second seat: sign the plan this acceptance leaves the book at, and let the trade book.

    The signature is over the TICKET, so it reaches this quote and no other, and the trader's own
    retry is what books - one seat accepts and a second signs, which is the whole of four eyes.
    """
    signed = binding.approve_quote(table.standing['quote'], RISK)
    table.did(RISK, 'approve_quote', recorded=signed['recorded']['lsn'], ticket=signed['ticket'])
    booked = binding.book_quote(table.standing['quote'], actor=TRADER)
    table.did(TRADER, 'book_quote (retry)', written=booked['written'],
              recorded=booked['recorded']['lsn'])


def confirm_the_trade(table):
    """Confirmations: move the state of what the book now owes, against the diary's own keys."""
    rows = [row for row in binding.book_diary()['rows']
            if row['key'] and row['key'] not in table.standing['settled']]
    table.did(CONFIRMATIONS, 'book_diary', lane=CURIOSITY, rows=len(rows))
    for row in rows[:1]:
        landed = binding.file_status(row['key'], CONFIRMED, actor=CONFIRMATIONS)
        table.did(CONFIRMATIONS, 'file_status confirmed', recorded=landed['recorded']['lsn'])


def close_the_day(table):
    """Financial control: check the day, attest the close's own numbers, and declare it.

    The valuation behind a close is the one lane that MINTS - a run is recorded iff its output will
    be cited by a fact, and the close is about to cite this one - and it is posted over the
    transport because declaring a standing lane is not something the binding offers a model.
    """
    verdict = binding.close_check(CLOSE_DATE)
    table.did(CONTROL, 'close_check', legal=verdict['legal'])
    live = table.hub('GET', '/book')
    submitted = table.hub('POST', '/execute',
                          json=dict(live['document'], lane=STANDING, actor=CONTROL))
    ran = table.settled(submitted['result_id'])
    table.did(CONTROL, 'execute (standing)', lane=STANDING, replay=ran,
              recorded=ran.get('attested', {}).get('lsn'))
    closed = binding.declare_close(date=CLOSE_DATE, actor=CONTROL)
    table.standing['close'] = closed
    table.did(CONTROL, 'declare_close', recorded=closed['recorded']['lsn'],
              supersedes_lsn=closed['supersedes_lsn'])


def read_the_record(table):
    """Audit: read and file nothing. Four readings, none of which moves the head."""
    head = binding.book_activity()
    table.did(AUDIT, 'book_activity', rows=len(head['rows']), lsn=head['lsn'])
    table.did(AUDIT, 'book_markets', closes=len(binding.book_markets()['closes']))
    reconciled = binding.book_reconcile()
    table.did(AUDIT, 'book_reconcile', divergences=dict(
        (name, len(reconciled[name])) for name in DIVERGENCES))


#: The day, in the order it is played: `(role, seat, callable)`. A host driving `DV_MCP` plays it by
#: handing in a callable of its own in place of one of these.
DAY = (('financial control', CONTROL, open_the_desk),
       ('trader', TRADER, carry_the_position),
       ('settlements', SETTLEMENTS, strike_the_settlement_file),
       ('sales', SALES, quote_the_client),
       ('trader', TRADER, accept_the_price),
       ('second seat', RISK, sign_the_ticket),
       ('confirmations', CONFIRMATIONS, confirm_the_trade),
       ('financial control', CONTROL, close_the_day),
       ('audit', AUDIT, read_the_record))


def declare(log, seat):
    """Put the day's capabilities document and its workflow on a fresh home, under `seat`.

    The deployment's own act, before any seat sits down: the document that scopes them, then the
    tiers policy that routes what they book. Both through the ordinary writer.
    """
    from derivus_spine import policy

    blob = log.store.put(document())
    log.append('policy_declared', {'policy': CAPABILITIES_POLICY, 'blob': blob},
               actor=seat, blob_refs=(blob,))
    return policy.declare(log, seat, policy.TIERS_POLICY, TIERS)
