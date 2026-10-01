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

"""The bank: its seats, how the admin founds them through the CLI, and the days they play.

One book, `BANK`, and three desks declared as its nodes - FX, rates and commodities - each with a
head holding `admin` at its node who seats its own trader, salesperson and approver there by a
scoped declaration; beside them the firm's back office at `*`: product control marks, closes and
reads the P&L, legal keeps the paper, settlements, confirmations and collateral move money and
state, audit reads, and the hub's own seat runs what no request signs. Nothing founds the bank but
the verbs a deployment's operator types (`found`) - `DV_Spine init`, `enroll`, `grant`, `rewrap`,
`declare`, `portfolio`, `name` - and nothing plays it but the binding's tools.

A ROLE IS A CALLABLE over the table, which is the whole of the interface: the scripted players
below are functions, and a host driving `DV_MCP` against the same hub plays the same day by handing
one of its own in. Every seat that has a worklist reads it and acts on what it lists; where the
script still decides - which ticket an approver rejects, what a trader books - the callable says
so. One act has no binding verb and says so: a fixing is printed through the hub's writer (`file`).

The table writes the SCRIPT as it goes - who asked what, in which lane, and what came back -
because the record answers what HAPPENED and the oracle needs what was ASKED. The suite's own
fixtures are imported where they are used, so this module loads for its seats without the engine.
"""
import asyncio
import contextlib
import io
import json
import time
from pathlib import Path

from derivus import spine as seam
from derivus_mcp import server as binding
from derivus_spine import cli
from derivus_spine.capability import ANY_BOOK
from derivus_spine.verbs import CONFIRMED, CURIOSITY, SETTLED, STANDING
from derivus_spine.vocabulary import (
    ADMIN, APPROVE, BOOK, DOCUMENT, FIRM_CLASS, MARK, SETTLE, VALIDATE)
from mcp.server.mcpserver.exceptions import ToolError

#: The book, and the desks declared as its nodes.
BANK = 'BANK'
DESKS = ('FX', 'RATES', 'COMMODITIES')

#: The firm's seats. Pseudonymous references, as every subject in the record is.
FOUNDER = 'subject-admin'
HUB = 'subject-hub'
PRODUCT_CONTROL = 'subject-product-control'
LEGAL = 'subject-legal'
SETTLEMENTS = 'subject-settlements'
CONFIRMATIONS = 'subject-confirmations'
COLLATERAL = 'subject-collateral'
AUDIT = 'subject-audit'
#: Who is not at it. Named here so red and the oracle mean one subject by it.
STRANGER = 'subject-nobody'


def seat(desk, role):
    """The seat `role` holds at `desk`: `head`, `trader`, `sales` or `approver`."""
    return 'subject-{}-{}'.format(desk.lower(), role)


def node(desk, *below):
    """`desk`'s node of the book, or a path under it."""
    return '/'.join((BANK, desk) + below)


#: What the founder grants the firm: the back office at `*`, each desk's head `admin` at its node.
FIRM = dict({FOUNDER: (ADMIN,), HUB: (VALIDATE,), PRODUCT_CONTROL: (MARK, VALIDATE),
             LEGAL: (DOCUMENT,), SETTLEMENTS: (SETTLE, VALIDATE),
             CONFIRMATIONS: (SETTLE, VALIDATE), COLLATERAL: (SETTLE, VALIDATE),
             AUDIT: (VALIDATE,)})
#: What a head grants the seats of its own desk, at its node.
STAFF = {'trader': (BOOK, VALIDATE, APPROVE), 'sales': (VALIDATE,), 'approver': (APPROVE, VALIDATE)}
#: Who holds a key to every body - the founder, and audit's copy of the record.
READERS = (FOUNDER, AUDIT)
SEATS = tuple(FIRM) + tuple(seat(desk, role) for desk in DESKS
                            for role in ('head',) + tuple(STAFF))

#: The board the desk marks, closes and settles on.
OFFICIAL = 'official'
#: Each desk's workflow: the hub signs a ticket under its size cap, a second seat above it.
CAPS = {'FX': (500_000.0, 'USD'), 'RATES': (10_000_000.0, 'ZAR'),
        'COMMODITIES': (500_000.0, 'USD')}
POLICIES = {
    'tiers': {'tiers': [tier for desk in DESKS for tier in (
        {'name': desk.lower() + '-auto', 'scope': node(desk),
         'max_notional': {'amount': CAPS[desk][0], 'currency': CAPS[desk][1]}},
        {'name': desk.lower() + '-desk', 'scope': node(desk), 'four_eyes': True})],
        'designations': {'settlement_export': OFFICIAL, 'pnl': OFFICIAL}},
    # the day's board is stamped the day it is played in, years before the clock that plays it
    'firmness': {'pillar_seconds': 3.0e8},
    'fixings': {'sources': {'InterestRate.ZAR': ['SARB']}},
    'tolerance': {'tolerances': {'pnl': 1e-6}}}

#: The three days the bank closes: yesterday's book carried in, the day's trading, the settlement.
D0, D1, D2 = '2024-06-27', '2024-06-28', '2024-07-01'
#: The clients, each one agreement: `(entity, name, agreement, counterparty, currency,
#: threshold, minimum transfer)`.
CLIENTS = (('LEI-A', 'Client A Treasury', 'CSA-A', 'CPTY_A', 'USD', 100_000.0, 10_000.0),
           ('LEI-B', 'Client B Pension Fund', 'CSA-B', 'CPTY_B', 'ZAR', 500_000.0, 50_000.0),
           ('LEI-C', 'Client C Mining', 'CSA-C', 'CPTY_C', 'USD', 250_000.0, 25_000.0))
#: The structure sales quotes, and the node each head declares under its desk.
STRUCTURE = 'ZeroCostCollar'
BOOKS = {'FX': 'Options', 'RATES': 'Swaps', 'COMMODITIES': 'Metals'}
#: What each trade is booked as - its term sheet the harness's (`play.term_sheets`): `(desk, the
#: node below the desk, agreement, quantity, price per unit in dollars, the notional the ticket
#: states)`. The script decides these.
BOOKED = {'CARRIED': ('RATES', (), 'CSA-B', 1.0, 18_490_000.0, (1_000_000.0, 'ZAR')),
          'FWD-1': ('FX', (), 'CSA-A', 1.0, 0.0, (200_000.0, 'USD')),
          'DEP-1': ('RATES', (), 'CSA-B', 1.0, 0.0, (5_000_000.0, 'ZAR')),
          'FRA-1': ('RATES', ('Swaps',), 'CSA-B', 1.0, 0.0, (50_000_000.0, 'ZAR')),
          'SWAP-1': ('RATES', ('Swaps',), 'CSA-B', 1.0, 0.0, (100_000_000.0, 'ZAR')),
          'METAL-1': ('COMMODITIES', ('Metals',), 'CSA-C', 1.0, 1_150_000.0, (1_150_000.0, 'USD'))}
#: What `/book/reconcile` can say the file and the record disagree about.
DIVERGENCES = ('in_record_not_in_file', 'in_file_not_in_record', 'quantity_mismatch',
               'terms_mismatch')
#: How long a seat waits on the hub's compute before it says so rather than hanging the day.
WAIT_SECONDS = 600.0


def spine(*argv):
    """One `DV_Spine` verb as its operator types it, in this process: `(exit code, the JSON it
    printed, or the refusal it wrote)`."""
    printed, complained = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(printed), contextlib.redirect_stderr(complained):
        code = cli.main([str(part) for part in argv])
    return code, json.loads(printed.getvalue()) if not code else complained.getvalue().strip()


def refused(act, refusal=ToolError):
    """`act()`'s refusal in its own words - the binding's, or `refusal`'s where another is named -
    or what it answered instead of refusing."""
    try:
        return 'NOT REFUSED: {!r}'.format(act())
    except refusal as said:
        return str(said)


def grants(*holders):
    """The capabilities document granting each `(seat, verbs, node)` of `holders`, with the
    firm's read rows."""
    return {'grants': sorted(({'subject': subject, 'verb': verb, 'book': where}
                              for subject, held, where in holders for verb in held),
                             key=lambda row: (row['subject'], row['verb'], row['book'])),
            'read': [{'subject': subject, 'class': FIRM_CLASS} for subject in READERS]}


def firm():
    """What the founder declares: the back office over `*`, each head `admin` at its node."""
    return [(subject, held, ANY_BOOK) for subject, held in FIRM.items()] + [
        (seat(desk, 'head'), (ADMIN,), node(desk)) for desk in DESKS]


def staffed(desks):
    """The firm's rows and the seats the heads of `desks` granted at their nodes."""
    return firm() + [(seat(desk, role), held, node(desk))
                     for desk in desks for role, held in STAFF.items()]


def found(home, out):
    """Found the bank on a fresh home through the CLI, and answer the script's acts.

    The founder mints the home, enrolls every seat, grants the firm, wraps the class key to its
    readers, declares the workflow, the staleness window, the fixings' sources and the P&L's
    tolerance, and the three desks as nodes; then each desk's head declares the document again
    with its own desk's seats added at its node - a scoped declaration, which the writer admits
    because it moves nothing beyond that node; last, every seat's display name. Each document is
    a file under `out`, as an operator hands one in.
    """
    out.mkdir(parents=True, exist_ok=True)
    acts, home = [], str(home)

    def run(actor, *argv):
        code, answer = spine(*argv[:1], '--home', home, *argv[1:])
        assert not code, '{} refused: {}'.format(' '.join(map(str, argv)), answer)
        acts.append({'seat': actor, 'act': 'DV_Spine ' + argv[0], 'cli': [str(part) for part in
                                                                         argv]})

    def written(name, document):
        path = out / '{}.json'.format(name)
        path.write_text(json.dumps(document, indent=2, sort_keys=True), newline='\n')
        return path

    run(FOUNDER, 'init', '--actor', FOUNDER)
    for subject in SEATS:
        run(FOUNDER, 'enroll', '--subject', subject, '--actor', FOUNDER)
    run(FOUNDER, 'grant', '--file', written('capabilities', grants(*firm())), '--actor', FOUNDER)
    run(FOUNDER, 'rewrap', '--actor', FOUNDER)
    for name, document in POLICIES.items():
        run(FOUNDER, 'declare', name, written(name, document), '--actor', FOUNDER)
    for desk in DESKS:
        run(FOUNDER, 'portfolio', node(desk), '--actor', FOUNDER)
    for at, desk in enumerate(DESKS):
        run(seat(desk, 'head'), 'grant', '--file', written(desk.lower(), grants(*staffed(
            DESKS[:at + 1]))), '--actor', seat(desk, 'head'))
    for subject in SEATS:
        run(FOUNDER, 'name', subject, '--display', subject[len('subject-'):].replace('-', ' '))
    return acts


class Table:
    """The bank mid-day: the hub every seat books through, and the script it writes as it goes.

    ONE TRANSPORT for every seat, bound to the hub wherever it is served - what makes a seat a
    seat here is the `actor` a call names and never a connection of its own, which is also the
    posture the page states: `actor` is attribution, and the honest control is that the hub is
    bound to localhost.
    """

    def __init__(self, home, sheets):
        self.url, self.home, self.acts, self.sheets = None, Path(home), [], sheets
        #: what one role leaves for the next: the quote struck, the diary's rows, the P&L read
        self.standing = {'diary': {}}

    def point(self, url):
        """Bind every seat to the hub at `url`."""
        self.url = url
        binding.configure(base_url=url)

    def serve(self):
        """Serve the hub's app in this process on an ephemeral port, and point every seat at it."""
        from derivus import service
        from test_spine_doorbell import serving

        url, *self.serving = serving(service.app)
        self.point(url)

    def stop(self):
        """Stop serving the hub in this process."""
        server, running, held = self.serving
        server.should_exit = True
        running.join(timeout=WAIT_SECONDS)
        held.close()

    def did(self, seat, act, **marks):
        """Record one act in the script and answer it.

        `marks` is what the oracle reads: `lane` and `replay` for a run, `denied` for an append the
        writer refused, `refused` for one turned away short of it, `instructed` for the rows a
        settlement file instructed and `called` for the collateral calls a seat read.
        """
        self.acts.append(dict({'seat': seat, 'act': act}, **marks))
        return self.acts[-1]

    def file(self, seat, event_type, body, effective_time=None):
        """Append one fact through the hub's own writer, under `seat` - a PRINT, which no endpoint
        takes. The writer still adjudicates the seat, so it is the hub's append and not a second
        writer's."""
        with seam.writing() as log:
            return log.append(event_type, body, actor=seat, book=BANK,
                              effective_time=effective_time)

    def read(self, fold):
        """`fold(log)` over the hub's home, on a handle this closes."""
        return seam.folded(fold)

    def settled(self, result_id):
        """The run at `result_id` once it stops moving - the binding's summary, replay tuple
        included."""
        deadline = time.monotonic() + WAIT_SECONDS
        while time.monotonic() < deadline:
            ran = binding.poll_result(result_id)
            if ran['status'] not in ('queued', 'running'):
                return ran
            time.sleep(0.25)
        raise AssertionError('{} was still running after {:.0f}s'.format(result_id, WAIT_SECONDS))

    def work(self, seat, listed):
        """The rows of the `listed` list of `seat`'s worklist, the read recorded."""
        waiting = binding.worklist(actor=seat)
        self.did(seat, 'worklist', counts=waiting['counts'])
        return waiting[listed]

    def script(self, day):
        """The script as it goes on disk beside the run: what was asked, by whom, and the answer."""
        return {'day': day, 'hub': self.url, 'home': str(self.home), 'acts': self.acts}


# ------------------------------------------------------------------------------------------------
# The acts, seat by seat.

def book(table, reference, **changed):
    """The desk's trader books `reference` as `BOOKED` says, `changed` overriding what it states;
    the answer, recorded with its ticket and what it waits on."""
    desk, below, agreement, quantity, price, (amount, currency) = BOOKED[reference]
    stated = dict(deal=table.sheets[reference], agreement=agreement, portfolio=node(desk, *below),
                  quantity=quantity, execution_reference='EXEC-' + reference,
                  actor=seat(desk, 'trader'), price=price, notional=amount,
                  notional_currency=currency)
    stated.update(changed)
    booked = binding.book_deal(**stated)
    table.did(stated['actor'], 'book_deal ' + reference, written=booked['written'],
              recorded=(booked.get('recorded') or {}).get('lsn'), ticket=booked.get('ticket'),
              tier=(booked.get('tier') or {}).get('name'), waits_on=booked.get('waits_on'),
              refused=booked.get('refused'))
    return booked


def path_of(reference):
    """The positional path the live book carries `reference` at, read back rather than assumed."""
    return next(deal['deal_path'] for deal in binding.read_book()['deals']
                if deal['reference'] == reference)


def quote(table, seat_asking, named):
    """Quote the day's collar for client A under `seat_asking`, which files NOTHING, and answer
    the quote id - the one place a structure is solved here."""
    from test_service import SPOT

    struck = asyncio.run(binding.solve_structure(
        STRUCTURE, {'pair': 'USDZAR', 'expiry': '1Y', 'notional': 1_000_000.0,
                    'notional_currency': 'USD', 'floor': 1.0 / (SPOT * 0.95)},
        netting_set='CSA-A', portfolio=node('FX', BOOKS['FX']), actor=seat_asking))
    table.did(seat_asking, 'solve_structure {} for {}'.format(STRUCTURE, named), lane=CURIOSITY,
              quote_id=struck['quote_id'])
    return struck['quote_id']


def open_the_day(day, spot=None):
    """Product control: roll the book onto `day` - the first trading day's vol board ticked in
    before it, the surface the roll re-bootstraps - move the rand where `spot` says, and mark the
    board: the close and the settlement file stand on one number."""
    def act(table):
        if day == D1:
            from test_service import dump, fx_vol_quotes

            binding.update_market_quotes(json.loads(dump(fx_vol_quotes())))
            table.did(PRODUCT_CONTROL, 'update_market_quotes FXVolPrices.USD.ZAR')
        if day != D0:
            rolled = binding.set_base_date(day)
            table.did(PRODUCT_CONTROL, 'set_base_date ' + day, written=rolled.get('written'))
        if spot is not None:
            binding.patch_market_values({'FxRate.ZAR': {'Spot': spot}})
            table.did(PRODUCT_CONTROL, 'patch_market_values FxRate.ZAR {}'.format(spot))
        marked = binding.declare_market(OFFICIAL, actor=PRODUCT_CONTROL)
        table.did(PRODUCT_CONTROL, 'declare_market ' + OFFICIAL,
                  recorded=marked['recorded']['lsn'])

    return act


def keep_the_paper(table):
    """Legal: declare each client and its one agreement - a collateralised netting set, stating
    no collateral rows, which the engine cannot yet carry without their amounts."""
    for entity, name, agreement, counterparty, currency, threshold, minimum in CLIENTS:
        declared = binding.declare_legal_entity(entity, name, actor=LEGAL)
        table.did(LEGAL, 'declare_legal_entity ' + entity, recorded=declared['recorded']['lsn'])
        dials = dict((dial, {'.CreditSupportList': [[1, amount]]}) for dial, amount in (
            ('Independent_Amount', 0.0), ('Received_Threshold', threshold),
            ('Posted_Threshold', -threshold), ('Minimum_Received', minimum),
            ('Minimum_Posted', minimum)))
        declared = binding.declare_agreement(agreement, entity, 'ISDA 2002 with CSA', {
            'Object': 'NettingCollateralSet', 'Netted': 'True', 'Collateralized': 'True',
            'Agreement_Currency': currency, 'Balance_Currency': currency,
            'Funding_Rate': currency, 'Liquidation_Period': 0.0, 'Settlement_Period': 0.0,
            'Credit_Support_Amounts': dict(dials, Counterparty=counterparty)}, actor=LEGAL)
        table.did(LEGAL, 'declare_agreement ' + agreement, recorded=declared['recorded']['lsn'])


def plant_the_tree(table):
    """Each head: declare the one node its desk books its book under, at its own node."""
    for desk in DESKS:
        declared = binding.declare_portfolio(node(desk, BOOKS[desk]), actor=seat(desk, 'head'))
        table.did(seat(desk, 'head'), 'declare_portfolio ' + node(desk, BOOKS[desk]),
                  recorded=declared['recorded']['lsn'])


def carry_the_book(table):
    """The rates trader books the position the bank opens holding: there is no import verb, so a
    legacy trade reaches the record the way every other does."""
    book(table, 'CARRIED')


def trade_fx(table):
    """FX: sales quotes the collar, the trader accepts it - over the automatic tier's cap, so it
    books PENDING - books a small forward the hub signs itself, and is refused a booking into a
    node its desk never declared."""
    table.standing['quote'] = quote(table, seat('FX', 'sales'), 'client A')
    accepted = binding.book_quote(table.standing['quote'], actor=seat('FX', 'trader'))
    table.did(seat('FX', 'trader'), 'book_quote', accepted=accepted['accepted']['lsn'],
              written=accepted['written'], waits_on=accepted.get('waits_on'))
    book(table, 'FWD-1')
    table.did(seat('FX', 'trader'), 'book_deal into an undeclared node', refused=refused(
        lambda: book(table, 'FWD-1', execution_reference='EXEC-FWD-EXOTIC',
                     portfolio=node('FX', 'Exotics'))))


def trade_rates(table):
    """Rates: a deposit starting on the next day under the automatic tier, an FRA and a par swap -
    one reset a coupon, the first today - above it, waiting for the desk's approver."""
    for reference in ('DEP-1', 'FRA-1', 'SWAP-1'):
        book(table, reference)


def trade_commodities(table):
    """Commodities: a metal forward above the cap, restruck while it waits - the restrike is a
    ticket of its own, which the approver then signs."""
    book(table, 'METAL-1')
    amended = binding.amend_deal(path_of('METAL-1'), {'Units': 400.0}, reference='METAL-1',
                                 actor=seat('COMMODITIES', 'trader'), notional=920_000.0,
                                 notional_currency='USD')
    table.did(seat('COMMODITIES', 'trader'), 'amend_deal METAL-1 while pending',
              recorded=amended['recorded']['lsn'], ticket=amended.get('ticket'),
              waits_on=amended.get('waits_on'))


def approve_what_waits(desk, rejecting=()):
    """The desk's approver: every ticket its worklist lists as pending, approved - but those whose
    execution the script names in `rejecting`, which it rejects and leaves standing."""
    def act(table):
        approver = seat(desk, 'approver')
        for row in table.work(approver, 'pending'):
            rejected = row['what'].startswith(rejecting) if rejecting else False
            ruled = (binding.reject_ticket(row['key'], 'over the desk\'s appetite', approver)
                     if rejected else binding.approve_ticket(row['key'], approver))
            table.did(approver, 'reject_ticket' if rejected else 'approve_ticket',
                      what=row['what'], recorded=ruled['recorded']['lsn'], ticket=row['key'],
                      status=ruled['status'])

    return act


def confirm_what_waits(table):
    """Confirmations: every clip its worklist lists unconfirmed, confirmed against its own key."""
    for row in table.work(CONFIRMATIONS, 'unconfirmed'):
        landed = binding.file_status(row['key'], CONFIRMED, actor=CONFIRMATIONS)
        table.did(CONFIRMATIONS, 'file_status confirmed ' + row['what'].split()[0],
                  recorded=landed['recorded']['lsn'])


def settle_what_is_due(day):
    """Settlements: read the book's diary, strike the settlement file for `day` where the worklist
    lists payments due, and file each row it instructs with the money it moved."""
    def act(table):
        rows = binding.book_diary(actor=SETTLEMENTS)['rows']
        # what the diary determines of each row, the latest reading standing
        table.standing['diary'].update((row['key'], {
            'amount': row['amount'] if row['determined'] else None, 'currency': row['currency']})
            for row in rows if row['key'])
        table.did(SETTLEMENTS, 'book_diary', lane=CURIOSITY, rows=len(rows))
        if not table.work(SETTLEMENTS, 'payments'):
            return
        exported = binding.export_settlements(due_before=day, actor=SETTLEMENTS)
        instructed = [row for row in exported['rows'] if row['key']]
        table.did(SETTLEMENTS, 'export_settlements ' + day, lane=CURIOSITY,
                  instructed=[row['key'] for row in instructed])
        for number, row in enumerate(instructed):
            paid = dict(subject=row['key'], status=SETTLED, actor=SETTLEMENTS,
                        amount=row['amount'], asset=row['currency'], kind='payment',
                        value_date=row['due_date'], reference='PAY-{}-{}'.format(day, number))
            landed = binding.file_status(**paid)
            table.standing['paid'] = (paid, landed['recorded']['lsn'])
            table.did(SETTLEMENTS, 'file_status payment {} {}'.format(row['amount'],
                                                                      row['currency']),
                      recorded=landed['recorded']['lsn'])

    return act


def charge_a_fee(table):
    """Settlements: the broker's fee on the forward, filed on its instrument."""
    held = next(row for row in binding.book_positions(actor=SETTLEMENTS)['positions']
                if row['reference'] == 'FWD-1')
    landed = binding.file_status(held['instrument'], SETTLED, actor=SETTLEMENTS, amount=-250.0,
                                 asset='USD', kind='fee', reference='FEE-FWD-1', value_date=D1)
    table.did(SETTLEMENTS, 'file_status fee on FWD-1', recorded=landed['recorded']['lsn'])


def print_the_fixing(table):
    """Product control: the swap's first reset, printed by the source the policy names - the one
    act no binding verb takes, so it goes through the hub's writer."""
    landed = table.file(PRODUCT_CONTROL, 'fixing_observed', {
        'index': 'InterestRate.ZAR', 'date': D1, 'source': 'SARB', 'value': 0.02})
    table.did(PRODUCT_CONTROL, 'fixing_observed InterestRate.ZAR (no binding verb)',
              recorded=landed['lsn'])


def close_and_mark(day):
    """Product control: check the day, declare its close, and mark the book at it where the
    worklist lists the close unmarked."""
    def act(table):
        verdict = binding.close_check(day)
        table.did(PRODUCT_CONTROL, 'close_check ' + day, legal=verdict['legal'],
                  outstanding=[row['kind'] for row in verdict['outstanding']])
        closed = binding.declare_close(date=day, actor=PRODUCT_CONTROL)
        table.did(PRODUCT_CONTROL, 'declare_close ' + day, recorded=closed['recorded']['lsn'])
        for _ in table.work(PRODUCT_CONTROL, 'unmarked'):
            mark(table, day)

    return act


def mark(table, day):
    """Product control: mark the book at the close standing on `day` - the standing run every P&L
    of the day reads - and wait for it."""
    marking = binding.mark_book(actor=PRODUCT_CONTROL)
    ran = table.settled(marking['result_id'])
    table.did(PRODUCT_CONTROL, 'mark_book ' + day, lane=STANDING, replay=ran, status=ran['status'])


def post_the_calls(day):
    """Collateral: read the calls on the day's marks where the worklist lists any, and settle each
    that moves, as collateral under its agreement."""
    def act(table):
        if not table.work(COLLATERAL, 'calls'):
            return
        read = binding.collateral_calls(date=day, actor=COLLATERAL)
        table.did(COLLATERAL, 'collateral_calls ' + day, called=[
            dict(call, date=read['date'], marks=read['marks'], lsn=read['lsn'])
            for call in read['calls']])
        for call in read['calls']:
            if call['direction']:
                landed = binding.file_status(
                    call['agreement'], SETTLED, actor=COLLATERAL, amount=call['call'],
                    asset=call['currency'], kind='collateral', value_date=day,
                    reference='COL-{}-{}'.format(call['agreement'], day))
                table.did(COLLATERAL, '{} {:.2f} {} under {}'.format(
                    call['direction'], call['call'], call['currency'], call['agreement']),
                    recorded=landed['recorded']['lsn'])

    return act


def read_the_pnl(table):
    """Product control: what the book made - each day, the two days' window with the second's
    explain, and each desk over the window - kept for the oracle's additive question."""
    def pnl(start, end, **scope):
        return binding.book_pnl(start=start, end=end, actor=PRODUCT_CONTROL, **scope)

    days = [pnl(D0, D1), pnl(D1, D2, explain=True)]
    window = pnl(D0, D2)
    table.standing['pnl'] = {'window': window, 'days': days, 'portfolios': [
        pnl(D0, D2, portfolio=node(desk)) for desk in DESKS]}
    table.did(PRODUCT_CONTROL, 'book_pnl', windows=[(one['start']['day'], one['end']['day'],
                                                     one['total']['pnl'], one['complete'])
                                                    for one in days + [window]],
              residual=days[1]['explain']['residual'])


def leave_the_rejected(table):
    """The rates trader: its worklist names the swap its approver rejected, which still stands -
    closing it out is a trade of its own, and the day leaves it standing."""
    table.did(seat('RATES', 'trader'), 'rejected, left standing', rows=[
        row['what'] for row in table.work(seat('RATES', 'trader'), 'rejected')])


def read_the_record(table):
    """Audit: read and file nothing - the strip, the closes, where the file and the record part,
    and the positions with what their tickets read."""
    head = binding.book_activity(actor=AUDIT)
    table.did(AUDIT, 'book_activity', rows=len(head['rows']), lsn=head['lsn'])
    table.did(AUDIT, 'book_markets', closes=len(binding.book_markets()['closes']))
    reconciled = binding.book_reconcile(actor=AUDIT)
    table.did(AUDIT, 'book_reconcile', divergences=dict(
        (name, len(reconciled[name])) for name in DIVERGENCES))
    table.did(AUDIT, 'book_positions', statuses=sorted(
        (row['reference'], row['status']) for row in
        binding.book_positions(actor=AUDIT)['positions']))


#: The days, in the order they are played: `(role, seat, callable)`. A host driving `DV_MCP` plays
#: them by handing in a callable of its own in place of one of these.
DAY = (('product control', PRODUCT_CONTROL, open_the_day(D0)),
       ('legal', LEGAL, keep_the_paper),
       ('desk heads', seat('FX', 'head'), plant_the_tree),
       ('rates trader', seat('RATES', 'trader'), carry_the_book),
       ('product control', PRODUCT_CONTROL, close_and_mark(D0)),
       ('collateral', COLLATERAL, post_the_calls(D0)),
       ('product control', PRODUCT_CONTROL, open_the_day(D1)),
       ('fx desk', seat('FX', 'trader'), trade_fx),
       ('rates trader', seat('RATES', 'trader'), trade_rates),
       ('commodities trader', seat('COMMODITIES', 'trader'), trade_commodities),
       ('fx approver', seat('FX', 'approver'), approve_what_waits('FX')),
       ('rates approver', seat('RATES', 'approver'), approve_what_waits('RATES', 'EXEC-SWAP')),
       ('commodities approver', seat('COMMODITIES', 'approver'),
        approve_what_waits('COMMODITIES')),
       ('confirmations', CONFIRMATIONS, confirm_what_waits),
       ('settlements', SETTLEMENTS, settle_what_is_due(D1)),
       ('settlements', SETTLEMENTS, charge_a_fee),
       ('product control', PRODUCT_CONTROL, print_the_fixing),
       ('product control', PRODUCT_CONTROL, close_and_mark(D1)),
       ('collateral', COLLATERAL, post_the_calls(D1)),
       ('product control', PRODUCT_CONTROL, open_the_day(D2, spot=18.3)),
       ('settlements', SETTLEMENTS, settle_what_is_due(D2)),
       ('product control', PRODUCT_CONTROL, close_and_mark(D2)),
       ('collateral', COLLATERAL, post_the_calls(D2)),
       ('product control', PRODUCT_CONTROL, read_the_pnl),
       ('rates trader', seat('RATES', 'trader'), leave_the_rejected),
       ('audit', AUDIT, read_the_record))
