// Drives the rules a client's page reads by, where the eye cannot: whether a client is ready to
// trade, an agreement's terms in words off the netting set's declaration, which agreements and
// trades are its own, its mark's side, and which of the record's rows it reads. Every check names
// the MUTATION it kills - the change that would still typecheck, still render, and still be wrong.
//
//   node web/scripts/client_check.mjs        # exits 1 on a miss

import { check, done, load, node, support } from './check.mjs';

const client = await load('client');
// the file's rows, as the page reads a set's trades off them
const blotter = await load('blotter');

// ---- the declaration, trimmed to what the rules turn on ------------------------------------------

const field = (widget, value, description, extra = {}) => ({ widget, description, value, ...extra });
const ZERO = support([[1, 0]]);
const SET = 'NettingCollateralSet';
const SCHEMA = { Instrument: {
  types: { [SET]: ['Admin', 'Set.Fields'], FXForwardDeal: ['Admin', 'Forward.Fields'] },
  sections: {
    Admin: { Object: field('Text', '', 'The deal type'), Reference: field('Text', '', 'Reference') },
    'Forward.Fields': {},
    'Set.Fields': {
      Agreement_Currency: field('Text', '', 'Agreement currency, which every amount of the agreement is in', { convention: true }),
      Collateralized: field('Dropdown', 'False', 'Collateralized', { convention: true, values: ['True', 'False'] }),
      Liquidation_Period: field('Integer', 0, 'Liquidation period, in calendar days', { convention: true }),
      Credit_Support_Amounts: field('Container', {}, 'Credit Support Amounts', { convention: true, sub_fields: {
        Counterparty: field('Text', '', 'Counterparty: the entity the agreement is with'),
        Received_Threshold: field('Table', ZERO, "The level of the set's value above which the bank receives collateral, in the agreement currency: zero or positive", { convention: true }),
        Minimum_Posted: field('Table', ZERO, 'Minimum transfer the bank posts, in the agreement currency', { convention: true }),
      } }),
      Collateral_Assets: field('Container', {}, 'Collateral Assets', { convention: true, sub_fields: {
        Cash_Collateral: field('Table', 'null', 'Cash Collateral') } }),
    },
  },
} };
const read = (terms) => {
  const reading = client.termsReading(terms, SCHEMA);
  const rows = (list) => list.map((term) => [term.words, term.value, term.unit, term.convention]);
  return [reading.collateralised, reading.stated, reading.sides, reading.currency ?? null,
          rows(reading.terms), rows(reading.conventions)];
};
const CSA = { Object: SET, Reference: 'CSA-1', Collateralized: 'True', Agreement_Currency: 'EUR',
              Liquidation_Period: 10, Credit_Support_Amounts: {
                Counterparty: 'ALDER', Minimum_Posted: support([[1, 250000], [2, 500000]]) } };
const THRESHOLD = "The level of the set's value above which the bank receives collateral";

check("a collateralised set reads every collateral term first, at its convention or stated, then "
      + 'each other term stated, in the currency or unit its description names, the rest apart',
      'a collateral term read only where stated, which drops a zero threshold off a CSA; the '
      + "credit support list's last row read, 500,000 for the 250,000 a run reads; the unit read "
      + "as its phrase, 'the agreement currency' for EUR; or a term at its convention listed with "
      + 'the stated ones',
      read(CSA),
      [true, true, ['receives', 'posts'], 'EUR',
       [[THRESHOLD, '0', 'EUR', true], ['Minimum transfer the bank posts', '250,000', 'EUR', false],
        ['Agreement currency', 'EUR', '', false], ['Liquidation period', '10', 'calendar days', false],
        ['Counterparty', 'ALDER', '', false]],
       [['Cash Collateral', 'none', '', true]]]);

check('a set whose collateral is off reads only what it states, and says the flag is its convention',
      'the collateral terms of an uncollateralised set shown at their conventions, a threshold it '
      + 'does not have; or the flag read as stated where the terms say nothing',
      read({ Object: SET, Agreement_Currency: 'USD', Credit_Support_Amounts: { Counterparty: 'KES' } }),
      [false, false, ['receives', 'posts'], 'USD',
       [['Agreement currency', 'USD', '', false], ['Counterparty', 'KES', '', false]], []]);

const ONE_WAY = structuredClone(SCHEMA);
delete ONE_WAY.Instrument.sections['Set.Fields'].Credit_Support_Amounts.sub_fields.Minimum_Posted;
check('the ways collateral moves are the sides its declared terms name, one where only one is named',
      'two-way taken as given, which calls a set declaring only what the bank receives two-way',
      client.termsReading({ ...CSA, Credit_Support_Amounts: {} }, ONE_WAY).sides, ['receives']);

// ---- which agreements, and is it ready -----------------------------------------------------------

const fwd = (reference) => node({ Object: 'FXForwardDeal', Reference: reference });
const KES_SET = { Object: SET, Reference: 'ISDA-KES', Credit_Support_Amounts: { Counterparty: 'KES' } };
const DOC = { Calc: { Deals: { Deals: { Children: [
  node(KES_SET, [fwd('F1'), fwd('F2'), fwd('F3')]),
  node({ Object: 'StructuredDeal', Reference: 'DESK' }, [
    node({ Object: SET, Reference: 'CSA-ALD', Credit_Support_Amounts: { Counterparty: 'ALD' } })]),
  node({ Object: SET, Reference: 'ISDA-OFF', Credit_Support_Amounts: { Counterparty: 'OFF' } }),
] } } } };
const ROWS = blotter.rowsByPath(blotter.toRows(DOC.Calc.Deals.Deals.Children, [], undefined));
const PAPER = [
  { agreement: 'ISDA-KES', entity: 'KES', kind: 'ISDA 2002', terms: KES_SET },
  { agreement: 'ISDA-NEW', entity: 'ALD', kind: 'ISDA 2002 with CSA', terms: { Object: SET } },
  { agreement: 'ISDA-OFF', entity: 'OFF', kind: 'ISDA 2002', terms: { Object: SET } },
];
check("a client's agreements are those legal declares with it, and on a desk keeping no record the "
      + 'netting sets naming it wherever they sit in the book',
      "another entity's agreement taken; the walk kept to the top level, which loses a set nested "
      + 'under a book node; or a set read off its type name rather than the credit support it states',
      [client.agreementsOf('ALD', PAPER, ROWS).map((row) => row.agreement),
       client.agreementsOf('ALD', null, ROWS).map((row) => [row.agreement, row.entity])],
      [['ISDA-NEW'], [['CSA-ALD', 'ALD']]]);

const ENTITIES = [{ entity: 'KES', name: 'Kestrel', parent: null },
                  { entity: 'ALD', name: 'Alderby', parent: 'KES' },
                  { entity: 'MAR', name: 'Marrick', parent: null },
                  { entity: 'OFF', name: 'Offside', parent: null }];
const RECONCILE = { terms_mismatch: [{ agreement: 'ISDA-OFF', deal_path: '2', fields: ['Netted'] }] };
const lines = (id, entities, curve) => client.readiness(
  id, entities, client.agreementsOf(id, entities && PAPER, ROWS), DOC, RECONCILE, curve)
  .map((line) => [line.tone, line.fix ?? null]);
check('a client is ready where nothing stops a booking: an agreement not yet in the book, or a set '
      + 'off its paper, books all the same; no agreement, no entity or no credit curve does not',
      'an agreement awaiting its first booking counted as stopping one, which calls a fresh client '
      + 'unready twice; the credit curve not asked for, which calls ready a client whose first '
      + 'trade the service refuses; a set off its paper read as in order; or a desk keeping no '
      + 'record asked for an entity and an agreement it has nowhere to declare',
      [lines('KES', ENTITIES, {}), lines('ALD', ENTITIES, undefined), lines('MAR', ENTITIES, {}),
       lines('OFF', ENTITIES, {}), lines('NOBODY', ENTITIES, {}), lines('KES', null, {}),
       ['KES', 'ALD', 'MAR'].map((id) => client.ready(client.readiness(
         id, ENTITIES, client.agreementsOf(id, PAPER, ROWS), DOC, null,
         id === 'ALD' ? null : {})))],
      [[['ok', null], ['ok', null], ['ok', null]],
       [['ok', null], ['note', null], ['todo', 'curve']],
       [['ok', null], ['todo', 'agreement'], ['ok', null]],
       [['ok', null], ['warn', null], ['ok', null]],
       [['todo', 'entity'], ['todo', 'agreement'], ['ok', null]],
       [['note', null], ['ok', null]],
       [true, false, false]]);

const curveless = (id, entities) => client.readiness(
  id, entities, client.agreementsOf(id, entities && PAPER, ROWS), DOC, null, undefined).at(-1).text;
check("a client with no credit curve reads its first trade refused where the book holds no set "
      + 'naming it, and a trade still booking where the book already prices one without a curve',
      'the refusal said of a set the book already holds, where the service books all the same - it '
      + 'refuses only a factor the book newly lacks; or said of neither, which tells an officer a '
      + 'fresh client trades',
      [curveless('ALD', ENTITIES), curveless('KES', null)].map((text) => /first trade is refused/.test(text)),
      [true, false]);

// ---- what the desk has on with it ----------------------------------------------------------------

const position = (path, quantity, extra = {}) => ({
  instrument: `i${path}${quantity}`, agreement: 'ISDA-KES', portfolio: 'DESK', book: 'DESK',
  counterparty: 'KES', quantity, clips: 1, first_lsn: 1, last_lsn: 1,
  deal_paths: path === null ? [] : [path], reference: 'F', object: 'FXForwardDeal', expired: null,
  settles: null, status: 'approved', pending: 0, tickets: [], ...extra });
const held = (positions) => client.heldUnder('0', ROWS, positions).map((trade) => [
  trade.row?.reference ?? null, trade.row?.units ?? null, trade.position?.quantity ?? null,
  trade.closed]);
check("an agreement's trades are its set's rows, each at the units the record holds - a row it "
      + 'holds nothing of closed - and a position the file does not hold under the set beside them',
      'a node the record holds nothing of dropped, which hides a trade closed today; the written '
      + 'amount kept where the record holds half; one node held in two portfolios read once; or a '
      + 'position the file lost dropped, so nothing says it does not price',
      [held([position('0/0', 0.5), position('0/2', 1, { portfolio: 'A' }),
             position('0/2', 2, { portfolio: 'B' }), position(null, 3)]), held(null)],
      [[['F1', 0.5, 0.5, false], ['F2', null, null, true], ['F3', 1, 1, false],
        ['F3', 2, 2, false], [null, null, 3, false]],
       [['F1', null, null, false], ['F2', null, null, false], ['F3', null, null, false]]]);
check('a position the record holds netted to nothing is a closed trade, beside one still on',
      'the position read as live because the record still lists it, which counts a trade unwound '
        + 'today among the trades on',
      held([position('0/0', 0), position('0/2', 1)]),
      [['F1', 0, 0, true], ['F2', null, null, true], ['F3', 1, 1, false]]);

check("a mark is said as the desk's side: worth something to the desk is in its favour",
      "the sign read the client's way, which tells a desk owed money that it owes it",
      [client.markText(1250.5, 'USD'), client.markText(-20213.34, 'USD'), client.markText(0, null)],
      ["1,250.50 USD, in the desk's favour", "-20,213.34 USD, in the client's favour", '0, flat']);

// ---- what happened -----------------------------------------------------------------------------

const row = (lsn, at, agreement, entity = null) => ({ lsn, record_time: at, agreement, entity,
  deal_path: null, summary: '', by: '', actor: '', event_type: '', book: null, effective_time: null });
const RECORD = [row(19, '2026-11-19T20:27:19Z', null, 'KES'), row(20, '2026-11-19T20:27:19Z', null, 'ALD'),
                row(22, '2026-11-19T20:27:20Z', 'ISDA-KES', 'KES'), row(23, '2026-11-19T20:27:20Z', 'ISDA-NEW', 'ALD'),
                row(29, '2026-11-19T20:27:28.518Z', 'ISDA-KES'), row(31, '2026-11-19T20:33:21.227Z', 'ISDA-KES'),
                row(32, '2026-11-19T20:33:22Z', 'CSA-ALD'), row(33, '2026-11-19T20:33:23Z', null)];
// the reads a client's page asks: by its entity, and by each of its agreements
const BY_ENTITY = RECORD.filter((entry) => entry.entity === 'KES');
const BY_AGREEMENT = RECORD.filter((entry) => entry.agreement === 'ISDA-KES');
check("a client's history is its reads together - by its entity and by each agreement - each row "
        + 'once though two reads answer it, newest first',
      'the reads laid end to end, which lists the agreement declared with the entity twice; or the '
        + "record's order kept, oldest first",
      [client.touching([BY_ENTITY, BY_AGREEMENT]).map((entry) => entry.lsn),
       client.touching([BY_ENTITY, []]).map((entry) => entry.lsn)],
      [[31, 29, 22, 19], [22, 19]]);

const ticket = (id, at) => ({ ticket: id, at, lsn: 0, fill: 0, actor: '', execution_reference: '',
  quantity: 1, status: 'approved', tier: null, fell_past: null, waits_on: null });
const RAN = { reference: 'ISDA-KES', as_of: '2026-11-19T22:27:28.743+02:00' };
const BOOKED = [{ agreement: 'ISDA-KES', tickets: [ticket('t1', '2026-11-19T20:27:20Z'),
                                                   ticket('t2', '2026-11-19T20:33:21Z')] },
                // a restrike's one ticket books a second position
                { agreement: 'ISDA-KES', tickets: [ticket('t2', '2026-11-19T20:33:21Z'),
                                                   ticket('t3', '2026-11-19T20:40:00Z')] },
                { agreement: 'CSA-ALD', tickets: [ticket('t4', '2026-11-19T21:00:00Z')] }];
check("trades booked under a set since its run are the tickets filed under its agreement after the "
        + "run's own instant, each once, and say so in a few words; none for a set never run",
      "a ticket of another agreement counted; one before the run counted, or the two stamps compared "
        + "as written - the record's UTC against the run's local time; or a restrike's one ticket "
        + 'counted once per position it books',
      [client.bookedSince(RAN, BOOKED), client.bookedSince({ ...RAN, as_of: null }, BOOKED),
       client.bookedText(2), client.bookedText(1)],
      [2, 0, '2 trades booked since', '1 trade booked since']);

// ---- every client as a credit officer reads it -------------------------------------------------

const CSA_KES = { Object: SET, Reference: 'CSA-KES', Collateralized: 'True',
                  Credit_Support_Amounts: { Counterparty: 'KES' } };
const CREDIT_DOC = { Calc: { Deals: { Deals: { Children: [
  node(KES_SET, [fwd('F1'), fwd('F2'), fwd('F3')]),
  node({ Object: 'StructuredDeal', Reference: 'DESK' }, [node(CSA_KES, [fwd('C1')])]),
  node({ Object: SET, Reference: 'KES-OLD', Credit_Support_Amounts: { Counterparty: 'KES' } }, [fwd('F9')]),
] } } } };
const CREDIT_ROWS = blotter.rowsByPath(blotter.toRows(CREDIT_DOC.Calc.Deals.Deals.Children, [], undefined));
const run = (reference, deal_path, extra = {}) => ({
  reference, deal_path, counterparty: 'KES', collateralized: false, note: null, status: 'done',
  cva: 10, fva: 1, currency: 'USD', as_of: '2026-11-19T22:27:28.743+02:00', result_id: reference,
  error: null, recalc: null, ...extra });
const RUNS = [run('ISDA-KES', '0'),
              run('CSA-KES', '1/0', { status: 'never run', as_of: null }),
              run('KES-OLD', '2', { status: 'failed', cva: null, fva: null,
                                     error: 'Cannot find SurvivalProb.KES' }),
              run('KES-GONE', null, { note: 'left the book', status: 'failed', cva: null, error: 'boom' }), run('OFF-1', '3', { counterparty: 'OFF' })];
const KES_PAPER = [PAPER[0], { agreement: 'CSA-KES', entity: 'KES', kind: 'ISDA 2002 with CSA', terms: CSA_KES },
                   { agreement: 'ISDA-NEW-KES', entity: 'KES', kind: 'ISDA 2002', terms: { Object: SET } }];
const CREDIT = {
  sets: RUNS, rows: CREDIT_ROWS, agreements: KES_PAPER,
  positions: [position('0/0', 1, { tickets: [ticket('t1', '2026-11-19T20:27:20Z'),
                                              ticket('t2', '2026-11-19T20:33:21Z')] }),
              position('0/2', 2, { pending: 2, status: 'pending' }),
              position('1/0/0', 1, { agreement: 'CSA-KES' })],
  marks: [{ reference: 'ISDA-KES', deal_path: '0', value: 1000 },
                          { reference: 'DESK', deal_path: '1', value: 1500 },
                          { reference: 'KES-OLD', deal_path: '2', value: -200 }],
  calls: [{ agreement: 'CSA-KES', currency: 'EUR', balance: 400, fx: { EUR: 1, USD: 0.8 } }],
};
const lined = (lines) => lines.map((line) => [line.reference, line.paper, line.collateralised,
  line.trades, line.pending, line.value, line.call?.agreement ?? null]);
const PLAIN = { ...CREDIT, agreements: client.agreementsOf('KES', null, CREDIT_ROWS),
                positions: null, calls: null };
check("a client's lines are every set naming it, under its own agreement where one is in force, then "
      + 'each agreement nothing is booked under, worth nothing; on a desk keeping no record each set '
      + 'its own agreement, its trades as the file writes them',
      'a trade the record holds nothing of counted, or a ticket that is not pending; a set the risk '
      + "marks only inside its top-level node given that node's value; a set no agreement covers "
      + 'read through the record, which counts its trades as none; an agreement nothing is booked '
      + "under read as unvalued, which blanks the client's total; or another client's set taken",
      [lined(client.creditLines('KES', CREDIT, SCHEMA)), lined(client.creditLines('KES', PLAIN, SCHEMA))],
      [[['ISDA-KES', 'ISDA 2002', false, 2, 1, 1000, null],
        ['CSA-KES', 'ISDA 2002 with CSA', true, 1, 0, null, 'CSA-KES'],
        ['KES-OLD', null, false, 1, 0, -200, null],
        ['KES-GONE', null, false, 0, 0, null, null],
        ['ISDA-NEW-KES', 'ISDA 2002', false, 0, 0, 0, null]],
       [['ISDA-KES', 'its own terms', false, 3, 0, 1000, null],
        ['CSA-KES', 'its own terms', true, 1, 0, null, null],
        ['KES-OLD', 'its own terms', false, 1, 0, -200, null],
        ['KES-GONE', null, false, 0, 0, null, null]]]);

check("a run set's line counts the trades booked under it since its run off the record's positions; "
        + 'a set with no run counts none, and a desk keeping no record nothing',
      "a set's count read off another set's tickets, or a desk keeping no record given a count",
      [client.creditLines('KES', CREDIT, SCHEMA).map((line) => [line.reference, line.booked]),
       client.creditLines('KES', PLAIN, SCHEMA).map((line) => line.booked)],
      [[['ISDA-KES', 1], ['CSA-KES', 0], ['KES-OLD', 0], ['KES-GONE', 0], ['ISDA-NEW-KES', null]],
       [null, null, null, null]]);

const after = (value, collateralised, balance) => client.exposureAfter({
  value, collateralised, call: balance === undefined ? null
    : { balance, fx: { EUR: 1, USD: 0.8 } } }, 'USD');
check('the exposure after collateral is the value to the desk less the collateral held, crossed at '
      + "the close's spots, what the bank posted adding to it, nothing where the client is owed, and "
      + "unread where a collateralised set's call is",
      'the balance taken in its own currency, or crossed the wrong way; the floor dropped, which '
      + 'reads a client owed money as a negative exposure; posted collateral subtracted; or a '
      + 'collateralised set with no close marked read as if it held nothing',
      [after(1000, false), after(-200, false), after(1500, true, 400), after(1500, true, -400),
       after(300, true, 400), after(1500, true), after(1500, true, null), after(null, false)],
      [1000, 0, 1000, 2000, 0, null, null, null]);

const SPOTS = { 'System Parameters': { Base_Currency: 'USD' }, 'Price Factors': {
  'FxRate.USD': { Spot: 1 }, 'FxRate.EUR': { Spot: 1.25 }, 'FxRate.ZAR': { Spot: 0.05 } } };
const stated = (terms) => {
  const held = client.statedHolding({ Reference: 'S', ...terms }, { Opening_Balance: { value: 0 } }, SPOTS);
  return [held.currency, held.balance, held.fx.USD, held.call,
          client.exposureAfter({ value: 1000, collateralised: true, call: held }, 'USD')];
};
check("on a desk keeping no record a set holds what its terms state - its opening balance, at its "
      + 'convention where unstated - in its balance currency, else its agreement\'s, else the base, '
      + "crossed at the market's spots; no call is worked",
      "the agreement currency read for the balance's; the spots inverted; or an unstated balance "
      + 'read as unread rather than at its convention',
      [stated({ Agreement_Currency: 'EUR', Opening_Balance: 400 }),
       stated({ Balance_Currency: 'ZAR', Agreement_Currency: 'EUR', Opening_Balance: 10000 }), stated({})],
      [['EUR', 400, 0.8, null, 500], ['ZAR', 10000, 20, null, 500], ['USD', 0, 1, null, 1000]]);

check('a collateral read refused for want of a close reads no close marked - no market designated '
      + 'for the P&L, none declared under its name, no marks for the day - and any other refusal not',
      "a record that never declared its close read as unread rather than unmarked; or a seat's "
      + 'refusal read as no close marked',
      ["this record designates no market for 'pnl', so there is nothing to strike it on",
       "no market is declared under the name 'official' in this record",
       "no marks stand for 2026-11-16 on 'official' - the days marked are none",
       'No seat is set, and this desk files every act under the seat that asks', null]
        .map(client.unmarked),
      [true, true, true, false, false]);

const KES_LINES = client.creditLines('KES', CREDIT, SCHEMA);
check("a client's sums are over every one of its book's sets or none - an unread set blanks it, and "
      + 'adjustments in two currencies, or a set that left the book, are never added',
      "an unread set's value skipped, which says a client owes less than it does; two currencies "
      + "added; or a set that left the book counted in its client's CVA",
      [client.summed([1, null, 2]), client.summed([1, 2]),
       client.runSum([KES_LINES[0], KES_LINES[3]], 'cva'), client.runSum(KES_LINES, 'cva'),
       client.runSum([KES_LINES[0], { run: { ...RUNS[0], currency: 'EUR' } }], 'fva')],
      [null, 3, { currency: 'USD', value: 10 }, { currency: 'USD', value: null },
       { currency: null, value: null }]);

check('a failed run names who puts it right off the factor the engine could not find: a credit curve '
      + "whoever marks, any other factor the market's owner",
      "every failure sent to the market's owner, which leaves a missing credit curve with nobody "
      + 'who marks one; or a sentence naming no factor given an owner it does not name',
      ['Cannot find SurvivalProb.KES', 'Cannot find InterestRate.ZAR-OIS', 'skipped a deal']
        .map((error) => client.ownerOf(error).split(':')[0]),
      ['no credit curve', 'a market factor is missing', "the engine's sentence names no factor"]);

check("what is wrong under a client's sets, worst first: a failed run, trades no agreement covers, "
      + 'trades waiting on a second seat, trades never run - nothing of a set that left the book or of '
      + 'a desk keeping no record that it cannot tell',
      "the flags in the sets' order rather than the kinds'; a set that left the book flagged; a "
      + 'never-run set with nothing on flagged; or a set on a desk keeping no record called unpapered',
      [client.wrongWith(KES_LINES).map((wrong) => [wrong.kind, wrong.reference]),
       client.wrongWith(client.creditLines('KES', PLAIN, SCHEMA)).map((wrong) => [wrong.kind, wrong.reference])],
      [[['failed', 'KES-OLD'], ['unpapered', 'KES-OLD'], ['pending', 'ISDA-KES'], ['unrun', 'CSA-KES']],
       [['failed', 'KES-OLD'], ['unrun', 'CSA-KES']]]);

check('trades booked under a set since its run are a fact of its run, never something to act on: '
        + 'they neither list it as wrong nor sort its client first',
      'a set booked under since its run listed among what is wrong, which sends its client to the '
        + 'top of the table for a book the close refreshes',
      client.wrongWith([{ reference: 'S', paper: 'ISDA', run: RUNS[0], trades: 1, pending: 0,
                          booked: 3 }]), []);

check('clients with something to act on come first, then the largest exposure after collateral, an '
      + 'unread one last, then by name',
      'the clients to act on sorted last; or an unread exposure read as the largest',
      client.actionFirst([{ name: 'B', wrong: [], after: 10 }, { name: 'A', wrong: [], after: null },
                          { name: 'C', wrong: [{}], after: 0 }, { name: 'D', wrong: [], after: 50 },
                          { name: 'E', wrong: [], after: 10 }]).map((row) => row.name),
      ['C', 'D', 'B', 'E', 'A']);

done();
