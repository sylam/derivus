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
const read = (terms, collateralized) => {
  const reading = client.termsReading({ terms, collateralized }, SCHEMA);
  const rows = (list) => list.map((term) => [term.words, term.value, term.unit, term.convention]);
  return [reading.currency ?? null, rows(reading.terms), rows(reading.conventions)];
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
      read(CSA, true),
      ['EUR',
       [[THRESHOLD, '0', 'EUR', true], ['Minimum transfer the bank posts', '250,000', 'EUR', false],
        ['Agreement currency', 'EUR', '', false], ['Liquidation period', '10', 'calendar days', false],
        ['Counterparty', 'ALDER', '', false]],
       [['Cash Collateral', 'none', '', true]]]);

check('a set whose collateral is off reads only what it states',
      'the collateral terms of an uncollateralised set shown at their conventions, a threshold it '
      + 'does not have',
      read({ Object: SET, Collateralized: 'False', Agreement_Currency: 'USD',
             Credit_Support_Amounts: { Counterparty: 'KES' } }, false),
      ['USD',
       [['Agreement currency', 'USD', '', false], ['Counterparty', 'KES', '', false]], []]);

check('a set reads as the service says a run collateralises it, whatever its flag spells: a set '
      + 'stating "Yes" the service reads uncollateralised reads only what it states',
      'the reading deciding off the flag again, collateralised unless it says "False", which shows '
      + 'the set thresholds no run holds it to',
      read({ ...CSA, Collateralized: 'Yes' }, false)[1].map(([words]) => words),
      ['Agreement currency', 'Liquidation period', 'Counterparty', 'Minimum transfer the bank posts']);

check('a collateralised set reads as collateralised and no more: its reading names no way its '
      + 'collateral moves',
      "a direction read off the sides the type's collateral terms name, which calls every "
      + 'collateralised set two-way: the type declares both sides for every set, and says '
      + 'nothing of this one',
      Object.keys(client.termsReading({ terms: CSA, collateralized: true }, SCHEMA)).sort(),
      ['conventions', 'currency', 'terms']);

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
      + 'netting sets naming it wherever they sit in the book, collateralised and lacking as its XVA '
      + 'row says',
      "another entity's agreement taken; the walk kept to the top level, which loses a set nested "
      + 'under a book node; a set read off its type name rather than the credit support it states; '
      + "or another set's row read for its collateral or what it lacks",
      [client.agreementsOf('ALD', PAPER, ROWS, []).map((row) => row.agreement),
       client.agreementsOf('ALD', null, ROWS, [
         { deal_path: '0', collateralized: false, lacks: ['FxRate.SEK'] },
         { deal_path: '1/0', collateralized: true, lacks: ['FxRate.CHF'] }])
         .map((row) => [row.agreement, row.entity, row.collateralized, row.lacks])],
      [['ISDA-NEW'], [['CSA-ALD', 'ALD', true, ['FxRate.CHF']]]]);

const ENTITIES = [{ entity: 'KES', name: 'Kestrel', parent: null },
                  { entity: 'ALD', name: 'Alderby', parent: 'KES' },
                  { entity: 'MAR', name: 'Marrick', parent: null },
                  { entity: 'OFF', name: 'Offside', parent: null }];
const RECONCILE = { terms_mismatch: [{ agreement: 'ISDA-OFF', deal_path: '2', fields: ['Netted'] }] };
// the market as far as a client's readiness reads it: its credit curve, where one is stated
const curved = (id, curve) => (curve ? { [`SurvivalProb.${id}`]: curve } : {});
const lines = (id, entities, curve) => client.readiness(
  id, entities, client.agreementsOf(id, entities && PAPER, ROWS, []), DOC, RECONCILE, curved(id, curve))
  .map((line) => [line.tone, line.fix ?? null]);
check('a client is ready where nothing is wanting: an agreement not yet in the book, or a set off its '
      + 'paper, is a fact and no want; no agreement, no entity or no credit curve is wanting',
      'an agreement awaiting its first booking counted as wanting, which calls a fresh client '
      + 'unready twice; the credit curve not asked for; a set off its paper read as in order; or a '
      + 'desk keeping no record asked for an entity and an agreement it has nowhere to declare',
      [lines('KES', ENTITIES, {}), lines('ALD', ENTITIES, undefined), lines('MAR', ENTITIES, {}),
       lines('OFF', ENTITIES, {}), lines('NOBODY', ENTITIES, {}), lines('KES', null, {}),
       ['KES', 'ALD', 'MAR'].map((id) => client.ready(client.readiness(
         id, ENTITIES, client.agreementsOf(id, PAPER, ROWS, []), DOC, null,
         curved(id, id === 'ALD' ? null : {}))))],
      [[['ok', null], ['ok', null], ['ok', null]],
       [['ok', null], ['note', null], ['todo', 'curve']],
       [['ok', null], ['todo', 'agreement'], ['ok', null]],
       [['ok', null], ['warn', null], ['ok', null]],
       [['todo', 'entity'], ['todo', 'agreement'], ['ok', null]],
       [['note', null], ['ok', null]],
       [true, false, false]]);

// ---- its paper against the market ----------------------------------------------------------------

const DECLARED = structuredClone(SCHEMA);
const set = DECLARED.Instrument.sections['Set.Fields'];
set.Agreement_Currency.factors = ['FxRate'];
set.Funding_Rate = field('Text', '', 'Funding rate', { convention: true, factors: ['InterestRate'] });
set.Credit_Support_Amounts.sub_fields.Counterparty.factors = ['SurvivalProb'];
set.Collateral_Assets.sub_fields.Cash_Collateral.factors = {
  Currency: ['FxRate'], Funding_Rate: ['InterestRate'] };
set.Collateral_Assets.sub_fields.Equity_Collateral = field('Table', 'null', 'Equity Collateral', {
  factors: { Equity: ['EquityPrice', 'DividendRate'] } });
const NAMING = { Object: SET, Reference: 'CSA-MAR', Agreement_Currency: 'CHF', Funding_Rate: 'USD-FUND',
  Credit_Support_Amounts: { Counterparty: 'MAR' }, Collateral_Assets: {
    Cash_Collateral: [{ Currency: 'SEK', Funding_Rate: 'USD-FUND' }, { Currency: 'CHF' }, { Currency: '' }],
    Equity_Collateral: [{ Equity: 'ACME' }] } };
const MAR_PAPER_NONE = { agreement: 'CSA-MAR', entity: 'MAR', kind: 'ISDA 2002', terms: {},
                         lacks: null };
check("a client's readiness says the gaps went unread where the book's calculation refused the walk, "
      + "in the calculation's own words, and nothing of it where the walk was asked",
      'the refusal left unsaid, which reads a client whose gaps nobody read as having none',
      [null, ["Greeks is 'Maybe'"]].map((unwalked) => client.readiness(
        'MAR', ENTITIES, [{ ...MAR_PAPER_NONE }], DOC, null, { 'SurvivalProb.MAR': {} }, unwalked)
        .filter((line) => line.tone === 'warn').map((line) => line.text)),
      [[], ["What its terms name that the market lacks is not read, the book's calculation "
            + "refusing it: Greeks is 'Maybe'."]]);

const worded = (terms, collateralized = false) => {
  const reading = client.termsReading({ terms, collateralized }, DECLARED);
  return [...reading.terms, ...reading.conventions].map((term) => term.key);
};
check("a set's own funding curve is no term a client's page reads, stated or at its convention",
      'the term read off the declaration whole, which words the curve of an adjustment no screen '
      + 'shows',
      [worded(NAMING), worded({ ...CSA }, true).includes('Funding_Rate')],
      [['Agreement_Currency', 'Credit_Support_Amounts.Counterparty',
        'Collateral_Assets.Cash_Collateral', 'Collateral_Assets.Equity_Collateral'], false]);

const MAR_PAPER = [{ agreement: 'CSA-MAR', entity: 'MAR', kind: 'ISDA 2002 with CSA', terms: {},
                     lacks: ['FxRate.CHF', 'FxRate.SEK', 'SurvivalProb.MAR'] }];
const said = (paper, factors) => client.readiness('MAR', ENTITIES, paper, DOC, null, factors)
  .slice(2).map((line) => [line.tone, line.text]);
check("a client's readiness says each factor its paper names that the market lacks as the service "
      + "walked it, a thing to put right, its credit curve on its own line alone - each a fact, "
      + 'nothing said of what a booking would do - and nothing where no book is served to walk',
      'a lacked factor dropped, which calls ready a client whose paper the market cannot price; the '
      + 'credit curve said twice; a booking sentence said again; or an unwalked agreement read as '
      + 'lacking everything',
      [said(MAR_PAPER, {}), said([{ ...MAR_PAPER[0], lacks: null }], { 'SurvivalProb.MAR': {} }),
       [MAR_PAPER, [{ ...MAR_PAPER[0], lacks: [] }]].map((paper) => client.ready(
         client.readiness('MAR', ENTITIES, paper, DOC, null, { 'SurvivalProb.MAR': {} })))],
      [[['todo', 'No credit curve SurvivalProb.MAR: its CVA cannot be read.'],
        ['todo', 'CSA-MAR names FxRate.CHF, which the market does not hold.'],
        ['todo', 'CSA-MAR names FxRate.SEK, which the market does not hold.']],
       [['ok', 'Its credit curve SurvivalProb.MAR is stated.']], [false, true]]);

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
      'the position read as live because the record still lists it, or an unfiled zero position '
        + 'hardcoded open',
      held([position('0/0', 0), position('0/2', 1), position(null, 0)]),
      [['F1', 0, 0, true], ['F2', null, null, true], ['F3', 1, 1, false], [null, null, 0, true]]);

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
  cva: 10, fva: 1, after_collateral: 900, currency: 'USD', as_of: '2026-11-19T22:27:28.743+02:00',
  result_id: reference, error: null, recalc: null, ...extra });
const RUNS = [run('ISDA-KES', '0'),
              run('CSA-KES', '1/0', { status: 'never run', as_of: null, collateralized: true,
                                      after_collateral: null }),
              run('KES-OLD', '2', { status: 'failed', cva: null, fva: null, after_collateral: null,
                                     error: 'Cannot find SurvivalProb.KES',
                                     missing: { factor: 'SurvivalProb.KES', supply: null,
                                                note: 'the CVA desk writes it from its spread' } }),
              run('KES-GONE', null, { note: 'left the book', status: 'failed', cva: null, error: 'boom' }), run('OFF-1', '3', { counterparty: 'OFF' })];
const KES_PAPER = [{ ...PAPER[0], collateralized: false },
                   { agreement: 'CSA-KES', entity: 'KES', kind: 'ISDA 2002 with CSA', terms: CSA_KES,
                     collateralized: true },
                   { agreement: 'ISDA-NEW-KES', entity: 'KES', kind: 'ISDA 2002', terms: { Object: SET },
                     collateralized: false }];
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
const PLAIN = { ...CREDIT, agreements: client.agreementsOf('KES', null, CREDIT_ROWS, RUNS),
                positions: null, calls: null };
check("a client's lines are every set naming it, under its own agreement where one is in force, then "
      + 'each agreement nothing is booked under, worth nothing; on a desk keeping no record each set '
      + 'its own agreement, its trades as the file writes them',
      'a trade the record holds nothing of counted, or a ticket that is not pending; a set the risk '
      + "marks only inside its top-level node given that node's value; a set no agreement covers "
      + 'read through the record, which counts its trades as none; an agreement nothing is booked '
      + "under read as unvalued, which blanks the client's total; or another client's set taken",
      [lined(client.creditLines('KES', CREDIT)), lined(client.creditLines('KES', PLAIN))],
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
      [client.creditLines('KES', CREDIT).map((line) => [line.reference, line.booked]),
       client.creditLines('KES', PLAIN).map((line) => line.booked)],
      [[['ISDA-KES', 1], ['CSA-KES', 0], ['KES-OLD', 0], ['KES-GONE', 0], ['ISDA-NEW-KES', null]],
       [null, null, null, null]]);

const closed = (market, stands, marked = false) => ({
  lsn: 4, events_behind: 0, positions_behind: 0, paper: 1, close: { market, stands, marked } });
const UNMARKED = [
  client.unmarked(closed(null, false), true), client.unmarked(closed('official', false), true),
  client.unmarked(closed('official', true), true),
  client.unmarked(closed('official', true, true), true),
  client.unmarked(closed('official', true)), client.unmarked(null, true)];
check('a close not there to read is a state of the desk known before asking off the record, said '
      + 'as what is missing and who puts it right - no market designated for the P&L, which a '
      + "seat holding admin designates, none declared on it, or for a read of the book's day no "
      + 'marks for it - and a read that may be asked, or a desk keeping no record, says nothing',
      "a record that never designated a market or declared its close asked anyway, its refusal in a "
      + "red box; a close declared read as none; the book's day marked read as unmarked, or its "
      + "marks judged for the P&L, which asks no day; the admin's part said of a close nobody "
      + 'declared; or a state said in the words of the machine',
      [UNMARKED.map((state) => state?.title ?? null), /A seat holding admin designates it/.test(
        UNMARKED[0]?.hint), UNMARKED.slice(1).some((state) => /admin/.test(state?.hint ?? '')),
       UNMARKED.some((state) => /designations|tiers|policy|mark_book|\{/.test(
         `${state?.title}${state?.hint}`))],
      [['No market is designated for the P&L.',
        'No close stands on the market designated for the P&L.',
        'The book is not marked for the day asked.', null, null, null], true, false, false]);
check('what a collateral number is read on is said one way on the strip, the column and a set\'s '
      + 'cell: the close of its day, that no close is marked where that is why none is read, '
      + 'and else that it is unread, in the place\'s own word',
      'a close read said to be none, or no close marked said of a read that failed, which sends '
      + 'a desk to declare a close that stands; or the place\'s own word for unread dropped',
      [client.closeRead('2026-11-20', false), client.closeRead(undefined, true),
       client.closeRead(undefined, false), client.closeRead(undefined, false, 'close unread'),
       client.closeRead('2026-11-20', true, 'close unread')],
      ['close of 2026-11-20', 'no close marked', 'unread', 'close unread', 'close of 2026-11-20']);

const sent = (said) => [
  / on the Back Office, on the market the desk designates for its P&L\./.test(said),
  /official|any other/.test(said)];
check('where no close stands the desk is sent to the Back Office, whose control declares on the '
      + 'market the desk designates for its P&L, once and for every desk; by a seat holding the '
      + 'verb the declaration demands, where the grants read name it',
      'the Back Office named for the market called official alone, which sends a desk '
      + 'designating another to the chat for a close its own control declares; or the verb left out',
      [sent(UNMARKED[1].hint), sent(client.declaredOn('mark')),
       client.declaredOn('mark').startsWith('A close is declared by a seat holding mark on the '),
       client.declaredOn().startsWith('A close is declared on the ')],
      [[true, false], [true, false], true, true]);

const UNREAD = 'the calculation policy in force states a CreditMonteCarlo block the engine refuses';
const funded = (cost, benefit, refused) =>
  client.fundedAt({ cost, benefit, ...(refused && { refused }) });
check('the line beside an FVA says what the next run funds at: the one curve where cost and benefit '
      + 'name the same, each where they differ, the one stated where the other is not, that the FVA '
      + "reads 0 where neither is, and the record's own sentence where the calculation is unread",
      'the benefit dropped where it differs, which says a run funding its benefit at USD funds at '
      + 'USD-FUND alone; a curve named where none is stated, "at null"; the side stating none '
      + 'named as the one funded; or the refusal swallowed, which reads a calculation nobody can '
      + 'read as one stating no curve',
      [funded('USD-FUND', 'USD-FUND'), funded('USD-FUND', 'USD'), funded('USD-FUND', null),
       funded(null, 'USD'), funded(null, null), funded(null, null, [UNREAD])],
      ['The next run funds its FVA at USD-FUND.',
       'The next run funds its FVA cost at USD-FUND and its benefit at USD.',
       'The next run funds its FVA cost at USD-FUND. No curve is stated for the benefit, so the '
       + 'benefit reads 0.',
       'The next run funds its FVA benefit at USD. No curve is stated for the cost, so the cost '
       + 'reads 0.',
       "No funding curve is stated, so the next run's FVA reads 0.", UNREAD]);

// ---- a credit limit, its usage and its headroom ------------------------------------------------

const limited = (entity, amount, extra) => ({
  entity, name: `${entity} Ltd`, clients: [entity], limit: { amount, currency: 'USD' }, sets: 2,
  unrun: 0,
  usage: 30000, headroom: amount - 30000, breached: 30000 > amount, not_comparable: null,
  ...extra });
const OTHER = 'this limit is stated in EUR and the exposure against it is reported in USD';
const LIMITS = [
  limited('WITHIN', 50000, { clients: ['WITHIN', 'OVER'] }), limited('OVER', 25000),
  limited('UNRUN', 50000, { sets: 1, unrun: 2, headroom: null }),
  limited('NEVER', 50000, { sets: 0, unrun: 1, usage: null, headroom: null, breached: null }),
  limited('OTHER', 50000, { usage: null, headroom: null, breached: null, not_comparable: OTHER }),
  limited('GROUP', 90000, { clients: ['GROUP', 'WITHIN', 'OVER'] })];
const lineOf = (read) => read && [read.figures.map((figure) => `${figure.label} ${figure.value}`)
  .join(', '), read.currency, read.flag, ...read.notes];
check('a credit limit reads as the limit, the usage of the sets run and the headroom, flagged where '
      + 'usage is over it; with a set unrun it says how many and states no headroom, and no usage '
      + "where none has run; not comparable, it is the limit and the service's sentence; a "
      + "parent's says how many entities under it its usage covers, as the service names them; "
      + 'and an entity with no limit reads none',
      'headroom stated with a set unrun, room promised against exposure nobody has run; the '
      + 'breach flag turned, a limit within its usage flagged and one over it not; the '
      + 'not-comparable sentence swallowed, a limit in another currency read as unused; a usage '
      + "said where the service answers none, no set having run; a parent's limit read as the "
      + "parent's alone; or "
      + 'the entity counted among those under it, which says every limit covers one more',
      [...LIMITS.map((entry) => lineOf(client.limitOf(entry.entity, LIMITS))),
       lineOf(client.limitOf('NONE', LIMITS)) ?? null],
      [['credit limit 50000, usage of 2 sets 30000, headroom 20000', 'USD', null,
        'It covers the entity under it.'],
       ['credit limit 25000, usage of 2 sets 30000, headroom -5000', 'USD', 'over its limit'],
       ['credit limit 50000, usage of 1 set 30000', 'USD', null,
        '2 of its 3 sets have no run that is done, so no headroom is stated.'],
       ['credit limit 50000', 'USD', null,
        'Its set has no run that is done, so neither usage nor headroom is stated.'],
       ['credit limit 50000', 'USD', null, OTHER],
       ['credit limit 90000, usage of 2 sets 30000, headroom 60000', 'USD', null,
        'It covers the 2 entities under it.'],
       null]);
const WITH = (without, amount = 50000) =>
  `credit limit ${amount}, usage without the trade ${without}, with it 30000`;
const charged = (amount, without, extra) => lineOf(client.limitRead(
  limited('KES', amount, { usage_without: without, breached_without: without > amount, ...extra })));
check('a credit charge reads each limit its set draws on as the usage without the trade, then '
      + 'with it, and the headroom after it, and says where the trade takes it over the limit - '
      + 'or that it is over with or without the trade, as the service answers whether it was',
      'with and without swapped, the trade read as taking usage down; the trade blamed for a '
      + 'breach that stands without it; a trade that would breach left unflagged; or the usage '
      + "without the trade set against the limit here, in place of the service's own answer",
      [charged(50000, 20000), charged(25000, 20000), charged(25000, 28000),
       charged(50000, 20000, { sets: 1, unrun: 1, headroom: null }),
       charged(25000, 20000, { breached_without: true })[2],
       charged(25000, 28000, { breached_without: false })[2]],
      [[`${WITH(20000)}, headroom after it 20000`, 'USD', null],
       [`${WITH(20000, 25000)}, headroom after it -5000`, 'USD',
        'the trade would take it over its limit'],
       [`${WITH(28000, 25000)}, headroom after it -5000`, 'USD',
        'over its limit with or without the trade'],
       [WITH(20000), 'USD', null,
        '1 of its 2 sets has no run that is done, so no headroom is stated.'],
       'over its limit with or without the trade', 'the trade would take it over its limit']);

const KES_LINES = client.creditLines('KES', CREDIT);
check("a client's sums are over every one of its book's sets or none - an unread set blanks it, and "
      + 'adjustments in two currencies, or a set that left the book, are never added; the exposure '
      + "after collateral is summed off the runs' own figure as an adjustment is",
      "an unread set's value skipped, which says a client owes less than it does; two currencies "
      + "added; a set that left the book counted in its client's CVA; or a set never run read as "
      + 'exposing nothing after collateral',
      [client.summed([1, null, 2]), client.summed([1, 2]),
       client.runSum([KES_LINES[0], KES_LINES[3]], 'cva'), client.runSum(KES_LINES, 'cva'),
       client.runSum([KES_LINES[0], { run: { ...RUNS[0], currency: 'EUR' } }], 'fva'),
       client.runSum([KES_LINES[0], KES_LINES[3]], 'after_collateral'),
       client.runSum(KES_LINES, 'after_collateral')],
      [null, 3, { currency: 'USD', value: 10 }, { currency: 'USD', value: null },
       { currency: null, value: null }, { currency: 'USD', value: 900 },
       { currency: 'USD', value: null }]);

check("a failed run names who puts it right off the factor its row says it wanted, in the service's "
      + 'note, and the sentence itself where the row names no factor',
      "the note dropped for a sentence of the client's own; or a row naming no factor given one",
      [{ factor: 'SurvivalProb.KES', supply: null, note: 'the CVA desk writes it from its spread' },
       { factor: 'InterestRate.ZAR-OIS', supply: { block: 'rates', key: 'ZAR-OIS' },
         note: "a seat holding mark sets it up off the desk's rates entry ZAR-OIS" }, null]
        .map((missing) => client.fixedBy(missing)),
      ['SurvivalProb.KES: the CVA desk writes it from its spread',
       "InterestRate.ZAR-OIS: a seat holding mark sets it up off the desk's rates entry ZAR-OIS",
       'read its sentence, then a recalc asked for through the chat']);

check("a failed set's line names who puts it right off the factor its row says it wanted",
      "the row's `missing` left unread, which names nobody for a run the CVA desk puts right",
      client.wrongWith(KES_LINES).filter((wrong) => wrong.kind === 'failed').map((wrong) => wrong.who),
      ['SurvivalProb.KES: the CVA desk writes it from its spread']);

check("what is wrong under a client's sets, worst first: a failed run, trades no agreement covers, "
      + 'trades waiting on a second seat, trades never run - nothing of a set that left the book or of '
      + 'a desk keeping no record that it cannot tell',
      "the flags in the sets' order rather than the kinds'; a set that left the book flagged; a "
      + 'never-run set with nothing on flagged; or a set on a desk keeping no record called unpapered',
      [client.wrongWith(KES_LINES).map((wrong) => [wrong.kind, wrong.reference]),
       client.wrongWith(client.creditLines('KES', PLAIN)).map((wrong) => [wrong.kind, wrong.reference])],
      [[['failed', 'KES-OLD'], ['unpapered', 'KES-OLD'], ['pending', 'ISDA-KES'], ['unrun', 'CSA-KES']],
       [['failed', 'KES-OLD'], ['unrun', 'CSA-KES']]]);

check('trades booked under a set since its run are a fact of its run, never something to act on: '
        + 'they neither list it as wrong nor sort its client first',
      'a set booked under since its run listed among what is wrong, which sends its client to the '
        + 'top of the table for a book the close refreshes',
      client.wrongWith([{ reference: 'S', paper: 'ISDA', run: RUNS[0], trades: 1, pending: 0,
                          booked: 3 }]), []);

const OVER = client.limitOf('OVER', LIMITS);
check('clients with something to act on - a client over its credit limit among them - come first, '
      + 'then the largest exposure after collateral, an unread one last, then by name',
      'the clients to act on sorted last; a breached limit not counted as something to act on, '
      + 'which sorts a client over its limit among those with nothing to do; or an unread exposure '
      + 'read as the largest',
      client.actionFirst([{ name: 'B', wrong: [], after: 10 }, { name: 'A', wrong: [], after: null },
                          { name: 'C', wrong: [{}], after: 0 }, { name: 'D', wrong: [], after: 50 },
                          { name: 'F', wrong: [], after: 0, limit: OVER },
                          { name: 'E', wrong: [], after: 10, limit: client.limitOf('WITHIN', LIMITS) }])
        .map((row) => row.name),
      ['C', 'F', 'D', 'B', 'E', 'A']);
check("the table's head counts what the clients have to act on kind by kind, worst first, each in "
      + 'its own words, a client over its credit limit counted last as over its limit',
      'a breach left out of the count, which tells a credit officer there is nothing to act on '
      + 'while a client is over its limit; or the kinds counted in the order the clients come',
      [client.actsText([{ wrong: [], limit: OVER },
                        { wrong: [{ kind: 'unrun' }, { kind: 'failed' }], limit: OVER },
                        { wrong: [{ kind: 'failed' }], limit: client.limitOf('WITHIN', LIMITS) }]),
       client.actsText([{ wrong: [], limit: client.limitOf('OTHER', LIMITS) }])],
      ['2 run failed · 1 never run · 2 over its limit', '']);

// ---- the credit table's export: a row per set, every number as itself ---------------------------

const RAN_AT = '2026-11-19T22:27:28.743+02:00';
const PEAKED = { ...KES_LINES[0], run: { ...RUNS[0], cva: 1234567.891, fva: 0.1 + 0.2, peaks: {
  EE: { value: 2500000.5, day: '2027-02-16' }, PFE_95: { value: 9000000.25, day: '2027-05-16' } } } };
const SHEET = client.creditExport([
  { name: 'Kestrel (KES)', spread: [1, 150], lines: [PEAKED, KES_LINES[1], KES_LINES[3]],
    limit: OVER },
  { name: 'Marrick (MAR)', lines: [], limit: client.limitOf('UNRUN', LIMITS) },
  { name: 'Ostrava (OST)', lines: [] }], 'USD');
check("the credit table exports a row per set under its client, a client with none a row of its own: "
      + "every amount the number itself, its currency and a peak's day in columns of their own, a "
      + 'PFE column per percentile kept, the note of a set that left the book, and the client\'s '
      + 'credit limit with the usage and headroom it states, in its currency',
      'an amount written as displayed - `formatNumber` over a CVA - which exports 1,234,567.89 as '
      + "text a spreadsheet cannot add and rounds what it keeps; a currency left in its amount's "
      + 'cell; a column written for one row and not the next, every value after it under the '
      + 'wrong head; the limit left out, which exports a breach as nothing; a headroom written '
      + "where a set has no run; or the exposure after collateral worked out of the value to the "
      + "desk and the collateral held, in place of its run's own figure",
      [SHEET[0], SHEET.slice(1), SHEET.every((line) => line.length === SHEET[0].length)],
      [['Client', 'Netting set', 'Spread bp', 'Spread tenor', 'Paper', 'Trades',
        'Waiting on a second seat', 'Value to the desk', 'Ccy', 'Collateral held', 'Call',
        'Collateral ccy', 'After collateral', 'Peak EE', 'Peak EE day', 'Peak PFE 95',
        'Peak PFE 95 day', 'CVA', 'FVA', 'Run ccy', 'Run', 'Note', 'Credit limit', 'Limit usage',
        'Headroom', 'Limit ccy'],
       [['Kestrel (KES)', 'ISDA-KES', 150, '1Y', 'uncollateralised', 2, 1, 1000, 'USD',
         null, null, null, 900, 2500000.5, '2027-02-16', 9000000.25, '2027-05-16', 1234567.891,
         0.30000000000000004, 'USD', RAN_AT, null, 25000, 30000, -5000, 'USD'],
        ['Kestrel (KES)', 'CSA-KES', 150, '1Y', 'collateralised', 1, 0, null, 'USD',
         400, null, 'EUR', null, null, null, null, null, 10, 1, 'USD', null, null, 25000, 30000, -5000,
         'USD'],
        ['Kestrel (KES)', 'KES-GONE', 150, '1Y', 'no agreement', 0, 0, null, 'USD',
         null, null, null, 900, null, null, null, null, null, 1, 'USD', RAN_AT, 'left the book', 25000,
         30000, -5000, 'USD'],
        ['Marrick (MAR)', ...Array(21).fill(null), 50000, 30000, null, 'USD'],
        ['Ostrava (OST)', ...Array(25).fill(null)]],
       true]);

// each agreement as the service answers it: whether a run collateralises it is its verdict
const PAPERS = [
  { agreement: 'ISDA-CSA', entity: 'A', kind: 'ISDA 2002 with CSA', collateralized: true,
    balance_currency: 'EUR', terms: {
      Collateralized: 'True', Agreement_Currency: 'USD', Balance_Currency: 'EUR' } },
  { agreement: 'ISDA-K', entity: 'K', kind: 'ISDA 2002', collateralized: false,
    terms: { Agreement_Currency: 'USD' } },
  { agreement: 'ISDA-SAYS-CSA', entity: 'S', kind: 'ISDA 2002 with CSA', collateralized: false,
    terms: { Collateralized: 'Yes' } },
  { agreement: 'CSA-USD', entity: 'U', kind: 'CSA', collateralized: true, balance_currency: 'USD',
    terms: { Collateralized: 'True', Agreement_Currency: 'USD' } },
];
check("the paper says its agreement and kind, and collateral as the service reads its terms, in "
        + 'the currency the service says its balance is kept in',
      "the flag read again off the terms, collateralised unless it says 'False', which calls the "
        + "set stating 'Yes' collateralised; collateral read off the document's name; or the "
        + "balance currency read off the terms again",
      ['ISDA-CSA', 'ISDA-K', 'ISDA-SAYS-CSA', 'CSA-USD', 'ISDA-X'].map(
        (set) => client.paperLine(set, PAPERS)),
      ['ISDA-CSA (ISDA 2002 with CSA), collateralised in EUR', 'ISDA-K (ISDA 2002), uncollateralised',
       'ISDA-SAYS-CSA (ISDA 2002 with CSA), uncollateralised', 'CSA-USD (CSA), collateralised in USD',
       'ISDA-X, which the record declares no agreement for']);

done();
