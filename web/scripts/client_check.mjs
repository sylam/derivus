// Drives the rules a client's page reads by, where the eye cannot: whether a client is ready to
// trade, an agreement's terms in words off the netting set's declaration, which agreements and
// trades are its own, its mark's side, and which of the record's rows it reads. Every check names
// the MUTATION it kills - the change that would still typecheck, still render, and still be wrong.
//
//   node web/scripts/client_check.mjs        # exits 1 on a miss

import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

async function bundled(module) {
  const out = await build({
    entryPoints: [fileURLToPath(new URL(`../src/${module}`, import.meta.url))],
    bundle: true, format: 'esm', write: false, logLevel: 'silent',
  });
  return import('data:text/javascript;base64,' + Buffer.from(out.outputFiles[0].text).toString('base64'));
}
const client = await bundled('client.ts');
// the file's rows, as the page reads a set's trades off them
const blotter = await bundled('blotter.ts');

let ran = 0;
const missed = [];

function check(name, kills, got, want) {
  ran += 1;
  const same = JSON.stringify(got) === JSON.stringify(want);
  if (!same) missed.push(name);
  console.log(`${same ? 'ok  ' : 'MISS'}  ${name}\n      kills: ${kills}`);
  if (!same) {
    console.log(`      got   ${JSON.stringify(got)}`);
    console.log(`      want  ${JSON.stringify(want)}`);
  }
}

// ---- the declaration, trimmed to what the rules turn on ------------------------------------------

const field = (widget, value, description, extra = {}) => ({ widget, description, value, ...extra });
const support = (rows) => ({ '.CreditSupportList': rows });
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

const node = (deal, children) => ({ Instrument: { '.Deal': deal }, ...(children ? { Children: children } : {}) });
const fwd = (reference) => node({ Object: 'FXForwardDeal', Reference: reference });
const KES_SET = { Object: SET, Reference: 'ISDA-KES', Credit_Support_Amounts: { Counterparty: 'KES' } };
const DOC = { Calc: { Deals: { Deals: { Children: [
  node(KES_SET, [fwd('F1'), fwd('F2'), fwd('F3')]),
  node({ Object: 'StructuredDeal', Reference: 'DESK' }, [
    node({ Object: SET, Reference: 'CSA-ALD', Credit_Support_Amounts: { Counterparty: 'ALD' } })]),
  node({ Object: SET, Reference: 'ISDA-OFF', Credit_Support_Amounts: { Counterparty: 'OFF' } }),
] } } } };
const PAPER = [
  { agreement: 'ISDA-KES', entity: 'KES', kind: 'ISDA 2002', terms: KES_SET },
  { agreement: 'ISDA-NEW', entity: 'ALD', kind: 'ISDA 2002 with CSA', terms: { Object: SET } },
  { agreement: 'ISDA-OFF', entity: 'OFF', kind: 'ISDA 2002', terms: { Object: SET } },
];
check("a client's agreements are those legal declares with it, and on a desk keeping no record the "
      + 'netting sets naming it wherever they sit in the book',
      "another entity's agreement taken; the walk kept to the top level, which loses a set nested "
      + 'under a book node; or a set read off its type name rather than the credit support it states',
      [client.agreementsOf('ALD', PAPER, DOC).map((row) => row.agreement),
       client.agreementsOf('ALD', null, DOC).map((row) => [row.agreement, row.entity])],
      [['ISDA-NEW'], [['CSA-ALD', 'ALD']]]);

const ENTITIES = [{ entity: 'KES', name: 'Kestrel', parent: null },
                  { entity: 'ALD', name: 'Alderby', parent: 'KES' },
                  { entity: 'MAR', name: 'Marrick', parent: null },
                  { entity: 'OFF', name: 'Offside', parent: null }];
const RECONCILE = { terms_mismatch: [{ agreement: 'ISDA-OFF', deal_path: '2', fields: ['Netted'] }] };
const lines = (id, entities, curve) => client.readiness(
  id, entities, client.agreementsOf(id, entities && PAPER, DOC), DOC, RECONCILE, curve)
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
         id, ENTITIES, client.agreementsOf(id, PAPER, DOC), DOC, null, id === 'ALD' ? null : {})))],
      [[['ok', null], ['ok', null], ['ok', null]],
       [['ok', null], ['note', null], ['todo', 'curve']],
       [['ok', null], ['todo', 'agreement'], ['ok', null]],
       [['ok', null], ['warn', null], ['ok', null]],
       [['todo', 'entity'], ['todo', 'agreement'], ['ok', null]],
       [['note', null], ['ok', null]],
       [true, false, false]]);

const curveless = (id, entities) => client.readiness(
  id, entities, client.agreementsOf(id, entities && PAPER, DOC), DOC, null, undefined).at(-1).text;
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
  settles: null, status: 'approved', pending: 0, ...extra });
const ROWS = blotter.rowsByPath(blotter.toRows(DOC.Calc.Deals.Deals.Children, [], undefined));
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
check("a client's rows are its declaration and every row naming one of its agreements, newest first",
      "the record's order kept, oldest first; or another client's declaration read",
      client.touching(RECORD, 'KES', ['ISDA-KES']).map((entry) => entry.lsn), [31, 29, 22, 19]);

check("a set moved since its XVA run by the record's rows naming it written after the run, each "
      + 'stamp read at its own offset',
      "the two stamps compared as written, the record's UTC against the run's local time, which "
      + 'sees nothing booked for two hours after every run; or a row of another set counted',
      client.since({ reference: 'ISDA-KES', as_of: '2026-11-19T22:27:28.743+02:00' }, RECORD)
        .map((entry) => entry.lsn), [31]);

console.log(`\n${ran} checks over 8 functions, ${missed.length} missed`);
if (missed.length) {
  console.log(missed.map((name) => `  ${name}`).join('\n'));
  process.exit(1);
}
