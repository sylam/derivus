// Drives the rules the Paper tab reads an agreement by, where the eye cannot: the sections a type's
// own terms are, when a set's collateral is off and what then reads, what paper never states, the
// currency an amount's description names, the terms a restatement moves, where an agreement
// stands in the book, and which parts of the paper a screen reads when the book moves.
// Every check names the MUTATION it kills - the change that would still typecheck, still render,
// and still be wrong.
//
//   node web/scripts/paper_check.mjs         # exits 1 on a miss

import { check, done, load, node, support } from './check.mjs';

const tokens = await load('tokens');
const { paperOwed, standing } = await load('state');

// ---- the declaration, trimmed to what the rules turn on ------------------------------------------

const field = (widget, value, extra = {}) => ({ widget, description: '', value, ...extra });
const TERMS = {
  Agreement_Currency: field('Text', '', { convention: true }),
  Collateralized: field('Dropdown', 'False', { convention: true, values: ['True', 'False'] }),
  Netted: field('Dropdown', 'True', { convention: true, values: ['True', 'False'] }),
  Liquidation_Period: field('Integer', 0, { convention: true }),
  Credit_Support_Amounts: field('Container', {}, { convention: true, sub_fields: {
    Counterparty: field('Text', ''),
    Minimum_Received: field('Table', support([[1, 0]]), { convention: true }),
  } }),
};
const ISDA = { Object: 'NettingCollateralSet', Reference: 'ISDA-1', Agreement_Currency: 'USD',
               Credit_Support_Amounts: { Counterparty: 'CPTY-1' } };
const CSA = { Object: 'NettingCollateralSet', Reference: 'CSA-1', Collateralized: 'True',
              Agreement_Currency: 'EUR', Liquidation_Period: 10,
              Credit_Support_Amounts: { Counterparty: 'CPTY-2',
                                        Minimum_Received: support([[1, 100000]]) } };

check("a type's own terms leave out the one section every type shares, and only that one",
      "the first type's sections taken whole, which puts Sales Margin and MtM on an agreement; or "
      + 'a section two types share left out, which drops a pricing block from a deal',
      tokens.sharedSections({ A: ['Admin', 'Pricing', 'A.Fields'],
                              B: ['Admin', 'Pricing', 'B.Fields'], C: ['C.Fields', 'Admin'] }),
      ['Admin']);

check('a set reads uncollateralised where its flag is off, stated or left at its convention',
      'the flag read only where stated, which shows an ISDA its thresholds; or a type declaring '
      + 'no flag read as off, which strips a deal to what it states',
      [tokens.uncollateralised(TERMS, ISDA), tokens.uncollateralised(TERMS, CSA),
       tokens.uncollateralised(TERMS, { ...CSA, Collateralized: 'False' }),
       tokens.uncollateralised({ Netted: TERMS.Netted }, {})],
      [true, false, true, false]);

const cut = tokens.statedOnly(TERMS, ISDA, [tokens.COLLATERAL]);
check('an uncollateralised read keeps what the terms state and the flag, a container its stated '
      + 'children',
      'a container kept whole, which shows an ISDA its minimum transfer at 0; or the flag dropped, '
      + 'which hides that the set is uncollateralised',
      [Object.keys(cut), Object.keys(cut.Credit_Support_Amounts.sub_fields)],
      [['Agreement_Currency', 'Collateralized', 'Credit_Support_Amounts'], ['Counterparty']]);

const table = (col_names, obj) =>
  field('Table', 'null', { col_names, obj, sub_types: obj.map(() => ({})) });
const SET = {
  ...TERMS, Opening_Balance: field('Float', 0, { convention: true }),
  Collateral_Assets: field('Container', {}, { sub_fields: {
    Cash_Collateral: table(['Currency', 'Amount', 'Haircut_Posted'], ['Text', 'Float', 'Percent']),
    Equity_Collateral: table(['Equity', 'Units'], ['Text', 'Float']) } }),
  Received_Threshold: field('Table', support([[1, 0]]), {
    col_names: ['Threshold Amount', 'Credit Rating'], obj: 'CreditSupportList' }),
};
const paper = tokens.onPaper(SET, { fields: ['Opening_Balance'],
                                    collateral_columns: ['Amount', 'Units'] });
const cash = paper.Collateral_Assets.sub_fields.Cash_Collateral;
check("an agreement's form leaves out the fields and the collateral columns the read names paper "
      + 'never states, each column with its form, and nothing else',
      'the balance offered, which the service refuses; a column dropped from the names alone, '
      + 'which files Haircut_Posted under the Amount form; the tables of a container left whole; '
      + "a column matched inside another's name, which strips a threshold table's Threshold "
      + 'Amount; or a table with nothing to cut rewritten',
      [Object.keys(paper).includes('Opening_Balance'), cash.col_names, cash.obj,
       cash.sub_types.length, paper.Collateral_Assets.sub_fields.Equity_Collateral.col_names,
       paper.Received_Threshold === SET.Received_Threshold, paper.Netted === SET.Netted],
      [false, ['Currency', 'Haircut_Posted'], ['Text', 'Percent'], 2, ['Equity'], true, true]);

const read = (values) => (key) => values[key];
check("an amount is in the currency field its description says it is in, by key or in words",
      "any mention of a currency field read, which tags a quanto rate 'payoff currency per unit' "
      + 'with the payoff currency; a blank currency shown; or the words left unread, which leaves '
      + "a threshold 'in the agreement currency' with none",
      [tokens.currencyOf('Minimum transfer the bank receives, in the agreement currency',
                         ['Agreement_Currency', 'Balance_Currency'],
                         read({ Agreement_Currency: 'EUR', Balance_Currency: 'ZAR' })),
       tokens.currencyOf('What the desk charged, in Sales_Margin_Currency', ['Sales_Margin_Currency'],
                         read({ Sales_Margin_Currency: 'USD' })),
       tokens.currencyOf('Cash in Currency earned by each fixing', ['Currency'], read({ Currency: 'ZAR' })),
       tokens.currencyOf('The rate, payoff currency per unit of the underlying currency',
                         ['Payoff_Currency', 'Currency'], read({ Payoff_Currency: 'USD', Currency: 'ZAR' })),
       tokens.currencyOf('Opening balance the bank holds, in the balance currency',
                         ['Agreement_Currency', 'Balance_Currency'],
                         read({ Agreement_Currency: 'EUR', Balance_Currency: '' })),
       tokens.currencyOf('Threshold, in the agreement currency', ['Currency'], read({ Currency: 'ZAR' }))],
      ['EUR', 'USD', 'ZAR', undefined, undefined, undefined]);

check('a restatement names each term it moves, a container by the term inside it',
      'a container compared whole, which names Credit_Support_Amounts where one minimum moved; a '
      + 'blank read as a statement, which names every field a form round-trips empty; or the '
      + 'terms a restatement leaves out unnamed, which reverts them silently',
      [tokens.termsMoved(CSA, { ...CSA, Liquidation_Period: 15, Netted: '', Credit_Support_Amounts: {
        ...CSA.Credit_Support_Amounts, Minimum_Received: support([[1, 250000]]) } }),
       tokens.termsMoved(CSA, { Object: 'NettingCollateralSet' }),
       tokens.termsMoved(CSA, structuredClone(CSA))],
      [['Credit_Support_Amounts.Minimum_Received', 'Liquidation_Period'],
       ['Agreement_Currency', 'Collateralized', 'Credit_Support_Amounts.Counterparty',
        'Credit_Support_Amounts.Minimum_Received', 'Liquidation_Period', 'Reference'],
       []]);

const DOC = { Calc: { Deals: { Deals: { Children: [
  node({ Object: 'FXForwardDeal', Reference: 'CSA-2' }),
  node(ISDA, [node({ Object: 'FXForwardDeal', Reference: 'FWD-1' })]),
  node({ Object: 'StructuredDeal', Reference: 'BOOK' }, [node(CSA)]),
] } } } };
const RECONCILE = { terms_mismatch: [{ agreement: 'CSA-1', deal_path: '2/0', fields: ['Netted'] }] };
check('an agreement stands where the book holds a set of its type under its id, off its paper on '
      + 'the fields the reconcile names, and nowhere before its first booking',
      'a set matched on its reference alone, which takes a trade named like the agreement for its '
      + 'set; the walk kept to the top level, which loses a set nested under a book node; or the '
      + "reconcile's fields read for another agreement",
      [standing(DOC, RECONCILE, { agreement: 'ISDA-1', terms: ISDA }),
       standing(DOC, RECONCILE, { agreement: 'CSA-1', terms: CSA }),
       standing(DOC, RECONCILE, { agreement: 'CSA-2', terms: CSA }),
       standing(null, null, { agreement: 'ISDA-1', terms: ISDA })],
      [{ path: '1', moved: [] }, { path: '2/0', moved: ['Netted'] }, { path: null, moved: [] },
       { path: null, moved: [] }]);

// ---- which parts of the paper a screen reads ----------------------------------------------------

const AT = { paper: 'e1', holdings: 'e1' };
const HELD = { entities: [], agreements: [] };
const owes = (tab, held, answered, clock) => {
  const parts = paperOwed(tab, held, answered, clock);
  return [parts.paper, parts.holdings];
};
check('a booking re-reads the paper on every screen naming clients, and the holdings only on '
      + 'one showing them; a screen with none held reads it all, one showing no paper nothing',
      'the holdings read on every screen naming a client, which sends the Paper tab four reads per '
      + 'booking for two it shows; the whole read dropped, which shows a desk holding nothing on '
      + 'the first screen that groups the book after the Paper tab; or a part read again under the '
      + 'clock it was read at',
      [owes('paper', HELD, AT, 'e2'), owes('portfolio', HELD, AT, 'e2'),
       owes('portfolio', HELD, { paper: 'e2', holdings: 'e1' }, 'e2'),
       owes('paper', null, {}, 'e2'), owes('portfolio', HELD, AT, 'e1'),
       owes('settings', null, {}, 'e2')],
      [[true, false], [true, true], [false, true], [true, true], [false, false], [false, false]]);

done();
