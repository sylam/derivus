// Drives `src/trade.ts` where the eye cannot: which node a path's trade is, which mark is its own,
// how the record is read whole and what of it is the trade's, who booked it, its size and dates,
// what an unwind files and says, and what a ticket to sign says to the seat reading it - and the
// store's way onto the page from a row naming a trade and back to where it was opened from.
// Every check names the MUTATION it kills - the change that would still typecheck, still render,
// and still be wrong.
//
//   node web/scripts/trade_check.mjs          # exits 1 on a miss
//
// The book is a client's netting set holding an option and a two-legged collar, and a forward at
// the book's root; the record is a fake reader paging rows as `GET /book/activity` does.

import { check, done, load, node } from './check.mjs';

const trade = await load('trade');
const blotter = await load('blotter');
const state = await load('state');

const ts = (day) => ({ '.Timestamp': day });
const leg = (role, type, buy, strike, amount = 1000000) => node({
  Object: 'FXOptionDeal', Reference: `COLLAR_${role}`, Structure_Reference: 'COLLAR',
  Currency: 'ZAR', Underlying_Currency: 'USD', Underlying_Amount: amount, Option_Type: type,
  Buy_Sell: buy, Strike_Price: strike, FX_Volatility: 'USD.ZAR', Delivery_Date: ts('2027-05-18'),
  Expiry_Date: ts('2027-05-14'), Trade_Date: ts('2026-11-16') });
const BOOK = [
  node({ Object: 'NettingCollateralSet', Reference: 'ISDA-A' }, [
    node({ Object: 'FXOptionDeal', Reference: 'CALL', Currency: 'ZAR', Underlying_Currency: 'USD',
           Underlying_Amount: 2000000, Buy_Sell: 'Buy', Strike_Price: 18.25 }),
    node({ Object: 'StructuredDeal', Reference: 'COLLAR', Currency: 'ZAR' }, [
      leg('protection', 'Put', 'Sell', 17.5), leg('financing', 'Call', 'Buy', 17.76)]),
  ]),
  node({ Object: 'FXForwardDeal', Reference: 'FWD', Buy_Currency: 'ZAR', Buy_Amount: 20091300,
         Sell_Currency: 'EUR', Sell_Amount: 1000000, Settlement_Date: ts('2027-02-16') }),
];

const BOOK_DOC = { Calc: { Deals: { Deals: { Children: BOOK } } } };

// --- which node is the trade
check('a leg names its structure, the trade, beside the set it sits in',
      'the walk returning the deepest node, so a leg opens a page of its own and not its trade\'s',
      trade.tradeAt(BOOK, '0/1/1'), { path: '0/1', set: 'ISDA-A' });
check('a netting set is no trade, and a path naming nothing is none either',
      'the set read as the trade it holds, which opens a page with no terms of its own',
      [trade.tradeAt(BOOK, '0'), trade.tradeAt(BOOK, '0/7')], [undefined, undefined]);
check('a trade at the book\'s root sits in no set',
      'the set\'s reference left from a walk that never met one, naming a client nobody declared',
      trade.tradeAt(BOOK, '1'), { path: '1', set: null });

// --- where its acts are judged
const sits = (portfolio, quantity = 1) => ({ portfolio, quantity });
check('a trade\'s acts are judged where its positions sit: one position\'s own node, the deepest '
      + 'node holding every position standing where there are several, and no node - the book - '
      + 'where the record holds none or only what it closed',
      'the book taken for every trade, which tells a seat granted book at a desk\'s node it may '
      + 'not amend or unwind what the record lets it; the first position\'s node taken for all, '
      + 'which offers a restrike of a trade two desks hold to a seat granted one; a closed '
      + 'position counted, which the record does not; or the nodes matched by their text, which '
      + 'reads HARBOURLINE/FX as holding HARBOURLINE/FXO',
      [trade.judgedAt([sits('HARBOURLINE/FX')]),
       trade.judgedAt([sits('HARBOURLINE/FX/Options'), sits('HARBOURLINE/FX/Spot')]),
       trade.judgedAt([sits('HARBOURLINE/FX'), sits('HARBOURLINE/RATES')]),
       trade.judgedAt([sits('HARBOURLINE/FX'), sits('HARBOURLINE/FXO')]),
       trade.judgedAt([sits('HARBOURLINE/FX/Options'), sits('HARBOURLINE/FX')]),
       trade.judgedAt([sits('HARBOURLINE/FX'), sits('HARBOURLINE/RATES', 0)]),
       trade.judgedAt([sits('HARBOURLINE/FX', 0)]), trade.judgedAt([])],
      ['HARBOURLINE/FX', 'HARBOURLINE/FX', 'HARBOURLINE', 'HARBOURLINE', 'HARBOURLINE/FX',
       'HARBOURLINE/FX', undefined, undefined]);

// --- the mark
const MARKS = [{ reference: 'ISDA-A', deal_path: '0', value: 5.5 },
               { reference: 'FWD', deal_path: '1', value: -2.25 },
               { reference: 'COLLAR', deal_path: '0/1', value: 1.5 }];
check('a trade has its own mark, at the top or inside a set, beside the set\'s net that holds it; '
        + 'a leg has the set\'s net alone',
      'the set\'s net returned as the trade\'s own mark, a number of three trades read as one; or '
        + 'the set\'s net dropped where the trade has its own, which loses what the mark sits in',
      [trade.markOf(MARKS, '1'), trade.markOf(MARKS, '0/1'), trade.markOf(MARKS, '0/1/0')],
      [{ own: MARKS[1] }, { own: MARKS[2], inside: MARKS[0] }, { inside: MARKS[0] }]);
const [DESK, FX, ALONE] = [['HARBOURLINE/desk', 'ISDA-A'], ['HARBOURLINE/fx', 'ISDA-A'],
  ['HARBOURLINE', 'ISDA-B']].map(([portfolio, agreement]) => ({ portfolio, agreement }));
// the risk answer's own share of each deal per position the record holds on it
const PER = [{ deal_path: '0/1', ...DESK, quantity: 1, unit: 1, value: 1 },
             { deal_path: '0/1', ...FX, quantity: 0.5, unit: 1, value: 0.5 },
             { deal_path: '1', ...ALONE, quantity: 3, unit: -0.75, value: -2.25 },
             { deal_path: '0', ...ALONE, quantity: 2, unit: null, value: null }];
const shown = (path, position, data = { per_position: PER }) => {
  const mark = trade.markShown({ marks: MARKS, data }, path, position);
  return mark === undefined ? 'none' : mark;
};
check("a position shows its own row of the risk answer's per-position marks, the record's share of "
        + "the deal, and a row that is no position the deal's own mark; a position the answer "
        + 'gives no value or no row, or no answer, shows none',
      "the deal's mark shown whole on each position, 1.5 units' worth read on a line holding "
        + "0.5; a position's row found by its deal alone, which reads another position's share; "
        + 'a deal netted to nothing read as a mark of nothing; or a row that is no position read '
        + 'off the positions',
      [shown('0/1', DESK), shown('0/1', FX), shown('1', ALONE), shown('0/1'), shown('0', ALONE),
       shown('0/7', DESK), shown('0/1', DESK, null)],
      [1, 0.5, -2.25, 1.5, 'none', 'none', 'none']);

// --- the record, read whole
const RECORD = Array.from({ length: 450 }, (_, at) => ({ lsn: at + 1 }));
const reader = (rows) => {
  const asked = [];
  const read = async (since) => {
    asked.push(since);
    if (asked.length > 9) throw new Error('the walk does not end');
    const page = rows.filter((row) => row.lsn > since).slice(0, 200);
    return { lsn: page.length ? page[page.length - 1].lsn : since, rows: page };
  };
  return { asked, read };
};
const walked = reader(RECORD);
const whole = await trade.wholeRecord(walked.read).catch((error) => String(error));
check('the record is read page by page from its first event, each from the cursor before',
      'one page taken for the whole record, which loses every event past the 200th',
      [Array.isArray(whole) ? whole.length : whole, walked.asked], [450, [0, 200, 400]]);
const exact = reader(RECORD.slice(0, 200));
await trade.wholeRecord(exact.read);
check('a full last page asks once more and stops on the empty one',
      'a full page read as the end, so a record of exactly a page\'s rows hides the next event',
      exact.asked, [0, 200]);

// one trade's rows, oldest first, as `GET /book/activity?deal_path=` pages them
const ROWS = [
  { lsn: 23, event_type: 'approval', by: 'tier desk-auto', deal_path: '0/0' },
  { lsn: 24, event_type: 'fill', by: 'sales-1', deal_path: '0/0' },
  { lsn: 31, event_type: 'amendment', by: 'sales-2', deal_path: '0/0' },
  { lsn: 33, event_type: 'fill', by: 'sales-2', deal_path: '0/0' },
];
const history = trade.historyOf(ROWS);
check('a trade\'s history is its rows newest first, the read it came from left as it was',
      'the rows left in the record\'s order, so the oldest act reads as the latest; or sorted in '
        + 'place, which reorders the page the next read merges onto',
      [history.map((row) => row.lsn), ROWS.map((row) => row.lsn)], [[33, 31, 24, 23], [23, 24, 31, 33]]);
check('who booked it is its first fill on the record, never a later clip or the restrike',
      'the newest fill taken, naming the seat that unwound it as the one that booked it',
      trade.bookedBy(history)?.lsn, 24);

// --- what a restrike moved
const TERMS = { Instrument: { types: { FXForwardDeal: ['Fwd'] }, sections: { Fwd: {
  Buy_Amount: { description: 'Buy Amount', sized: true }, Buy_Currency: { description: 'Buy Currency' },
  Settlement_Date: { description: 'Settlement Date' },
  Strike_Price: { description: 'Strike price on the ENGINE axis: the reporting currency per unit of '
    + 'Underlying_Currency, so a level of 17.50 on a USD book is authored as 1/17.50' },
  Fee: { description: 'A fee, in Fee_Currency' }, Fee_Currency: { description: 'Fee Currency' } } } } };
const MOVED = trade.movedTerms(['Buy_Amount', 'Settlement_Date', 'Strike_Price', 'Fee', 'Note']
  .map((field) => ({ field, was: 1, became: 2 })),
{ Object: 'FXForwardDeal', Buy_Currency: 'ZAR', Fee_Currency: 'EUR' }, TERMS);
check('a moved field reads in its declared words - its name where they are a paragraph - in the '
        + 'currency an amount is in by the size\'s rule, else the one its description names',
      'the paragraph shown as the field\'s name; an amount left without its currency because its '
        + 'description names none; or a field the type does not declare dropped',
      MOVED.map(({ field, words, unit, declared }) => [field, words, unit ?? null, !!declared]),
      [['Buy_Amount', 'Buy Amount', 'ZAR', true], ['Settlement_Date', 'Settlement Date', null, true],
       ['Strike_Price', 'Strike Price', null, true], ['Fee', 'A fee, in Fee_Currency', 'EUR', true],
       ['Note', 'Note', null, false]]);

// --- size and dates
const rows = blotter.rowsByPath(blotter.toRows(BOOK, ['NettingCollateralSet', 'StructuredDeal'],
                                               '2026-11-16'));
const sizeOf = (row) => { const size = trade.tradeSize(row, null);
                          return [size.amount?.value, size.of?.value]; };
const uneven = blotter.toRows([node({ Object: 'StructuredDeal', Reference: 'X' }, [
  leg('a', 'Put', 'Sell', 17.5), leg('b', 'Call', 'Buy', 18, 2000000)])], [], undefined)[0];
check('a structure\'s size is the one every leg states, a deal\'s its own, and legs that differ '
        + 'state none',
      'the first leg\'s size taken whatever the others state, a ratio collar dealing half its size',
      [sizeOf(rows.get('0/1')), sizeOf(rows.get('0/0')), sizeOf(uneven)],
      [[1000000, 'USD'], [2000000, 'USD'], [undefined, undefined]]);
check('a structure\'s dates are each field once where its legs agree, in date order',
      'every leg\'s dates listed again, a collar reading two expiries on one day; or the dates '
        + 'left in the order the deal writes its fields',
      trade.datesOf(rows.get('0/1')).map(({ key, date }) => `${key} ${date}`),
      ['Trade_Date 2026-11-16', 'Expiry_Date 2027-05-14', 'Delivery_Date 2027-05-18']);

// --- the unwind
check('an unwind fills toward nothing from either side, the cash in the desk\'s own sign',
      'the fill signed as a sale whatever is held, which doubles a short instead of closing it',
      [trade.unwindOf(1, 0.5, 15000), trade.unwindOf(-2, 2, 300)],
      [{ quantity: -0.5, left: 0.5, receives: 7500 }, { quantity: 2, left: 0, receives: -600 }]);
const CALL = { amount: { key: 'Underlying_Amount', value: 2000000 },
               of: { key: 'Underlying_Currency', value: 'USD' } };
const SPOT = { amount: 15000, currency: 'USD', day: '2026-11-20' };
check('a priced unwind settles on its day, and one at a price of nothing names none',
      'the day sent whatever the price, a value date on a fill with no consideration to date; or '
        + 'dropped, a priced unwind due on no day',
      [trade.unwindFill(1, 0.5, SPOT), trade.unwindFill(-1, 1, { ...SPOT, amount: 0 })],
      [{ quantity: -0.5, price: 15000, price_currency: 'USD', value_date: '2026-11-20' },
       { quantity: 1, price: 0, price_currency: 'USD' }]);
check('the confirm says how much of what is held, who pays whom what on which day, and what is '
        + 'left, in the trade\'s size',
      'what is left said as what was unwound, a confirm that cannot tell half from all; the '
        + 'client said as paying whatever the sign of the cash; or the day left out',
      [trade.unwindWords(1, 0.25, CALL, SPOT, 'Kestrel'),
       trade.unwindWords(-1, 1, {}, { amount: 0, currency: 'USD' }, 'Kestrel'),
       trade.unwindWords(-2, 1, {}, { amount: 300, currency: 'USD', day: '2026-11-18' },
                         'Kestrel')],
      ['Unwind 0.25 (500,000 USD) of the 1 (2,000,000 USD) held, at 15,000 USD a unit: Kestrel '
        + 'pays the desk 3,750 USD on 2026-11-20, and 0.75 (1,500,000 USD) is left.',
       'Unwind 1 of the 1 short held, at 0 USD a unit: no cash moves, and nothing is left.',
       'Unwind 1 of the 2 short held, at 300 USD a unit: the desk pays Kestrel 300 USD on '
        + '2026-11-18, and 1 short is left.']);
check("a mark is said as the desk's side, on a client's page, a trade's and an unwind's line: "
        + 'worth something to the desk is in its favour, and worth nothing is flat',
      "the sign read the client's way, which tells a desk owed money that it owes it; or a "
        + "mark of nothing said to stand in the desk's favour",
      [trade.markText(1250.5, 'USD'), trade.markText(-20213.34, 'USD'), trade.markText(0, null),
       trade.favour(0), trade.favour(-1)],
      ["1,250.50 USD, in the desk's favour", "-20,213.34 USD, in the client's favour", '0, flat',
       'flat', "in the client's favour"]);
const priced = (amount, currency = 'USD') => ({ amount, currency });
check('an unwind\'s price is set against the desk\'s mark of the units unwound, in the mark\'s '
        + 'currency and favour: the position\'s unit mark times the units unwound, '
        + 'a price over it is made and one under it given up, a short\'s cash is '
        + 'paid against a mark in the client\'s favour, a price in another currency is set '
        + 'against nothing, and no units read no mark',
      'the position\'s unit mark shown for a part unwind, 60,000 read against the price of a '
        + 'quarter unit; the sign turned, a price over the mark said as given up; the units '
        + 'unwound taken as long whatever is held, a short\'s liability read as worth to the '
        + 'desk; a difference struck across two currencies; or the favour dropped',
      [trade.unwindAgainst(1, 0.25, 60000, 'USD'),
       trade.unwindAgainst(1, 0.25, 60000, 'USD', priced(70000)),
       trade.unwindAgainst(1, 0.25, 60000, 'USD', priced(50000)),
       trade.unwindAgainst(1, 0.25, 60000, 'USD', priced(60000)),
       trade.unwindAgainst(-2, 1, 300, 'USD', priced(250)),
       trade.unwindAgainst(-2, 1, 300, 'USD', priced(350)),
       trade.unwindAgainst(1, 0.25, 60000, 'USD', priced(1300000, 'ZAR')),
       trade.unwindAgainst(1, 0, 60000, 'USD', priced(70000)) ?? 'none'],
      ['The desk marks the 0.25 unwound at 15,000 USD, in the desk\'s favour.',
       'The desk marks the 0.25 unwound at 15,000 USD, in the desk\'s favour, so at this price '
        + 'it makes 2,500 USD against its mark.',
       'The desk marks the 0.25 unwound at 15,000 USD, in the desk\'s favour, so at this price '
        + 'it gives up 2,500 USD against its mark.',
       'The desk marks the 0.25 unwound at 15,000 USD, in the desk\'s favour, and this price is '
        + 'that mark.',
       'The desk marks the 1 unwound at -300 USD, in the client\'s favour, so at this price it '
        + 'makes 50 USD against its mark.',
       'The desk marks the 1 unwound at -300 USD, in the client\'s favour, so at this price it '
        + 'gives up 50 USD against its mark.',
       'The desk marks the 0.25 unwound at 15,000 USD, in the desk\'s favour: a price in ZAR is '
        + 'not set against it.',
       'none']);
const READ = { data: {}, error: null };
check('a page reading no mark of a trade\'s own says why: the line holding the seat or the '
        + 'service\'s refusal, that the book is pricing, that the answer has no row of its own '
        + 'or that the trade is closed - and nothing where one is read; a netting set\'s page '
        + 'says the same with its own sentence for no row',
      'a held seat told the book is pricing, a wait for a read that is never asked; a mark in '
        + 'hand still said to be missing; or a set told it has no row while the book prices',
      [trade.noMark({ data: null, error: 'Needs validate: sales-9 holds it nowhere.' }),
       trade.noMark({ data: null, error: null }), trade.noMark(READ),
       trade.noMark(READ, undefined, true), trade.noMark(READ, MARKS[0]),
       trade.noMark(READ, undefined, false, 'nothing is booked under it'),
       trade.noMark({ data: null, error: null }, undefined, false, 'nothing is booked under it')],
      ['Needs validate: sales-9 holds it nowhere.', 'pricing the book…',
       'The risk answer holds no mark of this trade\'s own.',
       'Closed: the record holds none of it, so the book marks it at nothing.', null,
       'nothing is booked under it', 'pricing the book…']);

// --- the way there and back
const said = [];
state.openTrade((action) => said.push(action), '0/1/1');
const at = [{ type: 'TAB', tab: 'blotter' }, ...said].reduce(state.reducer, state.INITIAL);
check('a trade opens on its own page, which goes back to the tab it was opened from',
      'the tab arrived at kept as where to go back to, so back stays on the page',
      [said, at.selection.picks.trade, at.tab, at.back], [[
        { type: 'PICK', screen: 'trade', id: '0/1/1' }, { type: 'TAB', tab: 'trade' }],
      '0/1/1', 'trade', 'blotter']);
check('opening another trade from the page keeps the way back',
      'a tab asked for twice taking itself as where it came from',
      state.reducer(at, { type: 'TAB', tab: 'trade' }).back, 'blotter');

const opens = (deal, paper) => {
  const asked = [];
  state.opened((action) => asked.push(action), BOOK_DOC, deal, paper);
  return asked.map((action) => [action.type, action.tab ?? action.id ?? action.path]);
};
check('a row naming a trade or a leg opens the trade\'s page; a set, a position or an agreement '
        + 'opens where it did',
      'every deal path opened as a trade, a netting set landing on a page with no trade; or a '
        + 'trade opened on the Portfolio as before',
      [opens('0/1/1'), opens('1'), opens('0'), opens('position:["x","ISDA-A","BOOK/FX"]'),
       opens(null, 'ISDA-A')],
      [[['PICK', '0/1'], ['TAB', 'trade']], [['PICK', '1'], ['TAB', 'trade']],
       [['SELECT_DEAL', '0'], ['TAB', 'portfolio']],
       [['SELECT_DEAL', 'position:["x","ISDA-A","BOOK/FX"]'], ['TAB', 'portfolio']],
       [['PICK', 'ISDA-A'], ['TAB', 'paper']]]);

// --- a ticket to sign
const QUOTED = { structure: 'ZeroCostCollar', variation: 'floor' };
const SCHEMA = {
  Structure: { types: { ZeroCostCollar: { vernacular: 'zero-cost collar, range forward' } } },
  Instrument: { types: { NettingCollateralSet: ['Set'] }, sections: { Set: {
    Collateralized: { value: 'False', values: ['True', 'False'] },
    Agreement_Currency: { value: '' }, Balance_Currency: { value: '' } } } },
};
const collar = rows.get('0/1');
check('a structure booked from a quote is its sales name and form; any other trade its type, a '
        + 'structure\'s with its legs counted',
      'the structure\'s type named where its quote declares a sales name, a collar read as '
        + 'StructuredDeal; or an undeclared structure left nameless',
      [trade.tradeName(collar, QUOTED, SCHEMA),
       trade.tradeName(collar, { structure: 'Seagull', variation: null }, null),
       trade.tradeName(collar, undefined, SCHEMA), trade.tradeName(rows.get('0/0'), undefined, null)],
      ['zero-cost collar · floor', 'Seagull', 'StructuredDeal of 2 legs', 'FXOptionDeal']);

const OPTION = blotter.dealRow({
  Object: 'FXOptionDeal', Reference: 'USDZAR-CALL', Currency: 'ZAR', Underlying_Currency: 'USD',
  Underlying_Amount: 2000000, FX_Volatility: 'USD.ZAR', Option_Type: 'Call', Buy_Sell: 'Buy',
  Strike_Price: 18.25, Expiry_Date: ts('2027-05-16') });
const NDF = blotter.dealRow({
  Object: 'FXNonDeliverableForward', Reference: 'GBPZAR-NDF', Buy_Currency: 'GBP',
  Buy_Amount: -5000000, Sell_Currency: 'ZAR', Sell_Amount: -113800000,
  Settlement_Date: ts('2027-02-16') });
const CROSSES = { 'Price Factors': { 'FXVol.GBP.USD': {}, 'FXVol.USD.ZAR': {} } };
check('a ticket reads aloud as what it is, the desk\'s side and size, on what, its levels on the '
        + 'pair - a structure\'s leg by leg with the desk\'s side of each - and when it ends',
      'the legs\' levels said without the desk\'s side of each, a collar that cannot tell the put '
        + 'it sells from the call it buys; or a forward\'s outright said as a strike',
      [trade.ticketWords(OPTION, 'FXOptionDeal', null, null),
       trade.ticketWords(NDF, 'FXNonDeliverableForward', CROSSES, null),
       trade.ticketWords(collar, 'zero-cost collar · floor', null, null)],
      ['FXOptionDeal · desk buys 2,000,000 USD · on USDZAR call · strike 18.25 · expires 2027-05-16',
       'FXNonDeliverableForward · desk sells 5,000,000 GBP · on GBPZAR · at 22.76 · expires '
         + '2027-02-16',
       'zero-cost collar · floor · 1,000,000 USD · on USDZAR · desk sells put 17.5, desk buys call '
         + '17.76 · expires 2027-05-14']);

const local = (day, hour, minute) => new Date(2026, 10, day, hour, minute).getTime();
const stamp = (ms) => new Date(ms).toISOString();
const waited = (since, now) => trade.waitedText(since, now);
check('a wait reads in minutes within the hour, hours on the day, since yesterday, then days - '
        + 'counted on the calendar and floored',
      'days counted in 24-hour blocks: a ticket booked at nine last night read as 12 hours and '
        + 'not since yesterday, one from three mornings back as 2 days; or the hours rounded up',
      [waited(stamp(local(19, 9, 0)), local(19, 9, 0) + 30_000),
       waited(stamp(local(19, 9, 0)), local(19, 9, 12)),
       waited(stamp(local(18, 23, 50)), local(19, 0, 5)),
       waited(stamp(local(19, 1, 0)), local(19, 23, 30)),
       waited(stamp(local(18, 21, 0)), local(19, 9, 0)),
       waited(stamp(local(16, 10, 0)), local(19, 9, 0))],
      ['under a minute', '12 minutes', '15 minutes', '22 hours', 'since yesterday', '3 days']);
check('a wait reads the service\'s own stamps - microseconds and an offset - and a clock ahead of '
        + 'this one as no wait at all',
      'the offset dropped, a ticket booked a minute ago in another zone read hours old',
      [waited('2026-11-19T07:00:00.123456+02:00', Date.parse('2026-11-19T05:01:30Z')),
       waited('2026-11-19T07:00:00Z', Date.parse('2026-11-19T06:58:00Z')), waited(null, 0)],
      ['1 minute', 'under a minute', '']);

const PAPER = [
  { agreement: 'ISDA-CSA', entity: 'A', kind: 'ISDA 2002 with CSA', terms: {
    Collateralized: 'True', Agreement_Currency: 'USD', Balance_Currency: 'EUR' } },
  { agreement: 'ISDA-K', entity: 'K', kind: 'ISDA 2002', terms: { Agreement_Currency: 'USD' } },
  { agreement: 'ISDA-SAYS-CSA', entity: 'S', kind: 'ISDA 2002 with CSA', terms: {
    Collateralized: 'False' } },
  { agreement: 'CSA-USD', entity: 'U', kind: 'CSA', terms: {
    Collateralized: 'True', Agreement_Currency: 'USD' } },
];
check('the paper says its agreement and kind, and collateral by its own terms - the flag at its '
        + 'declared convention where unstated - in the currency its balance is kept in',
      'an unstated flag read as on, the declared convention ignored; or collateral read off the '
        + 'document\'s name, a set whose terms say no read as collateralised',
      ['ISDA-CSA', 'ISDA-K', 'ISDA-SAYS-CSA', 'CSA-USD', 'ISDA-X'].map(
        (set) => trade.paperText(set, PAPER, SCHEMA)),
      ['ISDA-CSA (ISDA 2002 with CSA), collateralised in EUR', 'ISDA-K (ISDA 2002), uncollateralised',
       'ISDA-SAYS-CSA (ISDA 2002 with CSA), uncollateralised', 'CSA-USD (CSA), collateralised in USD',
       'ISDA-X, which the record declares no agreement for']);

done();
