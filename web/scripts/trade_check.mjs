// Drives `src/trade.ts` where the eye cannot: which node a path's trade is, which mark is its own,
// how the record is read whole and what of it is the trade's, who booked it, its size and dates,
// and what an unwind files and says - and the store's way back to where the page was opened from.
// Every check names the MUTATION it kills - the change that would still typecheck, still render,
// and still be wrong.
//
//   node web/scripts/trade_check.mjs          # exits 1 on a miss
//
// The book is a client's netting set holding an option and a two-legged collar, and a forward at
// the book's root; the record is a fake reader paging rows as `GET /book/activity` does.

import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const load = async (module) => {
  const bundled = await build({
    entryPoints: [fileURLToPath(new URL(`../src/${module}`, import.meta.url))],
    bundle: true, format: 'esm', write: false, logLevel: 'silent',
  });
  return import('data:text/javascript;base64,' +
    Buffer.from(bundled.outputFiles[0].text).toString('base64'));
};
const trade = await load('trade.ts');
const blotter = await load('blotter.ts');
const state = await load('state.ts');

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

const ts = (day) => ({ '.Timestamp': day });
const node = (deal, children) => ({ Instrument: { '.Deal': deal },
                                   ...(children ? { Children: children } : {}) });
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

// --- the mark
const MARKS = [{ reference: 'ISDA-A', deal_path: '0', value: 5.5 },
               { reference: 'FWD', deal_path: '1', value: -2.25 }];
check('a top-level trade has its own mark; one inside a set has the set\'s net, said as that',
      'the set\'s net returned as the trade\'s own mark, a number of three trades read as one',
      [trade.markOf(MARKS, '1'), trade.markOf(MARKS, '0/1')],
      [{ own: MARKS[1] }, { inside: MARKS[0] }]);

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

const ROWS = [
  { lsn: 23, event_type: 'approval', by: 'tier desk-auto', deal_path: '0/0' },
  { lsn: 24, event_type: 'fill', by: 'sales-1', deal_path: '0/0' },
  { lsn: 25, event_type: 'fill', by: 'sales-1', deal_path: '0/1' },
  { lsn: 31, event_type: 'amendment', by: 'sales-2', deal_path: '0/0' },
  { lsn: 33, event_type: 'fill', by: 'sales-2', deal_path: '0/0' },
  { lsn: 34, event_type: 'quote_filed', by: 'sales-1', deal_path: null },
];
const history = trade.historyOf(ROWS, '0/0');
check('a trade\'s history is every row opening it, newest first',
      'the rows left in the record\'s order, so the oldest act reads as the latest',
      history.map((row) => row.lsn), [33, 31, 24, 23]);
check('who booked it is its first fill on the record, never a later clip or the restrike',
      'the newest fill taken, naming the seat that unwound it as the one that booked it',
      trade.bookedBy(history)?.lsn, 24);

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
check('an unwind states what it deals in the currency its size is of, and nothing for a size in '
        + 'the underlying\'s own units',
      'a volume stated as a notional in a currency named "volume", which a capped tier then reads',
      [trade.notionalOf(CALL, -0.5),
       trade.notionalOf({ amount: { key: 'Volume', value: 100 },
                          of: { key: 'Volume', value: 'volume' } }, 1)],
      [{ notional: 1000000, notional_currency: 'USD' }, {}]);
check('the confirm says how much of what is held, the cash and what is left, in the trade\'s size',
      'what is left said as what was unwound, a confirm that cannot tell half from all; or the '
        + 'cash said as received whatever its sign',
      [trade.unwindWords(1, 0.25, 15000, 'USD', CALL),
       trade.unwindWords(-1, 1, 0, 'USD', {}), trade.unwindWords(-2, 1, 300, 'USD', {})],
      ['Unwind 0.25 (500,000 USD) of the 1 (2,000,000 USD) held, at 15,000 USD a unit: the desk '
        + 'receives 3,750 USD, and 0.75 (1,500,000 USD) is left.',
       'Unwind 1 of the 1 short held, at 0 USD a unit: no cash moves, and nothing is left.',
       'Unwind 1 of the 2 short held, at 300 USD a unit: the desk pays 300 USD, and 1 short is '
        + 'left.']);

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

console.log(`\n${ran} checks over 12 functions, ${missed.length} missed`);
if (missed.length) {
  console.log(missed.map((name) => `  ${name}`).join('\n'));
  process.exit(1);
}
