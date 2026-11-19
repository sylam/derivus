// Drives `src/blotter.ts`'s size reading where the eye cannot: which amount a row shows, the
// currency that amount is an amount OF, and the side it is held on. Every check names the MUTATION
// it kills - the change that would still typecheck, still render, and still be wrong.
//
//   node web/scripts/blotter_check.mjs          # exits 1 on a miss
//
// The declarations are the service's own `GET /schema` answer, trimmed to the sections and fields
// a check turns on: `sized`, `side` and `settles` exactly as `F.descriptor` publishes them.

import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const bundled = await build({
  entryPoints: [fileURLToPath(new URL('../src/blotter.ts', import.meta.url))],
  bundle: true, format: 'esm', write: false, logLevel: 'silent',
});
const blotter = await import('data:text/javascript;base64,' +
  Buffer.from(bundled.outputFiles[0].text).toString('base64'));

// ---- the declarations ---------------------------------------------------------------------------

const text = (description) => ({ widget: 'Text', description, value: '' });
const side = (description) => ({ ...text(description), side: true });
const sized = (description) => ({ widget: 'Float', description, value: 0, sized: true });
const date = (description, ...settles) => (
  { widget: 'DatePicker', description, value: '', settles });
const cash = (currency, amount = null, sign = 1.0, otherwise = null) => (
  { currency, amount, sign, otherwise });

const SCHEMA = { Instrument: {
  containers: ['NettingCollateralSet', 'StructuredDeal'],
  types: {
    FXOptionDeal: ['FXAdmin', 'FXOptionBase.Fields', 'FXOptionDeal.Fields'],
    FXOneTouchOption: ['Pricing', 'FXAdmin', 'FXOneTouchOption.Fields', 'Touch'],
    FXNonDeliverableForward: ['FXNonDeliverableForward.Fields'],
    FXForwardDeal: ['FXForwardDeal.Fields'],
    FXSwapDeal: ['FXSwapDeal.Fields'],
    EquityDeal: ['EquityDeal.Fields'],
    NettingCollateralSet: ['NettingCollateralSet.Fields'],
    StructuredDeal: ['StructuredDeal.Fields'],
  },
  sections: {
    Pricing: { Currency: text('Currency') },
    FXAdmin: { Delivery_Date: date('Delivery Date',
                                   cash('Payoff_Currency', null, 1.0, 'Expiry_Date')) },
    'FXOptionBase.Fields': { Currency: text('Currency'),
                             Underlying_Currency: text('Underlying Currency'),
                             Buy_Sell: side('Buy Sell') },
    'FXOptionDeal.Fields': { Underlying_Amount: sized('Underlying Amount') },
    'FXOneTouchOption.Fields': { Payoff_Currency: text('Payoff Currency'),
                                 Underlying_Currency: text('Underlying Currency') },
    Touch: { Buy_Sell: side('Buy Sell'), Cash_Payoff: sized('Cash Payoff') },
    'FXNonDeliverableForward.Fields': {
      Sell_Currency: text('Sell Currency'), Sell_Amount: sized('Sell Amount'),
      Settlement_Date: date('Settlement Date', cash('Settlement_Currency')),
      Settlement_Currency: text('Settlement Currency'), Buy_Amount: sized('Buy Amount'),
      Buy_Currency: text('Buy Currency') },
    'FXForwardDeal.Fields': {
      Sell_Currency: text('Sell Currency'), Sell_Amount: sized('Sell Amount'),
      Settlement_Date: date('Settlement Date', cash('Buy_Currency', 'Buy_Amount'),
                            cash('Sell_Currency', 'Sell_Amount', -1.0)),
      Buy_Amount: sized('Buy Amount'), Buy_Currency: text('Buy Currency') },
    'FXSwapDeal.Fields': {
      Near_Settlement_Date: date('Near Settlement Date',
                                 cash('Near_Buy_Far_Sell_Ccy', 'Near_Buy_Amount'),
                                 cash('Near_Sell_Far_Buy_Ccy', 'Near_Sell_Amount', -1.0)),
      Far_Settlement_Date: date('Far Settlement Date',
                                cash('Near_Sell_Far_Buy_Ccy', 'Far_Buy_Amount'),
                                cash('Near_Buy_Far_Sell_Ccy', 'Far_Sell_Amount', -1.0)),
      Near_Buy_Far_Sell_Ccy: text('Near Buy Far Sell Ccy'),
      Near_Sell_Far_Buy_Ccy: text('Near Sell Far Buy Ccy'),
      Near_Buy_Amount: sized('Near Buy Amount'), Near_Sell_Amount: sized('Near Sell Amount'),
      Far_Buy_Amount: sized('Far Buy Amount'), Far_Sell_Amount: sized('Far Sell Amount') },
    'EquityDeal.Fields': { Currency: text('Currency'), Buy_Sell: side('Buy Sell'),
                           Units: sized('Units') },
    'NettingCollateralSet.Fields': {
      Balance_Currency: text('Balance Currency'),
      Opening_Balance: { widget: 'Float', description: 'Opening Balance', value: 0 } },
    'StructuredDeal.Fields': { Currency: text('Currency') },
  },
} };

// ---- the checks ---------------------------------------------------------------------------------

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

/** A row's size as a reader sees it: [side, what the amount is of, the amount]. */
const read = (object, deal, schema = SCHEMA) => {
  const size = blotter.sizeOf({ Object: object, ...deal }, object, schema);
  return [size.side?.value ?? null, size.of?.value ?? null, size.amount?.value ?? null];
};

check('an FX option\'s notional is of its underlying currency, whichever way round the pair is',
      'the deal\'s own Currency read ahead of the pairing, which prints USD 2m as ZAR 2m and ZAR '
      + '60m as EUR 60m - off by twenty times',
      [read('FXOptionDeal', { Currency: 'ZAR', Underlying_Currency: 'USD', Underlying_Amount: 2e6,
                              Buy_Sell: 'Buy' }),
       read('FXOptionDeal', { Currency: 'EUR', Underlying_Currency: 'ZAR', Underlying_Amount: 6e7,
                              Buy_Sell: 'Sell' })],
      [['Buy', 'USD', 2e6], ['Sell', 'ZAR', 6e7]]);

check('an amount a settles declaration pays is of the currency it names, where names pair nothing',
      'the settles declaration not read, which leaves an FX swap\'s near leg with no currency - '
      + 'its amounts and currencies share no word',
      read('FXSwapDeal', { Near_Buy_Far_Sell_Ccy: 'EUR', Near_Sell_Far_Buy_Ccy: 'USD',
                           Near_Buy_Amount: 1e6, Near_Sell_Amount: 1158000 }),
      ['Buy', 'EUR', 1e6]);

const TOUCH = { Currency: 'ZAR', Underlying_Currency: 'USD', Cash_Payoff: 1e6, Buy_Sell: 'Buy' };
check('a payout is of the payoff currency the deal states, and of its own Currency where blank',
      'the pairing read as `<Stem>_Amount` to `<Stem>_Currency` alone, which puts a one-touch\'s '
      + 'USD payout beside ZAR; or a blank currency read as a statement, which prints an empty Ccy',
      [read('FXOneTouchOption', { ...TOUCH, Payoff_Currency: 'USD' }),
       read('FXOneTouchOption', { ...TOUCH, Payoff_Currency: '' })],
      [['Buy', 'USD', 1e6], ['Buy', 'ZAR', 1e6]]);

check('a forward\'s direction is a word: a buy amount buys, and a negative one sells',
      'the side left to the sign, which shows a GBP sale as `-5 000 000` and nothing else; the '
      + 'word read off the field name alone, which calls a negative buy amount a Buy; or the sign '
      + 'kept beside the word, which reads as the sale of a negative',
      [read('FXNonDeliverableForward', { Buy_Currency: 'GBP', Buy_Amount: -5e6,
                                         Sell_Currency: 'ZAR', Sell_Amount: -1.138e8,
                                         Settlement_Currency: 'ZAR' }),
       read('FXForwardDeal', { Buy_Currency: 'ZAR', Buy_Amount: 20091300, Sell_Currency: 'EUR',
                               Sell_Amount: 1e6 })],
      [['Sell', 'GBP', 5e6], ['Buy', 'ZAR', 20091300]]);

check('a count of shares is of no currency, and says it counts units',
      'the unit test dropped, which prints 25 000 shares as ZAR 25 000',
      read('EquityDeal', { Currency: 'ZAR', Units: 25000, Buy_Sell: 'Buy' }),
      ['Buy', 'units', 25000]);

check('a netting set\'s balance is no trade size, nor a key a declared type does not declare',
      'the candidates read without the `sized` declaration, which shows a collateral balance of '
      + '2 500 000 as the size of the set, and a stray Underlying_Amount on a structure as its own; '
      + 'or the currency dropped with the amount, which blanks the Ccy of every list and structure',
      [read('NettingCollateralSet', { Opening_Balance: 2.5e6, Balance_Currency: 'EUR' }),
       read('StructuredDeal', { Currency: 'ZAR', Underlying_Amount: 4e6 })],
      [[null, null, null], [null, 'ZAR', null]]);

check('a type the schema does not declare still shows the amount it carries',
      'the `sized` test asked of a type nobody declared, which blanks the size of every type the '
      + 'schema does not know - the blotter renders value-first',
      [read('NewDeal', { Currency: 'USD', Principal: 1e6 }),
       read('NewDeal', { Currency: 'USD', Principal: 1e6 }, null)],
      [[null, 'USD', 1e6], [null, 'USD', 1e6]]);

check('a cell\'s hover is the field\'s declared description, the key where none is declared',
      'the sections not merged, which shows the raw key on every cell whose field is not in the '
      + 'type\'s first section',
      [blotter.describe(SCHEMA, 'FXOptionDeal', 'Underlying_Amount'),
       blotter.describe(SCHEMA, 'FXOptionDeal', 'Nothing_Declared'),
       blotter.describe(null, 'FXOptionDeal', 'Underlying_Amount')],
      ['Underlying Amount', 'Nothing_Declared', 'Underlying_Amount']);

// ---- the market's axis, what a trade is on, its size under the record, its terms ----------------
// Deals as the lab book writes them: an EURZAR collar on ZAR, a USDZAR one on USD, a forward and two
// NDFs naming no pair, a TARF, a range accrual, an Asian, a platinum barrier.

const MARKET = { 'Price Factors': { 'FXVol.EUR.ZAR': {}, 'FXVol.USD.ZAR': {}, 'FxRate.GBP': {} } };
const ON_ZAR = { Object: 'FXOptionDeal', Currency: 'EUR', Underlying_Currency: 'ZAR',
                 FX_Volatility: 'EUR.ZAR', Underlying_Amount: 6e7, Buy_Sell: 'Sell',
                 Option_Type: 'Call', Strike_Price: 0.05128205128205128 };
const ON_USD = { Object: 'FXOptionDeal', Currency: 'ZAR', Underlying_Currency: 'USD',
                 FX_Volatility: 'USD.ZAR', Underlying_Amount: 2e6, Buy_Sell: 'Sell',
                 Option_Type: 'Call', Strike_Price: 18.25 };
const FORWARD = { Object: 'FXForwardDeal', Buy_Currency: 'ZAR', Buy_Amount: 20091300,
                  Sell_Currency: 'EUR', Sell_Amount: 1e6 };
const NDF = { Object: 'FXNonDeliverableForward', Buy_Currency: 'USD', Buy_Amount: -3e6,
              Sell_Currency: 'ZAR', Sell_Amount: -5.49e7, Settlement_Currency: 'ZAR' };
const ROWS = (n) => Array.from({ length: n }, () => [{ '.Timestamp': '2027-01-16' }, null]);
const TARF = { ...ON_USD, Object: 'FXTARFOptionDeal', Option_Type: 'Put', Strike_Price: 18.6148,
               TargetLevel: 2.5, LeverageNotional: 8e6, Underlying_Amount: 4e6, Barrier: 0.0,
               InvertedTarget: false, TARF_ExpiryDates: ROWS(6) };
const RANGE = { Object: 'FXRangeAccrualDeal', Currency: 'EUR', Underlying_Currency: 'ZAR',
                FX_Volatility: 'EUR.ZAR', Buy_Sell: 'Sell', Lower_Barrier: 0.04587155963302752,
                Upper_Barrier: 0.050505050505050504, Payoff: 63151.84, Observation_Dates: ROWS(12) };
const XPT = { Object: 'CommodityBarrierOption', Currency: 'USD', Reference_Type: 'XPT',
              Buy_Sell: 'Sell', Option_Type: 'Call', Strike_Price: 1050, Units: 500,
              Barrier_Type: 'Up_And_Out', Barrier_Price: 1250 };

const node = (deal, children) => (
  { Instrument: { '.Deal': deal }, ...(children && { Children: children }) });
const rowOf = (deal, children) => blotter.toRows([node(deal, children)], ['StructuredDeal'],
                                                 '2026-11-16')[0];
/** A row as a reader sees it - [on, sense, strike, barrier, terms] - numbers as cells print them. */
const shown = (deal, children, market = MARKET, schema = null) => {
  const read = blotter.quoted(rowOf(deal, children), market, schema);
  return [read.on?.value ?? null, read.sense ?? null, blotter.cellText(read.strike?.value) ?? null,
          blotter.cellText(read.barrier?.value) ?? null, read.terms?.parts.join(' · ') ?? null];
};

check('a level written on the pair\'s quote currency reads one over it, its sense turned',
      'the level shown as written, which prints the EURZAR collar as 0.0512821 against a market '
      + 'quoting 19.5; or the sense not turned, which calls a call on the rand an EURZAR call when '
      + 'it is the put on the euro',
      shown(ON_ZAR), ['EURZAR', 'Put', '19.5', null, null]);

check('a level written on the pair\'s base currency reads as written, its sense as written',
      'every level of a deal naming a pair turned, which prints USDZAR 18.25 as 0.0547945 and '
      + 'its call as a put',
      shown(ON_USD), ['USDZAR', 'Call', '18.25', null, null]);

check('a level\'s hover says what is shown and which field it was read off, as written',
      'the hover left on the field\'s description, which never says the cell is 1 / Strike_Price',
      blotter.quoted(rowOf(ON_ZAR), MARKET).strike.title,
      'EURZAR 19.5, written as Strike_Price 0.0512821 EUR per ZAR');

check('a forward\'s outright is the quote currency\'s amount per the base\'s, on the market\'s '
      + 'pair',
      'the forward\'s rate read the wrong way up, which prints EURZAR 20.0913 as 0.0497728; or the '
      + 'pair read off the currencies\' order on the deal, which calls a ZAR-bought forward ZAREUR',
      [shown(FORWARD), shown(NDF)],
      [['EURZAR', null, '20.0913', null, null], ['USDZAR', null, '18.3', null, null]]);

check('a currency no surface of the market relates to the other is on no pair, and shows no rate',
      'a pair guessed from the deal\'s own order where the market says nothing, which prints '
      + 'GBPZAR off a Buy_Currency the deal sells',
      shown({ ...NDF, Buy_Currency: 'GBP' }), [null, null, null, null, null]);

// a market quoting the dollar legs and one cross of its own
const LEGS = { 'Price Factors': { 'FXVol.GBP.USD': {}, 'FXVol.USD.ZAR': {}, 'FXVol.EUR.USD': {},
                                  'FXVol.EUR.ZAR': {} } };
const GBPZAR = { ...NDF, Buy_Currency: 'GBP', Buy_Amount: -5e6, Sell_Amount: -1.138e8 };
check('a cross the market holds no surface for reads as its legs order it, its outright on that '
      + 'pair, whichever way the deal writes it',
      'the chain dropped - only a surface of the pair itself read - which leaves a GBPZAR NDF with '
      + 'no pair and no rate; or the deal\'s order read, which prints ZARGBP 0.0439 for one '
      + 'written buying the rand',
      [shown(GBPZAR, undefined, LEGS),
       shown({ ...GBPZAR, Buy_Currency: 'ZAR', Buy_Amount: 1.138e8, Sell_Currency: 'GBP',
               Sell_Amount: 5e6 }, undefined, LEGS)],
      [['GBPZAR', null, '22.76', null, null], ['GBPZAR', null, '22.76', null, null]]);
const on = (deal, market) => blotter.quoted(rowOf(deal), market).on;
check('the pair\'s hover names the surfaces it was read off, and a chained one says so',
      'the shortest chain not sought, which reads EURZAR through the dollar beside the EURZAR '
      + 'surface the market holds; or the hover left on one surface name the market does not hold',
      [on(GBPZAR, LEGS).key, on(GBPZAR, LEGS).title, on(FORWARD, LEGS).key,
       on(FORWARD, LEGS).title ?? null],
      ['FXVol.GBP.USD × FXVol.USD.ZAR', 'the market holds no GBPZAR surface, so the pair reads as '
        + 'its legs are quoted: GBPUSD × USDZAR', 'FXVol.EUR.ZAR', null]);
check('two currencies the surfaces order against a third but not against each other are on no pair',
      'the surfaces walked either way round, which strings EUR and GBP together through the dollar '
      + 'they are both quoted against and prints an EURGBP nobody\'s market says',
      shown({ ...FORWARD, Buy_Currency: 'GBP', Buy_Amount: 860000 }, undefined, LEGS),
      [null, null, null, null, null]);

const SWAP = { Object: 'FXSwapDeal', Near_Buy_Far_Sell_Ccy: 'EUR', Near_Sell_Far_Buy_Ccy: 'USD',
               Near_Buy_Amount: 1e6, Near_Sell_Amount: 1158000, Far_Buy_Amount: 1166500,
               Far_Sell_Amount: 1e6 };
check('a swap is on the pair its legs exchange, its far outright the strike and its near in its '
      + 'terms, whichever currency it buys near',
      'the exchanges read off the Buy_ and Sell_ words alone, which leaves a swap with no pair and '
      + 'no rate; the near shown as the strike beside the far leg\'s expiry; or the near dropped',
      [shown(SWAP, undefined, LEGS, SCHEMA),
       shown({ ...SWAP, Near_Buy_Far_Sell_Ccy: 'USD', Near_Sell_Far_Buy_Ccy: 'EUR',
               Near_Buy_Amount: 1158000, Near_Sell_Amount: 1e6, Far_Buy_Amount: 1e6,
               Far_Sell_Amount: 1166500 }, undefined, LEGS, SCHEMA),
       shown(FORWARD, undefined, LEGS, SCHEMA)],
      [['EURUSD', null, '1.1665', null, 'near 1.158'], ['EURUSD', null, '1.1665', null, 'near 1.158'],
       ['EURZAR', null, '20.0913', null, null]]);

check('a deal naming no pair - a metal - reads as written, on what it names',
      'the commodity read as a pair or turned, or what it is on left blank for want of a pair',
      shown(XPT), ['XPT', 'Call', '1,050', '1,250', 'up and out']);

check('a range reads on the pair\'s axis, its two bounds crossed by the turn, and its barrier '
      + 'direction turned with it',
      'the range shown as written (0.0458716 - 0.0505051); the bounds left in their written order, '
      + 'which prints 21.8 - 19.8; or an up-and-in on the rand left up-and-in on the euro',
      [shown(RANGE),
       shown({ ...ON_ZAR, Object: 'FXBarrierOption', Barrier_Type: 'Up_And_In',
               Barrier_Price: 0.045 })[4]],
      [['EURZAR', null, null, null,
        '12 fixings, amount per fixing · range 19.8\u00a0–\u00a021.8'], 'down and in']);

check('a strip states its fixings and that the amount is per fixing, its target in its unit and '
      + 'its leverage, and a barrier of nought is no barrier',
      'the barrier read whatever its value, which prints a TARF\'s 0 as a level; the target\'s unit '
      + 'dropped; or the leverage read off the leveraged notional alone, which prints 8,000,000',
      shown(TARF),
      ['USDZAR', 'Put', '18.6148', null,
       '6 fixings, amount per fixing · target 2.5 ZAR per USD · leverage 2×']);

check('an average states its fixings and never that its amount is per fixing, and a type nobody '
      + 'anticipated says nothing',
      'an Asian\'s samples read as a strip, which calls its whole notional a per-fixing amount; or '
      + 'a term invented for a type carrying none of the candidates',
      [shown({ ...ON_USD, Object: 'FXDiscreteExplicitAsianOption', Sampling_Data: ROWS(3) })[4],
       shown({ Object: 'NewDeal', Currency: 'USD', Principal: 1e6 })],
      ['average of 3 fixings', [null, null, null, null, null]]);

check('a structure is on the one pair its legs are on, and a set holding two pairs on none',
      'the container\'s pair read off its first leg alone, which files a set holding an EURZAR '
      + 'collar and a USDZAR TARF as EURZAR',
      [shown({ Object: 'StructuredDeal', Currency: 'EUR' },
             [node(ON_ZAR), node({ ...ON_ZAR, Option_Type: 'Put' })])[0],
       shown({ Object: 'StructuredDeal', Currency: 'EUR' }, [node(ON_ZAR), node(TARF)])[0]],
      ['EURZAR', null]);

/** A size under the record: [side, amount, the amount's hover]. */
const held = (deal, units) => {
  const size = blotter.sizeOf(deal, deal.Object, SCHEMA, units);
  return [size.side?.value ?? null, size.amount?.value ?? null, size.amount?.title ?? null];
};

check('under a position the amount is the written one times the units held, said in its hover',
      'the amount shown as written under a position, which prints 4,000,000 beside a Qty of 0.5',
      held({ ...ON_USD, Underlying_Amount: 4e6 }, 0.5),
      ['Sell', 2e6, '4,000,000 written, 0.5 held']);

check('a short position holds the other side, its amount unsigned, a word side as a field side',
      'the side kept under a short position, or the sign carried into the amount, which prints '
      + 'Sell -2,000,000 for a bought option unwound past flat',
      [held(ON_USD, -1), held(FORWARD, -1)],
      [['Buy', 2e6, '2,000,000 written, -1 held'],
       ['Sell', 20091300, '20,091,300 written, -1 held']]);

const BOOK = blotter.toRows([node({ Object: 'NettingCollateralSet', Reference: 'ISDA-1' }, [
  node({ Object: 'StructuredDeal', Reference: 'FE' }, [node(ON_USD), node(TARF)]),
  node(FORWARD)])], ['NettingCollateralSet', 'StructuredDeal'], '2026-11-16');
const POSITION = { instrument: 'a'.repeat(64), agreement: 'ISDA-1', portfolio: 'desk',
                   counterparty: 'K', quantity: 0.5, deal_paths: ['0/0'], reference: 'FE' };
const units = (rows) => rows.map((row) => [row.units ?? null, units(row.children)]);

check('every leg of a structure holds the units its position does, in the file\'s tree and the '
      + 'record\'s, and a node no position carries holds none',
      'the legs left at the written amount under a halved structure - the units read off the '
      + 'position\'s own node alone - or a node no position carries scaled by its neighbour\'s',
      [units(blotter.heldRows(BOOK, new Map([['0/0', [POSITION,
                                                      { ...POSITION, quantity: 0.25 }]]]))),
       units(blotter.groupedRows([{ id: 'p', label: 'FE' }], blotter.rowsByPath(BOOK),
                                 new Map([['p', POSITION]]), '2026-11-16'))],
      [[[null, [[0.75, [[0.75, []], [0.75, []]]], [null, []]]]],
       [[0.5, [[0.5, []], [0.5, []]]]]]);

const SHORT = { ...POSITION, quantity: -1 };
check('a structure held short is short in every leg, in the file\'s tree and the record\'s',
      'the sign of the units dropped on the way down to the legs, a short structure\'s legs then '
      + 'reading the side they were written on',
      [units(blotter.heldRows(BOOK, new Map([['0/0', [SHORT]]]))),
       units(blotter.groupedRows([{ id: 'p', label: 'FE' }], blotter.rowsByPath(BOOK),
                                 new Map([['p', SHORT]]), '2026-11-16'))],
      [[[null, [[-1, [[-1, []], [-1, []]]], [null, []]]]],
       [[-1, [[-1, []], [-1, []]]]]]);

check('a leg reads by its role under its structure, and any other reference whole',
      'every reference written whole, which clips a leg\'s role off a 1600 px screen; or a '
      + 'reference shortened by a structure it does not begin with',
      [blotter.nameOf({ reference: 'ZeroCostCollar-6a90461d_protection',
                        deal: { Structure_Reference: 'ZeroCostCollar-6a90461d' } }),
       blotter.nameOf({ reference: 'EURZAR-ZARCALL-REBOOKED-2027-05',
                        deal: { Structure_Reference: 'ZeroCostCollar-6a90461d' } }),
       blotter.nameOf({ reference: 'ZeroCostCollar-6a90461d', deal: {} })],
      ['protection', 'EURZAR-ZARCALL-REBOOKED-2027-05', 'ZeroCostCollar-6a90461d']);

console.log(`\n${ran} checks over 9 functions, ${missed.length} missed`);
if (missed.length) {
  console.log(missed.map((name) => `  ${name}`).join('\n'));
  process.exit(1);
}
