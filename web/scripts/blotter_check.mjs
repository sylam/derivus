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
      Near_Buy_Far_Sell_Ccy: text('Near Buy Far Sell Ccy'),
      Near_Sell_Far_Buy_Ccy: text('Near Sell Far Buy Ccy'),
      Near_Buy_Amount: sized('Near Buy Amount'), Near_Sell_Amount: sized('Near Sell Amount') },
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
      'the side left to the sign, which shows a GBP sale as `-5 000 000` and nothing else; or the '
      + 'word read off the field name alone, which calls a negative buy amount a Buy',
      [read('FXNonDeliverableForward', { Buy_Currency: 'GBP', Buy_Amount: -5e6,
                                         Sell_Currency: 'ZAR', Sell_Amount: -1.138e8,
                                         Settlement_Currency: 'ZAR' }),
       read('FXForwardDeal', { Buy_Currency: 'ZAR', Buy_Amount: 20091300, Sell_Currency: 'EUR',
                               Sell_Amount: 1e6 })],
      [['Sell', 'GBP', -5e6], ['Buy', 'ZAR', 20091300]]);

check('a count of shares is of no currency, and says it counts units',
      'the unit test dropped, which prints 25 000 shares as ZAR 25 000',
      read('EquityDeal', { Currency: 'ZAR', Units: 25000, Buy_Sell: 'Buy' }),
      ['Buy', 'units', 25000]);

check('a netting set\'s balance is no trade size, nor a key a declared type does not declare',
      'the candidates read without the `sized` declaration, which shows a collateral balance of '
      + '2 500 000 as the size of the set, and a stray Underlying_Amount on a structure as its own',
      [read('NettingCollateralSet', { Opening_Balance: 2.5e6, Balance_Currency: 'EUR' }),
       read('StructuredDeal', { Currency: 'ZAR', Underlying_Amount: 4e6 })],
      [[null, null, null], [null, null, null]]);

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

console.log(`\n${ran} checks over 2 functions, ${missed.length} missed`);
if (missed.length) {
  console.log(missed.map((name) => `  ${name}`).join('\n'));
  process.exit(1);
}
