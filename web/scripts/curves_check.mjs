// Drives `src/curves.ts` where the eye cannot: the row edits a desk makes, the request they build,
// and the fields a curve may be stated in. Every check names the MUTATION it kills - the change to
// the module that would still typecheck, still render, and still be wrong.
//
//   node web/scripts/curves_check.mjs        # exits 1 on a miss
//
// The shapes are the service's own answers, `GET /book/curve` on a book carrying one ZAR curve and
// the price factor store the solve writes into. The module is bundled rather than transformed,
// because it reads the wire tokens through `src/tokens.ts` rather than spelling them twice.

import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const bundled = await build({
  entryPoints: [fileURLToPath(new URL('../src/curves.ts', import.meta.url))],
  bundle: true, format: 'esm', write: false, logLevel: 'silent',
});
const curves = await import('data:text/javascript;base64,' +
  Buffer.from(bundled.outputFiles[0].text).toString('base64'));

// ---- the service's shapes ------------------------------------------------------------------------

const SEEDED = {
  currency: 'ZAR',
  conventions: {
    curve_day_count: 'ACT_365', spot_days: 0, front: 'fixings/3M', front_day_count: 'ACT_365',
    compounding: 'None', fixed_frequency: '3M', float_frequency: '3M',
    fixed_day_count: 'ACT_365', float_day_count: 'ACT_365', notional: 1000000.0, quote_scale: 1.0,
  },
  rows: [{ tenor: '3M', security: 'JIBA3M Index' }, { tenor: '1Y', security: 'SASW1 BGN Curncy' }],
};

const BLOCK = {
  curve: 'ZAR', currency: 'ZAR', discount_rate: '', interpolation: 'Linear',
  conventions: { ...SEEDED.conventions, calendar: '', near_interpolation: '', near_tenor: '' },
  rows: [{ tenor: '3M', security: 'JIBA3M Index', quote: 7.41, use: 'Yes' },
         { tenor: '1Y', security: 'SASW1 BGN Curncy', quote: 7.62, use: 'No' }],
};

// a block authored before it carried its definition: the read verb answers the quotes it can read,
// blank tenors, and a note where the conventions would be
const OLD_BLOCK = {
  curve: 'USD', currency: 'USD', discount_rate: '', interpolation: 'Linear',
  note: "authored without 'Spot_Days' - set it up again through POST /book/curve",
  rows: [{ tenor: '', security: 'US0003M Index', quote: 5.3, use: 'Yes' },
         { tenor: '2Y', security: 'USSW2 Curncy', quote: 4.1, use: 'Yes' }],
};

const ANSWER = { etag: 'e', curves: { 'InterestRatePrices.ZAR': BLOCK }, seeded: { ZAR: SEEDED } };
const BLANK_BOOK = { etag: 'e', curves: {}, seeded: { ZAR: SEEDED } };
const OLD_BOOK = { etag: 'e', curves: { 'InterestRatePrices.USD': OLD_BLOCK }, seeded: {} };

// `MarketPrices.types.InterestRatePrices`, in declaration order. Only the names matter here: what
// is asserted is WHICH of them a request may state, never how one renders.
const DECLARED = Object.fromEntries([
  'Currency', 'Day_Count', 'Discount_Rate', 'Calendar', 'Spot_Days', 'Fixed_Frequency',
  'Float_Frequency', 'Fixed_Day_Count', 'Float_Day_Count', 'Front_Day_Count', 'Compounding',
  'Near_Interpolation', 'Near_Tenor', 'N_Iter', 'Tol', 'Damping_Halvings', 'Points',
].map((name) => [name, { widget: 'Text', description: name, value: '' }]));

const FACTORS = {
  'FxRate.ZAR': { Domestic_Currency: null, Interest_Rate: 'ZAR', Spot: 18.5 },
  'InterestRate.ZAR': {
    Currency: 'ZAR', Day_Count: 'ACT_365',
    Curve: { '.Curve': { meta: [], data: [[0.0, 0.0741], [1.0, 0.0762]] } },
  },
  'InterestRate.ZAR-ZARONIA': {
    Currency: 'ZAR', Day_Count: 'ACT_365',
    Curve: { '.Curve': { meta: [], data: [[0.0, 0.0735]] } },
  },
};

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

const seededRows = [
  { tenor: '3M', security: 'JIBA3M Index', quote: null, use: 'Yes' },
  { tenor: '1Y', security: 'SASW1 BGN Curncy', quote: null, use: 'Yes' },
];

// --- prefill
check('prefill completes a seeded row with the fields a request takes',
      'the spread reversed ({...row, ...BLANK}), which blanks the quotes a block already carries',
      curves.prefill('ZAR', SEEDED), {
        curve: 'ZAR', currency: 'ZAR', discount_rate: '', rows: seededRows,
        stated: SEEDED.conventions, edits: {},
      });
check('prefill keeps a block\'s own quotes and hold-outs',
      'the same reversal, read from the other side: the held-out 1Y comes back as Yes',
      curves.prefill('ZAR', BLOCK).rows, BLOCK.rows);
check('prefill on nothing is an empty form',
      'a default of the curve name for the currency, which sends a new curve a currency nobody stated',
      curves.prefill('', {}),
      { curve: '', currency: '', discount_rate: '', rows: [], stated: {}, edits: {} });

// --- withRow
const rows = curves.prefill('ZAR', SEEDED).rows;
check('withRow patches the row it names',
      'the index test dropped, which applies one cell\'s edit to every row in the ladder',
      curves.withRow(rows, 1, { use: 'No' }),
      [seededRows[0], { ...seededRows[1], use: 'No' }]);
check('withRow removes the row it names',
      'the filter inverted (i === index), which deletes every row but the one a desk asked to drop',
      curves.withRow(rows, 0, null), [seededRows[1]]);
check('withRow appends past the end',
      'an index test of `index > rows.length`, which makes `add a row` do nothing at all',
      curves.withRow(rows, rows.length, {}),
      [...seededRows, { tenor: '', security: '', quote: null, use: 'Yes' }]);
check('withRow leaves the list it was handed standing',
      'a splice or an in-place assignment, which edits the array a table is rendering from',
      rows, seededRows);

// --- edited
const form = curves.prefill('ZAR', SEEDED);
check('edited files a declared request field as the request\'s own',
      'every field landing in `edits`, after which `currency` travels as a convention the emitter '
      + 'reads nothing of and the request carries no currency at all',
      [curves.edited(form, 'Currency', 'USD').currency, curves.edited(form, 'Currency', 'USD').edits],
      ['USD', {}]);
check('edited files everything else as a convention',
      'the routing reversed, which writes a calendar over the currency',
      [curves.edited(form, 'Calendar', 'ZAR').edits, curves.edited(form, 'Calendar', 'ZAR').currency],
      [{ Calendar: 'ZAR' }, 'ZAR']);

// --- curveRequest
let composed = curves.edited(curves.edited(form, 'Calendar', 'ZAR'), 'Spot_Days', 0);
composed = { ...composed, rows: curves.withRow(composed.rows, 2,
  { tenor: ' 2Y ', security: ' SASW2 BGN Curncy ', quote: 7.94 }) };
composed = { ...composed, rows: curves.withRow(composed.rows, 3, {}) };
const request = curves.curveRequest(composed);
check('curveRequest states only the conventions the desk moved, in the verb\'s spelling',
      '`.toLowerCase()` dropped (the verb reads `Calendar` as a name it knows nothing of and '
      + 'refuses the write) or the equality filter dropped (`spot_days` travels as a convention '
      + 'nobody stated, which is the same number today and a lie the day the seed moves)',
      Object.keys(request).filter((key) =>
        !['curve', 'currency', 'discount_rate', 'rows'].includes(key)),
      ['calendar']);
check('curveRequest leaves out the row nobody named, and trims the rest',
      'the blank-tenor filter dropped: an empty row a desk added and left behind reaches the '
      + 'emitter, whose grammar refuses `` by name and writes nothing',
      request.rows, [
        { tenor: '3M', security: 'JIBA3M Index', quote: null, use: 'Yes' },
        { tenor: '1Y', security: 'SASW1 BGN Curncy', quote: null, use: 'Yes' },
        { tenor: '2Y', security: 'SASW2 BGN Curncy', quote: 7.94, use: 'Yes' }]);
check('curveRequest names the curve and its currency outright',
      'either read off `edits`, which leaves the two the verb requires missing',
      [request.curve, request.currency, request.discount_rate, request.calendar],
      ['ZAR', 'ZAR', '', 'ZAR']);

// --- curveFields
const fields = curves.curveFields(DECLARED, ANSWER);
check('curveFields is the declaration crossed with the spellings the service answers in',
      'the comparison un-lowercased (no declared field is a convention and a desk can state '
      + 'none) or the filter dropped (`N_Iter` travels as a convention and the verb refuses the '
      + 'write). `Day_Count` is out because the conventions spell that one `curve_day_count`',
      fields,
      ['Currency', 'Discount_Rate', 'Calendar', 'Spot_Days', 'Fixed_Frequency', 'Float_Frequency',
       'Fixed_Day_Count', 'Float_Day_Count', 'Front_Day_Count', 'Compounding',
       'Near_Interpolation', 'Near_Tenor']);
check('curveFields reads the seeded entries too',
      'the union taken over `curves` alone, which leaves a desk with a book carrying no curve yet '
      + 'nothing to state at all. A seed stating no `calendar` is why that field is the one '
      + 'missing here',
      curves.curveFields(DECLARED, BLANK_BOOK),
      ['Currency', 'Discount_Rate', 'Spot_Days', 'Fixed_Frequency', 'Float_Frequency',
       'Fixed_Day_Count', 'Float_Day_Count', 'Front_Day_Count', 'Compounding']);

// --- curveValues
check('curveValues reads a block through the same lower-case spelling',
      'the map dropped, after which every field renders at its declared default and a desk edits '
      + 'against a baseline the block never stated',
      curves.curveValues(fields, BLOCK), {
        Currency: 'ZAR', Discount_Rate: '', Calendar: '', Spot_Days: 0, Fixed_Frequency: '3M',
        Float_Frequency: '3M', Fixed_Day_Count: 'ACT_365', Float_Day_Count: 'ACT_365',
        Front_Day_Count: 'ACT_365', Compounding: 'None', Near_Interpolation: '', Near_Tenor: '',
      });
check('curveValues leaves out what the source does not state',
      'a `?? \'\'` fallback, which paints a blank over the declaration\'s own default and sends it '
      + 'back as an edit',
      Object.keys(curves.curveValues(fields, SEEDED)),
      ['Currency', 'Spot_Days', 'Fixed_Frequency', 'Float_Frequency', 'Fixed_Day_Count',
       'Float_Day_Count', 'Front_Day_Count', 'Compounding']);

// --- the block that states no conventions at all
check('the block with a note in place of its conventions still reads',
      'the conventions read by key rather than spread, which throws on the one block the read verb '
      + 'answers a note for and takes the whole screen down with it',
      [curves.curveFields(DECLARED, OLD_BOOK), curves.curveValues(fields, OLD_BLOCK),
       curves.prefill('USD', OLD_BLOCK).stated],
      [['Currency', 'Discount_Rate'], { Currency: 'USD', Discount_Rate: '' }, {}]);
check('its rows are stated again by the ones that name a tenor',
      'the blank-tenor filter dropped, which sends the emitter a row whose instrument nothing '
      + 'names - the note says that is what is missing',
      curves.curveRequest(curves.prefill('USD', OLD_BLOCK)).rows,
      [{ tenor: '2Y', security: 'USSW2 Curncy', quote: 4.1, use: 'Yes' }]);

// --- solvedFactor
check('solvedFactor picks the entry that carries a curve',
      'the curve test dropped, after which a spot filed under the same name is what a desk reads '
      + 'a curve\'s knots off',
      curves.solvedFactor(FACTORS, 'ZAR')[0], 'InterestRate.ZAR');
check('solvedFactor reads the whole name',
      '`includes` for `endsWith(\'.\' + curve)`, under which `ZARONIA` claims `InterestRate.'
      + 'ZAR-ZARONIA` and every curve claims the first factor whose name contains it',
      curves.solvedFactor(FACTORS, 'ZARONIA'), null);
check('solvedFactor answers nothing for a curve the store has not solved',
      'a fallback to the first entry, which shows one curve\'s knots under another\'s name',
      curves.solvedFactor(FACTORS, 'USD'), null);

console.log(`\n${ran} checks over 7 functions, ${missed.length} missed`);
if (missed.length) {
  console.log(missed.map((name) => `  ${name}`).join('\n'));
  process.exit(1);
}
