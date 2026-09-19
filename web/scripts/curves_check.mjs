// Drives `src/curves.ts` where the eye cannot: the commits a desk makes on a card, when a burst of
// them is due to post, the request they build, and the fields a curve may be stated in. Every check
// names the MUTATION it kills - the change to the module that would still typecheck, still render,
// and still be wrong.
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
  curve: 'ZAR', currency: 'ZAR', discount_rate: '',
  interpolation: 'Linear', interpolation_source: 'default',
  conventions: { ...SEEDED.conventions, calendar: '', near_interpolation: '', near_tenor: '',
                 interpolation: 'Linear' },
  rows: [{ tenor: '3M', security: 'JIBA3M Index', quote: 7.41, use: 'Yes' },
         { tenor: '1Y', security: 'SASW1 BGN Curncy', quote: 7.62, use: 'No' }],
};

// a block authored before it carried its definition: the read verb answers the quotes it can read,
// blank tenors, and a note where the conventions would be
const OLD_BLOCK = {
  curve: 'USD', currency: 'USD', discount_rate: '',
  interpolation: 'Linear', interpolation_source: 'default',
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
// the family declares NO `Interpolation`: a curve's own scheme is a rule in a section, so the
// panel's row for it comes off what the ANSWER names instead

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

// --- curveFields: what a card shows, and what every other check is held to
const fields = curves.curveFields(DECLARED, ANSWER);
check('curveFields is the declaration crossed with the spellings the service answers in',
      'the comparison un-lowercased (no declared field is a convention and a desk can state '
      + 'none) or the filter dropped (`N_Iter` travels as a convention and the verb refuses the '
      + 'write). `Day_Count` is out because the conventions spell that one `curve_day_count`',
      fields,
      ['Currency', 'Discount_Rate', 'Calendar', 'Spot_Days', 'Fixed_Frequency', 'Float_Frequency',
       'Fixed_Day_Count', 'Float_Day_Count', 'Front_Day_Count', 'Compounding',
       'Near_Interpolation', 'Near_Tenor', 'Interpolation']);
check('curveFields reads the seeded entries too',
      'the union taken over `curves` alone, which leaves a desk with a book carrying no curve yet '
      + 'nothing to state at all. A seed stating no `calendar` is why that field is the one '
      + 'missing here',
      curves.curveFields(DECLARED, BLANK_BOOK),
      ['Currency', 'Discount_Rate', 'Spot_Days', 'Fixed_Frequency', 'Float_Frequency',
       'Fixed_Day_Count', 'Float_Day_Count', 'Front_Day_Count', 'Compounding']);

// --- prefill
check('prefill completes a seeded row with the fields a request takes',
      'the spread reversed ({...row, ...BLANK}), which blanks the quotes a block already carries',
      curves.prefill('ZAR', SEEDED, fields), {
        curve: 'ZAR', currency: 'ZAR', discount_rate: '', interpolation: '', rows: seededRows,
        conventions: { Spot_Days: 0, Fixed_Frequency: '3M', Float_Frequency: '3M',
                       Fixed_Day_Count: 'ACT_365', Float_Day_Count: 'ACT_365',
                       Front_Day_Count: 'ACT_365', Compounding: 'None' },
      });
check('prefill keeps a block\'s own quotes and hold-outs',
      'the same reversal, read from the other side: the held-out 1Y comes back as Yes',
      curves.prefill('ZAR', BLOCK, fields).rows, BLOCK.rows);
check('prefill reads a block through the lower-case spelling and leaves out what it does not state',
      'the `.toLowerCase()` dropped, after which every field renders at its declared default and '
      + 'a desk edits against a baseline the block never stated; or a `?? \'\'` fallback, which '
      + 'paints a blank over the declaration\'s own default and posts it back as a convention',
      [curves.prefill('ZAR', BLOCK, fields).conventions,
       Object.keys(curves.prefill('ZAR', SEEDED, fields).conventions)],
      [{ Calendar: '', Spot_Days: 0, Fixed_Frequency: '3M', Float_Frequency: '3M',
         Fixed_Day_Count: 'ACT_365', Float_Day_Count: 'ACT_365', Front_Day_Count: 'ACT_365',
         Compounding: 'None', Near_Interpolation: '', Near_Tenor: '' },
       ['Spot_Days', 'Fixed_Frequency', 'Float_Frequency', 'Fixed_Day_Count', 'Float_Day_Count',
        'Front_Day_Count', 'Compounding']]);
check('prefill keeps the conventions clear of the fields the request names outright',
      'the `REQUEST_FIELDS` filter dropped, after which the block\'s RESOLVED scheme travels as a '
      + 'convention and a curve on the type\'s own method writes itself a rule on every commit',
      ['Currency' in curves.prefill('ZAR', BLOCK, fields).conventions,
       'Interpolation' in curves.prefill('ZAR', BLOCK, fields).conventions],
      [false, false]);
check('prefill on nothing is an empty form',
      'a default of the curve name for the currency, which sends a new curve a currency nobody stated',
      curves.prefill('', {}, fields),
      { curve: '', currency: '', discount_rate: '', interpolation: '', rows: [], conventions: {} });
check('prefill carries a curve\'s own rule back and not a scheme it merely resolved to',
      'the `interpolation_source` test dropped, after which every commit on a curve writes the '
      + 'type\'s own method onto it as a rule of its own',
      [curves.prefill('ZAR', BLOCK, fields).interpolation,
       curves.prefill('ZAR', { ...BLOCK, interpolation: 'HermiteRT', interpolation_source: 'curve' },
                      fields).interpolation],
      ['', 'HermiteRT']);

// --- commit
const form = curves.prefill('ZAR', SEEDED, fields);
check('commit patches the row it names',
      'the index test dropped, which applies one cell\'s edit to every row in the ladder',
      curves.commit(form, { row: 1, patch: { use: 'No' } }).rows,
      [seededRows[0], { ...seededRows[1], use: 'No' }]);
check('commit removes the row it names',
      'the filter inverted (i === row), which deletes every row but the one a desk asked to drop',
      curves.commit(form, { row: 0, patch: null }).rows, [seededRows[1]]);
check('commit appends past the end',
      'an index test of `row > rows.length`, which makes `add a row` do nothing at all',
      curves.commit(form, { row: form.rows.length, patch: {} }).rows,
      [...seededRows, { tenor: '', security: '', quote: null, use: 'Yes' }]);
check('commit leaves the form it was handed standing',
      'a splice or an in-place assignment, which edits the array a table is rendering from',
      form.rows, seededRows);
check('commit clears a quote to null rather than to the empty string',
      'the `\'\'` test dropped: an emptied quote cell reaches the emitter as a quote of `\'\'`, '
      + 'which is neither a number nor the null that sends the row to a terminal',
      [curves.commit(form, { row: 0, patch: { quote: '' } }).rows[0].quote,
       curves.commit(form, { row: 0, patch: { quote: 0 } }).rows[0].quote],
      [null, 0]);
check('commit files a declared request field as the request\'s own',
      'every field landing in `conventions`, after which `currency` travels as a convention the '
      + 'emitter reads nothing of and the request carries no currency at all',
      [curves.commit(form, { field: 'Currency', value: 'USD' }).currency,
       curves.commit(form, { field: 'Currency', value: 'USD' }).conventions.Currency],
      ['USD', undefined]);
check('commit files everything else as a convention',
      'the routing reversed, which writes a calendar over the currency',
      [curves.commit(form, { field: 'Calendar', value: 'ZAR' }).conventions.Calendar,
       curves.commit(form, { field: 'Calendar', value: 'ZAR' }).currency],
      ['ZAR', 'ZAR']);
const near = curves.commit(curves.commit(form, { field: 'Near_Tenor', value: '2Y' }),
                           { field: 'Near_Interpolation', value: 'LinearRT' });
check('commit clears the near scheme with the tenor it stops at',
      'the pairing dropped, after which clearing the tenor leaves a near scheme naming nothing '
      + 'and the emitter refuses the write - `a near scheme is a scheme AND where it stops`',
      [near.conventions.Near_Interpolation,
       curves.commit(near, { field: 'Near_Tenor', value: '' }).conventions.Near_Interpolation],
      ['LinearRT', '']);

// --- nearPair
check('nearPair shows the whole curve\'s scheme through where no tenor stops it',
      'a blank shown instead, which reads as a curve with no interpolation at all; or `stated` '
      + 'true on a blank tenor, which lets a desk state half a pair and be refused for it',
      curves.nearPair(form, 'Linear'), { tenor: '', scheme: 'Linear', stated: false });
check('nearPair makes the whole curve\'s scheme the near one when a tenor states the split',
      'the `|| whole` fallback dropped, after which stating a tenor alone posts a near tenor with '
      + 'no scheme and the emitter refuses that half too',
      [curves.nearPair(curves.commit(form, { field: 'Near_Tenor', value: '18M' }), 'Linear'),
       curves.nearPair(near, 'Linear')],
      [{ tenor: '18M', scheme: 'Linear', stated: true },
       { tenor: '2Y', scheme: 'LinearRT', stated: true }]);
check('nearPair reads the curve\'s OWN rule as the scheme that shows through',
      '`typeScheme` read where the curve states a rule, which shows a HermiteRT curve as Linear '
      + 'near the front and posts that as its near scheme',
      curves.nearPair({ ...form, interpolation: 'HermiteRT' }, 'Linear').scheme, 'HermiteRT');

// --- panelValues
check('panelValues shows the scheme the curve RESOLVES to, held to the fields on screen',
      'the `|| typeScheme` dropped (a curve on the type\'s own method renders blank, and the desk '
      + 'is looking at a field that says the curve has no interpolation) or the `fields` filter '
      + 'dropped, which paints the near pair on a panel that declares no row for either',
      curves.panelValues(fields, curves.prefill('ZAR', BLOCK, fields), 'Linear'),
      { Currency: 'ZAR', Discount_Rate: '', Calendar: '', Spot_Days: 0, Fixed_Frequency: '3M',
        Float_Frequency: '3M', Fixed_Day_Count: 'ACT_365', Float_Day_Count: 'ACT_365',
        Front_Day_Count: 'ACT_365', Compounding: 'None', Near_Interpolation: 'Linear',
        Near_Tenor: '', Interpolation: 'Linear' });
check('panelValues leaves out what the source does not state',
      'a `?? \'\'` fallback, which paints a blank over the declaration\'s own default',
      Object.keys(curves.panelValues(fields, curves.prefill('ZAR', SEEDED, fields), 'Linear')),
      ['Currency', 'Discount_Rate', 'Spot_Days', 'Fixed_Frequency', 'Float_Frequency',
       'Fixed_Day_Count', 'Float_Day_Count', 'Front_Day_Count', 'Compounding',
       'Near_Interpolation', 'Near_Tenor', 'Interpolation']);

// --- due
const editing = { form: curves.prefill('ZAR', BLOCK, fields), at: 1000, posted: 0, saving: false };
check('due posts a burst once it has gone quiet, and only once',
      'the `at > posted` test dropped, which re-posts the same edit every tick of the timer and '
      + 'leaves the service solving the market forever',
      [curves.due(editing, 1000), curves.due(editing, 1499), curves.due(editing, 1500),
       curves.due({ ...editing, posted: 1000 }, 9000)],
      [false, false, true, false]);
check('due waits for the post in flight',
      'the `saving` test dropped, after which a second solve is posted over the first and the '
      + 'book is written by whichever lands last',
      curves.due({ ...editing, saving: true }, 9000), false);
check('due holds a card that is not yet a curve',
      'the completeness test dropped, which posts a curve with no name, no currency or no row the '
      + 'moment a desk types the first character of one and is refused by name for it',
      [curves.due({ ...editing, form: { ...editing.form, curve: ' ' } }, 9000),
       curves.due({ ...editing, form: { ...editing.form, currency: '' } }, 9000),
       curves.due({ ...editing, form: { ...editing.form, rows: [] } }, 9000),
       curves.due({ ...editing, form: { ...editing.form, rows: [{ tenor: ' ' }] } }, 9000)],
      [false, false, false, false]);

// --- curveRequest
let composed = curves.commit(curves.commit(curves.prefill('ZAR', SEEDED, fields),
                                           { field: 'Calendar', value: 'ZAR' }),
                             { field: 'Spot_Days', value: 0 });
composed = curves.commit(composed, { row: 2,
  patch: { tenor: ' 2Y ', security: ' SASW2 BGN Curncy ', quote: 7.94 } });
composed = curves.commit(composed, { row: 3, patch: {} });
const request = curves.curveRequest(composed, 'Linear');
check('curveRequest states the conventions the card shows, in the verb\'s own spelling',
      '`.toLowerCase()` dropped, after which the verb reads `Calendar` as a name it knows nothing '
      + 'of and refuses the write; or only the moved ones sent, after which a convention edited on '
      + 'one commit is re-authored off the seed by the next and silently reverts',
      Object.keys(request).filter((key) =>
        !['curve', 'currency', 'discount_rate', 'interpolation', 'rows'].includes(key)),
      ['spot_days', 'fixed_frequency', 'float_frequency', 'fixed_day_count', 'float_day_count',
       'front_day_count', 'compounding', 'calendar']);
check('curveRequest always states the curve\'s own scheme',
      'the field sent only where the desk moved it, after which a commit clears the rule the '
      + 'curve was solved under and it silently drops to the type\'s method',
      [request.interpolation,
       curves.curveRequest({ ...composed, interpolation: 'HermiteRT' }, 'Linear').interpolation],
      ['', 'HermiteRT']);
check('curveRequest states the near pair only where a tenor states the split',
      'the pair sent unconditionally, which posts a blank scheme over a near split the seed '
      + 'declares for a curve whose card never showed one',
      [request.near_tenor, curves.curveRequest(near, 'Linear').near_tenor,
       curves.curveRequest(near, 'Linear').near_interpolation],
      [undefined, '2Y', 'LinearRT']);
check('curveRequest leaves out the row nobody named, and trims the rest',
      'the blank-tenor filter dropped: an empty row a desk added and left behind reaches the '
      + 'emitter, whose grammar refuses `` by name and writes nothing',
      request.rows, [
        { tenor: '3M', security: 'JIBA3M Index', quote: null, use: 'Yes' },
        { tenor: '1Y', security: 'SASW1 BGN Curncy', quote: null, use: 'Yes' },
        { tenor: '2Y', security: 'SASW2 BGN Curncy', quote: 7.94, use: 'Yes' }]);
check('curveRequest names the curve and its currency outright',
      'either read off the conventions, which leaves the two the verb requires missing',
      [request.curve, request.currency, request.discount_rate, request.calendar],
      ['ZAR', 'ZAR', '', 'ZAR']);

// --- the block that states no conventions at all
check('the block with a note in place of its conventions still reads',
      'the conventions read by key rather than spread, which throws on the one block the read verb '
      + 'answers a note for and takes the whole screen down with it',
      [curves.curveFields(DECLARED, OLD_BOOK), curves.prefill('USD', OLD_BLOCK, fields).conventions,
       curves.panelValues(fields, curves.prefill('USD', OLD_BLOCK, fields), 'Linear')],
      [['Currency', 'Discount_Rate'], {},
       { Currency: 'USD', Discount_Rate: '', Near_Interpolation: 'Linear', Near_Tenor: '',
         Interpolation: 'Linear' }]);
check('its rows are stated again by the ones that name a tenor',
      'the blank-tenor filter dropped, which sends the emitter a row whose instrument nothing '
      + 'names - the note says that is what is missing',
      curves.curveRequest(curves.prefill('USD', OLD_BLOCK, fields), 'Linear').rows,
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

console.log(`\n${ran} checks over 8 functions, ${missed.length} missed`);
if (missed.length) {
  console.log(missed.map((name) => `  ${name}`).join('\n'));
  process.exit(1);
}
