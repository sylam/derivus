// Drives `src/tokens.ts`'s one reading of a number, a tenor and an excel day where the eye cannot.
// Every check names the MUTATION it kills - the change that would still typecheck, still render,
// and still be wrong.
//
//   node web/scripts/tokens_check.mjs          # exits 1 on a miss
//
// The machine is set to German before the module loads, so a format that follows the machine's
// locale reads `1.060,56` here, whatever this box is set to.

import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const Native = Intl.NumberFormat;
Intl.NumberFormat = function (locales, options) { return new Native(locales ?? 'de-DE', options); };
Number.prototype.toLocaleString = function (locales, options) {
  return new Native(locales ?? 'de-DE', options).format(this);
};

const bundled = await build({
  entryPoints: [fileURLToPath(new URL('../src/tokens.ts', import.meta.url))],
  bundle: true, format: 'esm', write: false, logLevel: 'silent',
});
const tokens = await import('data:text/javascript;base64,' +
  Buffer.from(bundled.outputFiles[0].text).toString('base64'));
const { formatNumber: f } = tokens;

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

check('a number reads with a decimal point and commas between thousands on any machine',
      "the machine's locale read, which prints `1 060,555542` on one desk and `1.060,56` here",
      [f(1060.555542), f(-492851.709951, 'money'), f(60000000, 'money')],
      ['1,060.56', '-492,851.71', '60,000,000']);

check('money is to the cent, a whole amount without its cents, a large table in whole units',
      'six decimals on money; cents printed on a whole notional; or a sensitivity rounded to the '
      + 'cent where the table reads in units, or to a negative zero',
      [f(291.418121, 'money'), f(-239195.2, 'money'), f(4000000, 'money'),
       f(-228761191.758549, 'whole'), f(-0.4, 'whole'), f(-0.004, 'money')],
      ['291.42', '-239,195.20', '4,000,000', '-228,761,192', '0', '0.00']);

check('a plain number carries six significant digits and no float noise',
      'a fixed count of decimals, which prints `0,000299999999999999` and a strike as `0.05`; or '
      + "significant digits taken left of the point, which reads a 228m delta as 228,762,000",
      [f(0.000299999999999999), f(0.05128205128205128), f(17.284859), f(7.578220835818276),
       f(152.4), f(228761191.758549), f(0), f(1.2e-9)],
      ['0.0003', '0.0512821', '17.2849', '7.57822', '152.4', '228,761,191.76', '0', '1.200e-9']);

check('a percent and a basis point read with their unit',
      'the token read as a bare number, which drops the unit; or six decimals on a percent',
      [tokens.label({ '.Percent': 7.578220835818276 }), tokens.label({ '.Basis': 12.5 }),
       f(12.3456789, 'percent')],
      ['7.5782 %', '12.5 bp', '12.3457 %']);

check('an edit holds the exact value, whatever the display rounds it to',
      'the input started from the display, which files 0.0003 over 0.000299999999999999 on a '
      + 'blur that changed nothing',
      tokens.encodeScalar({ widget: 'Float', description: '', value: 0 },
                          tokens.editRaw(0.000299999999999999)),
      0.000299999999999999);

check('a number typed with the grouping the screens show is that number, any other comma left alone',
      'every comma stripped, which files 2,5 as 25; or none stripped, which refuses the 61,000,000 '
      + 'a reader copies off the screen',
      ['61,000,000', '-1,234.5', '2,5', '1,23,456', '1234.5'].map(tokens.ungrouped),
      ['61000000', '-1234.5', '2,5', '1,23,456', '1234.5']);

check('a tenor reads as a dealer says it',
      "the nearest of a fixed table, which reads 4M as `0.33Y`; or a tolerance in percent, which "
      + "misses a 28-day month and a 7-year point stamped from dates",
      [0, 2 / 365, 0.019178, 14 / 365, 28 / 365, 0.0833333, 0.166667, 0.25, 4 / 12, 0.75, 1, 1.5,
       7.005479, 30.0192, 1.37].map(tokens.tenorLabel),
      ['0D', '2D', '1W', '2W', '1M', '1M', '2M', '3M', '4M', '9M', '1Y', '18M', '7Y', '30Y',
       '1.37Y']);

check('a coordinate reads as its axis is named: a date as its date, an expiry as a tenor',
      'a date read off any name holding the letters, which turns an `Update_Count` into a date; or '
      + 'the excel day printed as `46,339`',
      [tokens.axisText('excel date', 46339), tokens.axisText('Payment_Date', 46316),
       tokens.axisText('Expiry', 0.25), tokens.axisText('tenor', 7.005479),
       tokens.axisText('moneyness', -0.0625), tokens.axisText('Update_Count', 3)],
      ['2026-11-13', '2026-10-21', '3M', '7Y', '-0.0625', '3']);

check("a declared field's number is money where the declaration sizes the instrument",
      "every field read plain, which prints a notional's cents to six digits; or a sized flag "
      + 'read off a Percent',
      [tokens.kindOf({ widget: 'Float', description: '', value: 0, sized: true }),
       tokens.kindOf({ widget: 'Float', description: '', value: 0, obj: 'Percent' }),
       tokens.kindOf({ widget: 'Float', description: '', value: 0 })],
      ['money', 'percent', 'plain']);

console.log(`\n${ran} checks over 7 functions, ${missed.length} missed`);
if (missed.length) {
  console.log(missed.map((name) => `  ${name}`).join('\n'));
  process.exit(1);
}
