// Drives `src/desk.ts`'s quote reading where the eye cannot: which rows the Risk screen shows when
// it reads the book in quotes. Every check names the MUTATION it kills - the change to the module
// that would still typecheck, still render, and still be wrong.
//
//   node web/scripts/desk_check.mjs          # exits 1 on a miss
//
// The answer is `GET /book/risk`'s shape on a book reading one USD curve its quotes build, an FX
// surface they build too, and a spot nothing quotes.

import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const bundled = await build({
  entryPoints: [fileURLToPath(new URL('../src/desk.ts', import.meta.url))],
  bundle: true, format: 'esm', write: false, logLevel: 'silent',
});
const desk = await import('data:text/javascript;base64,' +
  Buffer.from(bundled.outputFiles[0].text).toString('base64'));

const RISK = {
  as_of: '2026-09-24T12:00:00.000', etag: 'e', currency: 'USD', mtm: 1.0, per_deal: [],
  greeks: [
    { factor: 'FXVol.USD.ZAR', tenor: [0.025, 0.1671], value: 59017.6 },
    { factor: 'FxRate.ZAR', value: 5057752.5 },
    { factor: 'InterestRate.USD', tenor: [0.18], value: 45057.3 },
    { factor: 'InterestRate.USD-OIS', tenor: [0.5], value: 12.0 },
  ],
  quotes: [
    { block: 'InterestRatePrices.USD', quote: 'USD 1M', value: 131.0 },
    { block: 'InterestRatePrices.USD', quote: 'USD 2M', value: 426.6 },
    { block: 'InterestRatePrices.USD', quote: 'USD 10Y', value: 0 },
    { block: 'FXVolPrices.USD.ZAR', quote: 'ATM 0.1671', value: 119057.4 },
  ],
  quoted: ['FXVol.USD.ZAR', 'InterestRate.USD'],
  quote_note: null,
};

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

check("the quote reading is every quote the book moves with, in its block's order, then what "
      + 'nothing quotes',
      'the factor rows the quotes stand for kept, which reads the curve and the surface twice; '
      + 'every factor row dropped, which loses the spot delta nothing quotes; or the ladder past '
      + "the book's last date kept, a screen of zeros under the rows that move",
      desk.quoteView(RISK).map((row) => [row.factor, row.quote ?? null]),
      [['InterestRatePrices.USD', 'USD 1M'], ['InterestRatePrices.USD', 'USD 2M'],
       ['FXVolPrices.USD.ZAR', 'ATM 0.1671'], ['FxRate.ZAR', null],
       ['InterestRate.USD-OIS', null]]);

check('a curve read by name keeps its ladder order under the factor sort',
      'quote rows compared on a coordinate they do not have, which reorders a strip by its text',
      desk.sortGreeks(desk.quoteView(RISK), 'factor', false)
        .filter((row) => row.quote).map((row) => row.quote),
      ['ATM 0.1671', 'USD 1M', 'USD 2M']);

console.log(`\n${ran} checks over 2 functions, ${missed.length} missed`);
if (missed.length) {
  console.log(missed.map((name) => `  ${name}`).join('\n'));
  process.exit(1);
}
