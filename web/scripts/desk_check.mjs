// Drives `src/desk.ts`'s quote reading where the eye cannot: which rows the Risk screen shows when
// it reads the book in quotes. Every check names the MUTATION it kills - the change to the module
// that would still typecheck, still render, and still be wrong.
//
//   node web/scripts/desk_check.mjs          # exits 1 on a miss
//
// The answer is `GET /book/risk`'s shape on a book reading one USD curve its quotes build, an FX
// surface they build too, and a spot nothing quotes. The last check is what a named calculation
// shows for a dial it does not state.

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

// --- ownDials
check("a named calculation's unstated dial stands at the book's own, where its type declares it",
      'nothing inherited, which shows a declared default the run never reads, one batch over a '
      + "book's twenty; the block's own dials inherited too, which marks a saved dial the "
      + "book's; or the declaration dropped, which lays the book's type and its undeclared keys "
      + 'on the panel',
      desk.ownDials({ Object: 'CreditMonteCarlo', Simulation_Batches: 20, Batch_Size: 1024,
                      Output_Detail: 'Quantiles' },
                    { Object: 'CreditMonteCarlo', Batch_Size: 4096 },
                    { Simulation_Batches: {}, Batch_Size: {}, Antithetic: {} }),
      { Simulation_Batches: 20 });

// --- the reading of a stamp, a tenor and a quote
const stamps = (zone) => {
  process.env.TZ = zone;
  return ['2026-11-16T19:21:07.364541Z', '2026-11-16T21:21:07.123+02:00',
          '2026-11-16T21:21:07.123', 'not a stamp'].map(desk.stampText);
};
check("every stamp reads on the browser's clock with its offset named, whatever zone it was taken in",
      "the record's UTC and the service's local time shown as written, unlabelled, two hours "
      + "apart on one screen; or one offset hard-coded, which is wrong on any other desk",
      [stamps('Africa/Johannesburg'), stamps('America/New_York')],
      [['2026-11-16 21:21:07 UTC+2', '2026-11-16 21:21:07 UTC+2', '2026-11-16 21:21:07',
        'not a stamp'],
       ['2026-11-16 14:21:07 UTC-5', '2026-11-16 14:21:07 UTC-5', '2026-11-16 21:21:07',
        'not a stamp']]);

const FACTORS = { Factor: { types: {
  InterestRate: { Curve: { widget: 'Curve', description: 'Continuously compounded interest rate' } },
  ForwardPrice: { Curve: { widget: 'Curve', description:
    '(excel date, price) pairs giving the forward price at each contract expiry' } },
  FXVol: { ATM_Vol: { widget: 'Curve', description: 'Skew and SVI parameter' },
           Surface: { widget: 'Surface', description:
             '(moneyness, expiry, volatility) triples, flat extrapolated' } },
} } };
const tenor = (factor, coordinates) =>
  desk.tenorText(coordinates, desk.factorAxes(FACTORS, factor, coordinates.length));
check("a factor's coordinates read as its declaration names them",
      'the axes the declaration names left unread, which prints a tenor as its year fraction, '
      + '`0,5`, and a commodity curve\'s excel day as `46,316`',
      [tenor('InterestRate.ZAR', [0.5]), tenor('InterestRate.USD', [7.005479]),
       tenor('ForwardPrice.TIN', [46316]), tenor('FXVol.USD.ZAR', [-0.0625, 0.25])],
      ['6M', '7Y', '2026-10-21', '-0.0625 / 3M']);

check('a vol quote reads as a dealer names it, and any other quote as itself',
      'the engine\'s name shown as is, `RR 0.25 0.0833333`; or a curve\'s own `USD 1M` rewritten',
      ['ATM 0.166667', 'RR 0.25 0.0833333', 'BF 0.1 0.0191781', 'USD 1M', 'ATM 2M']
        .map(desk.quoteText),
      ['2M ATM', '1M 25D RR', '1W 10D BF', 'USD 1M', 'ATM 2M']);

console.log(`\n${ran} checks over 7 functions, ${missed.length} missed`);
if (missed.length) {
  console.log(missed.map((name) => `  ${name}`).join('\n'));
  process.exit(1);
}
