// Drives `src/pnl.ts` where the eye cannot: how the P&L screen names a position and a risk factor,
// what it asks the verb for and when it asks again, and how it reads the explain. Every check names
// the MUTATION it kills - the change that would still typecheck, still render, and still be wrong.
//
//   node web/scripts/pnl_check.mjs          # exits 1 on a miss

import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const bundled = await build({
  stdin: { contents: "export * from './pnl';",
           resolveDir: fileURLToPath(new URL('../src', import.meta.url)), loader: 'ts' },
  bundle: true, format: 'esm', write: false, logLevel: 'silent',
});
const pnl = await import('data:text/javascript;base64,' +
  Buffer.from(bundled.outputFiles[0].text).toString('base64'));

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

const moved = (named) => ({ delta: 2, start: 1, end: 1.5, move: 0.5, pnl: 1, ...named });

check('a risk factor reads as its quote in its block, or as the factor at its coordinates',
      'a quote shown by its block alone, which files every rung of a curve under one name; or a '
      + "spot given an empty coordinate, which reads as a curve's front point",
      [pnl.factorLabel(moved({ block: 'InterestRatePrices.ZAR', quote: '3Y' })),
       pnl.factorLabel(moved({ factor: 'InterestRate.ZAR', tenor: [0.5] })),
       pnl.factorLabel(moved({ factor: 'FXVol.ZAR', tenor: [1, 0.25] })),
       pnl.factorLabel(moved({ factor: 'FxRate.ZAR', tenor: [] }))],
      ['InterestRatePrices.ZAR · 3Y', 'InterestRate.ZAR · 0.5', 'FXVol.ZAR · 1 × 0.25',
       'FxRate.ZAR']);

const EXPLAIN = { existing: -200, carry: 5, market: -190, residual: -15, reserves: null,
                  factors: [], unknown: [], note: null };
check('the explain reads in the order it adds up, the residual last',
      'the lines reordered or one dropped, which leaves a reader summing three numbers to a fourth '
      + 'that is not theirs',
      pnl.explained(EXPLAIN).map((line) => [line.label, line.value]),
      [['Held positions', -200], ['Carry', 5], ['Market', -190], ['Residual', -15]]);

check("the residual's share is of what the held positions made, whatever its sign, and none "
      + 'where nothing was made or either is unknown',
      "a share of a signed P&L, which flips a loss day's residual; or a share of nothing, "
      + 'which reads as infinite',
      [pnl.residualShare(EXPLAIN), pnl.residualShare({ ...EXPLAIN, existing: 0 }),
       pnl.residualShare({ ...EXPLAIN, residual: null }),
       pnl.residualShare({ ...EXPLAIN, existing: null })],
      [-0.075, null, null, null]);

check('the verb is asked for the window and the scope stated, a blank end being now',
      "a blank end sent as `end=`, which the verb refuses as no day; a blank scope sent, which "
      + 'narrows the book to a portfolio named nothing; or the explain asked for unasked, which '
      + 'costs three valuations',
      [pnl.pnlQuery('2024-07-01', '', { portfolio: '', agreement: 'ISDA-1', client: '' }, false),
       pnl.pnlQuery('', '2024-07-02', { portfolio: 'desk/FX' }, true)],
      ['start=2024-07-01&agreement=ISDA-1', 'end=2024-07-02&portfolio=desk%2FFX&explain=true']);

check('only a window ending now falls behind the book, and only once the book has moved',
      'a struck day offered a re-read, which never moves; or every write reading the window again, '
      + 'which values the whole book per tick',
      [pnl.behind('', 'e1', 'e1'), pnl.behind('', 'e1', 'e2'), pnl.behind('2024-07-02', 'e1', 'e2'),
       pnl.behind('', undefined, 'e2')],
      [false, true, false, false]);

check('a position reads by the reference it was booked under, else by its address',
      'the address shown where a reference exists, which no desk can read',
      [pnl.positionLabel({ instrument: 'a'.repeat(64), reference: 'CF-A' }),
       pnl.positionLabel({ instrument: 'b'.repeat(64), reference: null })],
      ['CF-A', `${'b'.repeat(12)}…`]);

console.log(`\n${ran} checks over 6 functions, ${missed.length} missed`);
if (missed.length) {
  console.log(missed.map((name) => `  ${name}`).join('\n'));
  process.exit(1);
}
