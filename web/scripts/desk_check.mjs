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

// --- the risk read in a desk's units: `GET /book/risk`'s rows as the service reads them per move
const SCHEMA = { Factor: { types: {
  InterestRate: { Curve: { widget: 'Curve', description: 'Continuously compounded interest rate' } },
  FXVol: { Surface: { widget: 'Surface', description:
    '(moneyness, expiry, volatility) triples, flat extrapolated' } },
  ForwardPriceVol: { Surface: { widget: 'Surface', description:
    '(delivery, expiry, moneyness, volatility) quads, flat extrapolated' } },
} } };
const read = (factor, value, level, size, relative, per, tenor) => ({
  factor, value, level, risk: value * size * (relative ? level : 1), per,
  ...(tenor ? { tenor } : {}) });
const GREEKS = [
  read('FxRate.ZAR', 120695904.67, 0.0576369, 0.01, true, 'per 1%'),
  read('FxRate.GBP', -6000000.0, 1.312, 0.01, true, 'per 1%'),
  read('InterestRate.ZAR', 7181714.29, 0.0755, 1e-4, false, 'per bp', [3.0]),
  read('InterestRate.ZAR', -1882540.44, 0.07, 1e-4, false, 'per bp', [0.25]),
  read('FXVol.USD.ZAR', 192273.52, 0.1639, 0.01, false, 'per vol point', [-0.05, 0.5]),
  read('FXVol.USD.ZAR', 339027.15, 0.1597, 0.01, false, 'per vol point', [-0.025, 0.5]),
  read('FXVol.USD.ZAR', 1564.67, 0.1596, 0.01, false, 'per vol point', [-0.05, 0.25]),
  read('ForwardPriceVol.GOLD', 100.0, 0.2, 0.01, false, 'per vol point', [47073, 0.5, 1.0]),
  read('ForwardPriceVol.GOLD', 50.0, 0.21, 0.01, false, 'per vol point', [47073, 0.5, 1.1]),
  read('FXVol.EUR.USD.rho', 17851.2, -0.4, 1, false, 'per unit', [0.4]),
];
const reading = desk.riskReading(GREEKS, SCHEMA);
check('every spot reads together, largest per its move first, and a per-unit row stands apart',
      'the spots ranked on the per-unit derivative, which puts the rand at 120m over the pound at '
      + "6m though a 1% move of each is 69,566 against 78,720; or an SVI parameter, which has no "
      + 'move, filed among the spots or the surfaces',
      [reading.spots.map((row) => row.factor), reading.units.map((row) => row.factor)],
      [['FxRate.GBP', 'FxRate.ZAR'], ['FXVol.EUR.USD.rho']]);
check("a surface's vega is its nodes summed across moneyness, an expiry each, in expiry order",
      'the nodes summed across the axis read by position, which sums an FX smile across expiry or '
      + "a forward space across delivery; or the buckets in the answer's order, 6M before 3M",
      reading.surfaces.map((ladder) => [ladder.factor, ladder.per, ladder.buckets.map(
        (bucket) => [bucket.key, Math.round(bucket.risk), bucket.nodes.length])]),
      [['FXVol.USD.ZAR', 'per vol point', [[[0.25], 16], [[0.5], 5313, 2]].map(
        ([key, risk, count = 1]) => [key, risk, count])],
       ['ForwardPriceVol.GOLD', 'per vol point', [[[47073, 0.5], 2, 2]]]]);
check("a curve's ladder is a tenor each in tenor order, its total the parallel basis point",
      'the ladder in the order the answer lists it, 3Y above 3M; or the total of the per-unit '
      + 'derivatives, a parallel move of one hundred percent',
      reading.curves.map((ladder) => [ladder.factor, ladder.per, ladder.buckets.map(
        (bucket) => desk.tenorText(bucket.key, ladder.axes)), Math.round(ladder.total)]),
      [['InterestRate.ZAR', 'per bp', ['3M', '3Y'], 530]]);
check('a ranking by size ranks the move each row is read per',
      "the magnitude sort left on the per-unit value, which ranks the rand's 120m per unit over "
      + 'the pound, whose 1% move is the bigger',
      desk.sortGreeks(GREEKS.slice(0, 2).concat(GREEKS[5]), 'magnitude', true)
        .map((row) => row.factor),
      ['FxRate.GBP', 'FxRate.ZAR', 'FXVol.USD.ZAR']);
check("a table names one move in its header only where every row shares it, and a row's raw "
      + 'derivative reads per unit at its level',
      'the first row\'s move taken as the table\'s, which heads a per-unit row "per 1%"; or the '
      + 'hover showing the risk again instead of what it was read from',
      [desk.sharedPer(GREEKS.slice(0, 2)), desk.sharedPer([GREEKS[0], GREEKS[9]]),
       desk.perUnitText(GREEKS[1])],
      ['per 1%', null, '-6,000,000 per unit at 1.312']);
check('a quote row read by quote keeps its move and its words',
      'the quote rows rebuilt from `value` alone, which shows a vol quote per 100 vol points again',
      desk.quoteView({ ...RISK, quotes: [{ block: 'FXVolPrices.USD.ZAR', quote: 'ATM 0.5',
        value: 513753.0, level: 0.15, risk: 5137.53, per: 'per vol point' }] })[0],
      { quote: 'ATM 0.5', value: 513753.0, level: 0.15, risk: 5137.53, per: 'per vol point',
        factor: 'FXVolPrices.USD.ZAR' });

// --- what the numbers are on, and whether they are of the book on screen
const DOC = { Calc: { Calculation: { Base_Date: { '.Timestamp': '2026-11-16' }, Currency: 'USD' },
                      MergeMarketData: { MarketDataFile: 'C:\\desk\\files\\MarketData.json' } } };
const MARKET = {
  'Market Prices': { 'FXVolPrices.USD.ZAR': { instrument: { Points: [
    { Timestamp: { '.Timestamp': '2026-11-16T07:00:00+00:00' } },
    { Timestamp: { '.Timestamp': '2026-11-16T07:05:00+00:00' } }] } } },
  'Price Factors': { 'FXVol.EUR.USD': { Quote_Timestamp: { '.Timestamp': '2026-11-15T16:00:00+00:00' } },
                     'FxRate.ZAR': { Spot: 0.0576 } },
};
check("the book's day, and its market as far as the book says: the file it names, the latest quote",
      "the file's whole path from another machine; or the first quote's stamp, which reads a "
      + 'market five minutes older than it is',
      [desk.valuationDay(DOC), desk.marketSource(DOC, MARKET),
       desk.marketSource({ Calc: { Calculation: {} } }, {})],
      ['2026-11-16', { file: 'MarketData.json', seen: '2026-11-16T07:05:00+00:00' },
       { file: null, seen: null }]);
check('an answer read before the record moved is behind the book, though the file did not move',
      "the record's head left out of the clock, which never says a risk read before an approval "
      + 'or a corrected agreement is of an earlier book',
      [desk.lagsBook(desk.bookClock('e1', '33/0/0'), desk.bookClock('e1', '33/2/1')),
       desk.lagsBook(desk.bookClock('e1', '33/0/0'), desk.bookClock('e1', '33/0/0')),
       desk.lagsBook(desk.bookClock('e1', null), desk.bookClock('e2', null))],
      [true, false, true]);

// --- pending approval inside the marks, off `GET /book/positions`
const position = (reference, paths, pending) => ({ reference, deal_paths: paths, pending,
  instrument: reference, status: pending ? 'pending' : 'approved' });
const MARKS = [{ reference: 'ISDA-KESTREL', deal_path: '0', value: 1 },
               { reference: 'NDF-TOP', deal_path: '1', value: 1 },
               { reference: 'SPARE', deal_path: '10', value: 1 }];
const held = desk.pendingUnder(MARKS, [position('GBPZAR-NDF', ['0/1'], 1),
                                       position('USDZAR-CALL', ['0/0'], 0),
                                       position('NDF-TOP', ['1'], 2), position('X', ['10/0'], 1)]);
check('a marked row holds the trades pending beneath it or on itself, and only those',
      "a path matched on its prefix alone, which files 10/0 under 1; or a position counted for its "
      + 'status with nothing awaiting a second seat',
      [...held].map(([path, rows]) => [path, rows.map((row) => row.reference)]),
      [['0', ['GBPZAR-NDF']], ['1', ['NDF-TOP']], ['10', ['X']]]);

// --- XVA: a row's own currency
const xset = (reference, currency, cva, fva) => ({ reference, currency, cva, fva, status: 'done',
  note: null, recalc: null });
const VIEW = { as_of: '', path: '', sets: [xset('A', 'USD', 6114.49, 1941.65),
                                           xset('B', 'ZAR', 1295.26, 0), xset('C', null, 10, null)] };
check('amounts in two currencies are never added: one sum per currency the rows ran in',
      "every row's CVA added under the open book's currency, which reads 6,114.49 USD plus "
      + '1,295.26 ZAR as 7,409.75 ZAR',
      desk.xvaTotals(VIEW).sums.map((sum) => [sum.currency, sum.cva, sum.counted, sum.countedFva]),
      [['USD', 6114.49, 1, 1], ['ZAR', 1295.26, 1, 1], [null, 10, 1, 0]]);

console.log(`\n${ran} checks over 16 functions, ${missed.length} missed`);
if (missed.length) {
  console.log(missed.map((name) => `  ${name}`).join('\n'));
  process.exit(1);
}
