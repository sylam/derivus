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
// the store, for what a run says about the book it ran on
const stored = await build({
  entryPoints: [fileURLToPath(new URL('../src/state.ts', import.meta.url))],
  bundle: true, format: 'esm', write: false, logLevel: 'silent',
});
const state = await import('data:text/javascript;base64,' +
  Buffer.from(stored.outputFiles[0].text).toString('base64'));

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

check('two stamps written at different offsets name one instant, a stamp naming no zone read as UTC',
      "the offset's sign turned, which puts the service's local time four hours from the record's "
      + 'UTC; or the offset ignored, two hours',
      ['2026-11-16T19:21:07.123Z', '2026-11-16T21:21:07.123+02:00', '2026-11-16T14:21:07.123-05:00',
       '2026-11-16T19:21:07.123', 'not a stamp'].map(desk.instantMs),
      [...Array(4).fill(Date.UTC(2026, 10, 16, 19, 21, 7, 123)), undefined]);

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
  { factor: 'EquityPrice.NOLEVEL', value: 42.0, level: null, risk: null, per: 'per 1%' },
];
const reading = desk.riskReading(GREEKS, SCHEMA);
check('every spot reads together, largest per its move first, and a per-unit row stands apart',
      'the spots ranked on the per-unit derivative, which puts the rand at 120m over the pound at '
      + "6m though a 1% move of each is 69,566 against 78,720; an SVI parameter, which has no "
      + 'move, filed among the spots or the surfaces; or a spot whose level never came, its move '
      + 'unread, shown among the spots at its per-unit number under "per 1%"',
      [reading.spots.map((row) => row.factor), reading.units.map((row) => row.factor)],
      [['FxRate.GBP', 'FxRate.ZAR'], ['FXVol.EUR.USD.rho', 'EquityPrice.NOLEVEL']]);
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
const held = desk.positionsUnder(MARKS, [
  position('GBPZAR-NDF', ['0/1'], 1), position('USDZAR-CALL', ['0/0'], 0),
  position('NDF-TOP', ['1'], 2), position('X', ['10/0'], 1)]);
check('a marked row holds the positions beneath it or on itself, and only those',
      'a path matched on its prefix alone, which files 10/0 under 1',
      [...held].map(([path, rows]) => [path, rows.map((row) => row.reference)]),
      [['0', ['GBPZAR-NDF', 'USDZAR-CALL']], ['1', ['NDF-TOP']], ['10', ['X']]]);
const ENTITIES = [{ entity: 'KESTREL', name: 'Kestrel Mining Ltd', parent: null }];
const holds = desk.markHolds([
  ...held.get('0').map((row) => ({ ...row, counterparty: 'KESTREL' })),
  { ...position('Y', ['0/2'], 0), counterparty: 'NONAME' }], ENTITIES);
check('a marked row names its clients by name and id, once each, and the trades waiting in it',
      'the client by id alone, `KESTREL`, which a credit officer reads as a code; a client named '
      + 'once per trade; or a position counted pending for its status with nothing awaiting a '
      + 'second seat',
      [holds.clients, holds.pending.map((row) => row.reference)],
      [['Kestrel Mining Ltd (KESTREL)', 'NONAME'], ['GBPZAR-NDF']]);
check('an entity reads as its name with its id, and a file as its name off any server path',
      "the counterparty's id shown bare; or the XVA file's whole path on the server",
      [desk.entityText(ENTITIES, 'KESTREL'), desk.entityText(ENTITIES, 'ALDERBY'),
       desk.fileName('C:\\desk\\home\\xva.json'), desk.fileName('/srv/desk/xva.json')],
      ['Kestrel Mining Ltd (KESTREL)', 'ALDERBY', 'xva.json', 'xva.json']);

// --- what a run's table is in
const SETS = { Instrument: { types: { NettingCollateralSet: ['Fields'], FXOptionDeal: ['Option'] },
  sections: { Fields: { Collateralized: { widget: 'Dropdown', value: 'False' } }, Option: {} } } };
const set = (reference, collateralized, children = []) => ({ Instrument: { '.Deal': {
  Object: 'NettingCollateralSet', Reference: reference,
  ...(collateralized === undefined ? {} : { Collateralized: collateralized }) } },
  Children: children });
const RATES = { 'Price Factors': { 'FxRate.ZAR': { Spot: 0.0576369 } } };
const RAN = { Calc: { Calculation: { Currency: 'USD' }, Deals: { Deals: { Children: [
  set('ISDA-KESTREL', undefined, [{ Instrument: { '.Deal': { Object: 'FXOptionDeal' } } }]),
  set('ISDA-CSA-ALDERBY', 'True')] } } } };
check('a table states its currency, what its exposure is net of, and its columns in words',
      'every table read in the report currency, cashflows in rand among them; a set at its '
      + 'convention read as holding collateral; the collateral note on a table over no dates, or '
      + "on cashflows; a percentile column left as its code; or a simulation's numbered columns "
      + 'left unsaid, or read as paths on a table over no dates',
      [desk.tableReading('exposure_profile', ['EE', 'ENE', 'PFE_97.5'], true, RAN, SETS, RATES),
       desk.tableReading('cashflows/ZAR', [0, 1, 2], true, RAN, SETS, RATES),
       desk.tableReading('mtm', [0, 1], false, RAN, SETS, RATES)],
      [{ currency: 'USD', collateral: ['ISDA-CSA-ALDERBY'], columns: [
         ['EE', 'expected exposure, the mean of the mark where it is positive'],
         ['ENE', 'expected negative exposure, the mean of the mark where it is negative'],
         ['PFE_97.5', 'the 97.5th percentile of the mark']] },
       { currency: 'ZAR', collateral: [], columns: [['0–2', 'one simulated path each']] },
       { currency: 'USD', collateral: [], columns: [] }]);

// --- a run says whether the book on screen is the one it ran on
const book = (etag) => ({ type: 'DOC_LOADED', doc: { Calc: { Calculation: {} } },
                         source: { kind: 'book', etag, path: 'book.json' } });
const asked = (at) => state.reducer(at, { type: 'RUN_SUBMITTED', resultId: 'r1', origin: 'book',
                                          clock: state.clockOf(at), doc: at.doc });
const first = asked(state.reducer(state.INITIAL, book('e1')));
const moved = state.reducer(first, { ...book('e2'), refresh: true });
const reopened = state.reducer(first, book('e1'));
check('a run read after the book moved under it says it is of an earlier book; a book opened '
      + 'afresh drops it',
      "the run filed without the clock it was asked at, which reads a number of the first book as "
      + 'the second\'s; or the run dropped on a refresh, which takes a number off the screen that '
      + 'was true of the book it ran on',
      [desk.lagsBook(first.run.clock, state.clockOf(first)),
       desk.lagsBook(moved.run.clock, state.clockOf(moved)), moved.run.resultId,
       reopened.run.resultId],
      [false, true, 'r1', null]);

// --- an FX spot on its pair as the market quotes it, by the blotter's rule
const PAIRS = { 'System Parameters': { Base_Currency: 'USD' }, 'Price Factors': {
  'FXVol.USD.ZAR': {}, 'FXVol.EUR.USD': {}, 'FxRate.ZAR': { Spot: 0.0576369 },
  'FxRate.EUR': { Spot: 1.158 }, 'FxRate.GBP': { Spot: 1.312 },
  'FxRate.JPY': { Domestic_Currency: 'EUR', Spot: 0.0061 } } };
check("a spot reads on its pair: the base currency's price of a rand is USDZAR, turned",
      "the engine's level shown as is, USD per ZAR at 0.0576 where a desk reads 17.35; every "
      + 'spot turned, which reads EURUSD as 0.86; or a spot with no pair the market names given '
      + 'one',
      ['FxRate.ZAR', 'FxRate.EUR', 'FxRate.GBP', 'FxRate.JPY'].map((factor) => {
        const read = desk.spotQuoted(factor, PAIRS['Price Factors'][factor].Spot, PAIRS);
        return read ? [read.pair, Number(read.level.toFixed(4)), read.turned, read.held] : null;
      }),
      [['USDZAR', 17.35, true, 'USD per 1 ZAR'], ['EURUSD', 1.158, false, 'USD per 1 EUR'],
       null, null]);
const zar = desk.onPair(GREEKS[0], PAIRS);
check("a turned spot's delta is per 1% of the pair: the engine's number with its sign turned",
      'the 1% of FxRate.ZAR shown under USDZAR, a long rand read as long dollars; or the per-unit '
      + 'derivative left per unit of USD per ZAR, whose hover then reads at 17.35',
      [zar.pair, Number(zar.level.toFixed(4)), Math.round(zar.risk),
       Math.round(zar.value * zar.level * 0.01)],
      ['USDZAR', 17.35, -Math.round(GREEKS[0].risk), -Math.round(GREEKS[0].risk)]);

// --- a credit curve stated as spreads and read back as them. `POST /book/survival`'s own answers
// on a book valued 2026-11-16: 1Y/3Y/5Y at 100/150/200bp and a recovery of 0.35, and flat 21bp
// and 120bp at the convention's 0.4, each run on to fifty years
const BASE = '2026-11-16';
const ROUTE_ROWS = [[0, 0], [1, 0.015384615384615384], [3.0027397260273974, 0.06929399367755532],
                    [5.002739726027397, 0.15393045310853531], [50, 1.5384615384615385]];
const ROUTE_21 = [[0, 0], [1, 0.0035], [50, 0.17500000000000002]];
const ROUTE_120 = [[0, 0], [1, 0.02], [50, 1.0]];
const LAB = [[0, 0], [1, 0.03833333333333334], [3.0027397260273974, 0.13512328767123288],
             [5.002739726027397, 0.2501369863013699], [50, 2.5]];
const readBack = (curve, recovery) => desk.spreads(curve, recovery).map(([tenor, spread, survival]) =>
  [Number(tenor.toFixed(4)), Number((spread * 1e4).toFixed(6)), Number(survival.toFixed(6))]);
check('a credit curve reads as the spreads typed, each with the survival to its tenor',
      'the average hazard shown as the spread, 383bp where 230bp was typed; the cumulative -ln S '
      + 'taken for the hazard, which reads the 5Y at 1,501bp; the knot at nought divided by '
      + 'nought; the knot the curve runs on to read as a 50Y nobody typed; that knot left only '
      + 'where it equals the spread before it to the last bit, which reads a flat 21bp back as a '
      + '1Y and a 50Y; or the survival read as exp of the spread',
      [readBack(LAB, 0.4), readBack(ROUTE_21, 0.4)],
      [[[1, 230, 0.962392], [3.0027, 270, 0.873608], [5.0027, 300, 0.778694]],
       [[1, 21, 0.996506]]]);
check("a tenor's years are the engine's own: ACT/365 to the day the period moves the book's to",
      'a year read as 365 days flat, 3Y at 3.0 where the route writes 3.0027; a month as 30 days, '
      + '6M at 0.4932; the day not held to the month\'s end, 31 January and a month read as 3 '
      + 'March; a lower-case unit unread, or a repeated one summed where the engine keeps the '
      + 'last; or an unreadable tenor read as nought',
      [...['1Y', '3Y', '6M', '1Y6M', '2W', '10D', '3m', '1y1y', 'soon', '']
        .map((tenor) => desk.yearsTo(BASE, tenor)),
       desk.yearsTo('2026-01-31', '1M') * 365, desk.yearsTo('2028-02-29', '1Y') * 365],
      [1, 3.0027397260273974, 0.4958904109589041, 1.4986301369863013, 0.038356164383561646,
       0.0273972602739726, 0.25205479452054796, 1, NaN, NaN, 28, 365]);
const TERM = { recovery: '0.35', flat: false, rows: [
  { tenor: '1Y', spread: '100' }, { tenor: ' 3Y', spread: '150' }, { tenor: '', spread: '' },
  { tenor: '5Y', spread: '200' }] };
const FLAT = { recovery: '0.4', flat: true, rows: [{ tenor: '2Y', spread: '120' },
                                                  { tenor: '5Y', spread: '300' }] };
const factor = (curve, recovery) => ({ Recovery_Rate: recovery, Curve: { '.Curve': {
  meta: [], data: curve } } });
check('spreads typed, curve written, spreads read: the form and the read-back are one round trip',
      'a spread read in percent, or the recovery left out of the hazard, which writes a curve the '
      + 'route does not; a tenor\'s hazard not scaled by its years; a row nothing is typed on '
      + 'posted; a flat spread posted as a row, or restated as two; a grouped or a mistyped number '
      + 'sent as anything but what it reads as; a curve restated at the convention\'s recovery '
      + 'rather than its own; or a blank spread read as nought, which previews a survival of 100 %',
      [desk.spreadRequest(' MARRICK ', TERM),
       JSON.stringify(desk.survivalCurve(desk.spreadRequest('MARRICK', TERM), BASE))
         === JSON.stringify(ROUTE_ROWS.slice(0, -1)),
       readBack(ROUTE_ROWS, 0.35).map(([, spread]) => spread),
       desk.spreadForm(factor(ROUTE_ROWS, 0.35), 0.4),
       desk.spreadRequest('ALDERBY-AM', FLAT), desk.survivalCurve(
         desk.spreadRequest('ALDERBY-AM', FLAT), BASE),
       desk.spreadForm(factor(ROUTE_120, 0.4), 0.4), desk.spreadForm(undefined, 0.4),
       desk.survivalCurve(desk.spreadRequest('X', desk.spreadForm(undefined, 0.4)), BASE),
       desk.spreadRequest('X', { recovery: 'forty', flat: false, rows: [
         { tenor: '1Y', spread: '1,250' }, { tenor: '2Y', spread: '2,5' }] })],
      [{ counterparty: 'MARRICK', recovery: 0.35, spread: [
        { tenor: '1Y', spread: 100 }, { tenor: '3Y', spread: 150 }, { tenor: '5Y', spread: 200 }] },
       true, [100, 150, 200],
       { recovery: '0.35', flat: false, rows: [{ tenor: '1Y', spread: '100' },
         { tenor: '3Y', spread: '150' }, { tenor: '5Y', spread: '200' }] },
       { counterparty: 'ALDERBY-AM', recovery: 0.4, spread: 120 }, ROUTE_120.slice(0, -1),
       { recovery: '0.4', flat: true, rows: [{ tenor: '1Y', spread: '120' }] },
       { recovery: '0.4', flat: true, rows: [{ tenor: '', spread: '' }] }, [[0, 0], [1, NaN]],
       { counterparty: 'X', recovery: 'forty', spread: [{ tenor: '1Y', spread: 1250 },
         { tenor: '2Y', spread: '2,5' }] }]);
const ASKED = desk.spreadRequest('KESTREL', FLAT);
check("a refusal of a credit curve reads beside the box of the request field it opens with, any "
      + 'other by the write',
      'every refusal printed in one place; a field matched anywhere in the sentence, which files '
      + "the recovery's refusal by the counterparty it never names first, or a seat's by the "
      + 'spread; the first word taken without asking the request, which files the tenors by a '
      + "field 'the'; or a word matched by prefix or case, which files 'spreads' or 'Recovery' by "
      + 'a box',
      ["counterparty 'NOBODY' is no legal entity the record declares - it declares ALDERBY-AM, "
         + 'KESTREL, MARRICK: declare the entity first',
       "counterparty '' names nobody: a survival curve is written for the counterparty a netting "
         + 'set names',
       'spread -5 is no positive spread: a credit spread is basis points over nothing, one flat '
         + 'or rows [{tenor, spread}] of them',
       'recovery 1.4 is outside [0, 1): it is the share of a claim recovered on default, 0.4 by '
         + 'convention',
       "the tenors ['5Y', '3Y'] do not rise - a term structure is one spread per tenor, shortest "
         + 'first',
       "'soon' is not a period - a count and one of D, W, M or Y, largest unit first (3M, 1Y6M)",
       "actor 'sales-1' holds no mark scope over 'HARBOURLINE', so no survival curve is written: "
         + 'the spread and the recovery are the marking seat\'s',
       'No seat is set, and this desk files every act under the seat that asks',
       'spreads are no field', 'Recovery is not the field']
        .map((said) => desk.refusedAt(said, ASKED)),
      ['counterparty', 'counterparty', 'spread', 'recovery', 'act', 'act', 'act', 'act', 'act',
       'act']);

const CURVES = { 'Price Factors': {
  'InterestRate.ZAR': {}, 'InterestRate.USD-OIS': {}, 'InterestRate.EUR': {}, 'FxRate.ZAR': {},
  'FXVol.USD.ZAR': {}, 'EquityPrice.ACME': {}, 'DividendRate.ACME': {}, 'InterestRateVol.JPY': {},
  SurvivalProb: {} } };
const named = (factors) => ({ widget: 'Text', description: '', value: '', factors });
check("a field naming price factors offers the market's of its types, by the name after the "
      + 'type, each once and in order',
      'the whole factor name offered, InterestRate.ZAR where the deal states ZAR; a type matched '
      + 'by prefix, which offers InterestRateVol.JPY as a discount curve; the name cut at its last '
      + 'dot, which offers USD.ZAR as ZAR; an equity named by its price and dividend offered '
      + 'twice; a table\'s column map read as one list; or a field declaring none offered every '
      + 'factor',
      [desk.factorNames(named(['InterestRate']), CURVES),
       desk.factorNames(named(['FXVol']), CURVES),
       desk.factorNames(named(['EquityPrice', 'DividendRate']), CURVES),
       desk.factorNames(named({ Currency: ['FxRate'] }), CURVES),
       desk.factorNames({ widget: 'Text', description: '', value: '' }, CURVES),
       desk.factorNames(named(['InterestRate']), null)],
      [['EUR', 'USD-OIS', 'ZAR'], ['USD.ZAR'], ['ACME'], [], [], []]);

// --- what the seat may do: one rule over GET /book/seat
const verbs = (held) => ({ book: [], approve: [], mark: [], document: [], settle: [], confirm: [],
                            collateral: [], validate: [], admin: [], ...held });
const seated = (grants, seat = 'sales-1', extra = {}) => ({
  source: { kind: 'book' }, record: { spine: { lsn: 1 } }, seat,
  doc: { Calc: { Deals: { Reference: 'HARBOURLINE' } } },
  grants: grants && { asked: seat, read: { seat, verbs: verbs(grants) }, refused: null },
  ...extra });
const NO_SEAT = 'No seat is set, and this desk files every act under the seat that asks.';
check('a seat may act where a node it holds the act at is the node or above it, the book by '
      + 'default and the whole firm only by a grant over every node; and says, naming the act, '
      + 'what holds it elsewhere',
      'the held nodes matched by prefix, which lets a grant over HARBOURLINE act at HARBOURLINE-2; '
      + 'a grant at a node read as reaching the firm, which offers a declaration the record '
      + 'refuses; a node above the one held read as reached, which offers HARBOURLINE to a seat '
      + 'holding HARBOURLINE/fx; an empty list read as everywhere; or another act\'s grant read',
      [state.heldBack(seated({ mark: ['HARBOURLINE'] }), 'mark'),
       state.heldBack(seated({ book: ['HARBOURLINE'] }), 'book', 'HARBOURLINE/fx/spot'),
       state.heldBack(seated({ document: null }), 'document', null),
       state.heldBack(seated({ book: ['HARBOURLINE'] }), 'mark'),
       state.heldBack(seated({ book: ['HARBOURLINE'] }), 'book', 'HARBOURLINE-2'),
       state.heldBack(seated({ document: ['HARBOURLINE'] }), 'document', null),
       state.heldBack(seated({ book: ['HARBOURLINE/fx'] }), 'book'),
       state.heldBack(seated({ mark: ['HARBOURLINE'] }), 'mark', null)],
      [null, null, null, 'Needs mark: sales-1 holds it nowhere.',
       'Needs book over HARBOURLINE-2: sales-1 holds it only over HARBOURLINE.',
       'Needs document over the whole firm: sales-1 holds it only over HARBOURLINE.',
       'Needs book over HARBOURLINE: sales-1 holds it only over HARBOURLINE/fx.',
       'Needs mark over the whole firm: sales-1 holds it only over HARBOURLINE.']);
check('nothing is held back off a desk that records, before the grants of the seat on screen are '
      + 'read, or where nothing narrows the act; a refused read holds every act in its own words',
      'a plain desk held back for want of a record, which hides every control it can use; the '
      + "grants of the seat typed before read for the one typed now; a null act read as nowhere; "
      + 'or a refusal - no seat named - read as leave to act',
      [state.heldBack(seated({}, 'sales-1', { record: { spine: null } }), 'book'),
       state.heldBack(seated({}, 'sales-1', { source: { kind: 'local' } }), 'book'),
       state.heldBack({ ...seated({}), seat: 'fx-trader' }, 'approve'),
       state.heldBack(seated(null), 'book'),
       state.heldBack(seated({ approve: null }), 'approve'),
       state.heldBack(seated(null, '', { grants: { asked: '', read: null, refused: NO_SEAT } }),
                      'book')],
      [null, null, null, null, null, NO_SEAT]);
// --- every write of the market or its set-up names the seat: the requests a fetch stub catches,
// so nothing leaves this process - no tick, no verification and no set-up is ever sent
const api = await import('data:text/javascript;base64,' + Buffer.from((await build({
  entryPoints: [fileURLToPath(new URL('../src/api.ts', import.meta.url))],
  bundle: true, format: 'esm', write: false, logLevel: 'silent',
})).outputFiles[0].text).toString('base64'));
const caught = [];
globalThis.fetch = async (url, options) => {
  caught.push([url, JSON.parse(options.body ?? 'null')?.actor ?? null]);
  return { ok: true, json: async () => ({ written: false }) };
};
const asking = async (seat) => {
  globalThis.localStorage = { getItem: (key) => (key === 'derivus.seat' ? seat : null) };
  caught.length = 0;
  await api.patchMarket('FxRate.GBP', { Spot: 1.3 });
  await api.tickMarket({ 'FXVolPrices.USD.ZAR': {} });
  await api.retireMarket([['System Parameters', 'Base_Date']]);
  await api.tickBloomberg();
  await api.configureBook('Bootstrapper Configuration', 'FXVol', {});
  await api.configureCurve({ curve: 'ZAR', currency: 'ZAR', rows: [] });
  await api.stateSurvival({ counterparty: 'KESTREL', spread: 120 });
  await api.configureSecurities({ block: 'rates', key: 'ZAR', entry: null });
  await api.verifySecurities({ block: 'rates' });
  return [...caught];
};
const MARKET_WRITES = ['/book/market', '/book/market', '/book/market', '/book/bloomberg',
  '/book/configure', '/book/curve', '/book/survival', '/book/securities',
  '/book/securities/verify'];
check('every write of the market or its set-up names the seat the settings control holds, and '
      + 'none where no seat is set',
      "a market write sent unsigned - postMarket's patch, tick or retire, the bootstrap's dial, "
      + "the curve's set-up or the vocabulary's entry - which a recording desk refuses outright",
      [await asking('marker'), await asking('')],
      [MARKET_WRITES.map((path) => [path, 'marker']), MARKET_WRITES.map((path) => [path, null])]);

const NETTED = { Calc: { Deals: { Deals: { Children: [
  { Instrument: { '.Deal': { Object: 'NettingCollateralSet', Credit_Support_Amounts: {
    Counterparty: 'KESTREL' } } }, Children: [{ Instrument: { '.Deal': { Object: 'FXOptionDeal' } } },
    { Instrument: { '.Deal': { Object: 'NettingCollateralSet', Credit_Support_Amounts: {
      Counterparty: 'ALDERBY-AM' } } } }] },
  { Instrument: { '.Deal': { Object: 'NettingCollateralSet', Credit_Support_Amounts: {
    Counterparty: 'KESTREL' } } } },
  { Instrument: { '.Deal': { Object: 'NettingCollateralSet' } } }] } } } };
const HELD = { 'SurvivalProb.NOBODY': {}, 'SurvivalProb.KESTREL': {}, 'InterestRate.USD': {} };
check('a credit curve is stated for the entities a record declares, else for every counterparty '
      + 'the netting sets name or a curve is held for',
      "the record's entities passed over on a desk that records, which offers a name the route "
      + 'refuses; a set nested under another missed; a counterparty with a curve and no set left '
      + 'out, so its curve cannot be restated; or a name listed twice, or a set naming nobody '
      + 'listed as a blank',
      [desk.creditCounterparties([{ entity: 'MARRICK' }, { entity: 'ALDERBY-AM' }], NETTED, HELD),
       desk.creditCounterparties(null, NETTED, HELD)],
      [['ALDERBY-AM', 'MARRICK'], ['ALDERBY-AM', 'KESTREL', 'NOBODY']]);

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

// --- XVA: how large a run's exposure gets
const day = (date) => ({ '.Timestamp': date });
const PROFILE = { name: 'exposure_profile', rows: 3, offset: 0, columns: ['ENE', 'PFE_97.5', 'EE'],
                  index: [day('2026-11-16'), day('2026-12-16'), day('2027-02-16')],
                  data: [[0, 2, 1], [-1, 4, 5], [-90, 9, 3]] };
check("a profile's peaks are each column's largest number on the day it falls, the PFE at the "
      + 'percentile its own column names',
      "the last day read as the peak's; the PFE taken at 95 whatever the run asked; or the first "
      + 'column - here the negative exposure - read as the expected exposure',
      [desk.peaks(PROFILE), desk.peaks({ ...PROFILE, columns: ['ENE', 'X', 'EE'] })],
      [{ ee: { value: 5, day: '2026-12-16' }, pfe: { value: 9, day: '2027-02-16', quantile: 97.5 } },
       { ee: { value: 5, day: '2026-12-16' } }]);

console.log(`\n${ran} checks over 45 functions, ${missed.length} missed`);
if (missed.length) {
  console.log(missed.map((name) => `  ${name}`).join('\n'));
  process.exit(1);
}
