// Drives `src/desk.ts`'s quote reading where the eye cannot: which rows the Risk screen shows when
// it reads the book in quotes. Every check names the MUTATION it kills - the change to the module
// that would still typecheck, still render, and still be wrong.
//
//   node web/scripts/desk_check.mjs          # exits 1 on a miss
//
// The answer is `GET /book/risk`'s shape on a book reading one USD curve its quotes build, an FX
// surface they build too, and a spot nothing quotes. The last check is what a named calculation
// shows for a dial it does not state.

import { readdirSync, readFileSync } from 'node:fs';
import { check, done, drawer, load } from './check.mjs';

const desk = await load('desk');
// the store, for what a run says about the book it ran on
const state = await load('state');

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

check('a row age compares instants across zone offsets rather than wall clocks',
      'ageOf reading stampMs without offsets, which calls a run stamped in UTC two hours old '
      + 'against a +02:00 read taken the same second',
      desk.ageOf('2026-11-16T19:21:07Z', '2026-11-16T21:21:07+02:00').text, 'moments ago');

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

const who = (said) => said && [/the desk puts it right/.test(said), /its sender puts it right/.test(said),
                               said.includes('from-fo.json')];
check('a book that will not price, by its code, says who puts it right: the desk on its own book, '
      + 'the sender of a document opened from another system by its name; any other refusal says '
      + 'for itself',
      "the document's name left unread, which tells a desk to fix in its own book a file another "
      + 'system sent it; or every refusal read as a book that will not price, which says so under '
      + "the record's sentence to a seat it does not admit",
      [who(desk.unpriced('unpriced')), who(desk.unpriced('unpriced', 'from-fo.json')),
       who(desk.unpriced('not_granted', 'from-fo.json')) ?? null, desk.unpriced(null) ?? null],
      [[true, false, false], [false, true, true], null, null]);

// --- leaving a document opened here
const COPY = { etag: 'e1', path: 'inbox/from-fo.json', opened: 'from-fo.json' };
const asks = (served, seen) => {
  const said = desk.leavingAsks(served, seen);
  return said && [said.includes('from-fo.json'), /lost unless it is downloaded/.test(said)];
};
check('leaving an opened document asks first where its copy may hold edits made here, by its name: '
      + "an amended copy, and one this client did not see opened; not the desk's own book, nor a "
      + 'copy standing as its open answered',
      'the etag left unread, which replaces an amended copy without a word; the open this client '
      + 'saw left unread, which asks over a copy nothing touched; a copy opened elsewhere taken '
      + "for untouched, by an etag another file's open answered; or the desk's own book asked over",
      [asks({ ...COPY, etag: 'e2' }, COPY), asks(COPY, { ...COPY }), asks(COPY, null),
       asks(COPY, { etag: 'e1', path: 'inbox/other.json' }),
       asks({ etag: 'e2', path: 'book.json' }, COPY)],
      [[true, true], null, [true, true], [true, true], null]);

// --- a screen that reads the desk's own book, with none of it to read
const HEADER = readFileSync(new URL('../src/components/DocumentLoader.tsx', import.meta.url), 'utf8')
  .split(/\r?\n/).map((line) => line.trim());
// a label the header draws as written, or by the `desk` constant holding it
const pressed = ({ hint }) => {
  const control = /Press "([^"]+)" at the top/.exec(hint)?.[1];
  const owner = Object.keys(desk).find((name) => desk[name] === control);
  return [control, HEADER.includes(control) || HEADER.includes(`{${owner}}`)];
};
check('over a read-only copy of a file a screen names the control that leaves it for the book, and '
      + 'what it then shows; a desk serving no book names the control that opens a file; each by '
      + 'the label the header draws',
      "a control named by a word the header does not draw ('return to book'), which sends the "
      + 'reader to a button that is not there; or what the screen shows left out of the sentence',
      [pressed(desk.offBook("the desk's own risk")),
       desk.offBook("the desk's own risk").hint.endsWith("this screen then shows the desk's own risk."),
       pressed({ hint: desk.OPEN_FILE })],
      [['back to book', true], true, ['open a job file', true]]);
const RECORDED = "This desk's book is what its record holds, so no file is opened on it.";
check('a file a desk that records will not open is a read-only copy said in the service\'s own '
      + 'sentence, with the control that leaves it; any other refusal says for itself',
      'the refusal of a desk that records left to land in a red box over the copy; every refusal '
      + 'read as a desk that records, which says a file is a copy where it will not parse; or the '
      + 'control that leaves the copy dropped',
      [desk.recordedCopy({ error: RECORDED, code: 'recorded' }),
       desk.recordedCopy({ error: 'trades.json does not parse', code: 'refused' })],
      [`${RECORDED} Press "back to book" at the top to leave it.`, null]);

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
      "the counterparty's id shown bare; a null entity list throwing; or the XVA file's whole path",
      [desk.entityText(ENTITIES, 'KESTREL'), desk.entityText(ENTITIES, 'ALDERBY'),
       desk.entityText(null, 'KESTREL'),
       desk.fileName('C:\\desk\\home\\xva.json'), desk.fileName('/srv/desk/xva.json')],
      ['Kestrel Mining Ltd (KESTREL)', 'ALDERBY', 'KESTREL', 'xva.json', 'xva.json']);

// --- what a run's table is in
const RATES = { 'Price Factors': { 'FxRate.ZAR': { Spot: 0.0576369 } } };
const RAN = { Calc: { Calculation: { Currency: 'USD' }, Deals: { Deals: { Children: [] } } } };
// the sets the job's description says its run collateralises
const SETS = ['ISDA-CSA-ALDERBY'];
check('a table states its currency, what its exposure is net of, and its columns in words',
      'every table read in the report currency, cashflows in rand among them; the collateral note '
      + "on a table over no dates, or on cashflows; a percentile column left as its code; or a "
      + "simulation's numbered columns left unsaid, or read as paths on a table over no dates",
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
  'FXVol.USD.ZAR': {}, 'FXVol.EUR.USD': {}, 'FXVol.USD.JPY': {}, 'FxRate.ZAR': { Spot: 0.0576369 },
  'FxRate.EUR': { Spot: 1.158 }, 'FxRate.GBP': { Spot: 1.312 },
  'FxRate.JPY': { Domestic_Currency: 'EUR', Spot: 0.0061 } } };
check("a spot reads on its pair against the base: the base currency's price of a rand is USDZAR, "
      + 'turned, and a yen stating another domestic currency is still USDJPY',
      "the engine's level shown as is, USD per ZAR at 0.0576 where a desk reads 17.35; every "
      + 'spot turned, which reads EURUSD as 0.86; a spot with no pair the market names given '
      + 'one; or the domestic currency read, which the engine does not read',
      ['FxRate.ZAR', 'FxRate.EUR', 'FxRate.GBP', 'FxRate.JPY'].map((factor) => {
        const read = desk.spotQuoted(factor, PAIRS['Price Factors'][factor].Spot, PAIRS);
        return read ? [read.pair, Number(read.level.toFixed(4)), read.turned, read.held] : null;
      }),
      [['USDZAR', 17.35, true, 'USD per 1 ZAR'], ['EURUSD', 1.158, false, 'USD per 1 EUR'],
       null, ['USDJPY', 163.9344, true, 'USD per 1 JPY']]);
const zar = desk.onPair(GREEKS[0], PAIRS);
check("a turned spot's delta is per 1% of the pair: the engine's number with its sign turned",
      'the 1% of FxRate.ZAR shown under USDZAR, a long rand read as long dollars; or the per-unit '
      + 'derivative left per unit of USD per ZAR, whose hover then reads at 17.35',
      [zar.pair, Number(zar.level.toFixed(4)), Math.round(zar.risk),
       Math.round(zar.value * zar.level * 0.01)],
      ['USDZAR', 17.35, -Math.round(GREEKS[0].risk), -Math.round(GREEKS[0].risk)]);

// --- the exports: each table's rows under its own heads, every number as itself
const MARKED = { ...RISK, per_deal: [{ reference: 'ISDA-KESTREL', deal_path: '0', value: 1234567.891 },
                                     { reference: 'TWICE', deal_path: null, value: -(0.1 + 0.2) }] };
const inside = (path) => (path === '0' ? {
  clients: ['Kestrel (KES)', 'Alderby (ALD)'],
  pending: [{ reference: 'NDF-2', instrument: 'a'.repeat(64) }, { reference: null, instrument: 'b'.repeat(64) }],
} : { clients: [], pending: [] });
check('the marks export under the heads the table shows, each mark the number itself and its '
      + 'currency in a column of its own; on a desk that records, its clients and the trades '
      + 'waiting in it by name',
      'a mark written as displayed - `formatNumber` over the value - which exports 1,234,567.89 '
      + 'as text a spreadsheet cannot add and rounds what it keeps; the currency left in the head; '
      + "or the record's two columns headed on a desk keeping none, every value under the wrong head",
      [desk.marksExport(MARKED), desk.marksExport(MARKED, inside)],
      [[['Reference', 'Path', 'Mark', 'Ccy'], ['ISDA-KESTREL', '0', 1234567.891, 'USD'],
        ['TWICE', null, -0.30000000000000004, 'USD']],
       [['Reference', 'Client', 'Path', 'Waiting on a second seat', 'Mark', 'Ccy'],
        ['ISDA-KESTREL', 'Kestrel (KES), Alderby (ALD)', '0', `NDF-2, ${'b'.repeat(12)}…`, 1234567.891, 'USD'],
        ['TWICE', '', null, '', -0.30000000000000004, 'USD']]]);

const QUOTE_ROW = { factor: 'FXVolPrices.USD.ZAR', quote: 'ATM 0.5', value: 513753.0, level: 0.15,
                 risk: 5137.53, per: 'per vol point' };
check('the quote reading exports under its own heads: a quote as a dealer names it, a spot on its '
      + 'pair, each risk per its move the number itself and the report currency in its own column',
      'a risk written as displayed, which exports 5,137.53 as text; the per-unit derivative '
      + "written for the risk per the row's move; or a turned spot written under its engine name "
      + 'alone, its sign then reading the wrong way',
      desk.quotesExport([QUOTE_ROW, zar], SCHEMA, 'USD'),
      [['Block · Factor', 'Quote · Tenor', 'Risk', 'per', 'Ccy'],
       ['FXVolPrices.USD.ZAR', '6M ATM', 5137.53, 'per vol point', 'USD'],
       ['USDZAR FxRate.ZAR', '', zar.risk, 'per 1%', 'USD']]);

check('the desk reading exports as one table: every spot on its pair at its level, each curve and '
      + 'surface a row per bucket - a surface\'s the sum its row shows - then each row read per '
      + 'unit, every risk the number itself',
      'a risk written as displayed, which exports 78,720 as text and rounds a ladder to cents; a '
      + "surface's nodes written each under its bucket's coordinates, which a sum then counts as "
      + 'the screen does not; or the spots written before they are read on their pairs',
      desk.readingExport(GREEKS, SCHEMA, PAIRS, 'USD'),
      [['Factor', 'Coordinates', 'Level', 'Risk', 'per', 'Ccy'],
       ['FxRate.GBP', null, 1.312, GREEKS[1].risk, 'per 1%', 'USD'],
       ['USDZAR FxRate.ZAR', null, zar.level, zar.risk, 'per 1%', 'USD'],
       ['InterestRate.ZAR', '3M', null, GREEKS[3].risk, 'per bp', 'USD'],
       ['InterestRate.ZAR', '3Y', null, GREEKS[2].risk, 'per bp', 'USD'],
       ['FXVol.USD.ZAR', '3M', null, GREEKS[6].risk, 'per vol point', 'USD'],
       ['FXVol.USD.ZAR', '6M', null, GREEKS[4].risk + GREEKS[5].risk, 'per vol point', 'USD'],
       ['ForwardPriceVol.GOLD', '47,073 / 6M', null, GREEKS[7].risk + GREEKS[8].risk, 'per vol point', 'USD'],
       ['FXVol.EUR.USD.rho', '0.4Y', null, 17851.2, 'per unit', 'USD'],
       ['EquityPrice.NOLEVEL', '', null, 42, 'per unit', 'USD']]);

// --- a credit curve stated as spreads, and the form a restatement opens in: each curve as the
// service reads its quotes block (`GET /book/market`'s `credit`) - 1Y/3Y/5Y at 100/150/200bp at a
// recovery of 0.35, struck in USD on USD-OIS; flat 120bp at the convention's recovery; and a block
// another source wrote quoting a 5Y upfront
const quoted = (tenor, quote, type = 'Par_Spread') => ({ tenor, quote, type });
const CURVE = (rows, recovery = 0.4) => ({ recovery, currency: 'USD', discount_rate: 'USD-OIS',
                                           rows: rows.map((row) => quoted(...row)) });
const SERVED_ROWS = CURVE([['1Y', 100], ['3Y', 150], ['5Y', 200]], 0.35);
const SERVED_120 = CURVE([['1Y', 120]]);
const UPFRONT = CURVE([['5Y', 2.5, 'Upfront']]);
const CREDIT_HELD = { 'SurvivalProb.MARRICK': SERVED_ROWS, 'SurvivalProb.KESTREL': SERVED_120,
                      'SurvivalProb.UPFRONT': UPFRONT };
// the quotes family's declaration, as far as a blank form turns on it
const CREDIT = { MarketPrices: { types: { SurvivalProbPrices: { Recovery_Rate: { value: 0.4 } } } } };
const TERM = { recovery: '0.35', currency: ' USD', discount_rate: 'USD-OIS ', flat: false, rows: [
  { tenor: '1Y', spread: '100' }, { tenor: ' 3Y', spread: '150' }, { tenor: '', spread: '' },
  { tenor: '5Y', spread: '200' }] };
const FLAT = { recovery: '0.4', currency: 'USD', discount_rate: 'USD', flat: true,
               rows: [{ tenor: '2Y', spread: '120' }, { tenor: '5Y', spread: '300' }] };
check('the form a restatement opens in posts the quotes its block states again, a spread per tenor '
      + 'as they stand, at their recovery, currency and discount curve: a row nothing is typed on, '
      + 'a grouped or a mistyped number each sent as what it reads as; a curve with no quotes, or '
      + 'none a spread, opens flat and blank at the declared recovery',
      'a quote scaled from what the block states, or an upfront read as a spread; a row nothing is '
      + 'typed on posted; a flat spread posted as a row, or a stated quote restated flat at 1Y; the '
      + 'currency or the curve left off the request, or the block\'s not offered back; a grouped '
      + 'or a mistyped number sent as anything but what it reads as; or a curve restated at the '
      + 'declared recovery rather than its own',
      [desk.spreadRequest(' MARRICK ', TERM), desk.spreadForm(SERVED_ROWS, CREDIT),
       desk.spreadRequest('ALDERBY-AM', FLAT),
       desk.spreadForm(SERVED_120, CREDIT), desk.spreadForm(undefined, CREDIT),
       desk.spreadForm(UPFRONT, CREDIT),
       desk.spreadRequest('X', { recovery: 'forty', currency: '', discount_rate: '', flat: false,
                                 rows: [{ tenor: '1Y', spread: '1,250' }, { tenor: '2Y', spread: '2,5' }] })],
      [{ counterparty: 'MARRICK', recovery: 0.35, currency: 'USD', discount_rate: 'USD-OIS', spread: [
        { tenor: '1Y', spread: 100 }, { tenor: '3Y', spread: 150 }, { tenor: '5Y', spread: 200 }] },
       { recovery: '0.35', currency: 'USD', discount_rate: 'USD-OIS', flat: false, rows: [
         { tenor: '1Y', spread: '100' }, { tenor: '3Y', spread: '150' }, { tenor: '5Y', spread: '200' }] },
       { counterparty: 'ALDERBY-AM', recovery: 0.4, currency: 'USD', discount_rate: 'USD', spread: 120 },
       { recovery: '0.4', currency: 'USD', discount_rate: 'USD-OIS', flat: false,
         rows: [{ tenor: '1Y', spread: '120' }] },
       { recovery: '0.4', currency: '', discount_rate: '', flat: true, rows: [{ tenor: '', spread: '' }] },
       { recovery: '0.4', currency: 'USD', discount_rate: 'USD-OIS', flat: true,
         rows: [{ tenor: '', spread: '' }] },
       { counterparty: 'X', recovery: 'forty', currency: '', discount_rate: '', spread: [
         { tenor: '1Y', spread: 1250 }, { tenor: '2Y', spread: '2,5' }] }]);
const served = (credit) => ({ type: 'MARKET_LOADED', market: {
  etag: credit ? 'e1' : 'e2', stamp: credit ? 's1' : 's2', market: credit && {}, credit } });
const marketed = { ...state.reducer(state.reducer(state.INITIAL, book('e1')), served(CREDIT_HELD)),
                   schema: CREDIT };
check("a credit curve is read as the service read its quotes, and stands while the service "
      + 'answers no market, the one held being current; a document opened from a file reads none',
      'the credit taken from an answer with none, which empties every credit curve on screen at '
      + "each poll the market has not moved in; or a document opened from a file read with the "
      + "live book's credit",
      [state.heldSpreads(state.reducer(marketed, served(null)), 'SurvivalProb.KESTREL'),
       state.heldSpreads(state.reducer(marketed, served(null)), 'SurvivalProb.NOBODY'),
       state.heldSpreads({ ...marketed, source: { kind: 'local', name: 'job.json', note: null } },
                         'SurvivalProb.KESTREL')],
      [SERVED_120, undefined, undefined]);
const ASKED = desk.spreadRequest('KESTREL', FLAT);
check("a refusal of a credit curve reads beside the box of the request field its code names, any "
      + 'other - the tenors, a seat without the grant, none named - by the write',
      'every refusal printed in one place; or a code that names no field of the request, a '
      + "seat's or the record's, filed by a box",
      ['counterparty', 'spread', 'recovery', 'currency', 'discount_rate', 'refused', 'not_granted',
       'no_actor', null].map((code) => desk.refusedAt(code, ASKED)),
      ['counterparty', 'spread', 'recovery', 'currency', 'discount_rate', 'act', 'act', 'act',
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
let walked = 0;
const COUNTED = { get 'Price Factors'() { walked += 1; return CURVES['Price Factors']; } };
[{ widget: 'Float', description: '', value: 0 }, named([]), named({ Currency: ['FxRate'] })]
  .forEach((descriptor) => desk.factorNames(descriptor, COUNTED));
check('a field naming no factor never walks the market',
      'the gate on a field declaring no factor type dropped, which walks every factor of the '
      + 'market once per input on a screen - 90 ms a render at 3,000 factors and 300 inputs',
      [walked, desk.factorNames(named(['FxRate']), COUNTED), walked], [0, ['ZAR'], 1]);

// --- what the seat may do: one rule over GET /book/seat
// the map GET /book/seat answers where no capabilities document moves an act
const DEMANDS = { fill: 'book', amendment: 'book', quote_filed: 'book', approval: 'approve',
                  entity_declared: 'document',
                  agreement_declared: 'document', official_close_declared: 'mark',
                  status_transition: 'settle', confirmed: 'confirm', collateral: 'collateral' };
// the acts the record judges at the firm, as the service serves them
const FIRM = ['agreement_declared', 'entity_declared', 'official_close_declared'];
// a seat whose `GET /book/seat` answers `read` over a seat shown the whole book; null, unread
const seated = (read, seat = 'sales-1', extra = {}) => ({
  source: { kind: 'book' }, record: { spine: { lsn: 1 } }, seat,
  doc: { Calc: { Deals: { Reference: 'HARBOURLINE' } } },
  grants: read && { asked: seat, refused: null,
                    read: { seat, verbs: {}, demands: DEMANDS, firm: FIRM, may: {}, sees: null,
                            ...read } },
  ...extra });
// one whose answer gives `may`, its verdict per act and node
const mays = (may, extra = {}) => seated({ may }, 'sales-1', extra);
check('the seat a client acts as is the one the service answers its acts are filed under, a '
      + "token's with no seat in the header's box; the box's own until that is read for it, and "
      + 'on a desk that asks no grants',
      "the box's seat alone read, which leaves a seat a token names no trades of its own in the "
      + "blotter; or the answer read for the seat the box held before, which names that seat's",
      [state.actingSeat(seated({})), state.actingSeat({ ...seated({}, 'fx-trader'), seat: '',
                                                        grants: { ...seated({}, 'fx-trader').grants, asked: '' } }),
       state.actingSeat({ ...seated({}), seat: 'sales-2' }), state.actingSeat(seated(null))],
      ['sales-1', 'fx-trader', 'sales-2', 'sales-1']);

const NONE = 'The record holds none of it now';
check('a seat shown no position in a trade is told the record holds none only where it is shown '
      + 'the whole book; a seat seeing part of it, another book or nothing is told it is shown '
      + 'none, and what it sees',
      "the record said to hold none to a seat the service shows part of the book, which tells it "
      + 'a trade on the record is not; a seat shown nothing read as shown the whole book; or '
      + 'grants not yet read taken for no sight',
      [state.noneShown(seated({})), state.noneShown(seated(null)),
       state.noneShown(seated({ sees: ['HARBOURLINE/FX'] }, 'sales-9')),
       state.noneShown(seated({ sees: ['HARBOUR'] }, 'sales-9')),
       state.noneShown(seated({ sees: [] }, 'sales-9'))],
      [NONE, NONE, 'sales-9 is shown none of it, seeing only what sits under HARBOURLINE/FX',
       'sales-9 is shown none of it, seeing only what sits under HARBOUR',
       'sales-9 is shown none of it, holding no grant at any node']);

const YET = 'nothing has been recorded here yet';
const NO_QUOTE = 'No quote this desk gave landed here';
check("a seat shown the whole book is told each place's own sentence - no fill opens the trade, "
      + 'nothing is recorded yet, no quote landed here - and a seat shown none of it, whose list '
      + 'of quotes the service narrows as its positions, is told so in every place',
      "the place's sentence dropped for the trade's, which tells the reader of an empty strip "
      + "that the record holds none of it now; or the place's sentence said to a seat shown "
      + 'none, which tells it nothing has been recorded on a record that holds the book',
      [state.noneShown(seated({}), YET), state.noneShown(seated(null), YET),
       state.noneShown(seated({ sees: ['HARBOURLINE/FX'] }, 'sales-9'), YET),
       state.noneShown(seated({}), NO_QUOTE),
       state.noneShown(seated({ sees: ['HARBOURLINE/FX'] }, 'sales-9'), NO_QUOTE)],
      [YET, YET, 'sales-9 is shown none of it, seeing only what sits under HARBOURLINE/FX',
       NO_QUOTE, 'sales-9 is shown none of it, seeing only what sits under HARBOURLINE/FX']);
check('a read of the record not answered yet is said as being read and a refused one in its '
      + 'words, never as holding none; one answered, empty or not, leaves the place its own line',
      'the unread read let through as answered, which tells a trade on a slow first load that the '
      + 'record holds none of it and there is nothing to unwind; or a refusal said as being read',
      [state.unread({ data: null, error: null }), state.unread({ data: null }),
       state.unread({ data: null, error: 'refused' }), state.unread({ data: [], error: null }),
       state.unread({ data: { positions: [] } })],
      ['reading the record…', 'reading the record…', 'the record did not answer: refused', null,
       null]);

const DONE = "Every client's sets have a run that is done";
check('an XVA table with nothing to act on says every run is done, and on a desk that records '
      + 'that no trade waits, to a seat shown the whole book; a seat shown none of it is told that',
      'the sentence said whatever the seat is shown, which tells a seat shown no trade that sets '
      + 'never run have a run that is done; or the second seat named on a desk that has none',
      [state.nothingWrong(seated({})),
       state.nothingWrong(seated(null, 'sales-1', { record: { spine: null } })),
       state.nothingWrong(seated({ sees: ['HARBOURLINE/FX'] }, 'sales-9'))],
      [`${DONE}, and no trade waits on a second seat`, DONE,
       'sales-9 is shown none of it, seeing only what sits under HARBOURLINE/FX']);
check('a P&L answering no row says in its table\'s place that no position stood in the window and '
      + 'scope, to a seat shown the whole book; a seat shown none of it is told that',
      'the sentence said whatever the seat is shown, which tells a seat shown no position that '
      + 'the book held none; or an empty table left under a total of nothing',
      [state.noPositions(seated({})),
       state.noPositions(seated({ sees: ['HARBOURLINE/FX'] }, 'sales-9'))],
      ['No position stood in this window and scope',
       'sales-9 is shown none of it, seeing only what sits under HARBOURLINE/FX']);

const NO_SEAT ='No seat is set, and this desk files every act under the seat that asks.';
// the record's own sentences, as the service answers them for a seat working HARBOURLINE/fx
const SAID = { book: 'sales-1 holds no book over HARBOURLINE',
               firm: 'sales-1 holds no document over every book' };
const MAY = { fill: { HARBOURLINE: SAID.book, 'HARBOURLINE/fx': null },
              agreement_declared: { HARBOURLINE: SAID.firm, 'HARBOURLINE/fx': SAID.firm },
              run: { HARBOURLINE: null } };
check("a control is held by the service's verdict on its act at its node, the book where none is "
      + "named and for the firm's own facts, in the record's own sentence; a node or an act the "
      + 'answer does not name is offered, the act judged when it is sent',
      "the verdict read off another node or act; a node the answer does not name read off a node "
      + 'above it, which holds HARBOURLINE/fxo from a booking the service would judge itself; a '
      + "firm's fact read at no node; or the record's sentence spelled anew",
      [state.heldBack(mays(MAY), 'fill'), state.heldBack(mays(MAY), 'fill', 'HARBOURLINE/fx'),
       state.heldBack(mays(MAY), 'fill', 'HARBOURLINE/fx/spot'),
       state.heldBack(mays(MAY), 'fill', 'HARBOURLINE/fxo'),
       state.heldBack(mays(MAY), 'agreement_declared', null), state.heldBack(mays(MAY), 'run'),
       state.heldBack(mays(MAY), 'approval')],
      [SAID.book, null, null, null, SAID.firm, null, null]);
check('nothing is held back off a desk that records, before the grants of the seat on screen are '
      + 'read, or where nothing narrows the act; a refused read holds every act in its own words',
      'a plain desk held back for want of a record, which hides every control it can use; the '
      + 'verdicts on the seat typed before read for the one typed now; or a refusal - no seat '
      + 'named - read as leave to act',
      [state.heldBack(mays(MAY, { record: { spine: null } }), 'fill'),
       state.heldBack(mays(MAY, { source: { kind: 'local' } }), 'fill'),
       state.heldBack({ ...mays(MAY), seat: 'fx-trader' }, 'fill'),
       state.heldBack(seated(null), 'fill'),
       state.heldBack(seated(null, '', { grants: { asked: '', read: null, refused: NO_SEAT } }),
                      'fill')],
      [null, null, null, null, NO_SEAT]);
const SETTLES = (fx) => mays({ status_transition: { HARBOURLINE: SAID.book, 'HARBOURLINE/fx': fx } });
check('a list the seat clears none of says so in plain words where every node refuses its act, '
      + 'naming the verb a seat clearing it holds; one it clears at any node, or an act no node '
      + 'judges, says nothing; a refused read says its own words; the close, which the service '
      + 'says the record judges at the firm, says which seat declares it over the firm',
      "the record's denial shown on the list, which quotes a denial nothing filed and names an "
      + 'event type; a list the seat clears at one node said to be none of its; a refused read '
      + "said as a plain no; or the deed dropped, or the scope not read off the service's firm "
      + "acts, which sends a book's mark to declare the firm's close",
      [state.clearsNone(SETTLES(SAID.book), 'status_transition'),
       state.clearsNone(SETTLES(null), 'status_transition'), state.clearsNone(mays(MAY), 'confirmed'),
       state.clearsNone(seated(null, '', { grants: { asked: '', read: null, refused: NO_SEAT } }),
                        'status_transition'),
       state.clearsNone(mays({ official_close_declared: { HARBOURLINE: SAID.firm,
                                                          'HARBOURLINE/fx': SAID.firm } }),
                        'official_close_declared', 'declare the close')],
      ['sales-1 does not clear these: a seat holding settle over this book does', null, null,
       NO_SEAT, 'sales-1 does not declare the close: a seat holding mark over the firm does']);
const UNREAD = { asked: 'sales-1', read: null, refused: null };
check('a read the grants gate waits on a desk that records until they are read for the seat on '
      + 'screen - none read yet, or those of the seat typed before - and no longer once they are, '
      + 'refused or failed; a desk keeping no record, or showing a file, never waits',
      "the wait dropped, which asks the risk of a held seat before its grants say it is held and "
      + 'files its denial; a plain desk made to wait on grants it never reads; or a grants read '
      + 'that failed read as one still owed, which leaves the risk unread for good',
      [state.grantsOwed(seated(null)), state.grantsOwed({ ...seated({}), seat: 'sales-9' }),
       state.grantsOwed(seated({})),
       state.grantsOwed(seated(null, '', { grants: { asked: '', read: null, refused: NO_SEAT } })),
       state.grantsOwed(seated(null, 'sales-1', { grants: UNREAD })),
       state.grantsOwed(seated(null, 'sales-1', { record: { spine: null } })),
       state.grantsOwed(seated(null, 'sales-1', { source: { kind: 'local' } }))],
      [true, true, false, false, false, false, false]);
const WAIT = 'reading what this seat may do…';
const HOLD = 'sales-1 holds no validate over HARBOURLINE or a node under it';
const [RUNS, HELD_RUN] = [{ run: { HARBOURLINE: null } }, { run: { HARBOURLINE: HOLD } }];
check('a run the queue admits waits in its button\'s place while the seat\'s grants are being '
      + 'read, then is held where the service holds the seat from it and offered where not; a '
      + 'desk keeping no record never waits; a read given its own hold - a P&L between marked days, '
      + 'none - waits on the grants alike, then is held by that hold alone',
      'the button offered before the grants are read, which a seat they hold from the run can '
      + 'press and file a denial; the wait kept once they are read; a plain desk made to wait; or '
      + "the hold given ignored, which holds a seat's P&L between marked days from it as a what-if",
      [state.runHeld(seated(null)), state.runHeld({ ...mays(RUNS), seat: 'sales-9' }),
       state.runHeld(mays(HELD_RUN)), state.runHeld(mays(RUNS)),
       state.runHeld(seated(null, 'sales-1', { record: { spine: null } })),
       state.runHeld(seated(null), null), state.runHeld(mays(HELD_RUN), null)],
      [WAIT, WAIT, HOLD, null, null, WAIT, null]);
const PNL = { pnl_to_now: { HARBOURLINE: HOLD }, pnl_explain: { HARBOURLINE: SAID.book } };
check("a P&L read that is a run is held by the service's verdict on that run - explained, whatever "
      + 'its window, or to now - and a window between two marked days read plain by none',
      'the explain read as a read to now, which offers three valuations to a seat the service '
      + 'holds from them; the two verdicts turned; or a struck window held as a run, which keeps a '
      + 'seat from the read of the record it may make',
      [state.pnlHeld(mays(PNL), '', false), state.pnlHeld(mays(PNL), '2024-07-02', true),
       state.pnlHeld(mays(PNL), '', true), state.pnlHeld(mays(PNL), '2024-07-02', false)],
      [HOLD, SAID.book, SAID.book, null]);
const ROWS = [{ deal_path: '0', value: 5 }, { deal_path: '0/1', value: 2 }];
const READ = { data: { mtm: 5, per_deal: [ROWS[0]], in_sets: [ROWS[1]] }, error: null,
               loading: false, clock: 'c' };
const REFUSED = { data: null, error: 'the book will not price', loading: false, clock: null };
const AS_HELD = { held: true, tone: 'note', joiner: '. ' };
const AS_READ = { held: false, tone: 'hint', joiner: ': ' };
const READING = { data: null, error: null };
check("a read the seat may not ask shows what holds it in the read's place, nothing read before "
      + "it, and says it is held, a notice after a full stop; one it may ask stands as read, its "
      + "refusal the service's own, an error after a colon, and a hint while it is being read",
      'the answer in hand kept under the hold, which shows a seat the numbers the seat before it '
      + "read; the service's refusal kept over the hold; the hold not said to be one, or its "
      + 'tone or joiner turned, which leaves it in the red box or runs a refusal on after a '
      + 'full stop; a read not yet answered drawn in the red box; or a read the seat may ask '
      + 'blanked, or said to be held',
      [state.readAs(HOLD, READ), state.readAs(HOLD, REFUSED), state.readAs(null, READ),
       state.readAs(null, REFUSED), state.readAs(null, READING)],
      [{ ...READ, data: null, error: HOLD, ...AS_HELD }, { ...REFUSED, error: HOLD, ...AS_HELD },
       { ...READ, ...AS_READ }, { ...REFUSED, ...AS_READ, tone: 'error-box' },
       { ...READING, ...AS_READ }]);
const risked = (may, extra) => state.bookRisk({ ...mays(may, extra), risk: READ });
check('the risk is a what-if the queue admits: held where the service holds the seat from a run, '
      + 'and read where not, and on a desk keeping no record; a mark is looked up in its '
      + "top-level rows and each set's trades, none where it is held",
      'the risk judged by another act, which holds it from a seat pricing one node of the book and '
      + 'shows it to one that books and may not price; held on a desk that asks no grants; or '
      + "the trades a set holds left out of the rows, which reads no mark of a trade inside one",
      [risked(HELD_RUN), risked(RUNS), risked({ ...HELD_RUN, fill: { HARBOURLINE: null } }).held,
       risked(HELD_RUN, { record: { spine: null } }).held],
      [{ ...READ, data: null, error: HOLD, ...AS_HELD, marks: [] },
       { ...READ, ...AS_READ, marks: ROWS }, true, false]);
const booked = (etag, may) => mays(may, { source: { kind: 'book', etag } });
const priced = (etag, may = RUNS) => state.riskNow({
  ...booked(etag, may), risk: { ...READ, clock: state.clockOf(booked('e1', {})) } });
check('a mark set against what the record holds now is read off a risk of the book as it stands: '
      + 'the answer where the book has not moved since it was read, none once it has, and the '
      + 'line holding a seat from the read either way',
      'the answer kept after the book moved, which sets a mark of the units held before a fill '
      + 'against the units held after it; or blanked on a book that has not moved',
      [priced('e1').data, priced('e2').data, priced('e2').loading, priced('e1', HELD_RUN).error,
       priced('e1').marks, priced('e2').marks],
      [READ.data, null, false, HOLD, ROWS, []]);

const ACCEPT = { quote_filed: { HARBOURLINE: SAID.book, 'HARBOURLINE/fx': null },
                 fill: { HARBOURLINE: null, 'HARBOURLINE/fx': SAID.firm } };
check('an acceptance is held where either act it files is, the quote or its fill, at the node the '
      + 'quote books at and the book where it states none',
      'the accept asked by its fill alone, which shows an accept the record refuses for its '
      + 'quote; or the two acts judged at the book, which reads the book\'s verdict for a quote '
      + 'booking at HARBOURLINE/fx',
      [state.acceptHeld(mays(ACCEPT)), state.acceptHeld(mays(ACCEPT), 'HARBOURLINE/fx'),
       state.acceptHeld(mays({ quote_filed: { HARBOURLINE: null }, fill: { HARBOURLINE: null } }))],
      [SAID.book, SAID.firm, null]);
// --- every write of the market or its set-up names the seat: the requests a fetch stub catches,
// so nothing leaves this process - no tick, no verification and no set-up is ever sent
const api = await load('api');
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

// --- each write of the market held by the service's verdict on that write, by the route's name
const WRITES = { patchMarket: 'market_tick', tickMarket: 'market_tick', retireMarket: 'market_tick',
                 tickBloomberg: 'terminal_tick', configureBook: 'market_configuration',
                 configureCurve: 'curve', stateSurvival: 'survival_curve',
                 configureSecurities: 'ticker_vocabulary', verifySecurities: 'security_map',
                 savePaths: 'paths' };
const unheld = (folder) => readdirSync(new URL(`../src/${folder}/`, import.meta.url))
  .flatMap((file) => {
    const text = readFileSync(new URL(`../src/${folder}/${file}`, import.meta.url), 'utf8');
    const writers = /import \{([^}]*)\} from '\.\.\/api'/.exec(text)?.[1] ?? '';
    const held = new Set([...text.matchAll(/heldBack\([^,]+, '(\w+)'/g)].map((hit) => hit[1]));
    return Object.entries(WRITES).filter(([writer, key]) => !held.has(key)
      && new RegExp(`\\b${writer}\\b`).test(writers)).map(([writer]) => `${file} ${writer}`);
  });
check("every screen writing the market holds its control by the service's verdict on that write, "
      + 'by the route\'s own name, and a screen performing two writes by both',
      "a screen held by another write's verdict, which offers a seat the service holds from a "
      + 'survival curve its form; or a second write left to the first one\'s verdict, which '
      + 'offers a terminal tick to a seat the service holds from it',
      [...unheld('components'), ...unheld('views')], []);

// --- a refusal told apart by its code: what a read the service refuses throws, as a view renders it
const refusing = async (status, body) => {
  globalThis.fetch = async () => ({ ok: false, status, statusText: '', text: async () => body });
  return api.getBook().then(() => 'answered', api.failure);
};
const said = (detail, code) => JSON.stringify({ detail, code });
// a request the service's framework would not parse, which names no kind
const UNPARSED = JSON.stringify({ detail: [{ loc: ['body', 'date'], msg: 'field required' }] });
check('a refusal is told apart by the code the service sends beside its sentence, never by its '
      + 'words: a seat named by nobody is told to set one, any other refusal says its own sentence '
      + 'and keeps its code, and a body that is no envelope, or names no kind, reads as itself',
      'the code dropped from the error thrown, which leaves the risk screen unable to say who puts '
      + 'an unpriced book right; the rewording keyed on the words, which tells a seat its own '
      + 'sentence in words the service did not send; or the JSON envelope printed at the reader',
      [await refusing(422, said('the request names no seat', 'no_actor')),
       await refusing(422, said('no actor for this act', 'refused')),
       await refusing(422, said('the book will not price: no curve', 'unpriced')),
       await refusing(500, 'Internal Server Error'), await refusing(422, UNPARSED)],
      [{ status: 422, code: 'no_actor', error: api.NO_SEAT },
       { status: 422, code: 'refused', error: 'no actor for this act' },
       { status: 422, code: 'unpriced', error: 'the book will not price: no curve' },
       { status: 500, code: null, error: 'Internal Server Error' },
       { status: 422, code: null, error: UNPARSED }]);

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

// --- XVA: how large a run's exposure gets, off the peaks its row keeps
const PEAKS = { EE: { value: 5, day: '2026-12-16' }, 'PFE_97.5': { value: 9, day: '2027-02-16' },
                PFE_99: { value: 12, day: '2027-02-16' }, PFE_X: { value: 1, day: '2026-11-16' } };
check("a run's peaks are its row's: the expected exposure's and each PFE column's, each at the "
      + 'percentile its own column names, and none on a row that keeps none',
      'the PFE taken at 95 whatever the run asked; only the first PFE column read; a column naming '
      + 'no percentile read as one; or the expected exposure read off a PFE column',
      [desk.peaks(PEAKS), desk.peaks({ ENE: { value: 0, day: '2026-11-16' } }), desk.peaks(null)],
      [{ ee: { value: 5, day: '2026-12-16' },
         pfe: [{ value: 9, day: '2027-02-16', quantile: 97.5 },
               { value: 12, day: '2027-02-16', quantile: 99 }] }, { pfe: [] }, { pfe: [] }]);

// --- the primitives of `components/Desk.tsx`, as the markup they draw
const draw = await drawer('Desk');
check("a control the seat is held from is said in its button's place, its label then the line "
      + 'holding it, and a whole form with no label in a note; the control where nothing holds '
      + 'it, and nothing where none is given',
      'the control drawn under a hold, which offers a seat a button whose press files a denial; '
      + 'the line holding it dropped, which names a control and not why it is missing; the '
      + 'label said where nothing holds; or a held form said as a bare line beside the page',
      [draw('Gated', { label: 'run', hold: HOLD }, 'Run'),
       draw('Gated', { label: 'run', hold: null }, 'Run'), draw('Gated', { label: 'run', hold: null }),
       draw('Gated', { hold: HOLD }, 'Form'), draw('Gated', { hold: null }, 'Form')],
      [`<span class="hint">run</span><span>${HOLD}</span>`, 'Run', '', `<div class="note">${HOLD}</div>`,
       'Form']);
check("an amount's cell draws a negative in the loss colour, the exact value on its hover unless its "
      + 'title says what the figure is, the currency after the amount; one not known is a dash '
      + 'under what is missing, and an empty cell where that is not said',
      'a gain drawn as a loss or a loss as a gain, on the P&L, the risk and the XVA table alike; '
      + 'the rounded figure left without its exact value, or a ladder\'s summed cell without what '
      + 'it sums; or a zero drawn for an amount not known',
      [draw('Amount', { value: -1250.5, kind: 'money' }), draw('Amount', { value: 3, currency: 'USD' }),
       draw('Amount', { value: null, absent: 'a set is unread' }), draw('Amount', { value: null }),
       draw('Amount', { value: -2, kind: 'money', title: '2 nodes summed' })],
      ['<td class="n neg" title="-1250.5">-1,250.50</td>',
       '<td class="n" title="3">3 <span class="unit">USD</span></td>',
       '<td class="n"><span class="hint" title="a set is unread">—</span></td>', '<td class="n"></td>',
       '<td class="n neg" title="2 nodes summed">-2</td>']);
check("a stat tile draws its label, then its amount and currency, a negative in the loss colour and "
      + 'the exact value on its hover unless its title says what the tile is; one not known is a dash',
      "a loss drawn as a gain on the P&L's totals, its explain, the book's MTM and a trade's worth "
      + "alike; a zero drawn for a total not known; or a share drawn as money, which prints the "
      + "explain's residual share to two decimals with no percent sign",
      [draw('Stat', { label: 'P&L', value: -1250.5, currency: 'USD' }),
       draw('Stat', { label: 'worth', value: 3, title: 'to the desk' }),
       draw('Stat', { label: 'Realised', value: null, currency: 'USD' }),
       draw('Stat', { label: 'Residual share', value: -12.5, kind: 'percent' })],
      ['<div class="stat" title="-1250.5"><div class="label">P&amp;L</div><div class="value neg">'
        + '-1,250.50 <span class="unit">USD</span></div></div>',
       '<div class="stat" title="to the desk"><div class="label">worth</div><div class="value">'
        + '3</div></div>',
       '<div class="stat"><div class="label">Realised</div><div class="value">—</div></div>',
       '<div class="stat" title="-12.5"><div class="label">Residual share</div><div class="value neg">'
        + '-12.5 %</div></div>']);
check("a credit limit's line draws a negative figure in the loss colour, as a tile and a cell do",
      'a headroom gone below zero drawn as room left, or the limit itself drawn as a loss',
      draw('LimitLine', { read: { currency: 'USD', flag: null, notes: [], figures: [
        { label: 'credit limit', value: 25000 }, { label: 'headroom', value: -5000 }] } })
        .match(/<b[^>]*>/g),
      ['<b>', '<b class="neg">']);

// a table whose second column a screen replaces, every cell asked for once
const replacing = [];
const replaced = (await drawer('DataTable'))('DataTable', { columns: ['a', 'b'], index: [],
  data: [[1, 2], [3, 4]], cell: (r, c) => { replacing.push([r, c]); return c === 1 ? 'x' : undefined; } });
check('a table asks a screen once for each cell it may replace, and sets a heading right over its '
      + 'numbers unless the screen replaced that column',
      'the first row asked again for every heading, which builds each replaced cell of it twice per '
      + 'paint; or a replaced column headed as numbers',
      [replacing.length, [...replaced.matchAll(/<th( [^>]*)?>/g)].map(([, attributes]) => attributes ?? '')],
      [4, [' class="n"', '']]);

// a trade's cells, as each table draws them off its size and its levels
const picked = (key, value) => ({ key, value, title: key });
const SIZED = { side: picked('Buy_Sell', 'Pay'), amount: picked('Amount', 1e6),
               of: picked('Currency', 'USD') };
const LEVELS = { on: picked('Pair', 'USDZAR'), sense: 'Call', strike: picked('Strike', 18.25) };
const cells = (props) => [...draw('TradeCells', { object: 'X', schema: null, ...props })
  .matchAll(/<td([^>]*)>(.*?)<\/td>/g)].map(([, attributes, inner]) =>
  `${/class="([^"]*)"/.exec(attributes)?.[1] ?? ''}|${inner.replace(/ title="[^"]*"/g, '')}`);
check("a trade's cells read its side, what it is on, its amount then the currency it is in, its "
      + 'levels and terms; a paying side reads as a sale, a closed trade shows no side or amount, '
      + 'and a level the recipe solved is marked',
      "the currency drawn ahead of the amount, as the blotter once drew it; a side read as a sale "
      + "only where it says Sell, which draws a payer as buying; a closed trade's side and amount "
      + 'still drawn; or the solved level unmarked',
      [cells({ size: SIZED, read: LEVELS }), cells({ size: SIZED, read: LEVELS, closed: true }).slice(0, 3),
       cells({ size: SIZED, read: LEVELS, solved: 'strike' })[4]],
      [['|<span class="side sell">Pay</span>', '|USDZAR Call', 'n|1,000,000', '|USD', 'n|18.25',
        'n|', 'terms|'],
       ['|', '|USDZAR Call', 'n|'], 'n solved|18.25']);

done();
