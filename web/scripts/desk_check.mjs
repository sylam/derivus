// Drives `src/desk.ts`'s quote reading where the eye cannot: which rows the Risk screen shows when
// it reads the book in quotes. Every check names the MUTATION it kills - the change to the module
// that would still typecheck, still render, and still be wrong.
//
//   node web/scripts/desk_check.mjs          # exits 1 on a miss
//
// The answer is `GET /book/risk`'s shape on a book reading one USD curve its quotes build, an FX
// surface they build too, and a spot nothing quotes. The last check is what a named calculation
// shows for a dial it does not state.

import { readFileSync } from 'node:fs';
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

const UNPRICED = 'the book will not price a consolidated risk run: Cannot find InterestRate.ZAR-X';
const who = (said) => said && [/the desk puts it right/.test(said), /its sender puts it right/.test(said),
                               said.includes('from-fo.json')];
check('a book that will not price says who puts it right: the desk on its own book, the sender of '
      + 'a document opened from another system by its name; any other refusal says for itself',
      "the document's name left unread, which tells a desk to fix in its own book a file another "
      + 'system sent it; or every refusal read as a book that will not price, which says so under '
      + "the record's sentence to a seat it does not admit",
      [who(desk.unpriced(UNPRICED)), who(desk.unpriced(UNPRICED, 'from-fo.json')),
       who(desk.unpriced('sales-9 holds no validate over HARBOURLINE', 'from-fo.json')) ?? null],
      [[true, false, false], [false, true, true], null]);

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
const pressed = ({ hint }) => {
  const control = /Press "([^"]+)" at the top/.exec(hint)?.[1];
  return [control, HEADER.includes(control)];
};
check('over a read-only copy of a file a screen names the control that leaves it for the book, and '
      + 'what it then shows; a desk serving no book names the control that opens a file; each by '
      + 'the label the header draws',
      "a control named by a word the header does not draw ('return to book'), which sends the "
      + 'reader to a button that is not there; or what the screen shows left out of the sentence',
      [pressed(desk.offBook("the desk's own risk")),
       desk.offBook("the desk's own risk").hint.endsWith("this screen then shows the desk's own risk."),
       pressed(desk.NO_BOOK)],
      [['back to book', true], true, ['open a job file', true]]);
check('a file opened on a desk that records is a read-only copy the service is not asked for, '
      + 'said as the state it is with the control that leaves it; a desk keeping no record '
      + 'serves the file, and says nothing',
      'the open asked of a desk that records, whose refusal lands in a red box over the copy; '
      + 'or a desk keeping no record told its file is a copy, which opens none there',
      [desk.copyOnly(true, 'trades.json'), desk.copyOnly(false, 'trades.json')],
      ["This desk's book is what its record holds, so trades.json is shown as a read-only copy "
        + 'and not opened on the desk. Press "back to book" at the top to leave it.', null]);

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
      [[['Reference', 'Path', "Desk's value", 'Ccy'], ['ISDA-KESTREL', '0', 1234567.891, 'USD'],
        ['TWICE', null, -0.30000000000000004, 'USD']],
       [['Reference', 'Client', 'Path', 'Waiting on a second seat', "Desk's value", 'Ccy'],
        ['ISDA-KESTREL', 'Kestrel (KES), Alderby (ALD)', '0', `NDF-2, ${'b'.repeat(12)}`, 1234567.891, 'USD'],
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
let walked = 0;
const COUNTED = { get 'Price Factors'() { walked += 1; return CURVES['Price Factors']; } };
[{ widget: 'Float', description: '', value: 0 }, named([]), named({ Currency: ['FxRate'] })]
  .forEach((descriptor) => desk.factorNames(descriptor, COUNTED));
check('a field naming no factor never walks the market',
      'the gate on a field declaring no factor type dropped, which walks every factor of the '
      + 'market once per input on a screen - 90 ms a render at 3,000 factors and 300 inputs',
      [walked, desk.factorNames(named(['FxRate']), COUNTED), walked], [0, ['ZAR'], 1]);

// --- what the seat may do: one rule over GET /book/seat
const verbs = (held) => ({ book: [], approve: [], mark: [], document: [], settle: [], confirm: [],
                            collateral: [], validate: [], admin: [], ...held });
// the map GET /book/seat answers where no capabilities document moves an act
const DEMANDS = { fill: 'book', amendment: 'book', quote_filed: 'book', approval: 'approve',
                  entity_declared: 'document',
                  agreement_declared: 'document', official_close_declared: 'mark',
                  status_transition: 'settle', confirmed: 'confirm', collateral: 'collateral' };
const seated = (grants, seat = 'sales-1', extra = {}, demands = DEMANDS) => ({
  source: { kind: 'book' }, record: { spine: { lsn: 1 } }, seat,
  doc: { Calc: { Deals: { Reference: 'HARBOURLINE' } } },
  grants: grants && { asked: seat, read: { seat, verbs: verbs(grants), demands }, refused: null },
  ...extra });
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
      'the record said to hold none to a seat lacking sight of the node, which tells it a trade '
      + 'on the record is not; a node matched by prefix, which reads a grant over HARBOUR as the '
      + 'whole of HARBOURLINE; or a grant nothing narrows, or grants not yet read, taken for no '
      + 'sight',
      [state.noneShown(seated({ book: ['HARBOURLINE'] })), state.noneShown(seated({ validate: null })),
       state.noneShown(seated(null)),
       state.noneShown(seated({ book: ['HARBOURLINE/FX'], validate: ['HARBOURLINE/FX'] }, 'sales-9')),
       state.noneShown(seated({ book: ['HARBOUR'] }, 'sales-9')), state.noneShown(seated({}, 'sales-9'))],
      [NONE, NONE, NONE, 'sales-9 is shown none of it, seeing only what sits under HARBOURLINE/FX',
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
      [state.noneShown(seated({ book: ['HARBOURLINE'] }), YET), state.noneShown(seated(null), YET),
       state.noneShown(seated({ book: ['HARBOURLINE/FX'] }, 'sales-9'), YET),
       state.noneShown(seated({ book: ['HARBOURLINE'] }), NO_QUOTE),
       state.noneShown(seated({ book: ['HARBOURLINE/FX'] }, 'sales-9'), NO_QUOTE)],
      [YET, YET, 'sales-9 is shown none of it, seeing only what sits under HARBOURLINE/FX',
       NO_QUOTE, 'sales-9 is shown none of it, seeing only what sits under HARBOURLINE/FX']);

const DONE = "Every client's sets have a run that is done";
check('an XVA table with nothing to act on says every run is done, and on a desk that records '
      + 'that no trade waits, to a seat shown the whole book; a seat shown none of it is told that',
      'the sentence said whatever the seat is shown, which tells a seat shown no trade that sets '
      + 'never run have a run that is done; or the second seat named on a desk that has none',
      [state.nothingWrong(seated({ book: ['HARBOURLINE'] })),
       state.nothingWrong(seated(null, 'sales-1', { record: { spine: null } })),
       state.nothingWrong(seated({ book: ['HARBOURLINE/FX'] }, 'sales-9'))],
      [`${DONE}, and no trade waits on a second seat`, DONE,
       'sales-9 is shown none of it, seeing only what sits under HARBOURLINE/FX']);
check('a P&L answering no row says in its table\'s place that no position stood in the window and '
      + 'scope, to a seat shown the whole book; a seat shown none of it is told that',
      'the sentence said whatever the seat is shown, which tells a seat shown no position that '
      + 'the book held none; or an empty table left under a total of nothing',
      [state.noPositions(seated({ book: ['HARBOURLINE'] })),
       state.noPositions(seated({ book: ['HARBOURLINE/FX'] }, 'sales-9'))],
      ['No position stood in this window and scope',
       'sales-9 is shown none of it, seeing only what sits under HARBOURLINE/FX']);

const NO_SEAT ='No seat is set, and this desk files every act under the seat that asks.';
check('a seat may act where a node it holds the act\'s verb at is the node or above it, the book '
      + 'by default and the whole firm only by a grant over every node; and says, naming the '
      + 'verb, what holds it elsewhere',
      'the held nodes matched by prefix, which lets a grant over HARBOURLINE act at HARBOURLINE-2; '
      + 'a grant at a node read as reaching the firm, which offers a declaration the record '
      + 'refuses; a node above the one held read as reached, which offers HARBOURLINE to a seat '
      + 'holding HARBOURLINE/fx; an empty list read as everywhere; or another act\'s grant read',
      [state.heldBack(seated({ mark: ['HARBOURLINE'] }), 'mark'),
       state.heldBack(seated({ book: ['HARBOURLINE'] }), 'fill', 'HARBOURLINE/fx/spot'),
       state.heldBack(seated({ document: null }), 'agreement_declared', null),
       state.heldBack(seated({ book: ['HARBOURLINE'] }), 'mark'),
       state.heldBack(seated({ book: ['HARBOURLINE'] }), 'fill', 'HARBOURLINE-2'),
       state.heldBack(seated({ document: ['HARBOURLINE'] }), 'entity_declared', null),
       state.heldBack(seated({ book: ['HARBOURLINE/fx'] }), 'amendment'),
       state.heldBack(seated({ mark: ['HARBOURLINE'] }), 'official_close_declared', null)],
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
      [state.heldBack(seated({}, 'sales-1', { record: { spine: null } }), 'fill'),
       state.heldBack(seated({}, 'sales-1', { source: { kind: 'local' } }), 'fill'),
       state.heldBack({ ...seated({}), seat: 'fx-trader' }, 'approval'),
       state.heldBack(seated(null), 'fill'),
       state.heldBack(seated({ approve: null }), 'approval'),
       state.heldBack(seated(null, '', { grants: { asked: '', read: null, refused: NO_SEAT } }),
                      'fill')],
      [null, null, null, null, null, NO_SEAT]);
const UNREAD = { asked: 'sales-1', read: null, refused: null };
check('a read the grants gate waits on a desk that records until they are read for the seat on '
      + 'screen - none read yet, or those of the seat typed before - and no longer once they are, '
      + 'refused or failed; a desk keeping no record, or showing a file, never waits',
      "the wait dropped, which asks the risk of a held seat before its grants say it is held and "
      + 'files its denial; a plain desk made to wait on grants it never reads; or a grants read '
      + 'that failed read as one still owed, which leaves the risk unread for good',
      [state.grantsOwed(seated(null)), state.grantsOwed({ ...seated({}), seat: 'sales-9' }),
       state.grantsOwed(seated({ validate: null })),
       state.grantsOwed(seated(null, '', { grants: { asked: '', read: null, refused: NO_SEAT } })),
       state.grantsOwed(seated(null, 'sales-1', { grants: UNREAD })),
       state.grantsOwed(seated(null, 'sales-1', { record: { spine: null } })),
       state.grantsOwed(seated(null, 'sales-1', { source: { kind: 'local' } }))],
      [true, true, false, false, false, false, false]);
const WAIT = 'reading what this seat may do…';
check('a run the queue admits waits in its button\'s place while the seat\'s grants are being '
      + 'read, then is held from a seat holding validate nowhere and offered to one holding it; a '
      + 'desk keeping no record never waits',
      'the button offered before the grants are read, which a seat they hold from the run can '
      + 'press and file a denial; the wait kept once they are read; or a plain desk made to wait',
      [state.runHeld(seated(null)),
       state.runHeld({ ...seated({ validate: ['HARBOURLINE'] }), seat: 'sales-9' }),
       state.runHeld(seated({})), state.runHeld(seated({ validate: ['HARBOURLINE'] })),
       state.runHeld(seated(null, 'sales-1', { record: { spine: null } }))],
      [WAIT, WAIT, 'Needs validate: sales-1 holds it nowhere.', null, null]);
const MOVED = { ...DEMANDS, approval: 'mark' };
check('a control is judged by the verb its act demands under the map in force and says that verb: '
      + 'where a stored map moves the approval to mark the sign control is the marking seat\'s '
      + 'and held from the approving one, and a verb the service asks itself is its own',
      'the act read as a verb\'s name, or its verb off the vocabulary and not the map GET '
      + '/book/seat answers, which shows the approving seat a sign control the record refuses and '
      + 'hides it from the marking seat that holds it',
      [state.heldBack(seated({ mark: ['HARBOURLINE'] }, 'sales-1', {}, MOVED), 'approval'),
       state.heldBack(seated({ approve: ['HARBOURLINE'] }, 'sales-1', {}, MOVED), 'approval'),
       state.heldBack(seated({ approve: ['HARBOURLINE'] }), 'approval'),
       state.heldBack(seated({ mark: ['HARBOURLINE'] }), 'approval'),
       state.heldBack(seated({ approve: ['HARBOURLINE'] }, 'sales-1', {}, MOVED), 'mark')],
      [null, 'Needs mark: sales-1 holds it nowhere.', null,
       'Needs approve: sales-1 holds it nowhere.', 'Needs mark: sales-1 holds it nowhere.']);
check('a what-if is admitted to a seat holding validate at the book, above it or at any node under '
      + 'it, as the queue asks it; every other act reaches down alone',
      'the what-if judged as a booking is, which holds the credit charge from a seat pricing one '
      + 'node of the book that the queue admits; a grant under another book read as under this '
      + 'one; or every act reaching up, which offers a booking at the book to a seat granted a desk',
      [state.heldBack(seated({ validate: ['HARBOURLINE/fx'] }), 'validate'),
       state.heldBack(seated({ validate: ['HARBOURLINE'] }), 'validate'),
       state.heldBack(seated({ validate: ['HARBOURLINE-2/fx'] }), 'validate'),
       state.heldBack(seated({}), 'validate'),
       state.heldBack(seated({ book: ['HARBOURLINE/fx'], validate: ['HARBOURLINE/fx'] }), 'fill')],
      [null, null, 'Needs validate over HARBOURLINE: sales-1 holds it only over HARBOURLINE-2/fx.',
       'Needs validate: sales-1 holds it nowhere.',
       'Needs book over HARBOURLINE: sales-1 holds it only over HARBOURLINE/fx.']);
const ROWS = [{ deal_path: '0', value: 5 }, { deal_path: '0/1', value: 2 }];
const READ = { data: { mtm: 5, per_deal: [ROWS[0]], in_sets: [ROWS[1]] }, error: null,
               loading: false, clock: 'c' };
const REFUSED = { data: null, error: 'the book will not price', loading: false, clock: null };
const HOLD = 'Needs validate: sales-1 holds it nowhere.';
const AS_HELD = { held: true, tone: 'note', joiner: '. ' };
const AS_READ = { held: false, tone: 'error-box', joiner: ': ' };
check("a read the seat may not ask shows what holds it in the read's place, nothing read before "
      + "it, and says it is held, a notice after a full stop; one it may ask stands as read, its "
      + "refusal the service's own, an error after a colon",
      'the answer in hand kept under the hold, which shows a seat the numbers the seat before it '
      + "read; the service's refusal kept over the hold; the hold not said to be one, or its "
      + 'tone or joiner turned, which leaves it in the red box or runs a refusal on after a '
      + 'full stop; or a read the seat may ask blanked, or said to be held',
      [state.readAs(HOLD, READ), state.readAs(HOLD, REFUSED), state.readAs(null, READ),
       state.readAs(null, REFUSED)],
      [{ ...READ, data: null, error: HOLD, ...AS_HELD }, { ...REFUSED, error: HOLD, ...AS_HELD },
       { ...READ, ...AS_READ }, { ...REFUSED, ...AS_READ }]);
const risked = (grants, extra) =>
  state.bookRisk({ ...seated(grants, 'sales-1', extra), risk: READ });
check('the risk is a what-if the queue admits: held from a seat holding validate nowhere, and '
      + 'read by one holding it at the book or a node under it, and on a desk keeping no record; '
      + "a mark is looked up in its top-level rows and each set's trades, none where it is held",
      'the risk judged as a booking, which holds it from a seat pricing one node of the book and '
      + 'shows it to one that books and may not price; held on a desk that asks no grants; or '
      + "the trades a set holds left out of the rows, which reads no mark of a trade inside one",
      [risked({}), risked({ validate: ['HARBOURLINE/fx'] }),
       risked({ book: ['HARBOURLINE'], mark: ['HARBOURLINE'] }).held,
       risked({}, { record: { spine: null } }).held],
      [{ ...READ, data: null, error: HOLD, ...AS_HELD, marks: [] },
       { ...READ, ...AS_READ, marks: ROWS }, true, false]);
const booked = (etag, grants) => seated(grants, 'sales-1', { source: { kind: 'book', etag } });
const priced = (etag, grants = { validate: ['HARBOURLINE'] }) => state.riskNow({
  ...booked(etag, grants), risk: { ...READ, clock: state.clockOf(booked('e1', {})) } });
check('a mark set against what the record holds now is read off a risk of the book as it stands: '
      + 'the answer where the book has not moved since it was read, none once it has, and the '
      + 'line holding a seat from the read either way',
      'the answer kept after the book moved, which sets a mark of the units held before a fill '
      + 'against the units held after it; or blanked on a book that has not moved',
      [priced('e1').data, priced('e2').data, priced('e2').loading, priced('e1', {}).error,
       priced('e1').marks, priced('e2').marks],
      [READ.data, null, false, HOLD, ROWS, []]);

const QUOTED = { ...DEMANDS, quote_filed: 'approve' };
check('an acceptance is held where either act it files is, the quote or its fill, at the node the '
      + 'quote books at and the book where it states none',
      'the accept asked by its fill alone, which shows a seat holding book an accept the record '
      + 'refuses where a stored map moves the quote to approve; or the two acts judged at the '
      + 'book, which holds a quote booking at HARBOURLINE/fx from the seat granted there',
      [state.acceptHeld(seated({ book: ['HARBOURLINE'] }, 'sales-1', {}, QUOTED)),
       state.acceptHeld(seated({ approve: ['HARBOURLINE'] }, 'sales-1', {}, QUOTED)),
       state.acceptHeld(seated({ book: ['HARBOURLINE'], approve: ['HARBOURLINE'] }, 'sales-1', {},
                               QUOTED)),
       state.acceptHeld(seated({ book: ['HARBOURLINE'] })),
       state.acceptHeld(seated({ book: ['HARBOURLINE/fx'] }), 'HARBOURLINE/fx'),
       state.acceptHeld(seated({ book: ['HARBOURLINE/fx'] }))],
      ['Needs approve: sales-1 holds it nowhere.', 'Needs book: sales-1 holds it nowhere.', null,
       null, null, 'Needs book over HARBOURLINE: sales-1 holds it only over HARBOURLINE/fx.']);
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
      + 'holding it; the control where nothing holds it, and nothing where none is given',
      'the control drawn under a hold, which offers a seat a button whose press files a denial; '
      + 'the line holding it dropped, which names a control and not why it is missing; or the '
      + 'label said where nothing holds',
      [draw('Gated', { label: 'run', hold: HOLD }, 'Run'),
       draw('Gated', { label: 'run', hold: null }, 'Run'), draw('Gated', { label: 'run', hold: null })],
      [`<span class="hint">run</span><span>${HOLD}</span>`, 'Run', '']);
check("an amount's cell draws a negative in the loss colour, the exact value on its hover, the "
      + 'currency after the amount; one not known is a dash under what is missing, and an empty '
      + 'cell where that is not said',
      'a gain drawn as a loss or a loss as a gain, on the P&L, the risk and the XVA table alike; '
      + 'the rounded figure left without its exact value; or a zero drawn for an amount not known',
      [draw('Amount', { value: -1250.5, kind: 'money' }), draw('Amount', { value: 3, currency: 'USD' }),
       draw('Amount', { value: null, absent: 'a set is unread' }), draw('Amount', { value: null })],
      ['<td class="n neg" title="-1250.5">-1,250.50</td>',
       '<td class="n" title="3">3 <span class="unit">USD</span></td>',
       '<td class="n"><span class="hint" title="a set is unread">—</span></td>', '<td class="n"></td>']);

done();
