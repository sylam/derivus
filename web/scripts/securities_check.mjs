// Drives `src/securities.ts` where the eye cannot: the edits a desk makes to one vocabulary entry,
// the two requests they build, and the join an IPV reader reads. Every check names the MUTATION it
// kills - the change to the module that would still typecheck, still render, and still be wrong.
//
//   node web/scripts/securities_check.mjs     # exits 1 on a miss
//
// The shapes are the service's own: `GET /book/securities` on a home carrying a hand-authored map,
// a seed completed the way the read verb completes one, and a book of one ZAR curve and one USD.
// The module is bundled rather than transformed, because it reads `src/tokens.ts` rather than
// spelling an object test twice.

import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const bundled = await build({
  entryPoints: [fileURLToPath(new URL('../src/securities.ts', import.meta.url))],
  bundle: true, format: 'esm', write: false, logLevel: 'silent',
});
const securities = await import('data:text/javascript;base64,' +
  Buffer.from(bundled.outputFiles[0].text).toString('base64'));

// ---- the service's shapes -----------------------------------------------------------------------

const ZAR = {
  prefix: 'SASW', expect: 'ZAR SWAP QTR', years: [3, 5, 7], weeks: true,
  overnight: { security: 'ZARONIA Index', expect: 'South African Overnight' },
  conventions: { curve_day_count: 'ACT_365', spot_days: 0, front: 'fixings/3M',
                 compounding: 'None', notional: 1000000.0, quote_scale: 1.0 },
};

const SEED = {
  fx_vol: { pairs: ['USDZAR', 'EURZAR'], expiries: { '1M': 0.0822, '1Y': 1.0 },
            pillars: [0.1, 0.25], leverage_prior: { USDZAR: -0.4 } },
  fx_spot: { pairs: ['EURUSD', 'USDZAR'] },
  rates: { ZAR },
  swaption: { ZAR: { prefix: 'SASN', expect: 'ZAR SWPT NVOL', tenor_years: [1, 2] } },
};

const evidence = (security, name) =>
  ({ security, name, last_update: '2024-06-27', verified: '2024-06-28' });

// `author_map`'s own shape: entries filed under their paths, the ledger keyed by ticker, and an
// fx_vol block carrying its `expiries` table BESIDE its entries - metadata a walk goes past.
const BLOCKS = {
  rates: { ZAR: { fixings: { '3M': evidence('JIBA3M Index', 'JIBA3M Index') },
                  strip: { '1Y': evidence('SASW1 BGN Curncy', 'ZAR SWAP QTR (VS 3M) 1Y') } } },
  fx_vol: { USDZAR: { quotes: { '1M': { ATM: evidence('USDZARV1M BGN Curncy', 'USD-ZAR OPT VOL') } },
                      expiries: { '1M': 0.0822 } } },
};

const REJECTED = {
  'SASW5 BGN Curncy': { verdict: 'dead', name: 'ZAR SWAP QTR (VS 3M) 5Y',
                        last_update: '2007-03-26', error: null },
  'EURZARV1M BGN Curncy': { verdict: 'invalid', name: null, last_update: null,
                            error: 'Unknown/Invalid Security' },
};

const USED = [
  { curve: 'ZAR', tenor: '3M', security: 'JIBA3M Index', quote: 7.41, timestamp: '2024-06-28',
    use: 'Yes', evidence: { name: 'JIBA3M Index', last_update: '2024-06-27',
                            verified: '2024-06-28' } },
  { curve: 'ZAR', tenor: '5Y', security: 'SASW5 BGN Curncy', quote: 8.02,
    timestamp: '2024-06-28', use: 'Yes', evidence: REJECTED['SASW5 BGN Curncy'] },
  { curve: 'ZAR', tenor: '10Y', security: 'SASW10 BGN Curncy', quote: 8.6,
    timestamp: '2024-06-28', use: 'No', evidence: { verdict: 'unmapped' } },
  { curve: 'USD', tenor: '1Y', security: 'USOSFR1 BGN Curncy', quote: 5.1,
    timestamp: '2024-06-27', use: 'Yes',
    evidence: { name: 'USD OIS 1Y', last_update: '2024-06-26', verified: '2024-06-28' } },
];

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

// --- setAt
const standing = JSON.stringify(ZAR);
check('setAt writes the field its path names and leaves its siblings',
      'the recursion stopped one level in (`{...node, [path[0]]: value}`), which writes the number '
      + '2 over the whole conventions block and sends the verb a curve stating nothing',
      securities.setAt(ZAR, ['conventions', 'spot_days'], 2).conventions,
      { ...ZAR.conventions, spot_days: 2 });
check('setAt leaves the entry it was handed standing',
      'the copy dropped (`node[key] = value; return node`), which writes into the entry the card '
      + 'is rendering from - the edit then shows as saved before the verb has seen it, and a '
      + 'refusal leaves a desk reading a file it does not have',
      JSON.stringify(ZAR), standing);
check('setAt builds the dicts on the way that the entry does not state',
      '`node[key]` read without the object test, which throws on the first field of a block a '
      + 'seed omits and takes the card down with it',
      securities.setAt({ prefix: 'GATE' }, ['overnight', 'security'], 'GATEON Index'),
      { prefix: 'GATE', overnight: { security: 'GATEON Index' } });
check('setAt at no path is the value itself',
      'the empty path falling through to the spread, which turns a pair LIST into '
      + '{0: ..., 1: ...} the moment a block keyed by a field is saved',
      securities.setAt(SEED.fx_spot.pairs, [], ['EURUSD']), ['EURUSD']);

// --- withMember
check('withMember replaces the member it names',
      'the index test dropped, which writes one year over every year in the strip',
      securities.withMember(ZAR.years, 1, 4), [3, 4, 7]);
check('withMember removes the member it names',
      'the filter inverted (`i === index`), which drops every pair but the one a desk asked to drop',
      securities.withMember(SEED.fx_vol.pairs, 0, null), ['EURZAR']);
check('withMember appends past the end',
      'a test of `index > list.length`, which makes `add` do nothing at all',
      securities.withMember(ZAR.years, 3, 10), [3, 5, 7, 10]);
check('withMember leaves the list it was handed standing',
      'a splice or an in-place assignment, which edits the array the inputs are rendering from',
      ZAR.years, [3, 5, 7]);

// --- seedDescriptor
check('seedDescriptor reads the widget off the value\'s own shape',
      'the widget read off the KEY - a rule of the client\'s about what a `spot_days` is, which '
      + 'is finance in a client that is meant to carry none',
      ['prefix', 'spot_days', 'weeks', 'quote_scale'].map((key) =>
        securities.seedDescriptor(key, { prefix: 'SASW', spot_days: 0, weeks: true,
                                         quote_scale: 1.0 }[key]).widget),
      ['Text', 'Float', 'Checkbox', 'Float']);
check('seedDescriptor calls a whole number a Float',
      '`Integer` for a value that reads whole, under which `encodeScalar` truncates: a quote '
      + 'scale moved from 1.0 to 0.01 saves as 0 and every quote on that curve reads zero',
      securities.seedDescriptor('spot_days', 0),
      { widget: 'Float', description: 'spot_days', value: 0 });

// --- mergeRequest
check('mergeRequest names the block, the key and the entry under it',
      'the entry spread into the body, which the verb reads as a nameless request and refuses',
      securities.mergeRequest('rates', 'ZAR', ZAR), { block: 'rates', key: 'ZAR', entry: ZAR });
check('mergeRequest sends an absent entry as an explicit null',
      '`entry` left undefined, which `JSON.stringify` DROPS - the verb then reads `entry` as None '
      + 'and every save of an entry the card could not read removes it from the desk\'s file',
      JSON.stringify(securities.mergeRequest('rates', 'GATE', undefined)),
      '{"block":"rates","key":"GATE","entry":null}');
check('mergeRequest trims the key',
      'the trim dropped, which files a second entry under `ZAR ` beside the one the desk meant',
      securities.mergeRequest('rates', ' ZAR ', {}).key, 'ZAR');

// --- verifyRequest
check('verifyRequest sends nothing for the whole map',
      'blank fields travelling anyway, which the verb refuses by name (`\'\' is no vocabulary '
      + 'block`) for the one button that asked for everything',
      [securities.verifyRequest({}), securities.verifyRequest({ block: '', key: 'ZAR' })],
      [{}, {}]);
check('verifyRequest carries a block, and a key only inside its block',
      'the key sent alone, which the verb refuses by name for a scope no button on the screen can '
      + 'even state',
      [securities.verifyRequest({ block: 'rates' }),
       securities.verifyRequest({ block: 'rates', key: 'ZAR' }),
       securities.verifyRequest({ key: 'ZAR' })],
      [{ block: 'rates' }, { block: 'rates', key: 'ZAR' }, {}]);
check('verifyRequest carries named tickers, and an empty list is no scope',
      '`securities: []` travelling, which is a scope covering nothing and a terminal trip for it',
      [securities.verifyRequest({ securities: ['SASW1 BGN Curncy'] }),
       securities.verifyRequest({ securities: [] })],
      [{ securities: ['SASW1 BGN Curncy'] }, {}]);

// --- verdictOf
check('verdictOf answers the word the service answered',
      'the verdict spelled here instead - a rule of the client\'s about what is live, which is '
      + 'the one thing a screen over evidence must never have',
      [securities.verdictOf(USED[0].evidence), securities.verdictOf(USED[1].evidence),
       securities.verdictOf(USED[2].evidence)],
      [{ text: 'verified', tone: 'done' }, { text: 'dead', tone: 'error' },
       { text: 'unmapped', tone: '' }]);
check('verdictOf tones an unmapped quote as itself',
      '`unmapped` toned as an error, which paints every knot a desk quotes by hand red and trains '
      + 'a reader to ignore the colour',
      securities.verdictOf({ verdict: 'unmapped' }).tone, '');

// --- usedRows
check('usedRows flattens one knot into the columns a reader compares',
      'the map\'s `verified` and the book\'s `timestamp` folded into one column, which reads as a '
      + 'quote printed the day it was verified - the two dates that must never be the same cell',
      securities.usedRows(USED, 'USD'),
      [{ curve: 'USD', tenor: '1Y', security: 'USOSFR1 BGN Curncy', use: 'Yes', quote: 5.1,
         print: '2024-06-27', verdict: 'verified', tone: 'done', name: 'USD OIS 1Y',
         verified: '2024-06-28' }]);
check('usedRows keeps the ladder\'s own order and narrows to one curve',
      'a sort, which moves the knots out of the order the curve was solved in; or the filter '
      + 'dropped, which files another curve\'s knots under this one\'s name',
      securities.usedRows(USED, 'ZAR').map((row) => [row.tenor, row.verdict]),
      [['3M', 'verified'], ['5Y', 'dead'], ['10Y', 'unmapped']]);
check('usedRows with no curve is the whole book',
      'the filter reading a blank curve as a name, which answers an empty table for the pane that '
      + 'shows everything',
      securities.usedRows(USED).length, 4);
check('usedRows leaves a rejection\'s columns empty rather than borrowing the ledger\'s date',
      '`last_update` falling into `verified`, which dates a dead ticker as verified in 2007',
      securities.usedRows(USED, 'ZAR')[1],
      { curve: 'ZAR', tenor: '5Y', security: 'SASW5 BGN Curncy', use: 'Yes', quote: 8.02,
        print: '2024-06-28', verdict: 'dead', tone: 'error',
        name: 'ZAR SWAP QTR (VS 3M) 5Y', verified: '' });

// --- mapRows
check('mapRows walks to the leaf however deep the family files one',
      'the walk stopped at a fixed depth, which answers nothing at all for a rates strip - the '
      + 'one block whose entries sit four names down',
      securities.mapRows(BLOCKS.rates, ['rates']).map(({ path, entry }) => [path, entry.security]),
      [['rates/ZAR/fixings/3M', 'JIBA3M Index'], ['rates/ZAR/strip/1Y', 'SASW1 BGN Curncy']]);
check('mapRows walks past the metadata a block files beside its entries',
      'the leaf test dropped (every dict a leaf), which lists the `quotes` container and the '
      + '`expiries` table as entries carrying no security at all',
      securities.mapRows(BLOCKS.fx_vol, ['fx_vol']).map(({ path }) => path),
      ['fx_vol/USDZAR/quotes/1M/ATM']);
check('mapRows on an empty map is an empty list',
      'a throw on the home that has never been verified, which is the one home the read verb '
      + 'answers an empty map for',
      [securities.mapRows({}), securities.mapRows(undefined)], [[], []]);

// --- mapPages
const SPOTS = { EURUSD: evidence('EURUSD BGN Curncy', 'EUR-USD X-RATE'),
                USDZAR: evidence('USDZAR BGN Curncy', 'USD-ZAR X-RATE') };
check('mapPages files a block of containers a page per container, and a block of entries whole',
      'every child a page, which files the spot block one page per security - a page per row; or '
      + 'every block whole, which lays every curve\'s strip on one page',
      securities.mapPages({ ...BLOCKS, fx_spot: SPOTS }).map(({ block, key }) => [block, key]),
      [['fx_spot', null], ['fx_vol', 'USDZAR'], ['rates', 'ZAR']]);

// --- rejectedRows
check('rejectedRows is the ledger in ticker order, carrying the terminal\'s own error',
      'the error dropped, which leaves `invalid` on screen with nothing saying what the terminal '
      + 'said; or the order left to the file, which moves a row every time the map is rewritten',
      securities.rejectedRows(REJECTED),
      [{ security: 'EURZARV1M BGN Curncy', verdict: 'invalid', name: '', last_update: '',
         error: 'Unknown/Invalid Security' },
       { security: 'SASW5 BGN Curncy', verdict: 'dead', name: 'ZAR SWAP QTR (VS 3M) 5Y',
         last_update: '2007-03-26', error: '' }]);

console.log(`\n${ran} checks over 9 functions, ${missed.length} missed`);
if (missed.length) {
  console.log(missed.map((name) => `  ${name}`).join('\n'));
  process.exit(1);
}
