// Drives `src/files.ts` where the eye cannot: the rows the Settings card offers for the two maps
// this workstation reads a job's files through, and the body one edit posts. Every check names the
// MUTATION it kills - the change to the module that would still typecheck, still render, and still
// be wrong.
//
//   node web/scripts/files_check.mjs         # exits 1 on a miss
//
// The answer is `GET /paths` on a box holding one folder mapping and one file-name mapping, serving
// a job written on another system that names a market data file in a folder this box maps and a
// calendar in one it does not.

import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const bundled = await build({
  entryPoints: [fileURLToPath(new URL('../src/files.ts', import.meta.url))],
  bundle: true, format: 'esm', write: false, logLevel: 'silent',
});
const files = await import('data:text/javascript;base64,' +
  Buffer.from(bundled.outputFiles[0].text).toString('base64'));

// ---- the service's answer -----------------------------------------------------------------------

const MARKET = '//elsewhere/drop/MarketData';
const CALENDARS = '//elsewhere/drop/Calendars';

const PATHS = {
  path_transform: { [MARKET]: 'D:/desk/MarketData', '//elsewhere/archive': 'D:/desk/archive' },
  file_transform: { 'market_eod.json': 'market.json' },
  named: [
    { key: 'MarketDataFile', name: `${MARKET}/market_eod.json`, folder: MARKET,
      file: 'market_eod.json', path: 'D:/desk/MarketData/market.json', digest: 'ab'.repeat(32),
      unread: null },
    { key: 'CalendDataFile', name: `${CALENDARS}/calendars.json`, folder: CALENDARS,
      file: 'calendars.json', path: `${CALENDARS}/calendars.json`, digest: null,
      unread: `No such file or directory: '${CALENDARS}/calendars.json'` },
  ],
};

// what the answer said before any edit was made from it
const HELD = JSON.stringify(PATHS);

// the same job with both files in the folder this box does not map
const ONE_FOLDER = {
  ...PATHS,
  named: PATHS.named.map((named) => ({ ...named, folder: CALENDARS })),
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

const pairs = (rows) => rows.map((row) => [row.written, row.here]);

// --- mapRows
check('a folder the book names with no entry is offered as a row, its right side empty',
      'the named keys left out of the rows, which leaves a job\'s folder unmappable until its key '
      + 'is typed by hand somewhere this card has no input for',
      pairs(files.mapRows(PATHS, 'path_transform')),
      [[MARKET, 'D:/desk/MarketData'], ['//elsewhere/archive', 'D:/desk/archive'],
       [CALENDARS, '']]);

check('an entry is listed once where a named file carries its key',
      'the named keys appended without asking the map, which lists a mapped folder a second time, '
      + 'empty, and an edit of that row removes the mapping',
      pairs(files.mapRows(PATHS, 'path_transform')).filter(([key]) => key === MARKET).length, 1);

check('two files named in one unmapped folder offer it once',
      'the unmapped keys not made distinct, which offers the folder twice and edits it twice',
      pairs(files.mapRows(ONE_FOLDER, 'path_transform')),
      [[MARKET, 'D:/desk/MarketData'], ['//elsewhere/archive', 'D:/desk/archive'],
       [CALENDARS, '']]);

check('the file names offer each named file\'s name, the folders each named file\'s folder',
      'one key read for both maps, which offers a folder as a file name to map',
      pairs(files.mapRows(PATHS, 'file_transform')),
      [['market_eod.json', 'market.json'], ['calendars.json', '']]);

// --- edited
check('an edit sets its one entry, the other entries standing in their order',
      'the map rebuilt from the edited entry alone, which posts every other mapping away',
      files.edited(PATHS, 'path_transform', CALENDARS, 'D:/desk/Calendars').path_transform,
      { [MARKET]: 'D:/desk/MarketData', '//elsewhere/archive': 'D:/desk/archive',
        [CALENDARS]: 'D:/desk/Calendars' });

check('an empty value removes the key and leaves the other entries',
      'the empty value posted as an entry, which maps a folder to nothing; or the removal applied '
      + 'to the whole map',
      files.edited(PATHS, 'path_transform', MARKET, '').path_transform,
      { '//elsewhere/archive': 'D:/desk/archive' });

check('an edit never changes the other map',
      'the entry written into both maps, or the other map left out of the body, which the '
      + 'service replaces whole with nothing',
      [files.edited(PATHS, 'path_transform', CALENDARS, 'D:/desk/Calendars').file_transform,
       files.edited(PATHS, 'file_transform', 'calendars.json', '').path_transform],
      [PATHS.file_transform, PATHS.path_transform]);

check('the body is the two maps and nothing else',
      'the answer posted whole, `named` included, which hands the service rows it reports back to '
      + 'it as a request',
      Object.keys(files.edited(PATHS, 'file_transform', 'calendars.json', 'cal.json')).sort(),
      ['file_transform', 'path_transform']);

check('an edit leaves the answer it was made from as it was',
      'the map edited in place, which changes the paths on screen before the service has answered',
      JSON.stringify(PATHS), HELD);

console.log(`\n${ran} checks over 2 functions, ${missed.length} missed`);
if (missed.length) {
  console.log(missed.map((name) => `  ${name}`).join('\n'));
  process.exit(1);
}
