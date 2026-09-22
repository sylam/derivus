// Drives `src/spine.ts` where the eye cannot: the page a poll merges onto the strip it holds, the
// verdict the banner reads off the reconcile lists, what a fold at the head is asked for, the two
// transitions the store holds it by, and the shape the markets panel renders. Every check names
// the MUTATION it kills - the change to the module that would still typecheck, still render, and
// still be wrong.
//
//   node web/scripts/spine_check.mjs         # exits 1 on a miss
//
// The shapes are the service's own answers on the design's synthetic home - `GET /book/activity`,
// `GET /book/markets` and `GET /book/reconcile` - with the rows trimmed to the ones a check turns
// on. The module is bundled rather than transformed, because it reads `src/desk.ts` rather than
// spelling a plural twice.

import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const bundled = await build({
  entryPoints: [fileURLToPath(new URL('../src/spine.ts', import.meta.url))],
  bundle: true, format: 'esm', write: false, logLevel: 'silent',
});
const spine = await import('data:text/javascript;base64,' +
  Buffer.from(bundled.outputFiles[0].text).toString('base64'));

// ---- the service's shapes ------------------------------------------------------------------------

const ACTOR = 'subject-desk-one';

function row(lsn, event_type, summary, book = 'FX-VANILLA') {
  return { lsn, record_time: `2026-08-26T08:0${lsn % 10}:00.000000Z`, effective_time: null,
           actor: ACTOR, event_type, book, summary };
}

// the head of the synthetic home's strip: a fill, the close that restated the day, and the two
// firm-level facts under it
const HELD = [row(18, 'official_close_declared', 'the official close was declared', null),
              row(19, 'rehash_declared', 'a hash algorithm was declared', null),
              row(20, 'snapshot_registered', 'a snapshot was registered')];
const PAGE = { lsn: 22, rows: [row(21, 'break_glass_used', 'the recovery seat was used', null),
                               row(22, 'fill', 'a clip was booked')] };

// one row per market, which is what the fold answers, each list in the order the service sorts
// it: `official` restated by a second close, a second market closed once, two declared names and
// two snapshots - so a list left in the fold's own order is visible
const MARKETS = {
  lsn: 21,
  names: [{ name: 'eur-close', values_hash: '7e1686c9e58a2753', actor: ACTOR,
            effective_time: '2026-08-26T07:00:00.000000Z', lsn: 12 },
          { name: 'official', values_hash: '7e1686c9e58a2753', actor: ACTOR,
            effective_time: '2026-08-26T08:00:00.000000Z', lsn: 13 }],
  closes: [{ market: 'eur-close', values_hash: 'b471376808000000', supersedes_lsn: null,
             effective_time: '2026-08-26T14:00:00.000000Z', lsn: 14 },
           { market: 'official', values_hash: 'b4713768088f9a06', supersedes_lsn: 15,
             effective_time: '2026-08-26T16:30:00.000000Z', lsn: 17 }],
  snapshots: [{ blob: '70305a91ff4aa328', book: 'FX-VANILLA', lsn: 19 },
              { blob: '70305a91ff000000', book: 'FX-VANILLA', lsn: 20 }],
};

const ETAG = 'e1', MOVED = 'e2';
const PINNED = { lsn: 20, head: 'e3b0c44298fc1c14', events_behind: 0, positions_behind: 0 };
const BEHIND = { ...PINNED, events_behind: 2, positions_behind: 2 };
// a fixings policy past the pin: the record moved and no row can follow it
const POLICY = { ...PINNED, events_behind: 1, positions_behind: 0 };

const CLEAN = { lsn: 20, events_behind: 0, positions_behind: 0,
                in_record_not_in_file: [], in_file_not_in_record: [], quantity_mismatch: [] };
const RECORD_ONLY = [{ instrument: 'a1', netting_set: 'CLIENT_A', quantity: 250000.0,
                       last_lsn: 21 },
                     { instrument: 'a2', netting_set: 'CLIENT_A', quantity: 500000.0,
                       last_lsn: 22 }];
const AHEAD = { ...CLEAN, events_behind: 2, positions_behind: 2,
                in_record_not_in_file: RECORD_ONLY };
const DRIFTED = { ...AHEAD, in_file_not_in_record: [
  { instrument: 'b1', deal_path: '0/1', reference: 'BY_HAND' }] };
// the desk's own delete verb: the record holds a live position the file has lost, and the pin has
// seen every event there is
const DELETED = { ...CLEAN, in_record_not_in_file: [RECORD_ONLY[0]] };

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

const lsns = (rows) => rows.map((entry) => entry.lsn);

// --- mergeActivity
check('a page lands after the rows the strip already holds',
      'the page prepended, or the sort dropped, which reads the newest event as the oldest and '
      + 'puts a booking under the close that preceded it',
      lsns(spine.mergeActivity(HELD, PAGE)), [18, 19, 20, 21, 22]);
check('a row the client already holds is not held twice',
      'the merge concatenating, which renders one event once per poll for as long as a cursor '
      + 'answered off a fold that moved keeps repeating it',
      lsns(spine.mergeActivity(HELD, { lsn: 20, rows: HELD.slice(1) })), [18, 19, 20]);
check('a repeated row is the page\'s own, not the one held',
      'the map written before the held rows, which keeps a stale copy of a row the service '
      + 'answered again',
      spine.mergeActivity(HELD, { lsn: 20, rows: [{ ...HELD[2], actor: 'subject-desk-two' }] })[2]
        .actor, 'subject-desk-two');
check('the cap keeps the NEWEST rows',
      'the cap taken from the start (`slice(0, cap)`), which pins a strip to the genesis events '
      + 'and never shows a desk what just happened',
      lsns(spine.mergeActivity(HELD, PAGE, 2)), [21, 22]);
check('a page carrying nothing leaves the strip itself standing',
      'a new array built anyway, which is the same rows and a different object - and most beats '
      + 'of a poll bring no event, so the desk would repaint every two seconds for nothing',
      [spine.mergeActivity(HELD, null) === HELD,
       spine.mergeActivity(HELD, { lsn: 20, rows: [] }) === HELD], [true, true]);
check('the pin reads as one comparable value, and no record as none',
      'the counts left out of it, after which a record moving past a standing pin is the same '
      + 'state and the banner never appears',
      [spine.pinnedAt(null), spine.pinnedAt(PINNED) === spine.pinnedAt(PINNED),
       spine.pinnedAt(BEHIND) === spine.pinnedAt(PINNED)], [null, true, false]);
check('the merge leaves the rows it was handed standing',
      'an in-place sort or push, which reorders the array the strip is rendering from',
      lsns(HELD), [18, 19, 20]);

// --- reconcileVerdict: THE LISTS DECIDE
const none = spine.reconcileVerdict(null, null);
check('no record is a state of the screen and not an empty reconcile',
      'a null pin read as clean, which tells a desk that records nothing that its file agrees '
      + 'with a record it does not keep',
      [none.state, none.line], ['none', '']);
check('a file the three lists say nothing about is clean, whatever the counts say',
      'the verdict read off the counts, after which a fixings policy past the pin lights an amber '
      + 'banner on every screen of a desk whose file and record agree exactly',
      [spine.reconcileVerdict(PINNED, CLEAN).state, spine.reconcileVerdict(POLICY, CLEAN).state,
       spine.reconcileVerdict(PINNED, null).state,
       spine.reconcileVerdict(POLICY, CLEAN).line], ['clean', 'clean', 'clean', '']);
const behind = spine.reconcileVerdict(BEHIND, AHEAD);
check('rows the record holds and the pin accounts for are a LAG, and the banner says how far',
      '`positions_behind` read off `events_behind`, after which a policy or a fixing past the pin '
      + 'reads as two trades the file has lost - the two counts exist precisely so it does not',
      [behind.state, behind.line, behind.events, behind.positions, behind.recordOnly],
      ['behind', 'record ahead by 2 events (2 positions)', 2, 2, 2]);
check('one event past the pin reads as one event',
      'the plural spelled into the sentence, which says `1 events` at the one moment a desk is '
      + 'being told something it has to act on',
      spine.reconcileVerdict({ ...BEHIND, events_behind: 1, positions_behind: 1 },
                             { ...CLEAN, in_record_not_in_file: [RECORD_ONLY[0]] }).line,
      'record ahead by 1 event (1 position)');
const drifted = spine.reconcileVerdict(BEHIND, DRIFTED);
check('a deal the file holds that nobody booked is DRIFT, not a lag',
      'the drift read off `in_record_not_in_file` alone, which calls every record that is one '
      + 'trade ahead of its file a drift and leaves a desk no word for the edit it cannot undo',
      [drifted.state, drifted.line],
      ['drifted', 'file and record disagree: 2 in record not in file, 1 in file not in record, '
        + '0 quantity mismatches']);
check('a clip count the two disagree on is drift with nothing else wrong',
      'the mismatch list left out of the test, which reconciles a deal booked twice in the file '
      + 'against one clip in the record as merely behind',
      spine.reconcileVerdict(BEHIND, { ...AHEAD, quantity_mismatch: [
        { instrument: 'a1', record_clips: 1, file_nodes: 2, record_quantity: 250000.0 }] }).state,
      'drifted');
check('a trade the file LOST under a pin that has seen everything is drift',
      'the `positions === 0` arm dropped, after which a deal deleted through the desk\'s own verb '
      + '- which records nothing and re-pins at the head - is invisible on every screen, forever',
      [spine.reconcileVerdict(PINNED, DELETED).state,
       spine.reconcileVerdict(PINNED, DELETED).line],
      ['drifted', 'file and record disagree: 1 in record not in file, 0 in file not in record, '
        + '0 quantity mismatches']);
check('a lost trade beside a lagging fill is still drift',
      'the drift arm asking for zero positions behind, so one late fill beside a deletion reads '
      + 'the deleted trade into the lag\'s own line',
      spine.reconcileVerdict({ ...PINNED, events_behind: 1, positions_behind: 1 },
                             { ...CLEAN, in_record_not_in_file: RECORD_ONLY }).state,
      'drifted');
check('a pin nothing has written reads clean rather than as a null lag',
      'the null counts read as anything but zero, which puts a banner naming `null events` over '
      + 'the one book that has nothing to compare',
      spine.reconcileVerdict(
        { lsn: null, head: null, events_behind: null, positions_behind: null }, null).state,
      'clean');

// --- wantsReconcile: what a fold at the head is asked for
check('the fold is asked for where the record holds positions the pin has not seen',
      '`events_behind` read instead, after which every fixing, checkpoint and attestation a '
      + 'recorded desk takes pays for a full head fold and repaints the screen',
      [spine.wantsReconcile(PINNED, ETAG, ETAG), spine.wantsReconcile(POLICY, ETAG, ETAG),
       spine.wantsReconcile(BEHIND, ETAG, ETAG)], [false, false, true]);
check('and where the FILE has moved under the answer in hand',
      'the etag test dropped, after which a hand edit or a delete - which move the file and no '
      + 'count at all - is never looked for; or the answer never marked, which re-folds per beat',
      [spine.wantsReconcile(PINNED, null, ETAG), spine.wantsReconcile(PINNED, ETAG, MOVED),
       spine.wantsReconcile(PINNED, ETAG, ETAG)], [true, true, false]);
check('a desk that records nothing is never asked',
      'the null test dropped, which 404s once per beat on the owner\'s own book',
      [spine.wantsReconcile(null, null, ETAG), spine.wantsReconcile(null, ETAG, MOVED)],
      [false, false]);

// --- the store's two transitions, which are this module's and answer the SAME OBJECT
const held = { spine: PINNED, rows: HELD, reconcile: AHEAD, reconciledAt: ETAG };
check('a beat that moved neither the rows nor the pin answers the record it was handed',
      'the guard made vacuous, or dropped for a fresh object every time, which repaints the desk '
      + '- a vol surface included - every two seconds for as long as a tab is open',
      [spine.readRecord(held, { ...PINNED }, null) === held,
       spine.readRecord(held, { ...PINNED }, { lsn: 20, rows: [] }) === held,
       spine.readRecord(held, BEHIND, null) === held,
       spine.readRecord(held, PINNED, PAGE) === held],
      [true, true, false, false]);
check('and what it answers is the pin and the rows merged, the rest left standing',
      'the merge dropped or the pin not carried, which paints a strip that never moves',
      [spine.readRecord(held, BEHIND, PAGE).spine.events_behind,
       lsns(spine.readRecord(held, BEHIND, PAGE).rows),
       spine.readRecord(held, BEHIND, PAGE).reconciledAt],
      [2, [18, 19, 20, 21, 22], ETAG]);
check('a reconcile answer already in hand answers the record it was handed',
      'identity alone, which a fetched answer never has - every fold is a new object and the whole '
      + 'app repaints; or the etag dropped, which holds an answer about a file that has moved',
      [spine.reconciledRecord(held, JSON.parse(JSON.stringify(AHEAD)), ETAG) === held,
       spine.reconciledRecord(held, AHEAD, MOVED) === held,
       spine.reconciledRecord(held, DRIFTED, ETAG) === held,
       spine.reconciledRecord(held, DELETED, ETAG).reconcile === DELETED],
      [true, false, false, true]);

// --- marketsView
const view = spine.marketsView(MARKETS);
check('the closes read newest first',
      'the sort reversed or dropped, which puts the close a desk is looking for under the one it '
      + 'restated - and the service answers them by market, not by when',
      view.closes.map((close) => [close.lsn, close.supersedes]),
      [[17, 'supersedes LSN 15'], [14, 'the first close on this market']]);
check('a close that stands over nothing says so rather than naming nothing',
      'the supersedes line built unconditionally, which prints `supersedes LSN null` on the first '
      + 'close of every market this desk has ever declared',
      view.closes[1].supersedes, 'the first close on this market');
check('the names and the snapshots are shaped the same way',
      'either list left as the answer gave it (the fold sorts names by NAME and snapshots by when '
      + 'they were filed), which is the one ordering rule this panel has',
      [view.names.map((name) => name.lsn), view.snapshots.map((shot) => shot.lsn)],
      [[13, 12], [20, 19]]);
check('the shaping leaves the answer it was handed standing',
      'an in-place `sort`, which reorders the answer under a second reader of it',
      [MARKETS.closes.map((close) => close.lsn), MARKETS.names.map((name) => name.lsn),
       MARKETS.snapshots.map((shot) => shot.lsn)], [[14, 17], [12, 13], [19, 20]]);

console.log(`\n${ran} checks over 8 functions, ${missed.length} missed`);
if (missed.length) {
  console.log(missed.map((name) => `  ${name}`).join('\n'));
  process.exit(1);
}
