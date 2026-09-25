// Drives `src/positions.ts` and the blotter's grouped reading where the eye cannot: how the
// record's positions file under their portfolio path and under their client, and how a grouped
// blotter keys the rows of two positions holding one node of the file. Every check names the
// MUTATION it kills - the change that would still typecheck, still render, and still be wrong.
//
//   node web/scripts/positions_check.mjs          # exits 1 on a miss
//
// The record: one set of terms held in two portfolios of one agreement, a legacy netting set's
// position under a counterparty nobody declared - its set named as the agreement is - a position
// the file lost, a client grouped under a parent legal never declared, an agreement nothing is
// booked under, and a parent chain that loops.

import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const bundled = await build({
  stdin: {
    contents: "export * from './positions'; export * as blotter from './blotter';"
      + " export * as tree from './tree';",
    resolveDir: fileURLToPath(new URL('../src', import.meta.url)), loader: 'ts',
  },
  bundle: true, format: 'esm', write: false, logLevel: 'silent',
});
const positions = await import('data:text/javascript;base64,' +
  Buffer.from(bundled.outputFiles[0].text).toString('base64'));
const { blotter, tree } = positions;

// ---- the record -----------------------------------------------------------------------------------

const hash = (letter) => letter.repeat(64);
const row = (instrument, agreement, portfolio, counterparty, quantity, paths, reference, object) =>
  ({ instrument: hash(instrument), agreement, portfolio, book: 'desk', counterparty, quantity,
     clips: 1, first_lsn: 5, last_lsn: 5, deal_paths: paths, reference, object });

const EM = row('a', 'ISDA-1', 'desk/Rates/EM', 'LEI-A', 1, ['1/0'], 'COLLAR1', 'StructuredDeal');
const TOP = row('a', 'ISDA-1', 'desk', 'LEI-A', 0.5, ['1/0'], 'COLLAR1', 'StructuredDeal');
const LEGACY = row('b', 'ISDA-1', 'desk/Rates', 'CPTY_B', -1, ['2/0'], 'FWD1', 'FXForwardDeal');
const LOST = row('c', 'ISDA-2', 'desk/FX', 'LEI-SUB', 2, [], null, null);
const POSITIONS = [EM, TOP, LEGACY, LOST];
const PAPER = {
  positions: POSITIONS,
  entities: [
    { entity: 'LEI-A', name: 'Client A', parent: 'LEI-GROUP' },
    { entity: 'LEI-SUB', name: 'Client Sub', parent: 'LEI-A' },
    { entity: 'LEI-IDLE', name: 'Idle Client', parent: null },
    { entity: 'LEI-X', name: 'Loop X', parent: 'LEI-Y' },
    { entity: 'LEI-Y', name: 'Loop Y', parent: 'LEI-X' }],
  agreements: [
    { agreement: 'ISDA-1', entity: 'LEI-A', kind: 'ISDA 2002' },
    { agreement: 'ISDA-3', entity: 'LEI-A', kind: 'ISDA 2002 with CSA' },
    { agreement: 'ISDA-2', entity: 'LEI-SUB', kind: 'ISDA 2002' }],
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

/** A tree as nested `[label, hint | children]` - what a reader sees, in the order it is shown. */
const seen = (nodes) => nodes.map((node) => (node.group
  ? [node.label, seen(node.children)] : [node.label, node.hint]));

const ids = (nodes) => nodes.flatMap((node) => [node.id, ...ids(node.children ?? [])]);

// --- the portfolio tree
const PORTFOLIOS = positions.portfolioTree(POSITIONS);
check('a position files under a folder per segment of its path, folders ahead of positions',
      'a folder per whole path, which puts desk/Rates beside desk rather than in it; or the '
      + 'positions filed in a folder listed ahead of the folders under it',
      seen(PORTFOLIOS),
      [['desk', [
        ['FX', [['cccccccccccc…', '×2 · not in the file']]],
        ['Rates', [['EM', [['COLLAR1', '×1 · ISDA-1']]], ['FWD1', '×-1 · ISDA-1']]],
        ['COLLAR1', '×0.5 · ISDA-1']]]]);

check('the same terms in two portfolios are two leaves with two ids',
      'a leaf keyed by its deal path, which gives both one id and selects them as one',
      [new Set(ids(PORTFOLIOS)).size === ids(PORTFOLIOS).length,
       positions.positionId(EM) !== positions.positionId(TOP)],
      [true, true]);

// --- the client tree
const CLIENTS = positions.clientTree(PAPER);
check('a client holds its agreements and then the entities declared under it, every declared one '
      + 'standing, a parent nobody declared a folder under its id',
      'the parent ignored, which puts Client A at the top; an agreement nothing is booked under '
      + 'dropped; the entities under a client listed ahead of its own paper; an undeclared parent '
      + 'dropped with everything under it; or an agreement folder keyed by its id alone, which files '
      + "the legacy set's position under the client the paper names",
      seen(CLIENTS),
      [['CPTY_B', [['ISDA-1', [['FWD1', '×-1 · desk/Rates']]]]],
       ['Idle Client', []],
       ['LEI-GROUP', [['Client A', [
         ['ISDA-1', [['COLLAR1', '×0.5 · desk'], ['COLLAR1', '×1 · desk/Rates/EM']]],
         ['ISDA-3', []],
         ['Client Sub', [['ISDA-2', [['cccccccccccc…', '×2 · not in the file']]]]]]]]],
       ['Loop X', []], ['Loop Y', []]]);

check('a client tree counts its positions and nothing else',
      'an agreement or a client counted as an item, which puts more positions on a desk than it holds',
      CLIENTS.reduce((sum, node) => sum + tree.leaves(node), 0), 4);

// --- the file's nodes
check('a node of the file holds every position whose instrument sits there; a lost one none',
      "a node's positions overwritten rather than gathered, which drops the second portfolio",
      [...positions.byPath(POSITIONS)].map(([path, held]) => [path, held.map((p) => p.portfolio)]),
      [['1/0', ['desk/Rates/EM', 'desk']], ['2/0', ['desk/Rates']]]);

check('a client is read by the name legal declared, else by the id the record holds',
      'the id shown where a name is declared, or a blank where none is',
      [positions.clientName(PAPER.entities, 'LEI-A'), positions.clientName(PAPER.entities, 'CPTY_B')],
      ['Client A', 'CPTY_B']);

// --- the grouped blotter
const deal = (object, reference, extra = {}) => ({ Instrument: { '.Deal': {
  Object: object, Reference: reference, ...extra } } });
const BOOK = blotter.toRows([
  deal('FixedCashflowDeal', 'CF1', { Payment_Date: { '.Timestamp': '2026-09-01' } }),
  { ...deal('NettingCollateralSet', 'ISDA-1'), Children: [
    { ...deal('StructuredDeal', 'COLLAR1'), Children: [
      deal('FXOptionDeal', 'PUT', { Expiry_Date: { '.Timestamp': '2026-05-01' } }),
      deal('FXOptionDeal', 'CALL', { Expiry_Date: { '.Timestamp': '2026-06-01' } })] }] },
  { ...deal('NettingCollateralSet', 'CSA-OLD'), Children: [
    deal('FXForwardDeal', 'FWD1', { Settlement_Date: { '.Timestamp': '2026-03-01' } })] }],
['NettingCollateralSet', 'StructuredDeal'], '2026-01-01');
const GROUPED = blotter.groupedRows(PORTFOLIOS, blotter.rowsByPath(BOOK), positions.byId(POSITIONS),
                                    '2026-01-01');
const flat = blotter.flatten(GROUPED, new Set());

check('two positions holding one node are keyed apart, legs included',
      'the file\'s rows reused as they are, which puts COLLAR1 and its legs in the table twice '
      + 'under one key',
      new Set(flat.map((r) => r.path)).size === flat.length, true);

check('every row of a position picks the position, a folder picks nothing',
      'a leg left picking its own deal path, which opens a leg the grouping never showed; or a '
      + 'folder picking its own id',
      flat.filter((r) => r.reference === 'PUT' || r.reference === 'COLLAR1' || r.object === '')
        .map((r) => [r.reference, r.target === null ? null : r.target === positions.positionId(EM)
          ? 'EM' : r.target === positions.positionId(TOP) ? 'TOP' : r.target]),
      [['desk', null], ['FX', null], ['Rates', null], ['EM', null], ['COLLAR1', 'EM'],
       ['PUT', 'EM'], ['COLLAR1', 'TOP'], ['PUT', 'TOP']]);

check('a folder rolls on the earliest thing under it, and a lost position says so',
      'a folder\'s roll left blank, which sorts the whole portfolio last; or a lost position dropped',
      [GROUPED[0].roll, GROUPED[0].days, flat.find((r) => r.target === positions.positionId(LOST))
        ?.object],
      ['2026-03-01', 59, 'not in the file']);

// --- the roll-off window
const held = (rows) => rows.map((r) => (r.children.length ? [r.reference, held(r.children)]
  : r.reference));
check('a row holding others survives a window with only what rolls under it',
      'a folder or a netting set surviving on the roll it inherits, which keeps every trade under '
      + 'it in a window only one of them rolls in',
      [held(blotter.filterRows(GROUPED, 60)), held(blotter.filterRows(BOOK, 60))],
      [[['desk', [['Rates', ['FWD1']]]]], [['CSA-OLD', ['FWD1']]]]);

check('the roll-off count counts trades, a structure once and what holds it never',
      'every row a window keeps counted, which puts the folders and netting sets above one '
      + "rolling trade into the count; or a structure's legs counted as trades, which counts a "
      + 'collar twice',
      [blotter.rollingCount(GROUPED, 60), blotter.rollingCount(BOOK, 60),
       blotter.rollingCount(GROUPED, 200), blotter.rollingCount(BOOK, 200)],
      [1, 1, 3, 2]);

console.log(`\n${ran} checks over 9 functions, ${missed.length} missed`);
if (missed.length) {
  console.log(missed.map((name) => `  ${name}`).join('\n'));
  process.exit(1);
}
