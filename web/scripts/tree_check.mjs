// Drives `src/tree.ts` where the eye cannot: how a screen's pages file under their folders, what a
// filter leaves standing, which folders stand open and which page a screen opens on. Every check
// names the MUTATION it kills - the change to the module that would still typecheck, still render,
// and still be wrong.
//
//   node web/scripts/tree_check.mjs          # exits 1 on a miss
//
// The pages are the Market Prices screen's on a book quoting three curves and a few hundred spot
// model blocks - the case the folders exist for - and the Portfolio screen's deal tree, whose ids
// are positions.

import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const bundled = await build({
  entryPoints: [fileURLToPath(new URL('../src/tree.ts', import.meta.url))],
  bundle: true, format: 'esm', write: false, logLevel: 'silent',
});
const tree = await import('data:text/javascript;base64,' +
  Buffer.from(bundled.outputFiles[0].text).toString('base64'));

// ---- the pages -----------------------------------------------------------------------------------

const page = (id) => {
  const [folder, ...rest] = id.split('.');
  return { id, folder, label: rest.join('.') };
};

const SPOT = Array.from({ length: 300 },
                        (_, i) => `LogVar2FJModelPrices.NAME${String(i).padStart(3, '0')}`);
const PRICES = ['InterestRatePrices.USD', 'InterestRatePrices.ZAR', 'InterestRatePrices.ZAR-ZARONIA',
                ...SPOT, 'FXVolPrices.USD.ZAR'].map(page);
const NODES = tree.grouped([...PRICES, { id: 'a new curve', label: 'a new curve', accent: true }]);

const DEALS = [
  { id: '0', type: 'NettingCollateralSet', label: 'CSA-1', children: [
    { id: '0/0', type: 'FXForwardDeal', label: 'FWD1' },
    { id: '0/1', type: 'FXOptionDeal', label: 'OPT1' },
  ] },
  { id: '1', type: 'SwapInterestDeal', label: 'SWP10' },
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

const shape = (nodes) => nodes.map((node) => [node.id, node.children?.length ?? null]);

// --- grouped
check('grouped files every page under its folder, a folder where its first member stood',
      'folders appended after the loose items, or sorted, which moves "a new curve" above the '
      + 'families and every folder the day a family is added',
      shape(NODES),
      [['group:InterestRatePrices', 3], ['group:LogVar2FJModelPrices', 300],
       ['group:FXVolPrices', 1], ['a new curve', null]]);

check('grouped keeps the order pages came in inside a folder, and takes the folder off the page',
      'members re-sorted, which files ZAR-ZARONIA above ZAR; or `folder` left on the node',
      NODES[0].children.map((node) => [node.id, node.label, 'folder' in node]),
      [['InterestRatePrices.USD', 'USD', false], ['InterestRatePrices.ZAR', 'ZAR', false],
       ['InterestRatePrices.ZAR-ZARONIA', 'ZAR-ZARONIA', false]]);

check('a folder and a page spelled the same are two rows',
      'the prefix dropped from a group id, which selects the folder when the page is picked',
      shape(tree.grouped([{ id: 'rates', label: 'rates' }, { id: 'rates/USD', folder: 'rates',
                                                              label: 'USD' }])),
      [['rates', null], ['group:rates', 1]]);

// --- leaves
check('a folder counts its members and a deal with children counts itself too, an action neither',
      'a folder counted as an item, which puts 301 on a family of 300; or an action counted, which '
      + 'puts 3 on a section configuring two families',
      [tree.leaves(NODES[1]), tree.leaves(DEALS[0]), tree.leaves(DEALS[1]),
       tree.leaves(tree.grouped([{ id: 'a', folder: 'f', label: 'a' },
                                 { id: 'f/', folder: 'f', label: 'another', accent: true }])[0])],
      [300, 3, 1, 1]);

// --- filtered
check('a filter on a currency finds its blocks across every family, and nothing else',
      'the folder name left out of what a page is matched on, or a match on the id, which '
      + 'every NAME### page carries',
      shape(tree.filtered(NODES, 'zar')),
      [['group:InterestRatePrices', 2], ['group:FXVolPrices', 1]]);

check('a filter on a family keeps the whole family',
      'members filtered against the folder\'s own match, which empties a family its name found',
      shape(tree.filtered(NODES, 'logvar2fj')), [['group:LogVar2FJModelPrices', 300]]);

check('a block\'s full name finds the block',
      'the folder not joined onto the name a page is matched on',
      tree.filtered(NODES, 'InterestRatePrices.ZAR-').flatMap((node) =>
        node.children.map((child) => child.id)),
      ['InterestRatePrices.ZAR-ZARONIA']);

check('a deal filters on its type and reference, never on its position',
      'the id matched, which keeps every deal under a position the text spells',
      [shape(tree.filtered(DEALS, 'opt')), shape(tree.filtered(DEALS, '0/'))], [[['0', 1]], []]);

check('no filter is the tree itself',
      'a filter of whitespace read as text, which empties the tree on a stray space',
      tree.filtered(NODES, '  ') === NODES, true);

// --- ancestors
check('the path to a page runs through its folder; a top-level page has none; a stranger null',
      'a page found at the top level answered as missing, which opens the first family instead',
      [tree.ancestors(NODES, 'InterestRatePrices.ZAR'), tree.ancestors(NODES, 'a new curve'),
       tree.ancestors(NODES, 'InterestRatePrices.EUR'), tree.ancestors(DEALS, '0/1')],
      [['group:InterestRatePrices'], [], null, ['0']]);

// --- isOpen
const [RATES, SPOTS] = NODES;
check('a small folder stands open and a big one closed until somebody says otherwise',
      'every folder open, which lays three hundred spot blocks over the three curves',
      [tree.isOpen(RATES, {}, [], false), tree.isOpen(SPOTS, {}, [], false),
       tree.isOpen(DEALS[0], {}, [], false)],
      [true, false, true]);

check('the page on screen opens the folder it sits in',
      'the path ignored, which leaves the picked block inside a closed family',
      tree.isOpen(SPOTS, {}, ['group:LogVar2FJModelPrices'], false), true);

check('the desk\'s own toggle wins over the page on screen, and a filter over both',
      'the path read first, which makes the folder the picked block sits in impossible to close; or '
      + 'the toggle read over a filter, which hides what the filter matched',
      [tree.isOpen(SPOTS, { 'group:LogVar2FJModelPrices': false }, ['group:LogVar2FJModelPrices'],
                   false),
       tree.isOpen(RATES, { 'group:InterestRatePrices': false }, [], true)],
      [false, true]);

// --- firstItem
check('a screen opens on its first page, inside the first folder',
      'the first NODE answered, which picks a folder and shows an empty panel',
      [tree.firstItem(NODES), tree.firstItem(tree.grouped([{ id: 'a new curve', label: 'x' }])),
       tree.firstItem([])],
      ['InterestRatePrices.USD', 'a new curve', null]);

console.log(`\n${ran} checks over 6 functions, ${missed.length} missed`);
if (missed.length) {
  console.log(missed.map((name) => `  ${name}`).join('\n'));
  process.exit(1);
}
