// Drives what the first paint downloads, where the eye cannot: the app bundled split as a build
// splits it, and the packages its entry loads before anything is drawn - the chart library left to
// the first chart drawn. Every check names the MUTATION it kills.
//
//   node web/scripts/paint_check.mjs        # exits 1 on a miss

import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';
import { check, done } from './check.mjs';

const { metafile: { outputs } } = await build({
  entryPoints: [fileURLToPath(new URL('../src/main.tsx', import.meta.url))],
  bundle: true, splitting: true, format: 'esm', write: false, metafile: true, logLevel: 'silent',
  packages: 'external', outdir: fileURLToPath(new URL('../dist', import.meta.url)),
});
const entry = Object.keys(outputs).find((name) => outputs[name].entryPoint?.endsWith('main.tsx'));

/** The chunks `start` loads with it, and the packages they import. */
function loads(start) {
  const chunks = new Set();
  const packages = [];
  const walk = (name) => {
    if (chunks.has(name)) return;
    chunks.add(name);
    for (const row of outputs[name].imports.filter((each) => each.kind === 'import-statement')) {
      if (row.external) packages.push(row.path);
      else walk(row.path);
    }
  };
  walk(start);
  return { chunks, packages };
}
/** The chunks holding the module `src/charts/<name>.ts`. */
const holding = (name) => Object.keys(outputs).filter((chunk) => Object.keys(outputs[chunk].inputs)
  .some((input) => input.split(/[\\/]/).slice(-3).join('/') === `src/charts/${name}.ts`));
const fetched = (row) => row.kind === 'dynamic-import'
  && holding('echarts').some((chunk) => loads(row.path).chunks.has(chunk));

check('the first paint downloads no chart library, and the chart hook fetches it when it draws',
      'a static import of echarts anywhere the entry reaches - the hook, a chart or its palette - '
      + "which puts 590 kB on every first paint; or the hook's fetch dropped, which draws nothing",
      [loads(entry).packages.some((name) => /^(echarts|zrender)/.test(name)),
       holding('useChart').some((name) => outputs[name].imports.some(fetched))],
      [false, true]);

done();
