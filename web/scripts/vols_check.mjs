// Drives `src/vols.ts` where the eye cannot: the expiries a surface carries, their tenor labels,
// how many nodes each smile stands on, and the column a term structure reads.
//
//   node web/scripts/vols_check.mjs [surface.json]
//
// The file is `{description, data}` - a `.Curve` payload's own two halves, as `GET /book` serves
// them. With none, only the synthetic arity-4 space below runs.

import { readFileSync } from 'node:fs';
import { transformSync } from 'esbuild';

const ts = readFileSync(new URL('../src/vols.ts', import.meta.url), 'utf8');
const js = transformSync(ts, { loader: 'ts', format: 'esm' }).code;
const vols = await import(`data:text/javascript;base64,${Buffer.from(js).toString('base64')}`);

function read(title, description, data) {
  const arity = data[0].length;
  const axes = vols.axisNames(description, arity);
  const keys = vols.sliceKeys(data);
  console.log(`\n== ${title}: ${data.length} rows, arity ${arity}, axes ${axes.join(' / ')}`);
  if (keys.length) console.log(`   ${axes[2]} slices: ${keys.join(', ')}`);
  for (const key of keys.length ? keys : [undefined]) {
    const shaped = vols.shape(vols.triples(data, key));
    const grid = vols.mesh(shaped);
    const x = vols.neutralX(shaped.xs);
    if (key !== undefined) console.log(`   -- ${axes[2]} = ${key}`);
    console.log(`   expiries ${shaped.expiries.length}: ` + shaped.smiles
      .map((s) => `${s.label}(${s.expiry.toFixed(4)}, ${s.points.length} nodes)`).join(' '));
    console.log(`   x grid ${shaped.xs.length}: ${shaped.xs[0]} .. ${shaped.xs[shaped.xs.length - 1]}`);
    console.log(`   mesh ${grid.expiries.length}x${grid.xs.length}, ${grid.filled} of ` +
      `${grid.expiries.length * grid.xs.length} cells filled between nodes`);
    console.log(`   term structure at ${axes[0]} ${x}: ` + vols.termStructure(shaped, x)
      .map(([e, v]) => `${vols.tenorLabel(e)} ${(v * 100).toFixed(3)}%`).join('  '));
  }
}

// a two-dimensional space: two slices, three expiries, four x nodes, every cell quoted
const SPACE = [];
for (const tenor of [1, 5]) {
  for (const expiry of [7 / 365, 0.5, 2]) {
    for (const x of [-0.2, 0, 0.2, 0.4]) {
      SPACE.push([x, expiry, tenor, 0.2 + 0.01 * tenor + 0.3 * x * x - 0.02 * expiry]);
    }
  }
}

const file = process.argv[2];
if (file) {
  const surface = JSON.parse(readFileSync(file, 'utf8'));
  read(file, surface.description, surface.data);
}
read('synthetic Space', '(moneyness, expiry, tenor, volatility) quads', SPACE);
