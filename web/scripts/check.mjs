// The check scripts' one harness: a module of `web/src` bundled with what it imports, a check
// naming the mutation it kills, the tally that fails a run on any miss, and the fixtures more than
// one script states.

import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

/** `web/src/<name>.ts`, bundled, as its exports. */
export async function load(name) {
  const out = await build({
    entryPoints: [fileURLToPath(new URL(`../src/${name}.ts`, import.meta.url))],
    bundle: true, format: 'esm', write: false, logLevel: 'silent',
  });
  return import(`data:text/javascript;base64,${
    Buffer.from(out.outputFiles[0].text).toString('base64')}`);
}

let ran = 0;
const missed = [];

/** One check: `got` against `want` as JSON, said with the mutation it `kills`. */
export function check(name, kills, got, want) {
  ran += 1;
  const same = JSON.stringify(got) === JSON.stringify(want);
  if (!same) missed.push(name);
  console.log(`${same ? 'ok  ' : 'MISS'}  ${name}\n      kills: ${kills}`);
  if (!same) {
    console.log(`      got   ${JSON.stringify(got)}`);
    console.log(`      want  ${JSON.stringify(want)}`);
  }
}

/** The tally, the run failing where any check missed. */
export function done() {
  console.log(`\n${ran} checks, ${missed.length} missed`);
  if (missed.length) {
    console.log(missed.map((name) => `  ${name}`).join('\n'));
    process.exit(1);
  }
}

/** A node of a book's deal tree. */
export const node = (deal, children) => (
  { Instrument: { '.Deal': deal }, ...(children ? { Children: children } : {}) });

/** A credit support list as the wire carries it. */
export const support = (rows) => ({ '.CreditSupportList': rows });
