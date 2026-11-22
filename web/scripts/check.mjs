// The check scripts' one harness: a module of `web/src` bundled with what it imports, a check
// naming the mutation it kills, the tally that fails a run on any miss, and the fixtures more than
// one script states.

import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const SRC = fileURLToPath(new URL('../src/', import.meta.url));

/** What `entry` names, bundled with what it imports, as its exports. */
async function bundled(entry) {
  const out = await build(
    { ...entry, bundle: true, format: 'esm', write: false, logLevel: 'silent' });
  return import(`data:text/javascript;base64,${
    Buffer.from(out.outputFiles[0].text).toString('base64')}`);
}

/** `web/src/<name>.ts`, bundled, as its exports. */
export const load = (name) => bundled({ entryPoints: [`${SRC}${name}.ts`] });

// no chart is drawn in a check, so the chart library a component reaches is left out of its bundle
const NO_CHARTS = { name: 'no-charts', setup(run) {
  run.onResolve({ filter: /^echarts/ }, ({ path }) => ({ path, namespace: 'none' }));
  run.onLoad({ filter: /.*/, namespace: 'none' }, () => (
    { contents: 'module.exports = new Proxy({}, { get: () => () => {} });' }));
} };

/** `web/src/components/<name>.tsx` as a function answering the markup one of its components
 * draws: the component's name, its props and its children. */
export async function drawer(name) {
  const { createElement, renderToStaticMarkup, ...parts } = await bundled({
    stdin: { contents: `export * from './components/${name}';\n`
      + "export { createElement } from 'react';\n"
      + "export { renderToStaticMarkup } from 'react-dom/server.browser';\n",
             resolveDir: SRC, loader: 'ts' },
    jsx: 'automatic', plugins: [NO_CHARTS],
  });
  return (part, props, ...children) =>
    renderToStaticMarkup(createElement(parts[part], props, ...children));
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
