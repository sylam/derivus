// The arithmetic behind a vol surface's four views - pure, free of React, the way `desk.ts` is for
// the two data views. Nothing here knows what a moneyness is: a surface is a `[x, expiry, value]`
// grid whose AXIS NAMES come off the factor's own declaration, and the grid is ragged, because a
// bootstrapped smile refines its own x nodes per expiry.

import { formatNumber, tenorLabel } from './tokens';
import type { Section } from './types';

/** Whether a vol block quotes absolute normal vols: its declared `Distribution_Type` reads Normal,
 * stated or at its default. */
export const quotesNormal = (fields?: Section, values?: Record<string, unknown>): boolean =>
  String(values?.Distribution_Type ?? fields?.Distribution_Type?.value) === 'Normal';

/** A vol as a desk reads one: a lognormal vol in percent, a normal vol - an absolute vol of a rate
 * - in basis points. */
export const volText = (value: number, normal = false): string =>
  (normal ? formatNumber(value * 1e4, 'basis') : formatNumber(value * 100, 'percent'));

/** Positional stand-ins, where the declaration's prose does not name one per column: a curve's
 * first axis is its tenor in years. */
const POSITIONAL: Record<number, string[]> = {
  2: ['tenor', 'value'], 3: ['x', 'expiry', 'value'], 4: ['x', 'expiry', 'tenor', 'value'],
};

/** One axis name per COLUMN, read off the descriptor: a declaration names the tuple its rows
 * carry — `(moneyness, expiry, volatility) triples`, `(excel date, price) pairs` — so the order is
 * the engine's and not a rule spelled here. */
export function axisNames(description: string | undefined, arity: number): string[] {
  const found = /\(([^)]*)\)/.exec(description ?? '');
  const names = found ? found[1].split(',').map((name) => name.trim()) : [];
  return names.length === arity ? names : POSITIONAL[arity] ?? POSITIONAL[3];
}

const sortedSet = (values: number[]) => [...new Set(values)].sort((a, b) => a - b);

/** The distinct third coordinates of an arity-4 grid — what a slice select offers — and `[]` on an
 * arity-3 one. */
export function sliceKeys(rows: number[][]): number[] {
  return rows[0]?.length === 4 ? sortedSet(rows.map((row) => row[2])) : [];
}

/** One `[x, expiry, value]` grid: an arity-3 surface as itself, an arity-4 one sliced at `key`. */
export function triples(rows: number[][], key?: number): number[][] {
  if (rows[0]?.length !== 4) return rows;
  return rows.filter((row) => row[2] === key).map((row) => [row[0], row[1], row[3]]);
}

export type Smile = { expiry: number; label: string; points: [number, number][] };

export type Shaped = {
  expiries: number[];
  /** Every x the grid names, sorted — the UNION over the expiries, which carry their own nodes. */
  xs: number[];
  smiles: Smile[];
};

/** The grid as one smile per expiry, each on its own x nodes, expiries ascending. */
export function shape(rows: number[][]): Shaped {
  const byExpiry = new Map<number, [number, number][]>();
  for (const [x, expiry, value] of rows) {
    const smile = byExpiry.get(expiry) ?? [];
    smile.push([x, value]);
    byExpiry.set(expiry, smile);
  }
  const expiries = sortedSet([...byExpiry.keys()]);
  return {
    expiries,
    xs: sortedSet(rows.map((row) => row[0])),
    smiles: expiries.map((expiry) => ({
      expiry,
      label: tenorLabel(expiry),
      points: byExpiry.get(expiry)!.sort((a, b) => a[0] - b[0]),
    })),
  };
}

/** One smile read at `x` — linear between its own nodes, flat past its ends, which is how a
 * `Surface` declares itself read. */
export function at(points: [number, number][], x: number): number {
  if (x <= points[0][0]) return points[0][1];
  const last = points[points.length - 1];
  if (x >= last[0]) return last[1];
  const upper = points.findIndex(([node]) => node >= x);
  const [x0, v0] = points[upper - 1];
  const [x1, v1] = points[upper];
  return x1 === x0 ? v1 : v0 + (v1 - v0) * ((x - x0) / (x1 - x0));
}

export type Mesh = {
  xs: number[];
  expiries: number[];
  /** `z[expiry][x]`, no holes. */
  z: number[][];
  /** Cells that are not a quoted node — the ragged grid's own measure of itself. */
  filled: number;
};

/** The ragged grid as one rectangle: every expiry read at every x the surface names. */
export function mesh(shaped: Shaped): Mesh {
  let filled = 0;
  const z = shaped.smiles.map((smile) => {
    const nodes = new Set(smile.points.map(([x]) => x));
    return shaped.xs.map((x) => {
      if (!nodes.has(x)) filled += 1;
      return at(smile.points, x);
    });
  });
  return { xs: shaped.xs, expiries: shaped.expiries, z, filled };
}

/** Vol against expiry at one x — the mesh's column, as `[expiry, value]` pairs. */
export function termStructure(shaped: Shaped, x: number): [number, number][] {
  return shaped.smiles.map((smile) => [smile.expiry, at(smile.points, x)]);
}

/** The x a term structure opens at: the neutral coordinate the grid BRACKETS — 0 where the nodes
 * span it, 1 where they span that instead, else the middle node. */
export function neutralX(xs: number[]): number {
  const span = (target: number) => xs[0] <= target && target <= xs[xs.length - 1];
  const target = span(0) ? 0 : span(1) ? 1 : xs[Math.floor(xs.length / 2)];
  return xs.reduce((a, b) => (Math.abs(b - target) < Math.abs(a - target) ? b : a));
}
