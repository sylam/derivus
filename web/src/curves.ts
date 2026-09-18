// The Curves screen's arithmetic - pure, free of React, the way `vols.ts` and `desk.ts` are for
// theirs. A curve here is a list of ROWS and the block of conventions they were stated in: nothing
// knows what a tenor means, only that the service reads one, and the fields a request may state
// are the schema's own declarations crossed with the spellings the service answers in - the same
// name in lower case, one for one.

import { curveOf, isObject } from './tokens';
import type { CurveRow, CurvesAnswer, Section } from './types';

/** A row nothing has been said about yet. */
const BLANK: CurveRow = { tenor: '', security: '', quote: null, use: 'Yes' };

/** The declared fields a request names outright, and what it calls them: `curve` is the block's
 * name and `rows` are the ladder, so these two are what a panel shows beside the conventions. */
const REQUEST_FIELDS: Record<string, 'currency' | 'discount_rate'> = {
  Currency: 'currency', Discount_Rate: 'discount_rate',
};

/** The form a request is composed in: the verb's own fields, and what the SOURCE stated, so an
 * edit can be told from a value that merely came along. */
export type CurveForm = {
  curve: string;
  currency: string;
  discount_rate: string;
  rows: CurveRow[];
  /** The conventions the seed or the block states, in the service's spelling. */
  stated: Record<string, unknown>;
  /** What the desk moved, keyed by the DECLARED spelling. */
  edits: Record<string, unknown>;
};

/** One row edit: a patch at an index. Past the end appends a blank row, `null` removes one, and
 * the list handed in is never touched. */
export function withRow(rows: CurveRow[], index: number, patch: Partial<CurveRow> | null) {
  if (patch === null) return rows.filter((_, i) => i !== index);
  if (index >= rows.length) return [...rows, { ...BLANK, ...patch }];
  return rows.map((row, i) => (i === index ? { ...row, ...patch } : row));
}

/** A seeded curve, or one the book carries, as the form that would author it: its own rows
 * completed with the fields a request takes, its conventions the baseline an edit is measured
 * against. */
export function prefill(curve: string, source: {
  currency?: string; discount_rate?: string;
  conventions?: Record<string, unknown>; rows?: Partial<CurveRow>[];
}): CurveForm {
  return {
    curve, currency: source.currency ?? '', discount_rate: source.discount_rate ?? '',
    rows: (source.rows ?? []).map((row) => ({ ...BLANK, ...row })),
    stated: source.conventions ?? {}, edits: {},
  };
}

/** One field of the form: the two a request names outright are its own, and everything else is a
 * convention, held until the request is built. */
export function edited(form: CurveForm, key: string, value: unknown): CurveForm {
  const own = REQUEST_FIELDS[key];
  if (own === undefined) return { ...form, edits: { ...form.edits, [key]: value } };
  return own === 'currency'
    ? { ...form, currency: String(value) } : { ...form, discount_rate: String(value) };
}

/** The request `POST /book/curve` takes: every row that names a tenor, and ONLY the conventions
 * the desk moved off what the source stated, the verb completing the rest from the seed. */
export function curveRequest(form: CurveForm): Record<string, unknown> {
  const moved = Object.entries(form.edits)
    .filter(([key, value]) => value !== form.stated[key.toLowerCase()])
    .map(([key, value]) => [key.toLowerCase(), value]);
  return {
    curve: form.curve.trim(), currency: form.currency.trim(),
    discount_rate: form.discount_rate.trim(),
    rows: form.rows.filter((row) => row.tenor.trim() !== '').map((row) => ({
      tenor: row.tenor.trim(), security: row.security.trim(), quote: row.quote, use: row.use,
    })),
    ...Object.fromEntries(moved),
  };
}

/** The fields a curve is stated in, in declaration order: the two the request names outright, and
 * every declared field whose lower-case spelling the service answers a conventions block in.
 * Nothing else in the family is one - a solver dial is not a convention, and the verb refuses a
 * name it reads nothing of. */
export function curveFields(declared: Section, answer: CurvesAnswer): string[] {
  const named = new Set<string>();
  for (const entry of [...Object.values(answer.curves), ...Object.values(answer.seeded ?? {})]) {
    Object.keys(entry.conventions ?? {}).forEach((key) => named.add(key));
  }
  return Object.keys(declared)
    .filter((key) => key in REQUEST_FIELDS || named.has(key.toLowerCase()));
}

/** Those fields against one block, for the panel that renders them. A field the block does not
 * state is left out, so the declaration's own default is what shows. */
export function curveValues(keys: string[], source: {
  currency?: string; discount_rate?: string; conventions?: Record<string, unknown>;
}): Record<string, unknown> {
  const stated: Record<string, unknown> = {
    currency: source.currency, discount_rate: source.discount_rate, ...source.conventions,
  };
  return Object.fromEntries(keys.filter((key) => stated[key.toLowerCase()] !== undefined)
    .map((key) => [key, stated[key.toLowerCase()]]));
}

/** The solved factor the market data store files under a curve's name - `[name, entry]`, or null.
 * The name is the store's own: the entry ending in this curve that CARRIES a curve, which is what
 * tells a solved rate factor from a spot filed under the same name. */
export function solvedFactor(factors: Record<string, unknown>, curve: string):
[string, Record<string, unknown>] | null {
  for (const [name, entry] of Object.entries(factors)) {
    if (name.endsWith('.' + curve) && isObject(entry) &&
        Object.values(entry).some((field) => curveOf(field))) return [name, entry];
  }
  return null;
}
