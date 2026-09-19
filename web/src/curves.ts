// The Curves screen's arithmetic - pure, free of React, the way `vols.ts` and `desk.ts` are for
// theirs. A curve here is a list of ROWS and the conventions they were stated in: nothing knows
// what a tenor means, only that the service reads one, and the fields a request may state are the
// schema's own declarations crossed with the spellings the service answers in - the same name in
// lower case, one for one.

import { curveOf, isObject } from './tokens';
import type { CurveRow, CurvesAnswer, Section } from './types';

/** A row nothing has been said about yet. */
const BLANK: CurveRow = { tenor: '', security: '', quote: null, use: 'Yes' };

/** The fields a request names outright, and what it calls them: `rows` are the ladder and
 * everything else is a convention, so these are what a panel shows beside them. `Interpolation`
 * is the curve's own scheme - a rule in a section rather than a column of the block, which is why
 * the family declares no field of that name. */
const REQUEST_FIELDS: Record<string, 'curve' | 'currency' | 'discount_rate' | 'interpolation'> = {
  Curve: 'curve', Currency: 'currency', Discount_Rate: 'discount_rate',
  Interpolation: 'interpolation',
};

/** How long a burst of edits settles for before the card posts. The verb re-solves the whole
 * market, so a desk typing through a ladder pays for one solve rather than one per field. */
export const QUIET_MS = 500;

/** The curve a card stands in: the fields the request names outright, and the conventions under
 * the DECLARED spelling the panel names them by. */
export type CurveForm = {
  curve: string;
  currency: string;
  discount_rate: string;
  /** The curve's OWN rule, blank where it takes the scheme every factor of its type takes. */
  interpolation: string;
  rows: CurveRow[];
  conventions: Record<string, unknown>;
};

/** One commit: a row's cell, a row added past the end or removed by index, or a field the panel
 * names. */
export type CurveEdit =
  | { row: number; patch: Record<string, unknown> | null }
  | { field: string; value: unknown };

/** What a card has in hand: the curve as the desk has edited it, when the last commit landed, the
 * clock of the last one POSTED, and whether that post is still in flight. */
export type Editing = { form: CurveForm; at: number; posted: number; saving: boolean };

/** A seeded curve, or one the book carries, as the form that would author it: the panel's own
 * fields filled in from what it states, its rows completed with the ones a request takes. */
export function prefill(curve: string, source: {
  currency?: string; discount_rate?: string; interpolation?: string; interpolation_source?: string;
  conventions?: Record<string, unknown>; rows?: Partial<CurveRow>[];
}, fields: string[] = []): CurveForm {
  const stated = source.conventions ?? {};
  return {
    curve, currency: source.currency ?? '', discount_rate: source.discount_rate ?? '',
    // only a scheme this curve's own RULE states comes back: one it merely resolved to is the
    // type's, and re-stating it would write a rule where the book carried none
    interpolation: source.interpolation_source === 'curve' ? source.interpolation ?? '' : '',
    rows: (source.rows ?? []).map((row) => ({ ...BLANK, ...row })),
    // a field the source does not state is left out, so the declaration's own default is what
    // shows and nothing the desk never touched travels as a convention it stated
    conventions: Object.fromEntries(fields
      .filter((key) => !(key in REQUEST_FIELDS) && stated[key.toLowerCase()] !== undefined)
      .map((key) => [key, stated[key.toLowerCase()]])),
  };
}

/** One commit folded into the card, the form handed in left standing. A cleared QUOTE is not a
 * quote of zero and travels as the null that says no number was stated - which is what sends the
 * row to the terminal, or refuses it. A cleared `Near_Tenor` clears the near scheme with it: a
 * near scheme is a scheme AND where it stops, and the emitter refuses half of one. */
export function commit(form: CurveForm, edit: CurveEdit): CurveForm {
  if ('field' in edit) {
    const own = REQUEST_FIELDS[edit.field];
    if (own !== undefined) return { ...form, [own]: String(edit.value ?? '') };
    const paired = edit.field === 'Near_Tenor' && !edit.value ? { Near_Interpolation: '' } : {};
    return { ...form, conventions: { ...form.conventions, [edit.field]: edit.value, ...paired } };
  }
  if (edit.patch === null) return { ...form, rows: form.rows.filter((_, i) => i !== edit.row) };
  const cell = (edit.patch.quote === ''
    ? { ...edit.patch, quote: null } : edit.patch) as Partial<CurveRow>;
  if (edit.row >= form.rows.length) return { ...form, rows: [...form.rows, { ...BLANK, ...cell }] };
  return { ...form, rows: form.rows.map((row, i) => (i === edit.row ? { ...row, ...cell } : row)) };
}

/** Whether a card is due to post: something committed since the last post, the burst quiet for
 * `QUIET_MS`, no post in flight - one curve solves at a time and whatever lands meanwhile rides
 * the next one - and the curve complete enough to BE one. A name, a currency and a row naming a
 * tenor are what the verb requires, which is what makes the commit completing a new card the
 * commit that creates the curve. */
export function due(edit: Editing, now: number, quiet: number = QUIET_MS): boolean {
  const { curve, currency, rows } = edit.form;
  return !edit.saving && edit.at > edit.posted && now - edit.at >= quiet
    && curve.trim() !== '' && currency.trim() !== '' && rows.some((row) => row.tenor.trim() !== '');
}

/** The near half of a curve as ONE pair, against the scheme the whole curve is built under. With
 * no `Near_Tenor` the near scheme is not the desk's to state and the whole curve's shows through;
 * stating a tenor makes that scheme the near one until the desk says otherwise. Applied before the
 * post, so the pairing the emitter refuses half of is never posted. */
export function nearPair(form: CurveForm, typeScheme: string):
{ tenor: string; scheme: string; stated: boolean } {
  const whole = form.interpolation || typeScheme;
  const tenor = String(form.conventions.Near_Tenor ?? '').trim();
  return tenor === ''
    ? { tenor: '', scheme: whole, stated: false }
    : { tenor, scheme: String(form.conventions.Near_Interpolation ?? '') || whole, stated: true };
}

/** The values a card's panel renders, held to the fields it shows: the conventions the curve
 * stands in, the ones the request names outright beside them, the curve's own scheme as the method
 * it RESOLVES to, and the near pair as one. */
export function panelValues(fields: string[], form: CurveForm, typeScheme: string):
Record<string, unknown> {
  const near = nearPair(form, typeScheme);
  const shown: Record<string, unknown> = {
    ...form.conventions, Currency: form.currency, Discount_Rate: form.discount_rate,
    Interpolation: form.interpolation || typeScheme,
    Near_Interpolation: near.scheme, Near_Tenor: near.tenor,
  };
  return Object.fromEntries(fields.filter((key) => shown[key] !== undefined)
    .map((key) => [key, shown[key]]));
}

/** The request `POST /book/curve` takes: every row that names a tenor, the conventions the card
 * shows in the verb's own spelling - the block IS the curve's definition, so what it stands in is
 * re-stated rather than completed from the seed - and the curve's own scheme, blank clearing its
 * rule. The near pair travels only where a tenor states one. */
export function curveRequest(form: CurveForm, typeScheme: string): Record<string, unknown> {
  const near = nearPair(form, typeScheme);
  return {
    curve: form.curve.trim(), currency: form.currency.trim(),
    discount_rate: form.discount_rate.trim(), interpolation: form.interpolation.trim(),
    rows: form.rows.filter((row) => row.tenor.trim() !== '').map((row) => ({
      tenor: row.tenor.trim(), security: row.security.trim(), quote: row.quote, use: row.use,
    })),
    ...Object.fromEntries(Object.entries(form.conventions)
      .map(([key, value]) => [key.toLowerCase(), value])),
    ...(near.stated ? { near_interpolation: near.scheme, near_tenor: near.tenor } : {}),
  };
}

/** The fields a curve is stated in, in declaration order: the ones the request names outright, and
 * every declared field whose lower-case spelling the service answers a conventions block in.
 * Nothing else in the family is one - a solver dial is not a convention, and the verb refuses a
 * name it reads nothing of. A request field the family declares NO column for comes last, and only
 * where the answer names it, which is how the curve's own scheme reaches the panel. */
export function curveFields(declared: Section, answer: CurvesAnswer): string[] {
  const named = new Set<string>();
  for (const entry of [...Object.values(answer.curves), ...Object.values(answer.seeded ?? {})]) {
    Object.keys(entry.conventions ?? {}).forEach((key) => named.add(key));
  }
  return Object.keys(declared)
    .filter((key) => key in REQUEST_FIELDS || named.has(key.toLowerCase()))
    .concat(Object.keys(REQUEST_FIELDS)
      .filter((key) => !(key in declared) && named.has(key.toLowerCase())));
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
