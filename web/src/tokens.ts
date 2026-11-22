// The wire tokens a job document carries (`CustomJsonEncoder` is the source of truth), the small
// formatters the read-only renderer shares, and the write half's encoder.

import type { Descriptor, Section, SettlementState } from './types';

export function token(value: unknown, name: string): unknown {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    return (value as Record<string, unknown>)[name];
  }
  return undefined;
}

export const isObject = (value: unknown): value is Record<string, unknown> =>
  !!value && typeof value === 'object' && !Array.isArray(value);

/** Whether a field value states nothing: absent, null or empty string. */
export const isBlank = (value: unknown): boolean =>
  value === undefined || value === null || value === '';

/** A field's label: its description where short enough to sit in a label column, else its key in
 * words with the full description kept for hover. */
export function fieldLabel(name: string, descriptor?: Descriptor): { text: string; title?: string } {
  const description = descriptor?.description ?? name;
  return description.length > 64 ? { text: name.replace(/_/g, ' '), title: description }
    : { text: description };
}

/** What one row of a credit support list is, as a run reads the list. */
export const SUPPORT_ROW = 'Each row is the amount that applies at its credit rating; a run reads '
  + 'the first row.';

/** A column, index or cell entry as text: tuples join, and every scalar wire token reads as what
 * it is - a date, a period, a percent or a basis with its unit. A table's cells carry the same
 * tokens a field does, so they are read in one place. */
export function label(value: unknown): string {
  if (Array.isArray(value)) return value.map(label).join(' / ');
  const stamp = token(value, '.Timestamp');
  if (stamp !== undefined) return String(stamp);
  if (token(value, '.DateOffset') !== undefined) return offsetText(value);
  const percent = token(value, '.Percent');
  if (percent !== undefined) return formatNumber(percent as number, 'percent');
  const basis = token(value, '.Basis');
  if (basis !== undefined) return formatNumber(basis as number, 'basis');
  if (value === null || value === undefined) return '';
  return String(value);
}

// ---- numbers, tenors and dates as a desk reads them ----

/** What a number is, which says how it shows: `money` to the cent, `whole` in whole units, a
 * `percent` to four decimals and a `basis` point to two with their units, and `plain` to six
 * significant digits, never rounding left of the point and to the cent past a thousand. */
export type Kind = 'money' | 'whole' | 'percent' | 'basis' | 'plain';

const FORMATS = new Map<string, Intl.NumberFormat>();

/** `value` to at most `most` decimals and at least `least`, in the one locale every screen uses:
 * a decimal point and commas between thousands, whatever the machine is set to. */
function digits(value: number, most: number, least = 0): string {
  const key = `${most}/${least}`;
  if (!FORMATS.has(key)) {
    FORMATS.set(key, new Intl.NumberFormat('en-US', {
      minimumFractionDigits: least, maximumFractionDigits: most }));
  }
  const text = FORMATS.get(key)!.format(value);
  return /^-[0.,]*$/.test(text) ? text.slice(1) : text;
}

/** A typed number with the grouping the screens themselves show taken out: `61,000,000` is
 * 61000000, while any other comma (`2,5`) is left to be refused, since it could be a decimal. */
export function ungrouped(text: string): string {
  return /^-?\d{1,3}(,\d{3})+(\.\d+)?$/.test(text) ? text.replace(/,/g, '') : text;
}

/** A number for display, never for an input - an edit holds the exact value. */
export function formatNumber(value: number, kind: Kind = 'plain'): string {
  if (!isFinite(value)) return String(value);
  const cents = Number.isInteger(value) ? digits(value, 0) : digits(value, 2, 2);
  switch (kind) {
    case 'money': return cents;
    case 'whole': return digits(value, 0);
    case 'percent': return `${digits(value, 4)} %`;
    case 'basis': return `${digits(value, 2)} bp`;
  }
  const size = Math.abs(value);
  if (size >= 1000) return cents;
  if (value !== 0 && size < 1e-6) return value.toExponential(3);
  return digits(value, value === 0 ? 0 : 5 - Math.floor(Math.log10(size)));
}

/** The currency an amount is stated in: the first `*Currency` field its description says it is
 * `in` - by key or in words, `in Sales_Margin_Currency`, `in the agreement currency` - that `read`
 * answers with one. */
export function currencyOf(description: string, keys: string[],
                           read: (key: string) => unknown): string | undefined {
  const at = (key: string) => description.search(
    new RegExp(`\\bin (the )?(${key}|${key.replace(/_/g, ' ')})\\b`, 'i'));
  return keys.filter((key) => /(^|_)Currency$/.test(key) && at(key) >= 0).map(read)
    .find((value): value is string => typeof value === 'string' && value !== '');
}

/** What a declared field's number is: an amount of the instrument (`sized`) is money, a Percent
 * or a Basis carries its unit, and anything else is plain. */
export const kindOf = (descriptor?: Descriptor): Kind => (descriptor?.sized ? 'money'
  : descriptor?.obj === 'Percent' ? 'percent' : descriptor?.obj === 'Basis' ? 'basis' : 'plain');

/** An excel day number as the ISO date it names. */
const excelDate = (serial: number): string =>
  new Date(Date.UTC(1899, 11, 30) + Math.round(serial) * 86_400_000).toISOString().slice(0, 10);

/** A length in years as a dealer says it - 3D, 1W, 2M, 18M, 7Y - a month count within half a week
 * (a hundredth on a long one) of a whole one; any other length reads in years. */
export function tenorLabel(years: number): string {
  const days = years * 365;
  if (Math.abs(days) < 0.5) return '0D';
  if (days > 0 && days < 6.5) return `${Math.round(days)}D`;
  const weeks = Math.round(days / 7);
  if (weeks >= 1 && weeks <= 3 && Math.abs(days - 7 * weeks) <= 1) return `${weeks}W`;
  const months = Math.round(years * 12);
  if (months >= 1 && Math.abs(days - months * 365 / 12) <= Math.max(3.5, days / 100)) {
    return months % 12 === 0 ? `${months / 12}Y` : `${months}M`;
  }
  return `${digits(years, 2)}Y`;
}

/** A coordinate as its axis is named: a date axis's excel day number as its date, an expiry or a
 * tenor in years as a dealer says it, anything else as a number. */
export function axisText(name: string, value: number): string {
  if (/(^|[^a-z])date([^a-z]|$)/i.test(name)) return excelDate(value);
  if (/^(expiry|tenor)$/i.test(name.trim())) return tenorLabel(value);
  return formatNumber(value);
}

/** `.Grid` entries and `.DateOffset` strings as one line of text. */
export function offsetText(value: unknown): string {
  const offset = token(value, '.DateOffset');
  const step = token(value, '.Offset');
  if (offset === undefined) return label(value);
  return step === undefined ? String(offset) : `${offset}(${step})`;
}

/** The `.Curve` payload, or undefined: `{meta, data}` with data a list of rows. */
export function curveOf(value: unknown): { meta: unknown[]; data: number[][] } | undefined {
  const curve = token(value, '.Curve');
  if (curve && isObject(curve) && Array.isArray((curve as { data?: unknown }).data)) {
    return curve as { meta: unknown[]; data: number[][] };
  }
  return undefined;
}

// ---- the write half: descriptor-driven, no field names anywhere ----
// This is the inverse of the read dispatch above, and the design correction to the old Jupyter
// app's `set_repr` whitelists: the DECLARATION (widget + obj token) decides the wire form, so a
// new field edits correctly the day it is declared.

const SCALAR_WIDGETS = new Set([
  'Text', 'Dropdown', 'Float', 'Integer', 'DatePicker', 'Checkbox',
]);

const SCALAR_TOKENS = ['.Timestamp', '.Percent', '.Basis', '.DateOffset'];

/** A field the edit slice can offer an input for: a declared scalar whose current value is a
 * scalar or a scalar wire token. Shapes, tables and containers stay read-only in this slice. */
export function isEditableScalar(descriptor: Descriptor | undefined, value: unknown): boolean {
  if (!descriptor || !SCALAR_WIDGETS.has(descriptor.widget)) return false;
  if (value === null || value === undefined || typeof value !== 'object') return true;
  return SCALAR_TOKENS.some((name) => token(value, name) !== undefined);
}

/** The current wire value as the string an input starts from. */
export function editRaw(value: unknown): string {
  for (const name of ['.Percent', '.Basis', '.Timestamp', '.DateOffset'] as const) {
    const inner = token(value, name);
    if (inner !== undefined) return String(inner);
  }
  if (value === null || value === undefined) return '';
  return String(value);
}

/** An input's string back into the wire form the DECLARATION names. `obj` outranks `widget`:
 * a Percent is a Float widget and a Period is a Text widget, and the token is what the file
 * carries. An empty string means "unset" and travels as itself. */
export function encodeScalar(descriptor: Descriptor, raw: string | boolean): unknown {
  if (typeof raw === 'boolean') return raw;
  if (raw === '') return '';
  switch (descriptor.obj) {
    case 'Percent': return { '.Percent': Number(raw) };
    case 'Basis': return { '.Basis': Number(raw) };
    case 'Period': return { '.DateOffset': raw };
  }
  switch (descriptor.widget) {
    case 'Float': return Number(raw);
    case 'Integer': return Math.trunc(Number(raw));
    case 'DatePicker': return { '.Timestamp': raw };
    case 'Checkbox': return raw === 'true';
    default: return raw;
  }
}

// ---- an agreement's terms: a deal's own sections, read off the declaration ----

/** The sections every instrument type carries - a trade's admin block - which no type's own
 * terms are. */
export function sharedSections(types: Record<string, string[]>): string[] {
  const all = Object.values(types);
  return (all[0] ?? []).filter((section) => all.every((sections) => sections.includes(section)));
}

/** The flag a netting set's collateral turns on. */
export const COLLATERAL = 'Collateralized';

/** The currency a netting set keeps its collateral balance in: Balance_Currency, else
 * Agreement_Currency, else blank. */
export const setCurrency = (terms: Record<string, unknown>): string =>
  label([terms.Balance_Currency, terms.Agreement_Currency].find(Boolean));

/** The curve a netting set's collateral adjustment funds at: experimental, so no screen shows
 * the term, and a set stating it keeps it through every edit. */
const SET_FUNDING = 'Funding_Rate';

/** One section of a deal as a screen shows it: its `fields` and the `deal`'s values under them,
 * `SET_FUNDING` in neither - a collateral table's column of that name stays - and `deal` whole. */
export function onScreen(fields: Section, deal: Record<string, unknown>):
  { fields: Section; values: Record<string, unknown> } {
  const cut = <T>(record: Record<string, T>): Record<string, T> => Object.fromEntries(
    Object.entries(record).filter(([key]) => key in fields && key !== SET_FUNDING));
  return { fields: cut(fields), values: cut(deal) };
}

/** `fields` cut to the ones `values` state - a container's to the children it states - and those
 * `also` names: what reads of terms whose conventions mean nothing. */
export function statedOnly(fields: Section, values: Record<string, unknown>,
                           also: string[] = []): Section {
  return Object.fromEntries(Object.entries(fields)
    .filter(([key]) => values[key] !== undefined || also.includes(key))
    .map(([key, field]) => [key, field.sub_fields && isObject(values[key])
      ? { ...field, sub_fields: statedOnly(field.sub_fields, values[key]) } : field]));
}

/** `fields` as an agreement's paper states them: the fields `unstated` names left out, and the
 * columns it names left out of every table - a container's among them - with their forms. */
export function onPaper(fields: Section, unstated: SettlementState): Section {
  return Object.fromEntries(Object.entries(fields)
    .filter(([key]) => !unstated.fields.includes(key))
    .map(([key, field]) => {
      if (field.sub_fields) {
        return [key, { ...field, sub_fields: onPaper(field.sub_fields, unstated) }];
      }
      const cut = (field.col_names ?? []).map((name) => unstated.collateral_columns.includes(name));
      const kept = <T>(list?: T[]) => list?.filter((_, at) => !cut[at]);
      return [key, !cut.some(Boolean) ? field : {
        ...field, col_names: kept(field.col_names), sub_types: kept(field.sub_types),
        obj: Array.isArray(field.obj) ? kept(field.obj) : field.obj }];
    }));
}

/** Every term two statements of a deal differ on, a container's by `Container.Term`: a blank or
 * a null is no statement, and a wire token is one value. */
export function termsMoved(held: Record<string, unknown>, edited: Record<string, unknown>,
                           at = ''): string[] {
  const said = (value: unknown) => (isBlank(value) ? undefined : value);
  const plain = (value: unknown): value is Record<string, unknown> | undefined =>
    value === undefined || (isObject(value) && !Object.keys(value).some((key) => key.startsWith('.')));
  return [...new Set([...Object.keys(held), ...Object.keys(edited)])].sort().flatMap((key) => {
    const [was, now] = [said(held[key]), said(edited[key])];
    if (plain(was) && plain(now)) return termsMoved(was ?? {}, now ?? {}, `${at}${key}.`);
    return JSON.stringify(was) === JSON.stringify(now) ? [] : [`${at}${key}`];
  });
}
