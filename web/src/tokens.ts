// The wire tokens a job document carries (`CustomJsonEncoder` is the source of truth), the small
// formatters the read-only renderer shares, and the write half's encoder.

import type { Descriptor } from './types';

export function token(value: unknown, name: string): unknown {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    return (value as Record<string, unknown>)[name];
  }
  return undefined;
}

export const isObject = (value: unknown): value is Record<string, unknown> =>
  !!value && typeof value === 'object' && !Array.isArray(value);

/** A column/index entry as text: tuples join, `.Timestamp` tokens read as their date. */
export function label(value: unknown): string {
  if (Array.isArray(value)) return value.map(label).join(' / ');
  const stamp = token(value, '.Timestamp');
  if (stamp !== undefined) return String(stamp);
  if (value === null || value === undefined) return '';
  return String(value);
}

export function formatNumber(value: number): string {
  if (!isFinite(value)) return String(value);
  if (value !== 0 && Math.abs(value) < 1e-4) return value.toExponential(4);
  return value.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 6 });
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
  'Text', 'Dropdown', 'Float', 'BoundedFloat', 'Integer', 'DatePicker', 'Checkbox',
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
    case 'Float': case 'BoundedFloat': return Number(raw);
    case 'Integer': return Math.trunc(Number(raw));
    case 'DatePicker': return { '.Timestamp': raw };
    case 'Checkbox': return raw === 'true';
    default: return raw;
  }
}
