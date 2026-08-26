// The wire tokens a job document carries (`CustomJsonEncoder` is the source of truth), and the
// small formatters the read-only renderer shares.

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
