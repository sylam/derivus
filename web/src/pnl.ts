// The desk's P&L as a reader takes it in - pure, free of React and of the network, the way
// `positions.ts` and `blotter.ts` are. What the book made between two marked days, per position
// and in total, and the explain of what the held positions made: the carry, the market per risk
// factor, and the residual left over.

/** Something the P&L could not know, named. */
export type Unknown = { instrument: string | null; what: string };

/** One position's P&L between the two marks, as `GET /book/pnl` answers it. A unit mark is the
 * close's own number, and `paid_*` what the unit paid that day, which the value takes out. */
export type PnlRow = {
  instrument: string; agreement: string; portfolio: string; counterparty: string | null;
  reference: string | null; quantity_start: number; quantity_end: number;
  unit_start: number | null; unit_end: number | null; paid_start: number | null;
  paid_end: number | null;
  value_start: number | null; value_end: number | null; premiums: number | null;
  payments: number | null; fees: number | null; existing: number | null;
  trading: number | null; pnl: number | null; realised: number | null;
  unrealised: number | null;
};

/** One risk factor the explain read: a quote the book's factors are built from, or a factor. */
export type Moved = {
  block?: string; quote?: string; factor?: string; tenor?: number[];
  delta: number; start: number | null; end: number | null; move: number | null;
  pnl: number | null;
};

export type Explain = {
  existing: number | null; carry: number | null; market: number | null;
  residual: number | null; reserves: number | null; factors: Moved[]; unknown: Unknown[];
  note: string | null;
};

export type Marks = { day: string; lsn: number; live?: boolean };

/** A total is null where any row's figure is - a sum over what nobody knows is no sum. */
export type Pnl = {
  currency: string | null; start: Marks; end: Marks; rows: PnlRow[];
  total: Record<string, number | null>; complete: boolean; unknown: Unknown[];
  realised_method: string; explain?: Explain;
};

/** The marked days a window can run between, on the market the desk designated for P&L. */
export type MarkedDays = { market: string; days: { day: string; lsn: number }[] };

/** The figures a position row shows, in the order a reader reads them. */
export const FIGURES: { key: keyof PnlRow; label: string }[] = [
  { key: 'value_start', label: 'Value start' },
  { key: 'value_end', label: 'Value end' },
  { key: 'premiums', label: 'Premiums' },
  { key: 'payments', label: 'Payments' },
  { key: 'fees', label: 'Fees' },
  { key: 'pnl', label: 'P&L' },
  { key: 'realised', label: 'Realised' },
  { key: 'unrealised', label: 'Unrealised' },
];

/** A position as a desk knows it: the reference it was booked under, else its address. */
export const positionLabel = (row: PnlRow): string =>
  row.reference ?? `${row.instrument.slice(0, 12)}…`;

/** A risk factor as a reader knows it: the quote and the block it sits in, or the factor and its
 * coordinates - a spot carrying none. */
export function factorLabel(row: Moved): string {
  if (row.quote !== undefined) return `${row.block} · ${row.quote}`;
  return row.tenor?.length ? `${row.factor} · ${row.tenor.join(' × ')}` : row.factor ?? '';
}

/** The explain's lines in the order they add up: what the held positions made, and the three it
 * is taken apart into - which sum to it by construction, the residual being what is left. */
export function explained(explain: Explain): { label: string; value: number | null }[] {
  return [
    { label: 'Held positions', value: explain.existing },
    { label: 'Carry', value: explain.carry },
    { label: 'Market', value: explain.market },
    { label: 'Residual', value: explain.residual },
  ];
}

/** The residual as a share of what the held positions made - null where either is unknown or
 * nothing was made, since a share of nothing is no reading. */
export function residualShare(explain: Explain): number | null {
  const { existing, residual } = explain;
  return existing === null || residual === null || existing === 0
    ? null : residual / Math.abs(existing);
}

/** Whether the reading in hand is behind the book: only a window ending now can be, since a day
 * struck never moves - and reading one again is a valuation of the whole book, so the screen
 * offers it rather than paying for it on every write. */
export function behind(end: string, readAt: string | undefined, now: string | undefined): boolean {
  return !end && readAt !== undefined && now !== readAt;
}

/** The query `GET /book/pnl` is asked with: a blank end is the book as it stands now, and a
 * blank scope is the whole book. */
export function pnlQuery(start: string, end: string, scope: Record<string, string>,
                         explain: boolean): string {
  const params = new URLSearchParams();
  if (start) params.set('start', start);
  if (end) params.set('end', end);
  for (const [key, value] of Object.entries(scope)) if (value) params.set(key, value);
  if (explain) params.set('explain', 'true');
  return params.toString();
}
