// The desk's P&L as a reader takes it in - pure, free of React and of the network, the way
// `positions.ts` and `blotter.ts` are. What the book made between two marked days, per position
// and in total, and the explain of what the held positions made: the carry, the market per risk
// factor, and the residual left over.

import { unrecorded } from './positions';

/** What the P&L page says in the P&L's place where no record is kept, null on a desk that
 * records: a P&L runs between marked closes, and where the `live` book's worth is read instead. */
export function noPnl(recording: boolean, live: boolean): { title: string; hint: string } | null {
  if (recording) return null;
  return {
    title: unrecorded(live, 'there is no P&L to read here', 'no P&L is read of it'),
    hint: `A P&L runs between two marked closes, and a close is marked on a record${
      live ? ': the Risk tab reads what the book is worth as it stands' : ''}.`,
  };
}

/** Something the P&L could not know, named. */
export type Unknown = { instrument: string | null; what: string };

/** One position's P&L between the two marks, as `GET /book/pnl` answers it. A unit mark is the
 * close's own number, and `paid_*` what the unit paid that day, which the value takes out. */
type PnlRow = {
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
type FactorMove = {
  block?: string; quote?: string; factor?: string; tenor?: number[];
  delta: number; start: number | null; end: number | null; move: number | null;
  pnl: number | null;
};

export type Explain = {
  existing: number | null; carry: number | null; market: number | null;
  residual: number | null; reserves: number | null; factors: FactorMove[]; unknown: Unknown[];
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

/** A risk factor as a reader knows it: the quote and the block it sits in, or the factor and its
 * coordinates - a spot carrying none. */
export function factorLabel(row: FactorMove): string {
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

/** The window the page opens on, a blank being the service's own - the last marks, to now: both
 * blank, and for a seat `held` from a read to now the last two of the marked `days`, which asks
 * no what-if - null where fewer than two are marked, when that seat has no window to read. */
export function opensOn(days: { day: string }[], held: boolean):
  { start: string; end: string } | null {
  if (!held) return { start: '', end: '' };
  const [start, end] = days.slice(-2).map((marked) => marked.day);
  return end === undefined ? null : { start, end };
}

/** What a seat `hold` keeps from a P&L to now is told where `opensOn` finds it no window among
 * the marked `days`: that line, and that a P&L between two marked days, which is not held from
 * it, reads once two are marked - beside the one day that is. */
export const noWindow = (hold: string, days: { day: string }[]): string =>
  `${hold} A P&L between two marked days is not held from this seat, and reads once two days `
  + `are marked: ${days.length ? `only ${days[0].day} is` : 'none is'}.`;

/** What one P&L read is of, asked again where it moves: the seat reading, which is shown the
 * positions under its own nodes, and the `query` asked (`pnlQuery`). */
export const pnlRead = (seat: string, query: string): string => `${seat}|${query}`;

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
