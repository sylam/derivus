// The desk's two data views, as arithmetic - kept out of the components the way `blotter.ts` is.
// `/book/risk` is the consolidated mark and the whole-book gradient; `/book/xva` is the last run
// per netting set. Nothing here touches React or the network, so everything the two screens SAY
// about a book - what lags, what is stale, what the rows add up to - is one pure function of the
// two answers, and can be run in node against the live service.
//
// BOTH VIEWS ARE READS. The risk verb computes on a cache miss and answers from the cache after;
// the XVA verb never runs anything at all. A recalc is asked for through the MCP verbs, and this
// module has no vocabulary for one on purpose.

import { formatNumber } from './tokens';
import type { BookRisk, BookXva, DealMark, GreekRow, JobDoc, XvaSet } from './types';

// ---- stamps -------------------------------------------------------------------------------------
// The service stamps every answer in local time to milliseconds. Two stamps from the same service
// are the only clocks compared here: an age is measured from a row's run to the READ that carried
// it, never to the browser's clock, which belongs to a different machine in a different timezone
// and would put a desk's numbers hours out for no reason it could see.

const STAMP = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?/;

/** A service stamp as milliseconds on one arbitrary clock, or undefined. Read through `Date.UTC`
 * rather than `Date.parse` so no local timezone - and no DST edge inside one - can move a
 * difference between two stamps by an hour. */
export function stampMs(stamp: string | null | undefined): number | undefined {
  const parts = stamp === null || stamp === undefined ? null : STAMP.exec(stamp);
  if (!parts) return undefined;
  const fraction = parts[7] === undefined ? 0 : Number(parts[7].slice(0, 3).padEnd(3, '0'));
  return Date.UTC(+parts[1], +parts[2] - 1, +parts[3], +parts[4], +parts[5], +parts[6], fraction);
}

/** A stamp as a desk reads it - `2026-08-28 20:51:34`, the milliseconds dropped. Anything that is
 * not a stamp reads as itself: a shape this does not recognise is still shown. */
export function stampText(stamp: string | null | undefined): string {
  const parts = stamp === null || stamp === undefined ? null : STAMP.exec(stamp);
  if (!parts) return stamp ?? '';
  return `${parts[1]}-${parts[2]}-${parts[3]} ${parts[4]}:${parts[5]}:${parts[6]}`;
}

function plural(count: number, unit: string): string {
  return `${count} ${unit}${count === 1 ? '' : 's'}`;
}

/** A duration as an age, coarsening as it grows: nobody reads a CVA to the minute once it is a
 * day old, and a stamp that precise invites a precision the number does not have. FLOORED at every
 * unit, so an age understates - `3 hours ago` for a row of three hours and fifty minutes is at
 * least true, where rounding it up to four would age a number that had not aged. */
export function ageText(ms: number): string {
  const minutes = Math.floor(ms / 60_000);
  if (minutes < 1) return 'moments ago';
  if (minutes < 60) return `${plural(minutes, 'minute')} ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${plural(hours, 'hour')} ago`;
  return `${plural(Math.floor(hours / 24), 'day')} ago`;
}

/** How old a projection row may be before this view calls it stale. A CVA recalc is a deliberate
 * act minutes of device time long, so a desk runs one and reads it for the rest of the session;
 * a row carried over from yesterday is a number about a different market. Half a day is that line,
 * and it is a READING of the number, not a claim the service makes. */
export const FRESH_MS = 12 * 3_600_000;

export type Age = { text: string; tone: 'fresh' | 'stale' | 'none'; title: string };

/** A row's age, measured against the read that carried it. `ran` null is the 'never run' case -
 * the book carries the set and no recalc has ever produced a row for it - which is not old, it is
 * absent, and says so. */
export function ageOf(ran: string | null, read: string): Age {
  if (ran === null) {
    return {
      text: 'never run', tone: 'none',
      title: 'the book carries this set, but no recalc has ever produced a row for it',
    };
  }
  const from = stampMs(ran);
  const to = stampMs(read);
  if (from === undefined || to === undefined) return { text: ran, tone: 'none', title: ran };
  // a row stamped after the read is a clock that moved, not a number from the future
  const ms = Math.max(0, to - from);
  return { text: ageText(ms), tone: ms < FRESH_MS ? 'fresh' : 'stale', title: stampText(ran) };
}

// ---- the consolidated risk ----------------------------------------------------------------------

export type GreekKey = 'factor' | 'magnitude';

/** A factor's coordinates as one cell: empty where the factor has none (the key is absent on a
 * scalar like an FxRate spot), one number for a curve point, two for a vol-surface node. */
export function tenorText(tenor?: number[]): string {
  return tenor === undefined ? '' : tenor.map(formatNumber).join(' / ');
}

/** Coordinates in reading order, a factor with none first - a spot sits above its own curve. */
function compareTenor(a: number[] | undefined, b: number[] | undefined): number {
  if (a === undefined || b === undefined) return a === b ? 0 : a === undefined ? -1 : 1;
  for (let index = 0; index < Math.max(a.length, b.length); index += 1) {
    const difference = (a[index] ?? -Infinity) - (b[index] ?? -Infinity);
    if (difference !== 0) return difference < 0 ? -1 : 1;
  }
  return 0;
}

/** The gradient sorted for reading. By MAGNITUDE - the default - it is a ranking: what the book
 * is most exposed to, sign carried but not sorted on, so the biggest short shows beside the
 * biggest long. By FACTOR it is a curve read in order: the factors alphabetical, and each one's
 * coordinates always ascending inside it, because a curve reversed is not a ranking of anything.
 * Never mutates the answer it is given. */
export function sortGreeks(rows: GreekRow[], key: GreekKey, descending: boolean): GreekRow[] {
  const direction = descending ? -1 : 1;
  return [...rows].sort((a, b) => {
    if (key === 'magnitude') {
      const difference = Math.abs(a.value) - Math.abs(b.value);
      if (difference !== 0) return direction * (difference < 0 ? -1 : 1);
      return a.factor.localeCompare(b.factor) || compareTenor(a.tenor, b.tenor);
    }
    const named = a.factor.localeCompare(b.factor);
    return named !== 0 ? direction * named : compareTenor(a.tenor, b.tenor);
  });
}

/** The family a factor name opens with - `FxRate`, `InterestRate`, `FXVol`. What a desk groups by
 * when it reads a gradient, and it is the name's own first segment, not a list kept here. */
export function factorFamily(factor: string): string {
  const stop = factor.indexOf('.');
  return stop === -1 ? factor : factor.slice(0, stop);
}

export function marksTotal(rows: DealMark[]): number {
  return rows.reduce((total, row) => total + row.value, 0);
}

/** Relative, because summing the same rows in a different order moves the last bits. */
export const RECONCILE_TOLERANCE = 1e-9;

/** Whether `mtm` is the sum of the per-deal rows, which the service promises it is on any book.
 * Checked rather than assumed: the marks table shows a total, and a total that did not agree with
 * the headline would be the one thing on the screen worth saying out loud. */
export function reconciles(risk: BookRisk): boolean {
  const scale = Math.max(1, Math.abs(risk.mtm));
  return Math.abs(risk.mtm - marksTotal(risk.per_deal)) <= RECONCILE_TOLERANCE * scale;
}

/** Whether the risk on screen was computed over an OLDER book than the one the app now holds.
 * Asked of the book etag the answer was FETCHED at and never of the risk's own `etag`: the two are
 * hashes of different things, and comparing them would read stale forever. */
export function lagsBook(fetchedAt: string | null, bookEtag: string | null | undefined): boolean {
  return fetchedAt !== null && !!bookEtag && fetchedAt !== bookEtag;
}

/** The report currency, off the document the app already holds - the same currency the risk
 * answer states, and the one a CVA is denominated in, which `/book/xva` does not repeat. */
export function currencyOf(doc: JobDoc | null): string {
  const currency = doc?.Calc?.Calculation?.['Currency'];
  return typeof currency === 'string' ? currency : '';
}

// ---- the XVA projection -------------------------------------------------------------------------

/** A row whose set has left the book. It keeps its last numbers - a number that was on the
 * blotter yesterday vanishing without a word is how a desk loses track of a position - and the
 * service sorts these after the book's own sets, an order this module never re-imposes. */
export function isOrphan(set: XvaSet): boolean {
  return set.note !== null;
}

export type Chip = { text: string; tone: string; title: string };

/** The status a row leads with. A recalc IN FLIGHT outranks the last run: what the desk wants to
 * know first is that the number is moving. The last run keeps its own chip beside it, so a done
 * row being recalculated never looks like a row with no number. */
export function statusChip(set: XvaSet): Chip {
  if (set.recalc) {
    return {
      text: set.recalc.status, tone: 'running',
      title: `a recalc is ${set.recalc.status} - result ${set.recalc.result_id}`,
    };
  }
  return lastRunChip(set);
}

export function lastRunChip(set: XvaSet): Chip {
  switch (set.status) {
    case 'done':
      return { text: 'done', tone: 'done', title: 'the last recalc of this set completed' };
    case 'failed':
      return { text: 'failed', tone: 'error', title: set.error ?? 'the last recalc failed' };
    case 'never run':
      return {
        text: 'never run', tone: '',
        title: 'the book carries this set, but no recalc has ever produced a row for it',
      };
    default:
      // a status this view has never heard of is still shown, as itself
      return { text: set.status, tone: '', title: set.status };
  }
}

export type XvaTotals = {
  /** Sets the book holds now. Orphans are counted apart, because they are not the book's. */
  sets: number;
  done: number;
  failed: number;
  neverRun: number;
  running: number;
  orphans: number;
  /** The CVA over the book's own sets that have RUN - never over an orphan, whose exposure has
   * left the book, and never over a set with no number. */
  cva: number;
  counted: number;
};

export function xvaTotals(view: BookXva): XvaTotals {
  const totals: XvaTotals = {
    sets: 0, done: 0, failed: 0, neverRun: 0, running: 0, orphans: 0, cva: 0, counted: 0,
  };
  for (const set of view.sets) {
    if (set.recalc) totals.running += 1;
    if (isOrphan(set)) {
      totals.orphans += 1;
      continue;
    }
    totals.sets += 1;
    if (set.status === 'done') totals.done += 1;
    else if (set.status === 'failed') totals.failed += 1;
    else if (set.status === 'never run') totals.neverRun += 1;
    if (set.status === 'done' && typeof set.cva === 'number') {
      totals.cva += set.cva;
      totals.counted += 1;
    }
  }
  return totals;
}

/** The replay tuple, labelled: what the row IS to the book over what the RUN was, in that order.
 * A null reads as absent rather than as a blank - the expander is where a desk goes to find out
 * whether a number can be reproduced, and half a tuple must look like half a tuple. */
export function replayRows(set: XvaSet): [string, string | null][] {
  return [
    ['Deal path', set.deal_path],
    ['Counterparty', set.counterparty],
    ['Collateralised', set.collateralized ? 'yes' : 'no'],
    ['Last run', set.as_of === null ? null : stampText(set.as_of)],
    ['Result id', set.result_id],
    ['Plan hash', set.plan_hash],
    ['Values hash', set.values_hash],
    ['Seed', set.seed === null ? null : String(set.seed)],
  ];
}
