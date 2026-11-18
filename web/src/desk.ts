// The desk's two data views, as arithmetic - kept out of the components the way `blotter.ts` is.
// `/book/risk` is the consolidated mark and the whole-book gradient; `/book/xva` is the last run
// per netting set. Nothing here touches React or the network, so everything the two screens SAY
// about a book - what lags, what is stale, what the rows add up to - is one pure function of the
// two answers, and can be run in node against the live service.
//
// BOTH VIEWS ARE READS. The risk verb computes on a cache miss and answers from the cache after;
// the XVA verb never runs anything at all. A recalc is asked for through the MCP verbs, and this
// module has no vocabulary for one on purpose.

import type { Position } from './positions';
import { axisText, formatNumber, isObject, label, tenorLabel, token } from './tokens';
import type {
  BookRisk, BookXva, DealMark, GreekRow, JobDoc, Market, Schema, XvaSet,
} from './types';
import { axisNames } from './vols';

// ---- stamps -------------------------------------------------------------------------------------
// The service stamps every answer on its own clock to milliseconds, with the offset it runs at. Two
// stamps from the same service are the only clocks compared here: an age is measured from a row's
// run to the READ that carried it, never to the browser's clock, which belongs to another machine.

const STAMP = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?(Z|[+-][\d:]{4,5})?/;

/** A service stamp as milliseconds on one arbitrary clock, or undefined. Read through `Date.UTC`
 * rather than `Date.parse` so no local timezone - and no DST edge inside one - can move a
 * difference between two stamps by an hour. */
export function stampMs(stamp: string | null | undefined): number | undefined {
  const parts = stamp === null || stamp === undefined ? null : STAMP.exec(stamp);
  if (!parts) return undefined;
  const fraction = parts[7] === undefined ? 0 : Number(parts[7].slice(0, 3).padEnd(3, '0'));
  return Date.UTC(+parts[1], +parts[2] - 1, +parts[3], +parts[4], +parts[5], +parts[6], fraction);
}

const pad = (value: number) => String(value).padStart(2, '0');

/** A stamp as a desk reads it: on this browser's clock with its offset named, the seconds kept -
 * `2026-08-28 22:51:34 UTC+2` - so every time on a screen compares with every other. A stamp naming
 * no zone reads as written; anything that is not a stamp reads as itself. */
export function stampText(stamp: string | null | undefined): string {
  const parts = stamp === null || stamp === undefined ? null : STAMP.exec(stamp);
  if (!parts) return stamp ?? '';
  if (!parts[8]) return `${parts[1]}-${parts[2]}-${parts[3]} ${parts[4]}:${parts[5]}:${parts[6]}`;
  const [hours, minutes] = parts[8] === 'Z' ? [0, 0] : parts[8].slice(1).split(/:?(?=\d{2}$)/);
  const east = (parts[8][0] === '-' ? -1 : 1) * (+hours * 60 + +minutes);
  const at = new Date(stampMs(stamp)! - east * 60_000);
  const west = at.getTimezoneOffset();
  const size = Math.abs(west);
  const zone = size === 0 ? 'UTC'
    : `UTC${west > 0 ? '-' : '+'}${Math.floor(size / 60)}${size % 60 ? `:${pad(size % 60)}` : ''}`;
  return `${at.getFullYear()}-${pad(at.getMonth() + 1)}-${pad(at.getDate())} ${
    pad(at.getHours())}:${pad(at.getMinutes())}:${pad(at.getSeconds())} ${zone}`;
}

export function plural(count: number, unit: string): string {
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
 * scalar like an FxRate spot), one for a curve point, two for a vol-surface node - each as its
 * axis is named, so a tenor reads as one and an excel day as its date. */
export function tenorText(tenor?: number[], axes: string[] = []): string {
  return tenor === undefined ? ''
    : tenor.map((value, at) => axisText(axes[at] ?? '', value)).join(' / ');
}

/** The names of a factor's coordinates, off the curve or the surface its type declares. */
export function factorAxes(schema: Schema | null, factor: string, count: number): string[] {
  const shape = Object.values(schema?.Factor.types[factorFamily(factor)] ?? {})
    .find((field) => field.widget === (count === 1 ? 'Curve' : 'Surface'));
  return axisNames(shape?.description, count + 1);
}

/** A vol quote as a dealer names it - `2M ATM`, `1M 25D RR` - off the engine's own name for it, its
 * type then its pillar delta then its expiry in years; any other name reads as itself. */
export function quoteText(quote: string): string {
  const [type, ...rest] = quote.split(' ');
  const numbers = rest.map(Number);
  if (!['ATM', 'RR', 'BF'].includes(type) || numbers.length !== (type === 'ATM' ? 1 : 2)
      || numbers.some((value) => !isFinite(value))) return quote;
  const expiry = tenorLabel(numbers[numbers.length - 1]);
  return type === 'ATM' ? `${expiry} ATM` : `${expiry} ${formatNumber(100 * numbers[0])}D ${type}`;
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
 * is most exposed to per the move each row is read per, sign carried but not sorted on, so the
 * biggest short shows beside the biggest long. By FACTOR it is a curve read in order: the factors
 * alphabetical, and each one's coordinates always ascending inside it, because a curve reversed is
 * not a ranking of anything. Never mutates the answer it is given. */
export function sortGreeks(rows: GreekRow[], key: GreekKey, descending: boolean): GreekRow[] {
  const direction = descending ? -1 : 1;
  return [...rows].sort((a, b) => {
    if (key === 'magnitude') {
      const difference = Math.abs(riskOf(a)) - Math.abs(riskOf(b));
      if (difference !== 0) return direction * (difference < 0 ? -1 : 1);
      return a.factor.localeCompare(b.factor) || compareTenor(a.tenor, b.tenor);
    }
    const named = a.factor.localeCompare(b.factor);
    return named !== 0 ? direction * named : compareTenor(a.tenor, b.tenor);
  });
}

/** The gradient read in QUOTE space: every quote the book moves with, under the block it came off
 * and in the order the block quotes them, then every factor row no quote block stands for - a
 * spot, a curve set up by hand - in its own units. A factor a block builds is read on its quotes
 * and never twice; a quote past everything the book holds reads exactly zero and is left off the
 * screen, the service's answer keeping the whole ladder. */
export function quoteView(risk: BookRisk): GreekRow[] {
  const quoted = new Set(risk.quoted);
  return [...risk.quotes.filter((row) => row.value !== 0)
            .map(({ block, ...row }) => ({ ...row, factor: block })),
          ...risk.greeks.filter((row) => !quoted.has(row.factor))];
}

/** A row's report-currency change per the move it is read per - the per-unit derivative on an
 * answer that reads none. */
export const riskOf = (row: GreekRow): number => row.risk ?? row.value;

/** The words of the move every row shares, or null where they differ: a table names one move in
 * its header, or each row beside itself. */
export function sharedPer(rows: GreekRow[]): string | null {
  const words = new Set(rows.map((row) => row.per ?? 'per unit'));
  return words.size === 1 ? [...words][0] : null;
}

/** What the raw derivative a row was read from says, for a reader who wants it: per unit of the
 * factor as stored, at the level it was taken. */
export function perUnitText(row: GreekRow): string {
  const at = row.level === null || row.level === undefined ? '' : ` at ${formatNumber(row.level)}`;
  return `${formatNumber(row.value)} per unit${at}`;
}

/** One bucket of a ladder: its coordinates less the one summed across, the risk of its nodes. */
export type Bucket = { key: number[]; risk: number; nodes: GreekRow[] };

/** One factor's buckets in coordinate order, the names of its axes - `across` the position of the
 * one its buckets sum over, -1 where none is - and its parallel move, every node at once. */
export type Ladder = {
  factor: string; per: string | null; axes: string[]; across: number; buckets: Bucket[];
  total: number;
};

/** The gradient as a desk reads it: every spot's delta together, largest first; each curve's
 * ladder by tenor; each surface's buckets by expiry - its nodes summed across the axis its type
 * declares as moneyness, the nodes kept on the bucket - and every row read per unit apart. */
export function riskReading(rows: GreekRow[], schema: Schema | null) {
  const spots: GreekRow[] = [];
  const units: GreekRow[] = [];
  const held = new Map<string, GreekRow[]>();
  for (const row of rows) {
    if ((row.per ?? 'per unit') === 'per unit') units.push(row);
    else if (row.tenor === undefined) spots.push(row);
    else held.set(row.factor, [...(held.get(row.factor) ?? []), row]);
  }
  const curves: Ladder[] = [];
  const surfaces: Ladder[] = [];
  for (const factor of [...held.keys()].sort()) {
    const nodes = held.get(factor)!;
    const count = nodes[0].tenor!.length;
    const axes = factorAxes(schema, factor, count).slice(0, count);
    const across = axes.findIndex((axis) => /moneyness/i.test(axis));
    const buckets = new Map<string, Bucket>();
    for (const node of nodes) {
      const key = node.tenor!.filter((_, at) => at !== across);
      const bucket = buckets.get(JSON.stringify(key)) ?? { key, risk: 0, nodes: [] };
      bucket.risk += riskOf(node);
      bucket.nodes.push(node);
      buckets.set(JSON.stringify(key), bucket);
    }
    (count === 1 ? curves : surfaces).push({
      factor, per: sharedPer(nodes), axes, across,
      buckets: [...buckets.values()].sort((a, b) => compareTenor(a.key, b.key)),
      total: nodes.reduce((sum, node) => sum + riskOf(node), 0),
    });
  }
  spots.sort((a, b) => Math.abs(riskOf(b)) - Math.abs(riskOf(a))
    || a.factor.localeCompare(b.factor));
  return { spots, curves, surfaces, units };
}

/** The positions awaiting a second seat inside each marked row, by its deal path: a position the
 * row's node holds, or one any node beneath it does. */
export function pendingUnder(marks: DealMark[], positions: Position[]): Map<string, Position[]> {
  const under = new Map<string, Position[]>();
  for (const mark of marks) {
    if (mark.deal_path === null) continue;
    const path = mark.deal_path;
    const held = positions.filter((position) => position.pending !== 0 && position.deal_paths
      .some((at) => at === path || at.startsWith(`${path}/`)));
    if (held.length) under.set(path, held);
  }
  return under;
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

/** The one clock an answer read off the live book is held against: the book's etag, and on a
 * desk that records how far the record stands (`pinnedAt`) - a booking moves the first, and an
 * approval, a declaration or a fill the record alone holds the second. */
export const bookClock = (etag: string, recordAt: string | null): string =>
  `${etag}@${recordAt ?? ''}`;

/** Whether an answer on screen was read at an OLDER clock than the one the app now holds. Asked of
 * the clock the answer was FETCHED at and never of the risk's own `etag`: the two are hashes of
 * different things, and comparing them would read stale forever. */
export function lagsBook(fetchedAt: string | null, now: string | null | undefined): boolean {
  return fetchedAt !== null && !!now && fetchedAt !== now;
}

/** The report currency, off the document the app already holds - the one the risk answer states
 * and a recalc would report in now; an XVA row states its own. */
export function currencyOf(doc: JobDoc | null): string {
  const currency = doc?.Calc?.Calculation?.['Currency'];
  return typeof currency === 'string' ? currency : '';
}

/** The day the book values on, its calculation's `Base_Date` as written. */
export const valuationDay = (doc: JobDoc | null): string =>
  label(doc?.Calc?.Calculation?.Base_Date);

/** Where the book says its market comes from: the file it names, null where its own block is all
 * of it, and the latest moment any quote of the market it prices on says it was seen. */
export function marketSource(doc: JobDoc | null, market: Market | null) {
  const named = doc?.Calc?.MergeMarketData?.MarketDataFile;
  const seen = [
    ...Object.values(market?.['Market Prices'] ?? {}).flatMap((block) => {
      const points = token(token(block, 'instrument'), 'Points');
      return Array.isArray(points) ? points.map((point) => token(point, 'Timestamp')) : [];
    }),
    ...Object.values(market?.['Price Factors'] ?? {})
      .map((block) => token(block, 'Quote_Timestamp')),
  ].map((stamp) => (isObject(stamp) ? String(stamp['.Timestamp']) : ''))
    .filter((stamp) => stampMs(stamp) !== undefined)
    .sort((a, b) => stampMs(a)! - stampMs(b)!);
  return { file: named ? named.split(/[\\/]/).pop()! : null, seen: seen.pop() ?? null };
}

// ---- a named calculation ------------------------------------------------------------------------

/** The dials a named calculation's type declares and its block does not state, at the book's
 * own: `POST /calculations/run` merges the block over the book's, so that is what they run at. */
export function ownDials(own: Record<string, unknown>, block: Record<string, unknown>,
                         declared: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(own).filter(([key]) => key in declared && !(key in block)));
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

/** One currency's sums: the CVA over the book's own sets that have RUN in it - never an orphan,
 * whose exposure has left the book, nor a set with no number - and the FVA over the rows that CARRY
 * one, counted apart since a row filed before the column existed reads null there. `currency` is
 * null for rows filed before the run kept one. */
export type XvaSum = {
  currency: string | null; cva: number; counted: number; fva: number; countedFva: number;
};

export type XvaTotals = {
  /** Sets the book holds now. Orphans are counted apart, because they are not the book's. */
  sets: number;
  done: number;
  failed: number;
  neverRun: number;
  running: number;
  orphans: number;
  /** One sum per currency the rows were run in: amounts in two currencies never add. */
  sums: XvaSum[];
};

export function xvaTotals(view: BookXva): XvaTotals {
  const totals: XvaTotals = {
    sets: 0, done: 0, failed: 0, neverRun: 0, running: 0, orphans: 0, sums: [],
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
    if (set.status !== 'done') continue;
    let sum = totals.sums.find((held) => held.currency === set.currency);
    if (!sum) {
      sum = { currency: set.currency, cva: 0, counted: 0, fva: 0, countedFva: 0 };
      totals.sums.push(sum);
    }
    if (typeof set.cva === 'number') {
      sum.cva += set.cva;
      sum.counted += 1;
    }
    if (typeof set.fva === 'number') {
      sum.fva += set.fva;
      sum.countedFva += 1;
    }
  }
  return totals;
}

/** The replay tuple, labelled: what the row IS to the book over what the RUN was, in that order.
 * A null reads as absent rather than as a blank - the expander is where a desk goes to find out
 * whether a number can be reproduced, and half a tuple must look like half a tuple. `FVA` is here
 * as well as in the table because its absence is a fact about the RUN: a null beside a `done` row
 * says the row predates the column, which is what an old `as_of` and a recalc would fix. */
export function replayRows(set: XvaSet): [string, string | null][] {
  return [
    ['Deal path', set.deal_path],
    ['Counterparty', set.counterparty],
    ['Collateralised', set.collateralized ? 'yes' : 'no'],
    ['Reported in', set.currency],
    ['FVA', set.fva === null ? null : formatNumber(set.fva, 'money')],
    ['Last run', set.as_of === null ? null : stampText(set.as_of)],
    ['Result id', set.result_id],
    ['Plan hash', set.plan_hash],
    ['Values hash', set.values_hash],
    ['Seed', set.seed === null ? null : String(set.seed)],
    ['Quasi seed', set.quasi_seed === null ? null : String(set.quasi_seed)],
  ];
}
