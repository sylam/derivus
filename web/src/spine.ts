// The record's three readers, as arithmetic - pure, free of React and of the network, the way
// `desk.ts` and `curves.ts` are for theirs. The strip merges the pages a poll brings, the banner
// reads one verdict off the reconcile lists, and the markets list is the record's own answer
// shaped for reading. The store's two render guards are here too, as the predicates the reducer
// asks, so what decides whether the desk repaints is checkable without a browser.
//
// EVERY READING REFUSES NOTHING, and a desk that records nothing is a STATE rather than an error:
// `spine` comes back null, the verdict is `none`, and the two screens render nothing at all.

import { plural } from './desk';
import { positionName, unrecorded } from './positions';
import type { Source } from './state';
import { formatNumber } from './tokens';
import type {
  ActivityPage, ActivityRow, BookMarkets, Doorbell, MarketClose, Reconcile, SpineBlock, Worklist,
} from './types';

/** How many strip rows a client holds. The service caps a page at 200 and this caps what has been
 * MERGED, so a poll running all day cannot grow the array without bound. */
const STRIP_ROWS = 200;

/** Whether the served book is asked every beat if it moved: the desk's own is, which any client
 * books into, and a document opened from a file is read once and again after each write this
 * screen makes. */
export function polled(source: Source | null): source is Source & { kind: 'book' } {
  return source?.kind === 'book' && !source.opened;
}

/** Whether a beat owes the desk's own book a read: `/book/status` returns `etag` without a path
 * (and `/book/open` is refused under a record), so a stamp change is the sole reload trigger. */
export const bookOwed = (held: Source | null, etag: string): boolean =>
  polled(held) && held.etag !== etag;

/** One read at a time: a call while one is in flight answers that one rather than sending a
 * second, so a stalled service never stacks a beat's reads. */
export function single<T>(read: () => Promise<T>): () => Promise<T> {
  let pending: Promise<T> | null = null;
  return () => pending ?? (pending = read().finally(() => { pending = null; }));
}

/** Where the file stands, as one comparable value - so a beat of the poll that moved nothing is
 * the same state and repaints nothing. Null throughout is a desk that records nothing. */
export function pinnedAt(spine: SpineBlock | null): string | null {
  return spine === null
    ? null : `${spine.lsn}/${spine.events_behind}/${spine.positions_behind}`;
}

/** A reconcile answer as one comparable value - the position it was folded at and every row of
 * the three lists. A fetched answer is a NEW object every time, so identity would repaint the
 * whole desk once per fold; what a desk reads is these rows. */
function reconciledAs(reconcile: Reconcile | null): string | null {
  return reconcile === null ? null : JSON.stringify([
    reconcile.lsn, reconcile.events_behind, reconcile.positions_behind,
    reconcile.in_record_not_in_file.map((row) => [row.instrument, row.netting_set, row.quantity]),
    reconcile.in_file_not_in_record.map((row) => [row.instrument, row.deal_path]),
    (reconcile.terms_mismatch ?? []).map((row) => [row.agreement, row.fields]),
  ]);
}

/** What this client holds of the record: where the file stands, the strip as it has been merged -
 * null until its first page is read - the reconcile answer with the head and file it was read
 * at (`reconcileAt`), and the one worklist every screen counts from with the head and seat it was
 * read at (`worklistAt`) or the read's refusal.
 * The two transitions below are the whole of what a beat can do to it. */
export type RecordHeld = {
  spine: SpineBlock | null;
  rows: ActivityRow[] | null;
  reconcile: Reconcile | null;
  reconciledAt: string | null;
  worklist: Worklist | null;
  worklistAt: string | null;
  worklistError: string | null;
};

/** The record after one beat's read - THE SAME OBJECT where neither the rows nor the pin moved.
 * The beat is two seconds and most beats bring nothing, so a transition that answered a new
 * object every time would repaint the desk, a vol surface included, for as long as a tab is
 * open. */
export function readRecord(held: RecordHeld, spine: SpineBlock | null,
                           page: ActivityPage | null): RecordHeld {
  const rows = mergeActivity(held.rows, page);
  return rows === held.rows && pinnedAt(spine) === pinnedAt(held.spine)
    ? held : { ...held, spine, rows };
}

/** The record after one reconcile answer - the same object where the lists and where they were
 * read are the ones already in hand. A fetched answer is a NEW object every time, so identity
 * alone would repaint the whole desk once per fold. */
export function reconciledRecord(held: RecordHeld, reconcile: Reconcile,
                                 at: string | null): RecordHeld {
  return at === held.reconciledAt && reconciledAs(reconcile) === reconciledAs(held.reconcile)
    ? held : { ...held, reconcile, reconciledAt: at };
}

/** The record's head: the file's pin and how far the record has moved past it, blank on a file
 * nothing has pinned. */
const headOf = (spine: SpineBlock): string =>
  (spine.lsn === null ? '' : String(spine.lsn + (spine.events_behind ?? 0)));

/** Where a reconcile answer is read: the record's head and the file's etag, so a declaration -
 * which moves the head and neither a position nor the file - asks again, and so does a file write. */
export function reconcileAt(spine: SpineBlock | null, etag: string | null): string | null {
  return spine === null ? null : `${headOf(spine)}|${etag ?? ''}`;
}

/** Whether the banner and the Paper owe a `/book/reconcile`: the head or the file moved since the
 * answer in hand. A fold at the head is what this costs, so it is never asked on the beat. */
export function wantsReconcile(spine: SpineBlock | null, heldAt: string | null,
                               etag: string | null): boolean {
  return spine !== null && heldAt !== reconcileAt(spine, etag);
}

/** What the banner says about the file it is looking at. `none` is a desk that records nothing. */
type VerdictState = 'none' | 'clean' | 'behind' | 'drifted';

type Reconciled = {
  state: VerdictState;
  /** The banner's one line, empty where it shows nothing. */
  line: string;
  events: number;
  positions: number;
  recordOnly: number;
  fileOnly: number;
  /** Netting sets whose paper is not what legal declared. */
  terms: number;
};

/** One disagreement as a desk reads it: the trade it is on, what disagrees, what to do, and what a
 * click opens - the deal the file holds, the position the record holds, or the agreement. */
type Divergence = {
  key: string; trade: string; what: string; todo: string;
  deal?: string; position?: { instrument: string; agreement: string }; agreement?: string;
};

const traded = (row: { reference: string | null; instrument: string; client: string | null }) =>
  `${positionName(row)}${row.client ? ` for ${row.client}` : ''}`;

/** The reconcile lists as the rows a desk reads, in their order. */
export function divergences(reconcile: Reconcile): Divergence[] {
  return [
    ...reconcile.in_record_not_in_file.map((row) => ({
      key: `record/${row.instrument}/${row.netting_set}`, trade: traded(row),
      what: `booked under ${row.netting_set}, quantity ${formatNumber(row.quantity)}, and missing `
        + 'from the book file, so nothing prices it',
      todo: 'send its booking again with the same execution reference - the record files '
        + 'nothing new and the file takes it back',
      position: { instrument: row.instrument, agreement: row.netting_set } })),
    ...reconcile.in_file_not_in_record.map((row) => ({
      key: `file/${row.deal_path}`, trade: traded(row),
      what: `in the book file at ${row.deal_path}${row.netting_set ? ` under ${row.netting_set}`
        : ''} and booked by nobody there, so it prices as written`,
      todo: 'book it where it sits, or delete it from the file', deal: row.deal_path })),
    ...(reconcile.terms_mismatch ?? []).map((row) => ({
      key: `paper/${row.deal_path}`, trade: row.agreement,
      what: `the netting set at ${row.deal_path} is off the paper legal declared: `
        + row.fields.join(', '),
      todo: 'runs read the declared terms - bring the set in the file to them, or restate the '
        + 'agreement', agreement: row.agreement })),
  ];
}

/** `held` with `page` merged into it: LSN order, no row twice, the newest `cap` kept; `held` null
 * until a first page is read, so a record not read yet is not one holding none.
 *
 * A page is what came after the `since` the client held, but a restarted service, a reconnect or
 * a cursor answered off a fold that moved can repeat one, so the merge is keyed by LSN rather
 * than trusting the cursor. Never mutates the rows it was handed, and a page carrying nothing
 * answers the rows held THEMSELVES - most beats of a poll bring no event, and a new array for each
 * would repaint a desk that nothing happened to. */
export function mergeActivity(held: ActivityRow[] | null, page: ActivityPage | null,
                              cap: number = STRIP_ROWS): ActivityRow[] | null {
  if (page === null || (page.rows.length === 0 && held !== null)) return held;
  const byLsn = new Map((held ?? []).map((row) => [row.lsn, row]));
  for (const row of page.rows) byLsn.set(row.lsn, row);
  return [...byLsn.values()].sort((a, b) => a.lsn - b.lsn).slice(-cap);
}

/** Whether a doorbell beat is worth a read: one naming a position past the cursor this client
 * holds is, and one at or behind it is a beat already read past - a duplicate, or one that
 * arrived out of order.
 *
 * A BEAT IS NEVER A DELIVERY. The page is what moves the strip and the cursor is what asks for
 * it, so a beat skipped here costs nothing the next beat or the poll does not cover, and a beat
 * acted on twice is one page that merges onto rows already held. */
export function ringsAhead(beat: Doorbell, cursor: number): boolean {
  return beat.lsn > cursor;
}

/** The banner's whole reading. THE LISTS DECIDE AND THE COUNTS NEVER DO: the counts say how far
 * the record has moved past the file's PIN, and every write to the file resets them - including
 * the write that is the other half of a divergence - so a desk gated on them is told `clean`
 * about a trade it deleted through its own verb.
 *
 * `drifted` is what the record moving FORWARD cannot explain: a deal the file holds that nobody
 * booked, a set off its paper, or a trade the record holds and the file has LOST that
 * `positions_behind` cannot account for - its line naming the trades, a lone one with what is
 * wrong. `behind` is the record being ahead - rows it holds that `positions_behind` accounts for.
 * `clean` is every list empty, whatever the counts say, so a fixings policy past the pin lights
 * nothing. */
export function reconcileVerdict(spine: SpineBlock | null | undefined,
                                 reconcile: Reconcile | null): Reconciled {
  const counts = {
    events: spine?.events_behind ?? 0,
    positions: spine?.positions_behind ?? 0,
    recordOnly: reconcile?.in_record_not_in_file.length ?? 0,
    fileOnly: reconcile?.in_file_not_in_record.length ?? 0,
    terms: reconcile?.terms_mismatch?.length ?? 0,
  };
  if (!spine) return { state: 'none', line: '', ...counts };
  if (reconcile && (counts.fileOnly > 0 || counts.terms > 0
      || counts.recordOnly > counts.positions)) {
    const found = divergences(reconcile);
    const named = found.slice(0, 3).map((row) => row.trade).join(', ');
    return {
      state: 'drifted', ...counts,
      line: found.length === 1
        ? `The book file and the record disagree on ${found[0].trade}: ${found[0].what}`
        : `The book file and the record disagree in ${found.length} places: ${named}${
          found.length > 3 ? ` and ${found.length - 3} more` : ''}`,
    };
  }
  if (counts.recordOnly === 0) return { state: 'clean', line: '', ...counts };
  return {
    state: 'behind', ...counts,
    line: `record ahead by ${plural(counts.events, 'event')}`
      + ` (${plural(counts.positions, 'position')})`,
  };
}

/** Where a worklist answer was read: the RECORD's head - the pin and how far the record has moved
 * past it - and the seat it was read as, so any move of the head asks again, an approval
 * included, which moves no position, and so does another seat named; a file write that re-pins
 * at the same head, a tick among them, asks nothing. */
export function worklistAt(spine: SpineBlock | null, seat = ''): string | null {
  return spine === null ? null : `${headOf(spine)}|${seat}`;
}

/** Whether the one worklist read is owed: the head or the seat moved since the answer in hand.
 * Never on the beat, and never on the file alone - the answer compiles the diary on a miss. */
export function wantsWorklist(spine: SpineBlock | null, seat: string,
                              heldAt: string | null): boolean {
  return spine !== null && heldAt !== worklistAt(spine, seat);
}

/** The worklist's lists, in the order the banner reads them and as the words it counts them in. */
export const WAITING: [keyof Worklist['counts'], string, string][] = [
  ['pending', 'ticket', 'to sign'], ['payments', 'payment', 'due unsettled'],
  ['unconfirmed', 'fill', 'to confirm'], ['calls', 'collateral call', 'due'],
  ['unmarked', 'close', 'to mark'], ['rejected', 'rejected trade', 'standing'],
];

/** The banner's one line: every list that holds anything, by its count - the rows are the newest
 * of them - and nothing where all six are empty. A row is a thing somebody has to do: this seat,
 * or - where no seat was named - anybody, which the line says. */
export function worklistLine(worklist: Worklist | null): string {
  const line = worklist === null ? '' : WAITING.filter(([list]) => worklist.counts[list])
    .map(([list, noun, rest]) => `${plural(worklist.counts[list], noun)} ${rest}`).join(' · ');
  return line && worklist?.seat === null
    ? `no seat is named, so this is what waits on anyone: ${line}` : line;
}

/** What the Back Office says in the worklist's place where no record is kept, null on a desk that
 * records: nothing waits on a second seat, and what a booking on the desk's `live` book does. */
export function noWork(recording: boolean, live: boolean): { title: string; hint: string } | null {
  if (recording) return null;
  return {
    title: unrecorded(live, 'nothing waits on a second seat', 'nothing waits on anyone for it'),
    hint: `${live ? 'A booking here writes the book file and stands as written: no' : 'No'} ticket `
      + 'waits to be signed, and no payment, confirmation or collateral call is tracked. A desk '
      + 'that keeps a record lists them here.',
  };
}

/** A close with what it stands over said out loud: a close is superseded by a NEW close rather
 * than corrected in place, so the first one on a market says that instead of naming nothing. */
type CloseRow = MarketClose & { supersedes: string };

const newestFirst = (a: { lsn: number }, b: { lsn: number }) => b.lsn - a.lsn;

/** The markets answer as the panel reads it - each list newest first, every close naming the close
 * of its day it restated. Never mutates the answer it was handed. */
export function marketsView(markets: BookMarkets): {
  closes: CloseRow[]; names: BookMarkets['names']; snapshots: BookMarkets['snapshots'];
} {
  return {
    closes: [...markets.closes].sort(newestFirst).map((close) => ({
      ...close,
      supersedes: close.supersedes_lsn === null
        ? 'the first close of its day' : `supersedes LSN ${close.supersedes_lsn}`,
    })),
    names: [...markets.names].sort(newestFirst),
    snapshots: [...markets.snapshots].sort(newestFirst),
  };
}
