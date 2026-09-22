// The record's three readers, as arithmetic - pure, free of React and of the network, the way
// `desk.ts` and `curves.ts` are for theirs. The strip merges the pages a poll brings, the banner
// reads one verdict off the reconcile lists, and the markets list is the record's own answer
// shaped for reading. The store's two render guards are here too, as the predicates the reducer
// asks, so what decides whether the desk repaints is checkable without a browser.
//
// EVERY READING REFUSES NOTHING, and a desk that records nothing is a STATE rather than an error:
// `spine` comes back null, the verdict is `none`, and the two screens render nothing at all.

import { plural } from './desk';
import type {
  ActivityPage, ActivityRow, BookMarkets, MarketClose, Reconcile, SpineBlock,
} from './types';

/** How many strip rows a client holds. The service caps a page at 200 and this caps what has been
 * MERGED, so a poll running all day cannot grow the array without bound. */
export const STRIP_ROWS = 200;

/** Where the file stands, as one comparable value - so a beat of the poll that moved nothing is
 * the same state and repaints nothing. Null throughout is a desk that records nothing. */
export function pinnedAt(spine: SpineBlock | null): string | null {
  return spine === null
    ? null : `${spine.lsn}/${spine.events_behind}/${spine.positions_behind}`;
}

/** A reconcile answer as one comparable value - the position it was folded at and every row of
 * the three lists. A fetched answer is a NEW object every time, so identity would repaint the
 * whole desk once per fold; what a desk reads is these rows. */
export function reconciledAs(reconcile: Reconcile | null): string | null {
  return reconcile === null ? null : JSON.stringify([
    reconcile.lsn, reconcile.events_behind, reconcile.positions_behind,
    reconcile.in_record_not_in_file.map((row) => [row.instrument, row.quantity]),
    reconcile.in_file_not_in_record.map((row) => [row.instrument, row.deal_path]),
    reconcile.quantity_mismatch.map((row) => [row.instrument, row.record_clips, row.file_nodes]),
  ]);
}

/** What this client holds of the record: where the file stands, the strip as it has been merged,
 * and the reconcile answer with the file etag it was folded against. The store holds exactly
 * this, so the two transitions below are the whole of what a beat can do to it. */
export type RecordHeld = {
  spine: SpineBlock | null;
  rows: ActivityRow[];
  reconcile: Reconcile | null;
  reconciledAt: string | null;
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

/** The record after one reconcile answer - the same object where the lists and the file it was
 * folded against are the ones already in hand. A fetched answer is a NEW object every time, so
 * identity alone would repaint the whole desk once per fold. */
export function reconciledRecord(held: RecordHeld, reconcile: Reconcile,
                                 etag: string | null): RecordHeld {
  return etag === held.reconciledAt && reconciledAs(reconcile) === reconciledAs(held.reconcile)
    ? held : { ...held, reconcile, reconciledAt: etag };
}

/** Whether the banner owes a `/book/reconcile`: the record holding positions the file's pin has
 * not seen, or a FILE that has moved since the answer in hand was fetched - the file being the
 * only thing that can drift under a record standing still, and `positions_behind` the only count
 * that can mean a row. A fold at the head is what this costs, so it is not asked on the beat. */
export function wantsReconcile(spine: SpineBlock | null, heldAt: string | null,
                               etag: string | null): boolean {
  return spine !== null && ((spine.positions_behind ?? 0) > 0 || heldAt !== etag);
}

/** What the banner says about the file it is looking at. `none` is a desk that records nothing. */
export type VerdictState = 'none' | 'clean' | 'behind' | 'drifted';

export type Verdict = {
  state: VerdictState;
  /** The banner's one line, empty where it shows nothing. */
  line: string;
  events: number;
  positions: number;
  recordOnly: number;
  fileOnly: number;
  mismatched: number;
};

/** `held` with `page` merged into it: LSN order, no row twice, the newest `cap` kept.
 *
 * A page is what came after the `since` the client held, but a restarted service, a reconnect or
 * a cursor answered off a fold that moved can repeat one, so the merge is keyed by LSN rather
 * than trusting the cursor. Never mutates the rows it was handed, and a page carrying nothing
 * answers the rows THEMSELVES - most beats of a poll bring no event, and a new array for each
 * would repaint a desk that nothing happened to. */
export function mergeActivity(held: ActivityRow[], page: ActivityPage | null,
                              cap: number = STRIP_ROWS): ActivityRow[] {
  if (page === null || page.rows.length === 0) return held;
  const byLsn = new Map(held.map((row) => [row.lsn, row]));
  for (const row of page.rows) byLsn.set(row.lsn, row);
  return [...byLsn.values()].sort((a, b) => a.lsn - b.lsn).slice(-cap);
}

/** The banner's whole reading. THE LISTS DECIDE AND THE COUNTS NEVER DO: the counts say how far
 * the record has moved past the file's PIN, and every write to the file resets them - including
 * the write that is the other half of a divergence - so a desk gated on them is told `clean`
 * about a trade it deleted through its own verb.
 *
 * `drifted` is what the record moving FORWARD cannot explain: a deal the file holds that nobody
 * booked, a clip count the two disagree on, or a trade the record holds and the file has LOST
 * that `positions_behind` cannot account for. `behind` is the record being ahead - rows it
 * holds that `positions_behind` accounts for. `clean` is all three lists empty, whatever the
 * counts say, so a fixings policy past the pin lights nothing. */
export function reconcileVerdict(spine: SpineBlock | null | undefined,
                                 reconcile: Reconcile | null): Verdict {
  const counts = {
    events: spine?.events_behind ?? 0,
    positions: spine?.positions_behind ?? 0,
    recordOnly: reconcile?.in_record_not_in_file.length ?? 0,
    fileOnly: reconcile?.in_file_not_in_record.length ?? 0,
    mismatched: reconcile?.quantity_mismatch.length ?? 0,
  };
  if (!spine) return { state: 'none', line: '', ...counts };
  if (counts.fileOnly > 0 || counts.mismatched > 0
      || counts.recordOnly > counts.positions) {
    return {
      state: 'drifted', ...counts,
      line: `file and record disagree: ${counts.recordOnly} in record not in file, `
        + `${counts.fileOnly} in file not in record, ${counts.mismatched} quantity mismatches`,
    };
  }
  if (counts.recordOnly === 0) return { state: 'clean', line: '', ...counts };
  return {
    state: 'behind', ...counts,
    line: `record ahead by ${plural(counts.events, 'event')}`
      + ` (${plural(counts.positions, 'position')})`,
  };
}

/** A close with what it stands over said out loud: a close is superseded by a NEW close rather
 * than corrected in place, so the first one on a market says that instead of naming nothing. */
export type CloseRow = MarketClose & { supersedes: string };

const newestFirst = (a: { lsn: number }, b: { lsn: number }) => b.lsn - a.lsn;

/** The markets answer as the panel reads it - each list newest first, every close naming the one
 * it restated. Never mutates the answer it was handed. */
export function marketsView(markets: BookMarkets): {
  closes: CloseRow[]; names: BookMarkets['names']; snapshots: BookMarkets['snapshots'];
} {
  return {
    closes: [...markets.closes].sort(newestFirst).map((close) => ({
      ...close,
      supersedes: close.supersedes_lsn === null
        ? 'the first close on this market' : `supersedes LSN ${close.supersedes_lsn}`,
    })),
    names: [...markets.names].sort(newestFirst),
    snapshots: [...markets.snapshots].sort(newestFirst),
  };
}
