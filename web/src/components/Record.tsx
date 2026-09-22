import { useEffect, useState } from 'react';
import { getBookReconcile } from '../api';
import { stampText } from '../desk';
import { reconcileVerdict, wantsReconcile } from '../spine';
import { useApp, type AppState } from '../state';
import type { Reconcile } from '../types';

/** The record's own block, and null where this client is not looking at the LIVE BOOK: the record
 * is the deployment's and says nothing about a copy opened from somebody's disk. */
function recorded(state: AppState) {
  return state.source?.kind === 'book' ? state.record.spine : null;
}

/** Where the book FILE stands against the record, in the app frame beside the book's own header.
 *
 * Nothing at all when the two agree or when this desk records nothing - a banner that said `clean`
 * on every screen would be a banner nobody reads. `behind` is a lag the next write closes;
 * `drifted` is the two disagreeing about a trade, which no beat of the poll will close.
 */
export function ReconcileBanner() {
  const { state, dispatch } = useApp();
  const [open, setOpen] = useState(false);
  const spine = recorded(state);
  const { reconcile, reconciledAt } = state.record;
  const verdict = reconcileVerdict(spine, reconcile);
  const etag = state.source?.kind === 'book' ? state.source.etag : null;
  const records = spine !== null;
  const positions = spine?.positions_behind ?? 0;

  // WHAT the two disagree about costs a fold at the head: asked for where the record holds
  // positions the pin has not seen or the FILE has moved under the answer in hand, never on the
  // beat, and never on `events_behind`, which a fixings policy moves and no row follows.
  useEffect(() => {
    if (!wantsReconcile(spine, reconciledAt, etag)) return;
    let live = true;
    getBookReconcile()
      .then((answer) => { if (live) dispatch({ type: 'RECONCILED', reconcile: answer, etag }); })
      .catch(() => undefined);
    return () => { live = false; };
    // the answer's own etag is not a dependency: it moves BECAUSE of this fetch
  }, [records, positions, etag, dispatch]);

  if (verdict.state === 'none' || verdict.state === 'clean') return null;

  return (
    <div className={`recordbar ${verdict.state}`}>
      <span>{verdict.line}</span>
      {reconcile && (
        <button className="ghost" onClick={() => setOpen(!open)}>
          {open ? 'hide' : 'show'} what the two hold
        </button>
      )}
      <span className="spacer" />
      <span className="mono">file pinned at LSN {spine?.lsn ?? '—'}</span>
      {open && reconcile && <Divergences reconcile={reconcile} />}
    </div>
  );
}

/** The three lists as one table, each row named by its INSTRUMENT ADDRESS - the identity both
 * sides are compared on, so a renamed deal is two rows rather than a clean reconcile. */
function Divergences({ reconcile }: { reconcile: Reconcile }) {
  return (
    <div className="rows">
      <table className="data">
        <tbody>
          {reconcile.in_record_not_in_file.map((row) => (
            <Divergence key={`record-${row.instrument}`} what="in the record, not in the file"
                        instrument={row.instrument} detail={row.netting_set ?? ''}
                        number={row.quantity} />
          ))}
          {reconcile.in_file_not_in_record.map((row) => (
            <Divergence key={`file-${row.deal_path}`} what="in the file, nobody booked"
                        instrument={row.instrument}
                        detail={`${row.reference ?? '(no reference)'} at ${row.deal_path}`} />
          ))}
          {reconcile.quantity_mismatch.map((row) => (
            <Divergence key={`clips-${row.instrument}`} what="the two count different clips"
                        instrument={row.instrument}
                        detail={`${row.file_nodes} in the file, ${row.record_clips} recorded`}
                        number={row.record_quantity} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Divergence(props: {
  what: string; instrument: string; detail: string; number?: number;
}) {
  return (
    <tr>
      <td>{props.what}</td>
      <td className="mono" title={props.instrument}>{props.instrument.slice(0, 12)}</td>
      <td>{props.detail}</td>
      <td className="n">{props.number === undefined ? '' : props.number}</td>
    </tr>
  );
}

/** The record's strip at the foot of the frame: one line per event, newest first, as far back as
 * this client has merged.
 *
 * It opens no body on the way here - the fold reads envelopes alone - so every type shows,
 * including one this deployment has no sentence for, which renders its own name. Collapsed it is
 * the last thing that happened; expanded it is the sequence.
 */
export function ActivityStrip() {
  const { state } = useApp();
  const [open, setOpen] = useState(false);
  if (recorded(state) === null) return null;

  const rows = [...state.record.rows].reverse();
  const newest = rows[0];

  return (
    <div className="recordbar foot">
      <button className="ghost" onClick={() => setOpen(!open)}>
        {open ? '▾' : '▸'} the record
      </button>
      {newest ? (
        <span>
          <b className="mono">LSN {newest.lsn}</b> {newest.summary}
          {' · '}{newest.actor}{' · '}{stampText(newest.record_time)}
        </span>
      ) : <span>nothing has been recorded here yet</span>}
      <span className="spacer" />
      <span className="mono">{rows.length} rows held</span>
      {open && (
        <div className="rows">
          <table className="data">
            <tbody>
              {rows.map((row) => (
                <tr key={row.lsn}>
                  <td className="n mono">{row.lsn}</td>
                  {/* the writer's own clock, in UTC; the title is when the fact is TRUE, which is
                      null where the fact carries no truth-time of its own */}
                  <td title={row.effective_time ?? 'no truth-time of its own'}>
                    {stampText(row.record_time)}
                  </td>
                  <td>{row.actor}</td>
                  <td>{row.summary}</td>
                  <td>{row.book ?? ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
