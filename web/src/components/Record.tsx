import { useEffect, useState } from 'react';
import { SEAT, TOKEN, failure, getBookReconcile, getWorklist, stored } from '../api';
import { stampText } from '../desk';
import { positionId } from '../positions';
import {
  divergences, reconcileVerdict, wantsReconcile, wantsWorklist, worklistAt, worklistLine,
} from '../spine';
import { opened, useApp, type AppState } from '../state';
import type { Reconcile, Worklist } from '../types';

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
      {reconcile && verdict.state === 'drifted' && (
        <button className="ghost" onClick={() => setOpen(!open)}>
          {open ? 'hide' : 'show'} which, and what to do
        </button>
      )}
      <span className="spacer" />
      <span className="mono">file pinned at LSN {spine?.lsn ?? '—'}</span>
      {open && reconcile && <Divergences reconcile={reconcile} />}
    </div>
  );
}

/** What waits on this seat, beside the reconcile banner: six lists read off what stands, so a row
 * leaves the moment its fact lands, and a rejected trade is named for as long as it stands. Asked
 * where the record's head moved (`wantsWorklist`), never on the beat or the file alone - the answer
 * compiles the diary - and nothing at all where nothing waits; a read that failed says so.
 */
export function WorklistBanner() {
  const { state } = useApp();
  const [open, setOpen] = useState(false);
  const [held, setHeld] = useState<{
    worklist: Worklist | null; error: string | null; at: string | null }>(
    { worklist: null, error: null, at: null });
  const spine = recorded(state);
  const at = worklistAt(spine);

  useEffect(() => {
    if (!wantsWorklist(spine, held.at)) return;
    let live = true;
    getWorklist()
      .then((worklist) => { if (live) setHeld({ worklist, error: null, at }); })
      .catch((error) => {
        if (live) setHeld({ worklist: null, error: failure(error).error, at });
      });
    return () => { live = false; };
    // the answer's own position is not a dependency: it moves BECAUSE of this fetch
  }, [at]);

  if (held.error) {
    return <div className="recordbar drifted"><span>worklist unread: {held.error}</span></div>;
  }
  const line = worklistLine(held.worklist);
  if (!line || !held.worklist) return null;
  const { date: _, counts: __, ...lists } = held.worklist;
  return (
    <div className="recordbar behind">
      <span>{line}</span>
      <button className="ghost" onClick={() => setOpen(!open)}>
        {open ? 'hide' : 'show'} what waits
      </button>
      {open && (
        <div className="rows">
          <table className="data">
            <tbody>
              {Object.values(lists).flat().map((row) => (
                <tr key={`${row.kind}/${row.key}/${row.lsn}`}>
                  <td>{row.kind}</td>
                  <td>{row.what}</td>
                  <td className="mono" title={row.key}>{row.key.slice(0, 12)}</td>
                  <td>{row.since ?? ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

/** Who this browser acts as where the service checks no token: the seat a write is filed under
 * and a read narrowed to, kept in this browser. A stored sign-in token names the seat instead. */
export function SeatControl() {
  const [seat, setSeat] = useState(stored(SEAT));
  if (stored(TOKEN)) return <span className="hint">signed in</span>;
  return (
    <input className="seat" placeholder="seat" value={seat} title="the seat this desk acts as"
           onChange={(event) => {
             setSeat(event.target.value);
             try {
               localStorage.setItem(SEAT, event.target.value);
             } catch { /* storage blocked: the seat lasts the page */ }
           }} />
  );
}

/** The disagreements as a desk reads them: the trade, what is wrong, what to do - each row opening
 * the deal the file holds, the position the record holds, or the agreement. */
function Divergences({ reconcile }: { reconcile: Reconcile }) {
  const { state, dispatch } = useApp();
  const positions = state.paper.data?.positions ?? [];
  return (
    <div className="rows">
      <table className="data">
        <tbody>
          {divergences(reconcile).map((row) => {
            const held = row.position && positions.find((position) => (
              position.instrument === row.position!.instrument
              && position.agreement === row.position!.agreement));
            return (
              <tr key={row.key} className="pick"
                  onClick={() => opened(dispatch, held ? positionId(held) : row.deal,
                                        row.agreement ?? row.position?.agreement)}>
                <td>{row.trade}</td>
                <td>{row.what}</td>
                <td>{row.todo}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/** The record's strip at the foot of the frame: one line per event, newest first, as far back as
 * this client has merged - each the service's sentence for what it is about and who did it, and a
 * row naming a deal, an agreement or an entity opens it. Every type shows, including one this
 * deployment has no sentence for, which renders its own name. Collapsed it is the last thing that
 * happened; expanded it is the sequence.
 */
export function ActivityStrip() {
  const { state, dispatch } = useApp();
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
          <b className="mono">LSN {newest.lsn}</b> {newest.by}: {newest.summary}
          {' · '}{stampText(newest.record_time)}
        </span>
      ) : <span>nothing has been recorded here yet</span>}
      <span className="spacer" />
      <span className="mono">{rows.length} rows held</span>
      {open && (
        <div className="rows">
          <table className="data">
            <tbody>
              {rows.map((row) => {
                const opens = row.deal_path ?? row.agreement ?? row.entity;
                return (
                  <tr key={row.lsn} className={opens ? 'pick' : ''}
                      title={opens ? `open ${opens}` : undefined}
                      onClick={() => opened(dispatch, row.deal_path, row.agreement ?? row.entity)}>
                    <td className="n mono">{row.lsn}</td>
                    {/* the writer's own clock, in UTC; the title is when the fact is TRUE, which
                        is null where the fact carries no truth-time of its own */}
                    <td title={row.effective_time ?? 'no truth-time of its own'}>
                      {stampText(row.record_time)}
                    </td>
                    <td>{row.by}</td>
                    <td>{row.summary}</td>
                    <td>{row.book ?? ''}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
