import { useEffect, useState } from 'react';
import { SEAT, TOKEN, failure, getBookReconcile, getWorklist, stored } from '../api';
import { stampText } from '../desk';
import { positionId } from '../positions';
import {
  WAITING, divergences, reconcileAt, reconcileVerdict, wantsReconcile, wantsWorklist, worklistAt,
  worklistLine,
} from '../spine';
import { opened, useApp, type AppState } from '../state';
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
  const at = reconcileAt(spine, etag);

  // WHAT the two disagree about costs a fold at the head: asked where the record's head or the
  // FILE moved under the answer in hand - a declaration among them - and never on the beat
  useEffect(() => {
    if (!wantsReconcile(spine, reconciledAt, etag)) return;
    let live = true;
    getBookReconcile()
      .then((answer) => { if (live) dispatch({ type: 'RECONCILED', reconcile: answer, at }); })
      .catch(() => undefined);
    return () => { live = false; };
    // where the answer in hand was read is not a dependency: it moves BECAUSE of this fetch
  }, [at, dispatch]);

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

/** What waits on this seat, beside the reconcile banner, read ONCE for every screen that counts
 * it: six lists read off what stands, so a row leaves the moment its fact lands, held in the store
 * and asked again where the record's head or the seat moved (`wantsWorklist`), never on the beat
 * or the file alone - the answer compiles the diary. Nothing at all where nothing waits; a read
 * that failed says so. A row opens where it is cleared - the Back Office, the P&L, the trade. */
export function WorklistBanner() {
  const { state, dispatch } = useApp();
  const [open, setOpen] = useState(false);
  const spine = recorded(state);
  const { worklist, worklistAt: heldAt, worklistError } = state.record;
  const at = worklistAt(spine, state.seat);

  useEffect(() => {
    if (!wantsWorklist(spine, state.seat, heldAt)) return;
    let live = true;
    getWorklist()
      .then((read) => {
        if (live) dispatch({ type: 'WORKLISTED', at, worklist: read, error: null });
      })
      .catch((error) => {
        if (live) dispatch({ type: 'WORKLISTED', at, worklist: null, error: failure(error).error });
      });
    return () => { live = false; };
    // the answer's own position is not a dependency: it moves BECAUSE of this fetch
  }, [at, dispatch]);

  if (worklistError) {
    return <div className="recordbar drifted"><span>worklist unread: {worklistError}</span></div>;
  }
  const line = worklistLine(worklist);
  if (!line || !worklist) return null;
  const clearedOn: Record<string, string> = { unmarked: 'pnl' };
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
              {WAITING.flatMap(([list]) => worklist[list].map((row) => (
                <tr key={`${row.kind}/${row.key}/${row.lsn}`} className="pick"
                    onClick={() => (list === 'rejected' ? opened(dispatch, row.deal_path)
                      : dispatch({ type: 'TAB', tab: clearedOn[list] ?? 'backoffice' }))}>
                  <td>{row.kind}</td>
                  <td>{row.what}</td>
                  <td>{stampText(row.since)}</td>
                </tr>
              )))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

/** Who this browser acts as where the service checks no token: the seat a write is filed under
 * and a read narrowed to, kept in this browser and in the store, so every read waiting on it asks
 * again. A stored sign-in token names the seat instead. */
export function SeatControl() {
  const { state, dispatch } = useApp();
  if (stored(TOKEN)) return <span className="hint">signed in</span>;
  return (
    <input className="seat" placeholder="seat" value={state.seat}
           title="the seat this desk acts as"
           onChange={(event) => {
             try {
               localStorage.setItem(SEAT, event.target.value);
             } catch { /* storage blocked: the seat lasts the page */ }
             dispatch({ type: 'SEAT', seat: event.target.value });
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
