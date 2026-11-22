import { useRef, useState } from 'react';
import { getMarkedDays, getPnl } from '../api';
import { declaredOn, unmarked } from '../client';
import { Amount, Gated, Stat } from '../components/Desk';
import { EmptyState } from '../components/EmptyState';
import {
  behind, explained, factorLabel, FIGURES, noPnl, noWindow, opensOn, pnlQuery, pnlRead,
  positionLabel, residualShare, type Pnl,
} from '../pnl';
import {
  grantsRead, noPositions, pnlHeld, readAs, runHeld, spineOf, useApp, useRead,
} from '../state';
import { formatNumber } from '../tokens';
import { entityText } from '../desk';
import { NO_PAPER } from '../positions';

/** What a figure nobody can know says on its dash. */
const UNKNOWN = 'not known: the lines above name what is missing';

/** The desk's P&L between two marked days, or from the last marks to the book as it stands - per
 * position and in total, narrowed by portfolio, agreement or client - and, asked for, the explain
 * of what the held positions made: the carry, the market per risk factor, and the residual.
 *
 * A READING OF THE RECORD. The marks are the closes' own numbers and nothing here writes; a figure
 * nobody can know is named rather than shown as a zero, and the explain is three more valuations
 * run only when it is asked for. */
export function PnlView() {
  const { state } = useApp();
  const recording = spineOf(state) !== null;
  const etag = state.source?.kind === 'book' ? state.source.etag : undefined;
  const [start, setStart] = useState('');
  const [end, setEnd] = useState('');
  const [scope, setScope] = useState({ portfolio: '', agreement: '', client: '' });
  // a path is typed a letter at a time and read once, where the reader commits it
  const [path, setPath] = useState('');
  const commit = () => { if (path !== scope.portfolio) setScope({ ...scope, portfolio: path }); };
  const [explain, setExplain] = useState(false);
  // the book as it stood when the answer in hand was read, and a count the reader bumps to read
  // again: a window ending now values the whole book, so a write offers a re-read, never runs one
  const current = useRef(etag);
  current.current = etag;
  const [again, setAgain] = useState(0);

  // no market designated, or no close declared on it, is known before anything is asked
  const closed = unmarked(spineOf(state));
  const marked = useRead(recording && !closed ? getMarkedDays : null, [recording, closed?.title]);
  const days = marked.data;
  const paper = state.paper.data ?? NO_PAPER;
  // a what-if's work - a read to now, an explain - only of a seat that may ask one: a seat held
  // from it opens on the last two marked days, and reads no window where fewer are marked
  const toNow = pnlHeld(state, '', false);
  const opens = opensOn(days?.days ?? [], toNow !== null);
  const [from, to] = [start || opens?.start || '', end || opens?.end || ''];
  const hold = pnlHeld(state, to, explain);
  // read again whenever the seat, the window or the scope moves, or the reader asks, with the
  // book it was read at
  const query = pnlQuery(from, to, scope, explain);
  const waits = runHeld(state, hold);
  const read = readAs(hold, useRead(recording && days?.days.length && !waits ? async () => ({
    asked: current.current, answer: await getPnl(query) }) : null,
  [recording, days, pnlRead(state.seat, query), again, waits]));
  const { answer, asked: readAt } = read.data ?? {};
  const { loading } = read;
  const error = marked.error ?? read.error;

  const none = noPnl(recording, state.source?.kind === 'book') ?? closed;
  if (none) return <EmptyState {...none} />;
  if (days && !days.days.length) {
    // the verb the declaration demands under the map in force, where the seat's grants are read
    const verb = grantsRead(state)?.read?.demands.official_close_declared;
    return (
      <EmptyState
        title="No close has been marked yet."
        hint={`The P&L runs between marked closes on '${days.market}'. ${declaredOn(verb)
        }: the first marks are where the P&L starts.`}
      />
    );
  }
  if (!answer) {
    const held = read.held && !marked.error;
    return error ? (
      <div className="main"><div className="panel">
        <div className={marked.error ? 'error-box' : read.tone}>
          {held && !opens ? noWindow(error, days?.days ?? []) : error}</div>
      </div></div>
    ) : <EmptyState title="Reading the P&L…" hint="The marks of both days, the trades and the cash between them." />;
  }

  const currency = answer.currency ?? '';
  const later = (days?.days ?? []).filter((marked) => marked.day > answer.start.day);
  return (
    <div className="main">
      <div className="panel">
        <div className="blotterbar">
          <label className="hint">From{' '}
            <select value={from || answer.start.day} onChange={(event) => setStart(event.target.value)}>
              {(days?.days ?? []).slice(0, toNow ? -1 : undefined).map((marked) => (
                <option key={marked.day} value={marked.day}>{marked.day}</option>))}
            </select>
          </label>
          <label className="hint">to{' '}
            <select value={to} onChange={(event) => setEnd(event.target.value)}>
              {!toNow && <option value="">now</option>}
              {later.map((marked) => <option key={marked.day} value={marked.day}>{marked.day}</option>)}
            </select>
          </label>
          <Gated label="to now, or explained" hold={toNow} />
          <input placeholder="portfolio path" value={path}
                 onChange={(event) => setPath(event.target.value)} onBlur={commit}
                 onKeyDown={(event) => { if (event.key === 'Enter') commit(); }} />
          <select value={scope.agreement}
                  onChange={(event) => setScope({ ...scope, agreement: event.target.value })}>
            <option value="">every agreement</option>
            {paper.agreements.map((row) => (
              <option key={row.agreement} value={row.agreement}>{row.agreement}</option>))}
          </select>
          <select value={scope.client}
                  onChange={(event) => setScope({ ...scope, client: event.target.value })}>
            <option value="">every client</option>
            {paper.entities.map((row) => (
              <option key={row.entity} value={row.entity}>{entityText(paper.entities, row.entity)}
              </option>))}
          </select>
          {!toNow && (
            <button className={`ghost${explain ? ' on' : ''}`} onClick={() => setExplain(!explain)}>
              explain
            </button>
          )}
          <span className="spacer" />
          {!loading && behind(to, readAt, etag) && (
            <button className="ghost" onClick={() => setAgain(again + 1)}>
              the book has moved - read again
            </button>
          )}
          {loading && <span className="hint">reading…</span>}
        </div>
        {error && <div className="error-box">{error}</div>}

        <div className="stats">
          {(['pnl', 'realised', 'unrealised'] as const).map((figure) => (
            <Stat key={figure} label={figure === 'pnl' ? 'P&L' : figure[0].toUpperCase() + figure.slice(1)}
                  value={answer.total[figure] ?? null} currency={currency} />
          ))}
          <div className="stat">
            <div className="label">Window</div>
            <div className="value">
              {answer.start.day} → {answer.end.live ? 'now' : answer.end.day}
            </div>
          </div>
          <div className="stat">
            <div className="label">Complete</div>
            <div className="value">{answer.complete ? 'yes' : 'no'}</div>
          </div>
        </div>
        <div className="hint">Realised at {answer.realised_method}.</div>
        {answer.unknown.map((named) => (
          <div className="banner" key={`${named.instrument}:${named.what}`}>{named.what}</div>))}

        {!answer.rows.length ? <div className="note">{noPositions(state)}.</div> : (
        <div className="tablewrap">
          <table className="data desk">
            <thead>
              <tr>
                <th>Position</th><th>Portfolio</th><th>Agreement</th><th>Quantity</th>
                {FIGURES.map((figure) => <th key={figure.key}>{figure.label}</th>)}
              </tr>
            </thead>
            <tbody>
              {answer.rows.map((row) => (
                <tr key={`${row.instrument}:${row.agreement}:${row.portfolio}`}>
                  <td title={row.instrument}>{positionLabel(row)}</td>
                  <td>{row.portfolio}</td>
                  <td>{row.agreement}</td>
                  <td className="n">
                    {formatNumber(row.quantity_start)} → {formatNumber(row.quantity_end)}
                  </td>
                  {FIGURES.map((figure) => (
                    <Amount key={figure.key} value={row[figure.key] as number | null} kind="money"
                            absent={UNKNOWN} />))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        )}

        {answer.explain && <Explained explain={answer.explain} currency={currency} />}
      </div>
    </div>
  );
}

/** What the held positions made, taken apart: the carry, the market per risk factor off the
 * start's own sensitivities, and the residual neither explains. */
function Explained({ explain, currency }: { explain: NonNullable<Pnl['explain']>; currency: string }) {
  const share = residualShare(explain);
  return (
    <>
      <h3>Explain</h3>
      <div className="stats">
        {explained(explain).map((line) => (
          <Stat key={line.label} label={line.label} value={line.value} currency={currency} />
        ))}
        <Stat label="Residual share" value={share === null ? null : 100 * share} kind="percent" />
      </div>
      <div className="hint">
        The carry is the start's book rolled to the end's day at the start's own quotes; the
        market is each risk factor's move times the start's sensitivity to it. Reserves are not
        carried.
      </div>
      {explain.note && <div className="banner">{explain.note}</div>}
      {explain.unknown.map((named) => (
        <div className="banner" key={named.what}>{named.what}</div>))}
      <div className="tablewrap">
        <table className="data desk">
          <thead>
            <tr><th>Risk factor</th><th>Sensitivity</th><th>Start</th><th>End</th><th>Move</th><th>P&L</th></tr>
          </thead>
          <tbody>
            {explain.factors.map((row) => (
              <tr key={factorLabel(row)}>
                <td>{factorLabel(row)}</td>
                <Amount value={row.delta} kind="money" />
                {[row.start, row.end, row.move].map((value, at) => (
                  <Amount key={at} value={value} kind="plain" absent={UNKNOWN} />))}
                <Amount value={row.pnl} kind="money" absent={UNKNOWN} />
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
