import { useEffect, useRef, useState } from 'react';
import { failure, getBookAgreements, getBookEntities, getMarkedDays, getPnl } from '../api';
import { EmptyState } from '../components/EmptyState';
import {
  behind, explained, factorLabel, FIGURES, pnlQuery, positionLabel, residualShare,
  type MarkedDays, type Pnl,
} from '../pnl';
import type { Agreement, Entity } from '../positions';
import { useApp } from '../state';
import { formatNumber } from '../tokens';

/** A figure as a cell: a number signed in colour, and a dash for what nobody can know. */
function Figure({ value }: { value: number | null | undefined }) {
  if (value === null || value === undefined) return <td className="n">—</td>;
  return <td className={`n${value < 0 ? ' neg' : ''}`}>{formatNumber(value)}</td>;
}

/** The desk's P&L between two marked days, or from the last marks to the book as it stands - per
 * position and in total, narrowed by portfolio, agreement or client - and, asked for, the explain
 * of what the held positions made: the carry, the market per risk factor, and the residual.
 *
 * A READING OF THE RECORD. The marks are the closes' own numbers and nothing here writes; a figure
 * nobody can know is named rather than shown as a zero, and the explain is three more valuations
 * run only when it is asked for. */
export function PnlView() {
  const { state } = useApp();
  const live = state.source?.kind === 'book';
  const etag = state.source?.kind === 'book' ? state.source.etag : undefined;
  const [days, setDays] = useState<MarkedDays | null>(null);
  const [paper, setPaper] = useState<{ entities: Entity[]; agreements: Agreement[] }>(
    { entities: [], agreements: [] });
  const [start, setStart] = useState('');
  const [end, setEnd] = useState('');
  const [scope, setScope] = useState({ portfolio: '', agreement: '', client: '' });
  // a path is typed a letter at a time and read once, where the reader commits it
  const [path, setPath] = useState('');
  const commit = () => { if (path !== scope.portfolio) setScope({ ...scope, portfolio: path }); };
  const [explain, setExplain] = useState(false);
  const [answer, setAnswer] = useState<Pnl | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  // the book as it stood when the answer in hand was read, and a count the reader bumps to read
  // again: a window ending now values the whole book, so a write offers a re-read, never runs one
  const current = useRef(etag);
  current.current = etag;
  const [readAt, setReadAt] = useState<string | undefined>(undefined);
  const [again, setAgain] = useState(0);

  useEffect(() => {
    if (!live) return;
    getMarkedDays().then(setDays).catch((reason) => setError(failure(reason).error));
    Promise.all([getBookEntities(), getBookAgreements()])
      .then(([legal, signed]) => setPaper(
        { entities: legal.entities, agreements: signed.agreements }))
      .catch(() => undefined);
  }, [live]);

  // read again whenever the window or the scope moves, or the reader asks - the explain only when
  // asked for
  useEffect(() => {
    if (!live || !days?.days.length) return;
    let wanted = true;
    const asked = current.current;
    setLoading(true);
    getPnl(pnlQuery(start, end, scope, explain))
      .then((read) => { if (wanted) { setAnswer(read); setReadAt(asked); setError(null); } })
      .catch((reason) => { if (wanted) setError(failure(reason).error); })
      .finally(() => { if (wanted) setLoading(false); });
    return () => { wanted = false; };
  }, [live, days, start, end, scope, explain, again]);

  if (!live) {
    return (
      <EmptyState
        title="The P&L is a reading of the live book's record."
        hint="This document was opened from a file. Start the service with --book and a spine home, mark a close, and the desk's P&L appears here."
      />
    );
  }
  if (days && !days.days.length) {
    return (
      <EmptyState
        title="No close has been marked yet."
        hint={`The P&L runs between marked closes on '${days.market}'. Declare a close and mark the book (POST /book/marks) - the first marks are where the P&L starts.`}
      />
    );
  }
  if (!answer) {
    return error ? (
      <div className="main"><div className="panel"><div className="error-box">{error}</div></div></div>
    ) : <EmptyState title="Reading the P&L…" hint="The marks of both days, the trades and the cash between them." />;
  }

  const currency = answer.currency ?? '';
  const later = (days?.days ?? []).filter((marked) => marked.day > answer.start.day);
  return (
    <div className="main">
      <div className="panel">
        <div className="blotterbar">
          <label className="hint">From{' '}
            <select value={start || answer.start.day} onChange={(event) => setStart(event.target.value)}>
              {(days?.days ?? []).map((marked) => (
                <option key={marked.day} value={marked.day}>{marked.day}</option>))}
            </select>
          </label>
          <label className="hint">to{' '}
            <select value={end} onChange={(event) => setEnd(event.target.value)}>
              <option value="">now</option>
              {later.map((marked) => <option key={marked.day} value={marked.day}>{marked.day}</option>)}
            </select>
          </label>
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
              <option key={row.entity} value={row.entity}>{row.name}</option>))}
          </select>
          <button className={`ghost${explain ? ' on' : ''}`} onClick={() => setExplain(!explain)}>
            explain
          </button>
          <span className="spacer" />
          {!loading && behind(end, readAt, etag) && (
            <button className="ghost" onClick={() => setAgain(again + 1)}>
              the book has moved - read again
            </button>
          )}
          {loading && <span className="hint">reading…</span>}
        </div>
        {error && <div className="error-box">{error}</div>}

        <div className="stats">
          {(['pnl', 'realised', 'unrealised'] as const).map((figure) => (
            <div className="stat" key={figure}>
              <div className="label">{figure === 'pnl' ? 'P&L' : figure[0].toUpperCase() + figure.slice(1)}</div>
              <div className={`value${(answer.total[figure] ?? 0) < 0 ? ' neg' : ''}`}>
                {answer.total[figure] === null ? '—' : formatNumber(answer.total[figure] as number)}
                {currency && <span className="unit">{currency}</span>}
              </div>
            </div>
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
                    <Figure key={figure.key} value={row[figure.key] as number | null} />))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

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
          <div className="stat" key={line.label}>
            <div className="label">{line.label}</div>
            <div className={`value${(line.value ?? 0) < 0 ? ' neg' : ''}`}>
              {line.value === null ? '—' : formatNumber(line.value)}
              {currency && <span className="unit">{currency}</span>}
            </div>
          </div>
        ))}
        <div className="stat">
          <div className="label">Residual share</div>
          <div className="value">{share === null ? '—' : `${formatNumber(100 * share)}%`}</div>
        </div>
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
                <Figure value={row.delta} /><Figure value={row.start} /><Figure value={row.end} />
                <Figure value={row.move} /><Figure value={row.pnl} />
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
