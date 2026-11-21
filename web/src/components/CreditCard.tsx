// Counterparty survival curve display and spread editor.

import { useEffect, useState } from 'react';
import { failure, stateSurvival } from '../api';
import {
  SURVIVAL, refusedAt, spreadForm, spreadRequest, spreads, survivalCurve, type SpreadForm,
} from '../desk';
import { actingSeat, heldBack, useApp, written } from '../state';
import { curveOf, formatNumber, tenorLabel, token } from '../tokens';
import { Whose } from './Whose';

/** Prefix identifying a credit page id in the Curves navigator. */
export const CREDIT = 'credit:';

const HOLDS = {
  record: 'On this desk a curve is written for an entity the record declares, by a seat holding '
    + 'mark over the book.',
  plain: 'This desk keeps no record: a curve is written for any counterparty named.',
};

/** Tenor, spread (bp) and survival probability table for a counterparty credit curve. */
export function SpreadTable({ rows }: { rows: [string, unknown, number][] }) {
  return (
    <table className="data">
      <thead>
        <tr><th>Tenor</th><th className="n">Spread</th><th className="n">Survival</th></tr>
      </thead>
      <tbody>
        {rows.map(([tenor, spread, survival], at) => (
          <tr key={at}>
            <td>{tenor}</td>
            <td className="n">{typeof spread === 'number' ? formatNumber(spread, 'basis')
              : String(spread)}</td>
            <td className="n">
              {isFinite(survival) ? formatNumber(100 * survival, 'percent') : '—'}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** Counterparty credit curve panel and flat or per-tenor spread entry form. */
export function CreditCard({ id, title, factor, base, convention, recording }: {
  id?: string; title: string; factor?: unknown; base: string; convention: number;
  recording: boolean;
}) {
  const { state, dispatch } = useApp();
  const hold = heldBack(state, 'mark');
  const [name, setName] = useState(id ?? '');
  const [form, setForm] = useState(() => spreadForm(factor, convention));
  const [said, setSaid] = useState<{ at: string; text: string } | null>(null);
  const [saving, setSaving] = useState(false);
  const incoming = JSON.stringify(factor ?? null);
  useEffect(() => setForm(spreadForm(factor, convention)), [incoming]);

  const request = spreadRequest(name, form);
  const knots = survivalCurve(request, base).slice(1);
  const typed = Array.isArray(request.spread) ? request.spread
    : [{ tenor: '1Y', spread: request.spread }];
  const curve = curveOf(token(factor, 'Curve'));
  const recovery = Number(token(factor, 'Recovery_Rate') ?? convention);
  const edit = (change: Partial<SpreadForm>) => { setSaid(null); setForm({ ...form, ...change }); };
  const row = (at: number, cell: Partial<SpreadForm['rows'][number]>) =>
    edit({ rows: form.rows.map((held, i) => (i === at ? { ...held, ...cell } : held)) });
  const refusal = (at: string) => said?.at === at && <span className="refusal">{said.text}</span>;

  async function write() {
    setSaving(true);
    try {
      const answer = await stateSurvival(request);
      await written(dispatch, answer);
      setSaid({ at: 'done', text: `wrote ${answer.factor}, its last spread run on to ${
        formatNumber(answer.horizon)} years` });
      if (id) return;
      setName('');
      setForm(spreadForm(undefined, convention));
      dispatch({ type: 'PICK', screen: 'curves', id: CREDIT + request.counterparty });
    } catch (error) {
      const text = failure(error).error;
      setSaid({ at: refusedAt(text, request), text });
    } finally {
      setSaving(false);
    }
  }

  return (
    <>
      <section className="card credit">
        <h3>{title} — {SURVIVAL}.{name || '…'}</h3>
        <div className="pager">
          {id && <Whose path={['Price Factors', `${SURVIVAL}.${id}`]} />}
          <span>{recording ? HOLDS.record : HOLDS.plain}</span>
        </div>
        {curve ? (
          <>
            <div className="hint">As the book holds it, at a recovery of{' '}
              {formatNumber(100 * recovery, 'percent')}:</div>
            <SpreadTable rows={spreads(curve.data, recovery).map(([tenor, spread, survival]) =>
              [tenorLabel(tenor), spread * 1e4, survival])} />
          </>
        ) : id && (
          <div className="note">No credit curve yet: a CVA against this counterparty has no default
            to read, and the first trade under a netting set naming it is refused until it has
            one.</div>
        )}
      </section>
      <section className="card credit">
        <h3>{curve ? 'Restate the curve' : 'State the curve'}</h3>
        {hold ? <div className="note">{hold}</div> : <>
        {!id && (
          <div className="statusrow">
            <input placeholder={recording ? 'an entity the record declares' : 'counterparty'}
                   value={name} onChange={(event) => {
                     setSaid(null); setName(event.target.value);
                   }} />
            {refusal('counterparty')}
          </div>
        )}
        <div className="statusrow">
          <span>recovery</span>
          <input value={form.recovery} inputMode="decimal"
                 onChange={(event) => edit({ recovery: event.target.value })} />
          <span className="hint">the factor's convention is {String(convention)}</span>
          {refusal('recovery')}
        </div>
        <div className="statusrow">
          <label><input type="radio" checked={form.flat} onChange={() => edit({ flat: true })} />
            {' '}one flat spread</label>
          <label><input type="radio" checked={!form.flat} onChange={() => edit({ flat: false })} />
            {' '}a spread per tenor</label>
          {form.flat && (
            <>
              <input value={form.rows[0].spread} inputMode="decimal" placeholder="spread"
                     onChange={(event) => row(0, { spread: event.target.value })} />
              <span>bp</span>
            </>
          )}
          {refusal('spread')}
        </div>
        {!form.flat && (
          <>
            <table className="data">
              <thead><tr><th>Tenor</th><th>Spread, bp</th><th /></tr></thead>
              <tbody>
                {form.rows.map((held, at) => (
                  <tr key={at}>
                    <td><input value={held.tenor} placeholder="5Y"
                               onChange={(event) => row(at, { tenor: event.target.value })} /></td>
                    <td><input value={held.spread} inputMode="decimal"
                               onChange={(event) => row(at, { spread: event.target.value })} /></td>
                    <td><button className="ghost" onClick={() => edit({
                      rows: form.rows.filter((_, i) => i !== at) })}>✕</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
            <button className="ghost" onClick={() => edit({
              rows: [...form.rows, { tenor: '', spread: '' }] })}>add a tenor</button>
          </>
        )}
        <div className="hint">What it will write: each spread read as a hazard rate, the spread
          divided by one less the recovery, counted ACT/365 from {base} to its tenor. A flat spread
          is stated at one year. Past the last tenor the curve runs on at its last spread.</div>
        <SpreadTable rows={typed.map((held, at) =>
          [held.tenor, held.spread, Math.exp(-knots[at][1])])} />
        <div className="statusrow">
          <button className="primary" onClick={() => void write()} disabled={saving}>
            write {SURVIVAL}.{request.counterparty || '…'}
          </button>
          {recording && actingSeat(state) && <span className="hint">as {actingSeat(state)}</span>}
          {saving && <span className="chip running">writing</span>}
          {said?.at === 'done' && <span className="chip done">{said.text}</span>}
          {refusal('act')}{id && refusal('counterparty')}
        </div>
        </>}
      </section>
    </>
  );
}
