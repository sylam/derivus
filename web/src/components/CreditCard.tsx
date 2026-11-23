// Counterparty survival curve display and spread editor.

import { useEffect, useState } from 'react';
import { failure, stateSurvival } from '../api';
import {
  SURVIVAL, heldSpreads, recoveryOf, refusedAt, spreadForm, spreadRequest, type SpreadForm,
} from '../desk';
import { actingSeat, heldBack, useApp, written } from '../state';
import { formatNumber, tenorLabel } from '../tokens';
import { Gated } from './Desk';
import { Whose } from './Whose';

/** Prefix identifying a credit page id in the Curves navigator. */
export const CREDIT = 'credit:';

const HOLDS = {
  record: 'On this desk a curve is written for an entity the record declares, by a seat holding '
    + 'mark over the book.',
  plain: 'This desk keeps no record: a curve is written for any counterparty named.',
};

/** A counterparty's credit curve as the book holds it (`heldSpreads`): the recovery it is read at,
 * and each tenor's spread with the survival to it - nothing where the factor holds no curve. */
export function HeldCurve({ factor }: { factor: unknown }) {
  const held = heldSpreads(factor, useApp().state.schema);
  if (!held) return null;
  return (
    <>
      <div className="hint">As the book holds it, at a recovery of{' '}
        {formatNumber(100 * held.recovery, 'percent')}:</div>
      <table className="data">
        <thead>
          <tr><th>Tenor</th><th className="n">Spread</th><th className="n">Survival</th></tr>
        </thead>
        <tbody>
          {held.rows.map(([tenor, spread, survival]) => (
            <tr key={tenor}>
              <td>{tenorLabel(tenor)}</td>
              <td className="n">{formatNumber(spread * 1e4, 'basis')}</td>
              <td className="n">{formatNumber(100 * survival, 'percent')}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

/** Counterparty credit curve panel and flat or per-tenor spread entry form. */
export function CreditCard({ id, title, factor, recording }: {
  id?: string; title: string; factor?: unknown; recording: boolean;
}) {
  const { state, dispatch } = useApp();
  const hold = heldBack(state, 'survival_curve');
  const [name, setName] = useState(id ?? '');
  const [form, setForm] = useState(() => spreadForm(factor, state.schema));
  const [said, setSaid] = useState<{ at: string; text: string } | null>(null);
  const [saving, setSaving] = useState(false);
  const incoming = JSON.stringify(factor ?? null);
  useEffect(() => setForm(spreadForm(factor, state.schema)), [incoming]);

  const request = spreadRequest(name, form);
  const stated = heldSpreads(factor, state.schema);
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
      setForm(spreadForm(undefined, state.schema));
      dispatch({ type: 'PICK', screen: 'curves', id: CREDIT + request.counterparty });
    } catch (error) {
      const { error: text, code } = failure(error);
      setSaid({ at: refusedAt(code, request), text });
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
        <HeldCurve factor={factor} />
        {!stated && id && (
          <div className="note">No credit curve yet: a CVA against this counterparty has no default
            to read.</div>
        )}
      </section>
      <section className="card credit">
        <h3>{stated ? 'Restate the curve' : 'State the curve'}</h3>
        <Gated hold={hold}>
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
          <span className="hint">the factor's convention is {
            String(recoveryOf(undefined, state.schema))}</span>
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
        <div className="statusrow">
          <button className="primary" onClick={() => void write()} disabled={saving}>
            write {SURVIVAL}.{request.counterparty || '…'}
          </button>
          {recording && actingSeat(state) && <span className="hint">as {actingSeat(state)}</span>}
          {saving && <span className="chip running">writing</span>}
          {said?.at === 'done' && <span className="chip done">{said.text}</span>}
          {refusal('act')}{id && refusal('counterparty')}
        </div>
        </Gated>
      </section>
    </>
  );
}
