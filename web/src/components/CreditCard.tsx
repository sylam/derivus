// Counterparty survival curve display and spread editor.

import { useEffect, useState } from 'react';
import { failure, stateSurvival } from '../api';
import {
  SURVIVAL, SURVIVAL_QUOTES, factorNames, recoveryConvention, refusedAt, spreadForm, spreadRequest,
  type SpreadForm,
} from '../desk';
import { actingSeat, heldBack, heldSpreads, marketOf, useApp, written } from '../state';
import { formatNumber } from '../tokens';
import { Gated } from './Desk';
import { Whose } from './Whose';

/** Prefix identifying a credit page id in the Curves navigator. */
export const CREDIT = 'credit:';

const HOLDS = {
  record: 'On this desk a curve is written for an entity the record declares, by a seat holding '
    + 'mark over the book.',
  plain: 'This desk keeps no record: a curve is written for any counterparty named.',
};

/** A counterparty's credit curve `SurvivalProb.<id>` as its quotes block states it (`heldSpreads`):
 * the currency, the discount curve and the recovery its default swaps are priced at, and each
 * tenor's quote; where the market holds no block, that its curve has no quotes, or that there is
 * no curve at all. */
export function HeldCurve({ id }: { id: string }) {
  const { state } = useApp();
  const factor = `${SURVIVAL}.${id}`;
  const held = heldSpreads(state, factor);
  if (!held) {
    return (
      <div className="note">{marketOf(state)?.['Price Factors']?.[factor]
        ? `Its curve has no quotes: the book holds ${factor} as it was stated, solved from none. `
          + 'Stating its spreads writes its quotes, and the curve is solved from them.'
        : 'No credit curve yet: a CVA against this counterparty has no default to read.'}</div>
    );
  }
  return (
    <>
      <div className="hint">Its quotes, struck in {held.currency} and discounting on{' '}
        {held.discount_rate}, at a recovery of {formatNumber(100 * held.recovery, 'percent')}:</div>
      <table className="data">
        <thead><tr><th>Tenor</th><th className="n">Quote</th><th>As</th></tr></thead>
        <tbody>
          {held.rows.map((row) => (
            <tr key={row.tenor}>
              <td>{row.tenor}</td>
              <td className="n">{row.type === 'Par_Spread' ? formatNumber(row.quote, 'basis')
                : formatNumber(row.quote)}</td>
              <td className="hint">{row.type === 'Par_Spread' ? 'par spread' : row.type}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

/** A select over the market's factors a field of the quotes block names, the value held among
 * them where the market lacks it. */
function FactorPick({ field, value, blank, onPick }: {
  field: string; value: string; blank: string; onPick: (value: string) => void;
}) {
  const { state } = useApp();
  const offered = factorNames(state.schema?.MarketPrices.types[SURVIVAL_QUOTES]?.[field],
                              marketOf(state));
  return (
    <select value={value} onChange={(event) => onPick(event.target.value)}>
      <option value="">{blank}</option>
      {[...new Set([value, ...offered])].filter(Boolean).map((name) => (
        <option key={name}>{name}</option>))}
    </select>
  );
}

/** Counterparty credit curve panel and flat or per-tenor spread entry form. */
export function CreditCard({ id, title, recording }: {
  id?: string; title: string; recording: boolean;
}) {
  const { state, dispatch } = useApp();
  const hold = heldBack(state, 'survival_curve');
  const held = id === undefined ? undefined : heldSpreads(state, `${SURVIVAL}.${id}`);
  const [name, setName] = useState(id ?? '');
  const [form, setForm] = useState(() => spreadForm(held, state.schema));
  const [said, setSaid] = useState<{ at: string; text: string } | null>(null);
  const [saving, setSaving] = useState(false);
  const incoming = JSON.stringify(held ?? null);
  useEffect(() => setForm(spreadForm(held, state.schema)), [incoming]);

  const request = spreadRequest(name, form);
  const edit = (change: Partial<SpreadForm>) => { setSaid(null); setForm({ ...form, ...change }); };
  const row = (at: number, cell: Partial<SpreadForm['rows'][number]>) =>
    edit({ rows: form.rows.map((held, i) => (i === at ? { ...held, ...cell } : held)) });
  const refusal = (at: string) => said?.at === at && <span className="refusal">{said.text}</span>;

  async function write() {
    setSaving(true);
    try {
      const answer = await stateSurvival(request);
      await written(dispatch, answer);
      setSaid({ at: 'done', text: `solved ${answer.factor} from its quotes` });
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
        {id && <HeldCurve id={id} />}
      </section>
      <section className="card credit">
        <h3>{held ? 'Restate the curve' : 'State the curve'}</h3>
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
          <span className="hint">the quotes' convention is {
            String(recoveryConvention(state.schema))}</span>
          {refusal('recovery')}
        </div>
        <div className="statusrow">
          <span>struck in</span>
          <FactorPick field="Currency" value={form.currency} blank="a currency"
                      onPick={(currency) => edit({ currency })} />
          {refusal('currency')}
          <span>discounting on</span>
          <FactorPick field="Discount_Rate" value={form.discount_rate} blank="a curve"
                      onPick={(discount_rate) => edit({ discount_rate })} />
          {refusal('discount_rate')}
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
