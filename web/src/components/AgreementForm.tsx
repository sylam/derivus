// Legal agreement terms display and declaration / restatement form.

import { useEffect, useRef, useState } from 'react';
import { declareAgreement, failure } from '../api';
import { SET, fieldsOf } from '../blotter';
import { entityText } from '../desk';
import type { Agreement, Entity } from '../positions';
import { heldBack, useApp } from '../state';
import { COLLATERAL, onPaper, statedOnly, termsMoved, uncollateralised } from '../tokens';
import type { Schema, SettlementState } from '../types';
import { Gated } from './Desk';
import { DealPanels } from './FieldView';

/** Declaration outcome: where it landed or the service's refusal sentence. */
export type Said = { refused: boolean; text: string } | null;

/** Agreement terms via DealPanels, omitting settlement state and inert collateral fields. */
export function AgreementTerms({ schema, terms, unstated, onChange }: {
  schema: Schema; terms: Record<string, unknown>; unstated: SettlementState;
  onChange?: (terms: Record<string, unknown>) => void;
}) {
  const bare = !onChange && uncollateralised(fieldsOf(schema, SET) ?? {}, terms);
  return (
    <DealPanels deal={terms} schema={schema} own tables={!!onChange}
                cut={(fields) => onPaper(bare ? statedOnly(fields, terms, [COLLATERAL]) : fields,
                                         unstated)}
                onAmend={onChange && (async (key, wire) => {
                  onChange({ ...terms, [key]: wire });
                  return null;
                })} />
  );
}

/** Form to declare a new agreement or restate an existing one, highlighting moved terms. */
export function AgreementForm({ schema, entities, agreements, unstated, from }: {
  schema: Schema; entities: Entity[]; agreements: Agreement[]; unstated: SettlementState;
  from?: Partial<Agreement>;
}) {
  const hold = heldBack(useApp().state, 'agreement_declared', null);
  const [named, setNamed] = useState({
    agreement: from?.agreement ?? '', entity: from?.entity ?? '', kind: from?.kind ?? '' });
  const [terms, setTerms] = useState<Record<string, unknown>>(from?.terms ?? { Object: SET });
  const [said, setSaid] = useState<Said>(null);
  const card = useRef<HTMLElement>(null);
  useEffect(() => { if (from) card.current?.scrollIntoView({ block: 'start' }); }, [from]);
  const held = agreements.find((row) => row.agreement === named.agreement);
  const moved = held ? termsMoved(held.terms, terms) : [];

  async function declare() {
    try {
      const filed = await declareAgreement({ ...named, terms });
      setSaid({ refused: false, text: `declared at LSN ${filed.recorded.lsn}` });
    } catch (error) {
      setSaid({ refused: true, text: failure(error).error });
    }
  }

  const words = (keys: string[]) => keys.map((key) => key.replace(/_/g, ' ')).join(', ');
  return (
    <section className="card" ref={card}>
      <h3>{held ? `Restate ${held.agreement}` : 'Declare an agreement'}</h3>
      <Gated hold={hold}>
      <div className="statusrow">
        <input placeholder="agreement" value={named.agreement}
               onChange={(event) => setNamed({ ...named, agreement: event.target.value })} />
        <select value={named.entity}
                onChange={(event) => setNamed({ ...named, entity: event.target.value })}>
          <option value="">with entity…</option>
          {entities.map((row) => (
            <option key={row.entity} value={row.entity}>{entityText(entities, row.entity)}</option>
          ))}
        </select>
        <input placeholder="kind (ISDA 2002 with CSA)" value={named.kind}
               onChange={(event) => setNamed({ ...named, kind: event.target.value })} />
        <button className="ghost" onClick={declare}>
          {held ? 'restate the agreement' : 'declare the agreement'}
        </button>
      </div>
      {held && (
        <div className="banner">
          {named.agreement} is in force: filing this restates it whole, every term left out going
          back to its convention. {moved.length
            ? `Terms that differ from the version in force: ${moved.join(', ')}.`
            : 'No term differs from the version in force yet.'}
        </div>
      )}
      {said && <div className={said.refused ? 'error-box' : 'hint'}>{said.text}</div>}
      {unstated.fields.length + unstated.collateral_columns.length > 0 && (
        <div className="note">Paper states no {words(unstated.fields)}, nor a collateral row's{' '}
          {words(unstated.collateral_columns)}: those are settlement state, which the record holds
          from each collateral settlement the Back Office files.</div>
      )}
      <AgreementTerms schema={schema} terms={terms} unstated={unstated} onChange={setTerms} />
      </Gated>
    </section>
  );
}
