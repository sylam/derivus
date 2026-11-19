import { useEffect, useRef, useState } from 'react';
import {
  declareAgreement, declareEntity, failure, getBookAgreements, getBookEntities,
} from '../api';
import { SET } from '../blotter';
import { DealPanels } from '../components/FieldView';
import { entityText } from '../desk';
import { openClient } from '../client';
import { EmptyState } from '../components/EmptyState';
import type { Agreement, Entity } from '../positions';
import { worklistAt } from '../spine';
import { heldBack, opened, standing, useApp } from '../state';
import { COLLATERAL, onPaper, statedOnly, termsMoved, uncollateralised } from '../tokens';
import type { Schema, SettlementState } from '../types';

/** What a declaration answered: where it landed, or the service's refusal in its own words. */
type Said = { refused: boolean; text: string } | null;

/** The paper the book trades under: every legal entity and agreement the record declares, each
 * agreement with where it stands on the desk and its terms rendered off the netting set's
 * declaration, and a form for each - an agreement's terms being the netting set itself, every
 * field at its convention until stated, an agreement amended from the terms in force. Declaring
 * is the acting seat's; the record's `document` grant and the engine's judge of the terms refuse,
 * in their own words. Read again at every move of the record's head. An agreement a row of the
 * record opened stands open, an entity it opened marked. */
export function PaperView() {
  const { state, dispatch } = useApp();
  const { schema } = state;
  const at = worklistAt(state.source?.kind === 'book' ? state.record.spine : null);
  const [paper, setPaper] = useState<{ entities: Entity[]; agreements: Agreement[];
                                       unstated: SettlementState; error: string | null }>(
    { entities: [], agreements: [], unstated: { fields: [], collateral_columns: [] },
      error: null });
  const [amending, setAmending] = useState<{ from: Agreement; at: number } | null>(null);
  const hold = heldBack(state, 'document', null);

  useEffect(() => {
    if (at === null) return;
    let live = true;
    Promise.all([getBookEntities(), getBookAgreements()])
      .then(([legal, signed]) => {
        if (live) {
          setPaper({ entities: legal.entities, agreements: signed.agreements,
                     unstated: signed.settlement_state, error: null });
        }
      })
      .catch((error) => {
        if (live) setPaper((held) => ({ ...held, error: failure(error).error }));
      });
    return () => { live = false; };
  }, [at]);

  if (at === null || !schema) {
    return (
      <EmptyState
        title="The paper is the live book's record."
        hint="Start the service with --book and a spine home: the counterparties and the agreements the book trades under appear here, and a seat holding `document` declares them."
      />
    );
  }
  const named = (id: string) => entityText(paper.entities, id);
  // what a row of the record opened here: an agreement, or an entity
  const picked = state.selection.picks.paper;
  return (
    <div className="main">
      <div className="panel">
        {paper.error && <div className="error-box">{paper.error}</div>}
        <section className="card">
          <h3>Legal entities ({paper.entities.length})</h3>
          <div className="tablewrap">
            <table className="data">
              <tbody>
                {paper.entities.map((row) => (
                  <tr key={row.entity} className={picked === row.entity ? 'selected' : ''}>
                    <td className="mono">{row.entity}</td>
                    <td>{row.name}</td>
                    <td>{row.parent ? `under ${named(row.parent)}` : ''}</td>
                    <td><button className="ghost" onClick={() => openClient(dispatch, row.entity)}>
                      open its page</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <EntityForm entities={paper.entities} />
        </section>
        <section className="card">
          <h3>Agreements ({paper.agreements.length})</h3>
          {paper.agreements.map((row) => {
            const { path, moved } = standing(state.doc, state.record.reconcile, row);
            return (
              <details className="fold" key={row.agreement}
                       open={picked === row.agreement || undefined}>
                <summary>
                  <span className="mono">{row.agreement}</span> · {named(row.entity)} · {row.kind}
                  {' · '}
                  <span className={moved.length ? 'chip error' : 'chip'}>
                    {path === null ? 'no trade booked under it yet: its netting set enters the '
                      + 'book at the first booking'
                      : moved.length ? `in the book, its set off the paper on ${moved.join(', ')}`
                        : 'in the book, as declared'}
                  </span>
                  {path !== null && (
                    <button className="ghost" onClick={(event) => {
                      event.preventDefault();
                      opened(dispatch, state.doc, path);
                    }}>open its set</button>
                  )}
                  {!hold && (
                    <button className="ghost" onClick={(event) => {
                      event.preventDefault();
                      setAmending({ from: row, at: Date.now() });
                    }}>amend</button>
                  )}
                </summary>
                <Terms schema={schema} terms={row.terms} unstated={paper.unstated} />
              </details>
            );
          })}
        </section>
        <AgreementForm key={amending?.at} schema={schema} entities={paper.entities}
                       agreements={paper.agreements} unstated={paper.unstated}
                       from={amending?.from} />
      </div>
    </div>
  );
}

/** An agreement's terms as the netting set declares them: its own sections, the way a deal's
 * panel renders, each term its convention where the terms state nothing - those of a set whose
 * collateral is off left out, being collateral it does not have, and what paper never states
 * (`unstated`) - and every term, a table's cells among them, editable where `onChange` takes the
 * terms back. */
function Terms({ schema, terms, unstated, onChange }: {
  schema: Schema; terms: Record<string, unknown>; unstated: SettlementState;
  onChange?: (terms: Record<string, unknown>) => void;
}) {
  const bare = !onChange && (schema.Instrument.types[SET] ?? []).some((section) =>
    uncollateralised(schema.Instrument.sections[section], terms));
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

/** Declaring a legal entity: its id as the legal system knows it, its name and the parent it is
 * grouped under. */
function EntityForm({ entities }: { entities: Entity[] }) {
  const hold = heldBack(useApp().state, 'document', null);
  const [entity, setEntity] = useState({ entity: '', name: '', parent: '' });
  const [said, setSaid] = useState<Said>(null);

  async function declare() {
    try {
      const filed = await declareEntity({
        entity: entity.entity, name: entity.name, ...(entity.parent ? { parent: entity.parent } : {}),
      });
      setSaid({ refused: false, text: `declared at LSN ${filed.recorded.lsn}` });
    } catch (error) {
      setSaid({ refused: true, text: failure(error).error });
    }
  }

  return hold ? <div className="note">{hold}</div> : (
    <div className="statusrow">
      {(['entity', 'name'] as const).map((field) => (
        <input key={field} placeholder={field} value={entity[field]}
               onChange={(event) => setEntity({ ...entity, [field]: event.target.value })} />
      ))}
      <select value={entity.parent}
              onChange={(event) => setEntity({ ...entity, parent: event.target.value })}>
        <option value="">under no parent</option>
        {entities.map((row) => (
          <option key={row.entity} value={row.entity}>{entityText(entities, row.entity)}</option>
        ))}
      </select>
      <button className="ghost" onClick={declare}>declare the entity</button>
      {said && <span className={said.refused ? 'refusal' : 'hint'}>{said.text}</span>}
    </div>
  );
}

/** Declaring an agreement with an entity: its id, the document's kind and its terms - the netting
 * set every position under it compiles into, edited off the declaration and posted as it stands.
 * Started `from` an agreement in force, it holds that one's terms - from an entity alone, a new
 * agreement with it; an id the record declares already is a restatement, which names every term it
 * moves before it is filed. What paper never states (`unstated`) is left out, and said once. */
export function AgreementForm({ schema, entities, agreements, unstated, from }: {
  schema: Schema; entities: Entity[]; agreements: Agreement[]; unstated: SettlementState;
  from?: Partial<Agreement>;
}) {
  const hold = heldBack(useApp().state, 'document', null);
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
      {hold ? <div className="note">{hold}</div> : <>
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
      <Terms schema={schema} terms={terms} unstated={unstated} onChange={setTerms} />
      </>}
    </section>
  );
}
