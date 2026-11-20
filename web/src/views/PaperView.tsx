import { useState } from 'react';
import { declareEntity, failure } from '../api';
import { entityText } from '../desk';
import { openClient } from '../client';
import { EmptyState } from '../components/EmptyState';
import { type Agreement, type Entity, NO_PAPER } from '../positions';
import { heldBack, opened, standing, useApp } from '../state';
import { AgreementForm, AgreementTerms, type Said } from '../components/Desk';

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
  const recording = state.source?.kind === 'book' && state.record.spine !== null;
  const [amending, setAmending] = useState<{ from: Agreement; at: number } | null>(null);
  const hold = heldBack(state, 'document', null);

  const paper = { ...(state.paper.data ?? NO_PAPER), error: state.paper.error };

  if (!recording || !schema) {
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
                <AgreementTerms schema={schema} terms={row.terms} unstated={paper.unstated} />
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
