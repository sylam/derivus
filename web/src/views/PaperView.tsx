import { useEffect, useState } from 'react';
import {
  declareAgreement, declareEntity, failure, getBookAgreements, getBookEntities,
} from '../api';
import { DataTable } from '../components/DataTable';
import { DescriptorPanel, EditableScalar } from '../components/FieldView';
import { EmptyState } from '../components/EmptyState';
import type { Agreement, Entity } from '../positions';
import { worklistAt } from '../spine';
import { useApp } from '../state';
import { isObject, supportRows, token } from '../tokens';
import type { Descriptor, Schema } from '../types';

/** The declaration an agreement's terms are: the netting set's own, as `/schema` publishes it. */
const TERMS = 'NettingCollateralSet';

/** What a declaration answered: where it landed, or the service's refusal in its own words. */
type Said = { refused: boolean; text: string } | null;

/** The paper the book trades under: every legal entity and agreement the record declares, each
 * agreement's terms rendered off the netting set's declaration, and a form for each - an
 * agreement's terms being the netting set itself, every field at its convention until stated.
 * Declaring is the acting seat's; the record's `document` grant and the engine's judge of the
 * terms refuse, in their own words. Read again at every move of the record's head. */
export function PaperView() {
  const { state } = useApp();
  const { schema } = state;
  const at = worklistAt(state.source?.kind === 'book' ? state.record.spine : null);
  const [paper, setPaper] = useState<{
    entities: Entity[]; agreements: Agreement[]; error: string | null }>(
    { entities: [], agreements: [], error: null });

  useEffect(() => {
    if (at === null) return;
    let live = true;
    Promise.all([getBookEntities(), getBookAgreements()])
      .then(([legal, signed]) => {
        if (live) setPaper({ entities: legal.entities, agreements: signed.agreements, error: null });
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
  const named = (id: string) => paper.entities.find((row) => row.entity === id)?.name ?? id;
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
                  <tr key={row.entity}>
                    <td className="mono">{row.entity}</td>
                    <td>{row.name}</td>
                    <td>{row.parent ? `under ${named(row.parent)}` : ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <EntityForm />
        </section>
        <section className="card">
          <h3>Agreements ({paper.agreements.length})</h3>
          {paper.agreements.map((row) => (
            <details className="fold" key={row.agreement}>
              <summary>
                <span className="mono">{row.agreement}</span> · {named(row.entity)} · {row.kind}
              </summary>
              <Terms schema={schema} terms={row.terms} />
            </details>
          ))}
        </section>
        <AgreementForm schema={schema} entities={paper.entities} />
      </div>
    </div>
  );
}

/** An agreement's terms as the netting set declares them: its sections, the way a deal's panel
 * renders, each field its convention where the terms state nothing, and every table the terms
 * carry editable a cell at a time where `onChange` takes the terms back. */
function Terms({ schema, terms, onChange }: {
  schema: Schema; terms: Record<string, unknown>;
  onChange?: (terms: Record<string, unknown>) => void;
}) {
  const sections = schema.Instrument.types[TERMS] ?? [];
  const stated = (key: string, wire: unknown) => onChange?.({ ...terms, [key]: wire });
  // the tables sit inside the containers, which take a table back whole
  const tables = sections.flatMap((section) => Object.entries(schema.Instrument.sections[section])
    .filter(([, field]) => field.widget === 'Container')
    .flatMap(([key, field]) => Object.entries(field.sub_fields ?? {})
      .filter(([, sub]) => sub.widget === 'Table')
      .map(([name, sub]) => ({ key, name, sub }))));
  return (
    <>
      {sections.map((section) => (
        <DescriptorPanel key={section} title={schema.Instrument.titles?.[section] ?? section}
                         fields={schema.Instrument.sections[section]} values={terms}
                         onAmend={onChange && (async (key, wire) => { stated(key, wire); return null; })} />
      ))}
      {onChange && tables.map(({ key, name, sub }) => {
        const held = isObject(terms[key]) ? terms[key] as Record<string, unknown> : {};
        return (
          <TableEdit key={`${key}.${name}`} name={`${key} · ${name}`} declared={sub}
                     value={name in held ? held[name] : sub.value}
                     onChange={(wire) => stated(key, { ...held, [name]: wire })} />
        );
      })}
    </>
  );
}

/** One table of the terms - a credit support list or a list of eligible collateral - its columns
 * and their forms the declaration's own, a cell edited at a time and a row added blank. */
function TableEdit({ name, declared, value, onChange }: {
  name: string; declared: Descriptor; value: unknown; onChange: (wire: unknown) => void;
}) {
  const columns = declared.col_names ?? [];
  const support = declared.obj === 'CreditSupportList';
  const forms = support ? ['Float', 'Integer'] : (declared.obj ?? []) as string[];
  const rows: unknown[][] = support
    ? supportRows((token(value, '.CreditSupportList') ?? []) as unknown[][])
    : (Array.isArray(value) ? value : []).map((row) => columns.map(
      (column) => (row as Record<string, unknown>)[column]));
  const wire = (next: unknown[][]) => (support
    ? { '.CreditSupportList': supportRows(next) }
    : next.map((row) => Object.fromEntries(columns.map((column, at) => [column, row[at]]))));
  const cell = (form: string, column: string): Descriptor => ({
    widget: ['Float', 'Percent'].includes(form) ? 'Float' : form === 'Integer' ? 'Integer' : 'Text',
    obj: form, description: column, value: null,
  });
  return (
    <section className="card">
      <h3>{name}</h3>
      <DataTable columns={columns} index={[]} data={rows} cell={(r, c) => (
        <EditableScalar name={columns[c]} descriptor={cell(forms[c], columns[c])} value={rows[r][c]}
                        onAmend={async (_, edited) => {
                          onChange(wire(rows.map((row, at) => (at === r
                            ? row.map((held, column) => (column === c ? edited : held)) : row))));
                          return null;
                        }} />
      )} />
      <div className="statusrow">
        <button className="ghost" onClick={() => onChange(wire([...rows, columns.map(() => null)]))}>
          add a row
        </button>
        {rows.length > 0 && (
          <button className="ghost" onClick={() => onChange(wire(rows.slice(0, -1)))}>
            drop the last row
          </button>
        )}
      </div>
    </section>
  );
}

/** Declaring a legal entity: its id as the legal system knows it, its name and the parent it is
 * grouped under. */
function EntityForm() {
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

  return (
    <div className="statusrow">
      {(['entity', 'name', 'parent'] as const).map((field) => (
        <input key={field} placeholder={field} value={entity[field]}
               onChange={(event) => setEntity({ ...entity, [field]: event.target.value })} />
      ))}
      <button className="ghost" onClick={declare}>declare the entity</button>
      {said && <span className={said.refused ? 'refusal' : 'hint'}>{said.text}</span>}
    </div>
  );
}

/** Declaring an agreement with an entity: its id, the document's kind and its terms - the netting
 * set every position under it compiles into, edited off the declaration and posted as it stands. */
function AgreementForm({ schema, entities }: { schema: Schema; entities: Entity[] }) {
  const [named, setNamed] = useState({ agreement: '', entity: '', kind: '' });
  const [terms, setTerms] = useState<Record<string, unknown>>({ Object: TERMS });
  const [said, setSaid] = useState<Said>(null);

  async function declare() {
    try {
      const filed = await declareAgreement({ ...named, terms });
      setSaid({ refused: false, text: `declared at LSN ${filed.recorded.lsn}` });
    } catch (error) {
      setSaid({ refused: true, text: failure(error).error });
    }
  }

  return (
    <section className="card">
      <h3>Declare an agreement</h3>
      <div className="statusrow">
        <input placeholder="agreement" value={named.agreement}
               onChange={(event) => setNamed({ ...named, agreement: event.target.value })} />
        <select value={named.entity}
                onChange={(event) => setNamed({ ...named, entity: event.target.value })}>
          <option value="">with entity…</option>
          {entities.map((row) => <option key={row.entity} value={row.entity}>{row.name}</option>)}
        </select>
        <input placeholder="kind (ISDA 2002 with CSA)" value={named.kind}
               onChange={(event) => setNamed({ ...named, kind: event.target.value })} />
        <button className="ghost" onClick={declare}>declare the agreement</button>
      </div>
      {said && <div className={said.refused ? 'error-box' : 'hint'}>{said.text}</div>}
      <Terms schema={schema} terms={terms} onChange={setTerms} />
    </section>
  );
}
