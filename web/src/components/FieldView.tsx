import { useEffect, useId, useRef, useState, type ReactNode } from 'react';
import { failure } from '../api';
import { factorNames } from '../desk';
import { marketOf, useApp } from '../state';
import {
  SUPPORT_ROW, currencyOf, curveOf, editRaw, encodeScalar, fieldLabel, formatNumber, isBlank,
  isEditableScalar, isObject, kindOf, label, offsetText, onScreen, sharedSections, token, ungrouped,
} from '../tokens';
import type { Descriptor, Schema, Section } from '../types';
import { quotesNormal } from '../vols';
import { CurveChart } from './CurveChart';
import { DataTable } from './DataTable';
import { JsonView } from './JsonView';
import { VolSurface } from './VolSurface';

/** One field, read-only, dispatched on the VALUE first and the descriptor second. The value is
 * authoritative: a document may carry a key the schema does not declare, and a viewer that hides
 * such a key is a viewer nobody can trust. The descriptor refines - table columns, dropdown
 * hints, formatting - and its absence only ever downgrades to the JSON fallback, never to
 * nothing. A `.Curve` branches on ROW ARITY, because the widget token folds Surface and Space;
 * `normal` is a surface of absolute normal vols. */
export function FieldView({ value, descriptor, normal }: {
  value: unknown; descriptor?: Descriptor; normal?: boolean;
}) {
  // --- wire tokens ---
  const curve = curveOf(value);
  if (curve) {
    const arity = curve.data[0]?.length ?? 0;
    if (arity === 2) {
      return <CurveChart data={curve.data} description={descriptor?.description} />;
    }
    if (arity === 3 || arity === 4) {
      return <VolSurface data={curve.data} descriptor={descriptor} normal={normal} />;
    }
    return <JsonView value={value} />;
  }
  for (const name of ['.Timestamp', '.DateOffset', '.Percent', '.Basis'] as const) {
    if (token(value, name) !== undefined) return <span className="num">{label(value)}</span>;
  }
  const grid = token(value, '.Grid');
  if (Array.isArray(grid)) {
    return <span className="num">{grid.map(offsetText).join(' ')}</span>;
  }
  for (const list of ['.DateList', '.DateEqualList'] as const) {
    const rows = token(value, list);
    if (Array.isArray(rows)) {
      return <DataTable columns={descriptor?.col_names ?? []} index={[]} data={rows} />;
    }
  }
  const support = token(value, '.CreditSupportList');
  if (Array.isArray(support)) {
    return (
      <>
        <DataTable columns={descriptor?.col_names ?? []} index={[]} data={support} />
        <div className="hint">{SUPPORT_ROW}</div>
      </>
    );
  }
  // 'null' is the wire blank of every table, what a table widget writes for an empty one
  if (value === 'null' && descriptor?.widget === 'Table') {
    return <DataTable columns={descriptor.col_names ?? []} index={[]} data={[]} />;
  }
  const params = token(value, '.ModelParams');
  if (isObject(params)) {
    return (
      <div>
        {(['modeldefaults', 'modelfilters'] as const).map((half) => (
          <div key={half}>
            <div className="pager">{half}</div>
            <JsonView value={params[half]} />
          </div>
        ))}
      </div>
    );
  }
  const descriptorToken = token(value, '.Descriptor');
  if (descriptorToken !== undefined) return <span className="num">{String(descriptorToken)}</span>;
  const deal = token(value, '.Deal');
  if (isObject(deal)) return <JsonView value={deal} />;
  if (isObject(value) && Object.keys(value).some((key) => key.startsWith('.'))) {
    return <JsonView value={value} />; // an unknown token must still be visible
  }

  // --- plain shapes, refined by the descriptor ---
  if (Array.isArray(value)) {
    if (descriptor?.widget === 'Table' || value.every((row) => Array.isArray(row))) {
      const names = descriptor?.col_names ?? [];
      // a declared table's rows travel as dicts, and the declared column ORDER is what makes one
      // a row: a quote ladder reads as its own columns rather than as JSON
      const rows = names.length > 0 && value.every(isObject)
        ? value.map((row) => names.map((name) => (row as Record<string, unknown>)[name]))
        : value;
      return <DataTable columns={names} index={[]} data={rows} />;
    }
    return <JsonView value={value} />;
  }
  if (isObject(value)) return <JsonView value={value} />;

  // --- scalars ---
  if (isBlank(value)) {
    return <span className="hint">—</span>;
  }
  if (typeof value === 'boolean') return <span>{value ? '✓ true' : '✗ false'}</span>;
  if (typeof value === 'number') {
    return (
      <span className="num" title={String(value)}>{formatNumber(value, kindOf(descriptor))}</span>
    );
  }
  return (
    <span>
      {String(value)}
      {descriptor?.values && descriptor.values.length > 1 && (
        <span className="hint">({descriptor.values.join(' | ')})</span>
      )}
    </span>
  );
}

/** Save one field's wire value; resolves to refusal messages, or null on success. */
export type AmendField = (key: string, wireValue: unknown) => Promise<string[] | null>;

/** One card of labelled fields: declared descriptors in declaration order with the document's
 * values over them, then every UNDECLARED key of the value dict - visible, marked, at the end.
 * With `onAmend`, declared SCALAR fields grow inputs (shapes and tables stay read-only), and a
 * container's own scalars edit THROUGH the container: the amendment names the container and
 * carries it whole, its other keys standing, because an amendment merges top-level keys and a
 * container is one of them. `editable` narrows WHICH declared fields may edit - the market view
 * passes the bind='value' test, so structure stays read-only where the engine says it must.
 * `aside` is what a top-level field's label carries beside it - whose the entry is. `ask` is
 * what filing an edit does, where that is more than a save: a changed field asks it first.
 * `tables` edits a table too, a cell at a time; `currency` reads the currency an amount's
 * description names, a container's children reading their deal's. `beside` is what a top-level
 * field's value carries beside it - how a desk reads it. */
export function DescriptorPanel({ title, fields, values, onAmend, editable, aside, beside, ask,
                                  tables, currency }: {
  title: string;
  fields?: Record<string, Descriptor>;
  values: Record<string, unknown>;
  onAmend?: AmendField;
  editable?: (key: string, descriptor: Descriptor) => boolean;
  aside?: (key: string) => ReactNode;
  beside?: (key: string) => ReactNode;
  ask?: string;
  tables?: boolean;
  currency?: (description: string) => string | undefined;
}) {
  const declared = Object.entries(fields ?? {});
  const undeclared = Object.keys(values).filter((key) => !(fields ?? {})[key]);
  const rows = [
    ...declared.map(([key, descriptor]) => ({ key, descriptor: descriptor as Descriptor | undefined })),
    ...undeclared.map((key) => ({ key, descriptor: undefined })),
  ].filter(({ key, descriptor }) => descriptor !== undefined || values[key] !== undefined);
  if (!rows.length) return null;
  const inCurrency = currency ?? ((description: string) => currencyOf(
    description, [...declared.map(([key]) => key), ...undeclared],
    (key) => values[key] ?? fields?.[key]?.value));

  return (
    <section className="card">
      <h3>{title}</h3>
      <div className="fields">
        {rows.map(({ key, descriptor }) => {
          const value = values[key] !== undefined ? values[key] : descriptor?.value;
          // a container's children are stated only where it states them
          const whole = isObject(values[key]) ? values[key] as Record<string, unknown> : {};
          return descriptor?.widget === 'Container' ? (
            <div key={key} style={{ gridColumn: '1 / -1' }}>
              <DescriptorPanel
                title={fieldLabel(key, descriptor).text
                  + (descriptor.convention && values[key] === undefined ? ' · convention' : '')}
                fields={descriptor?.sub_fields}
                values={whole} editable={editable} ask={ask} tables={tables} currency={inCurrency}
                onAmend={onAmend && (!editable || editable(key, descriptor!))
                  ? (field, wire) => onAmend(key, { ...whole, [field]: wire }) : undefined}
              />
            </div>
          ) : (
            <FieldRow key={key} name={key} descriptor={descriptor} value={value}
                      stated={!isBlank(values[key])}
                      declared={!!descriptor || !undeclared.includes(key)} tables={tables}
                      currency={descriptor && inCurrency(descriptor.description)}
                      onAmend={onAmend && (!descriptor || !editable || editable(key, descriptor))
                        ? onAmend : undefined} aside={aside?.(key)} beside={beside?.(key)}
                      normal={quotesNormal(fields, values)} ask={ask} />
          );
        })}
      </div>
    </section>
  );
}

/** A deal as its type declares it: a card per section holding what a screen shows - its `own`
 * alone, the sections every type shares left out - then every key no section covers, the viewer's
 * honesty panel; a type nobody declared is one card whole. `cut` narrows a section's fields, and
 * `hold` - what holds the seat from amending - reads in place of every input. */
export function DealPanels({ deal, schema, own, cut = (fields) => fields, hold, onAmend,
                             ...panel }: {
  deal: Record<string, unknown>; schema: Schema | null; own?: boolean;
  cut?: (fields: Section) => Section; hold?: string | null;
  onAmend?: AmendField; ask?: string; tables?: boolean; beside?: (key: string) => ReactNode;
}) {
  const object = String(deal.Object ?? '?');
  const sections = schema?.Instrument.types[object];
  if (!schema || !sections) {
    return <DescriptorPanel title={`${object} (undeclared type)`} values={deal} />;
  }
  const shared = own ? sharedSections(schema.Instrument.types) : [];
  const of = (section: string) => schema.Instrument.sections[section] ?? {};
  const extra = Object.fromEntries(Object.entries(deal)
    .filter(([key]) => !sections.some((section) => key in of(section))));
  return (
    <>
      {hold && <div className="note">{hold}</div>}
      {sections.filter((section) => !shared.includes(section)).map((section) => {
        const { fields, values } = onScreen(of(section), deal);
        return (
          <DescriptorPanel key={section} title={schema.Instrument.titles?.[section] ?? section}
                           fields={cut(fields)} values={values}
                           onAmend={hold ? undefined : onAmend} {...panel} />
        );
      })}
      <DescriptorPanel title="Fields the engine does not know" values={extra} />
    </>
  );
}

/** `stated` is whether the DOCUMENT says this key, which is not whether the row has a value: a
 * field that must be stated shows what it would default to, and the red `*` is the difference
 * between a term the author wrote and a placeholder standing in for one. A `null` or a blank is
 * what a form round-trips for nothing, so it is not a statement here either - which is what the
 * engine's own booking verdict says. `required` is the Instrument store's reading, MUST BE
 * STATED; on every other store it is still `default=REQUIRED`, and this panel renders both. A
 * convention left unstated reads as one, lighter and named so; `currency` is the amount's. */
function FieldRow({ name, descriptor, value, stated, declared, onAmend, aside, beside, normal, ask,
                    tables, currency }: {
  name: string; descriptor?: Descriptor; value: unknown; stated?: boolean; declared: boolean;
  onAmend?: AmendField; aside?: ReactNode; beside?: ReactNode; normal?: boolean; ask?: string;
  tables?: boolean; currency?: string;
}) {
  const editable = onAmend !== undefined && descriptor !== undefined &&
    isEditableScalar(descriptor, value);
  const { text, title } = fieldLabel(name, descriptor);
  const convention = descriptor?.convention && !stated;
  return (
    <>
      <div className="k" title={title}>
        {text}
        {descriptor?.required && !stated && <span className="required"> *</span>}
        {currency && <span className="chip">{currency}</span>}
        {convention && <span className="hint">convention</span>}
        {!declared && <span className="hint">(the engine has no such field)</span>}
        {aside}
      </div>
      <div className={convention ? 'v convention' : 'v'}>
        {tables && onAmend && descriptor?.widget === 'Table'
          ? <TableEdit name={name} declared={descriptor} value={value} onAmend={onAmend} />
          : editable
            ? <EditableScalar name={name} descriptor={descriptor!} value={value} onAmend={onAmend!}
                              ask={ask} />
            : <FieldView value={value} descriptor={descriptor} normal={normal} />}
        {beside}
      </div>
    </>
  );
}

/** One table edited a cell at a time - a credit support list or a list of collateral - its
 * columns and their forms the declaration's own, a row added blank and the last one dropped. */
function TableEdit({ name, declared, value, onAmend }: {
  name: string; declared: Descriptor; value: unknown; onAmend: AmendField;
}) {
  const columns = declared.col_names ?? [];
  const support = declared.obj === 'CreditSupportList';
  const forms = support ? ['Integer', 'Float'] : (declared.obj ?? []) as string[];
  const rows: unknown[][] = support
    ? (token(value, '.CreditSupportList') ?? []) as unknown[][]
    : (Array.isArray(value) ? value : []).map((row) => columns.map(
      (column) => (row as Record<string, unknown>)[column]));
  const file = (next: unknown[][]) => onAmend(name, support
    ? { '.CreditSupportList': next }
    : next.map((row) => Object.fromEntries(columns.map((column, at) => [column, row[at]]))));
  const cell = (form: string, column: string): Descriptor => ({
    widget: ['Float', 'Percent'].includes(form) ? 'Float' : form === 'Integer' ? 'Integer' : 'Text',
    obj: form, description: column, value: null,
    factors: (declared.factors as Record<string, string[]> | undefined)?.[column],
  });
  return (
    <div>
      <DataTable columns={columns} index={[]} data={rows} cell={(r, c) => (
        <EditableScalar name={columns[c]} descriptor={cell(forms[c], columns[c])} value={rows[r][c]}
                        onAmend={(_, edited) => file(rows.map((row, at) => (at === r
                          ? row.map((held, column) => (column === c ? edited : held)) : row)))} />
      )} />
      {support && <div className="hint">{SUPPORT_ROW}</div>}
      <div className="statusrow">
        <button className="ghost" onClick={() => void file([...rows, columns.map(() => null)])}>
          add a row
        </button>
        {rows.length > 0 && (
          <button className="ghost" onClick={() => void file(rows.slice(0, -1))}>
            drop the last row
          </button>
        )}
      </div>
    </div>
  );
}

/** One scalar input, saved on Enter or on leaving it where its VALUE changed - `2000000.00` over
 * `2000000` is no change - and with `ask` only on the confirm beside it, declining restoring the
 * value held. A save refreshes the book and the effect below re-syncs: the file is the truth. A
 * refusal renders verbatim over the value that stands, until the next edit. A field naming price
 * factors offers the market's of its types, and takes a name typed. Exported because a quote
 * ladder's value cells are the same input. */
export function EditableScalar({ name, descriptor, value, onAmend, ask }: {
  name: string; descriptor: Descriptor; value: unknown; onAmend: AmendField; ask?: string;
}) {
  const offered = factorNames(descriptor, marketOf(useApp().state));
  const list = useId();
  const incoming = editRaw(value);
  const [raw, setRaw] = useState(incoming);
  const [refused, setRefused] = useState<string[] | null>(null);
  const [saving, setSaving] = useState(false);
  const [asking, setAsking] = useState(false);
  const [editing, setEditing] = useState(false);
  // a ref, not state: the blur a browser fires on an input disabled mid-save reads it current
  const filing = useRef(false);
  useEffect(() => { setRaw(incoming); setRefused(null); setAsking(false); }, [incoming]);

  const numeric = ['Float', 'Integer'].includes(descriptor.widget)
    || descriptor.obj === 'Percent' || descriptor.obj === 'Basis';
  const wire = (text: string) => JSON.stringify(encodeScalar(descriptor, text));
  const restore = () => { setRaw(incoming); setAsking(false); };
  // a refused value goes back to the one standing, the refusal kept beside it
  const refuse = (said: string[] | null) => { if (said) restore(); setRefused(said); };
  const edit = (text: string) => { setRaw(text); setRefused(null); };

  function commit(typed: string) {
    const text = numeric ? ungrouped(typed.trim()) : typed;
    if (numeric && text !== '' && !isFinite(Number(text))) {
      refuse([`${text} is not a number: write it with a decimal point, as 1234.5 or 1,234.5`]);
    } else if (wire(text) === wire(incoming)) restore();
    else if (ask) setAsking(true);
    else void save(text);
  }

  async function save(text: string) {
    if (filing.current) return;
    filing.current = true;
    setAsking(false);
    setSaving(true);
    try {
      refuse(await onAmend(name, encodeScalar(descriptor, text)));
    } catch (error) {
      refuse([failure(error).error]);
    } finally {
      filing.current = false;
      setSaving(false);
    }
  }

  const suffix = descriptor.obj === 'Percent' ? ' %' : descriptor.obj === 'Basis' ? ' bp' : '';
  let input;
  if (descriptor.widget === 'Dropdown') {
    input = (
      <select value={raw} disabled={saving}
              onChange={(e) => { edit(e.target.value); commit(e.target.value); }}>
        {/* what the block states where it is no option - blank where a required menu is unset */}
        {!(descriptor.values ?? []).includes(raw) && <option value={raw}>{raw}</option>}
        {(descriptor.values ?? []).map((option) => <option key={option}>{option}</option>)}
      </select>
    );
  } else if (descriptor.widget === 'Checkbox') {
    input = (
      <input type="checkbox" checked={raw === 'true'} disabled={saving}
             onChange={(e) => { edit(String(e.target.checked));
                                commit(String(e.target.checked)); }} />
    );
  } else {
    // a number reads as the desk reads one until it is edited, and then as the exact value
    const shown = numeric && !editing && raw === incoming && raw !== ''
      ? formatNumber(Number(raw), descriptor.sized ? 'money' : 'plain') : raw;
    input = (
      <input
        type={descriptor.widget === 'DatePicker' ? 'date' : 'text'}
        list={offered.length ? list : undefined}
        inputMode={numeric ? 'decimal' : undefined} value={shown} disabled={saving}
        onFocus={() => setEditing(true)}
        onChange={(e) => edit(e.target.value)}
        onBlur={() => { setEditing(false); commit(raw); }}
        onKeyDown={(e) => { if (e.key === 'Enter') commit(raw); }}
      />
    );
  }
  return (
    <span className="editcell">
      {input}{suffix && <span className="hint">{suffix}</span>}
      {offered.length > 0 && (
        <datalist id={list}>
          {offered.map((option) => <option key={option} value={option} />)}
        </datalist>
      )}
      {saving && <span className="hint">saving…</span>}
      {asking && (
        <span className="banner">
          {ask}
          <button className="ghost" onClick={() => void save(raw)}>file it</button>
          <button className="ghost" onClick={restore}>keep it as it was</button>
        </span>
      )}
      {refused && refused.map((message, i) => (
        <span key={i} className="refusal">{message}</span>
      ))}
    </span>
  );
}
