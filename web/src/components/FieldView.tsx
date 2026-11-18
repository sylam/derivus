import { useEffect, useRef, useState, type ReactNode } from 'react';
import { failure } from '../api';
import {
  curveOf, editRaw, encodeScalar, formatNumber, isEditableScalar, isObject, kindOf, label,
  offsetText, supportRows, token,
} from '../tokens';
import type { Descriptor } from '../types';
import { axisNames } from '../vols';
import { CurveChart } from './CurveChart';
import { DataTable } from './DataTable';
import { JsonView } from './JsonView';
import { VolSurface } from './VolSurface';

/** One field, read-only, dispatched on the VALUE first and the descriptor second. The value is
 * authoritative: a document may carry a key the schema does not declare, and a viewer that hides
 * such a key is a viewer nobody can trust. The descriptor refines - table columns, dropdown
 * hints, formatting - and its absence only ever downgrades to the JSON fallback, never to
 * nothing. A `.Curve` branches on ROW ARITY, because the widget token folds Surface and Space. */
export function FieldView({ value, descriptor }: { value: unknown; descriptor?: Descriptor }) {
  // --- wire tokens ---
  const curve = curveOf(value);
  if (curve) {
    const arity = curve.data[0]?.length ?? 0;
    if (arity === 2) {
      return <CurveChart data={curve.data} axes={axisNames(descriptor?.description, 2)} />;
    }
    if (arity === 3 || arity === 4) {
      return <VolSurface data={curve.data} descriptor={descriptor} />;
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
  for (const list of ['.DateList', '.DateEqualList', '.CreditSupportList'] as const) {
    const rows = token(value, list);
    if (Array.isArray(rows)) {
      const data = list === '.CreditSupportList' ? supportRows(rows) : rows;
      return <DataTable columns={descriptor?.col_names ?? []} index={[]} data={data} />;
    }
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
  if (value === null || value === undefined || value === '') {
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

/** A field's label: its description where that is a label, its own name where the description is
 * a paragraph - which is then the hover, so the prose is one pointer away rather than the column. */
function labelOf(name: string, descriptor?: Descriptor): { text: string; title?: string } {
  const description = descriptor?.description ?? name;
  return description.length > 64 ? { text: name.replace(/_/g, ' '), title: description }
    : { text: description };
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
 * what filing an edit does, where that is more than a save: a changed field asks it first. */
export function DescriptorPanel({ title, fields, values, onAmend, editable, aside, ask }: {
  title: string;
  fields?: Record<string, Descriptor>;
  values: Record<string, unknown>;
  onAmend?: AmendField;
  editable?: (key: string, descriptor: Descriptor) => boolean;
  aside?: (key: string) => ReactNode;
  ask?: string;
}) {
  const declared = Object.entries(fields ?? {});
  const undeclared = Object.keys(values).filter((key) => !(fields ?? {})[key]);
  const rows = [
    ...declared.map(([key, descriptor]) => ({ key, descriptor: descriptor as Descriptor | undefined })),
    ...undeclared.map((key) => ({ key, descriptor: undefined })),
  ].filter(({ key, descriptor }) => descriptor !== undefined || values[key] !== undefined);
  if (!rows.length) return null;

  return (
    <section className="card">
      <h3>{title}</h3>
      <div className="fields">
        {rows.map(({ key, descriptor }) => {
          const value = values[key] !== undefined ? values[key] : descriptor?.value;
          const container = descriptor?.widget === 'Container';
          const whole = isObject(value) ? value : {};
          return container ? (
            <div key={key} style={{ gridColumn: '1 / -1' }}>
              <DescriptorPanel
                title={labelOf(key, descriptor).text}
                fields={descriptor?.sub_fields}
                values={whole} editable={editable} ask={ask}
                onAmend={onAmend && (!editable || editable(key, descriptor!))
                  ? (field, wire) => onAmend(key, { ...whole, [field]: wire }) : undefined}
              />
            </div>
          ) : (
            <FieldRow key={key} name={key} descriptor={descriptor} value={value}
                      stated={values[key] !== undefined && values[key] !== null && values[key] !== ''}
                      declared={!!descriptor || !undeclared.includes(key)}
                      onAmend={onAmend && (!descriptor || !editable || editable(key, descriptor))
                        ? onAmend : undefined} aside={aside?.(key)} ask={ask} />
          );
        })}
      </div>
    </section>
  );
}

/** `stated` is whether the DOCUMENT says this key, which is not whether the row has a value: a
 * field that must be stated shows what it would default to, and the red `*` is the difference
 * between a term the author wrote and a placeholder standing in for one. A `null` or a blank is
 * what a form round-trips for nothing, so it is not a statement here either - which is what the
 * engine's own booking verdict says. `required` is the Instrument store's reading, MUST BE
 * STATED; on every other store it is still `default=REQUIRED`, and this panel renders both. */
function FieldRow({ name, descriptor, value, stated, declared, onAmend, aside, ask }: {
  name: string; descriptor?: Descriptor; value: unknown; stated?: boolean; declared: boolean;
  onAmend?: AmendField; aside?: ReactNode; ask?: string;
}) {
  const editable = onAmend !== undefined && descriptor !== undefined &&
    isEditableScalar(descriptor, value);
  const { text, title } = labelOf(name, descriptor);
  return (
    <>
      <div className="k" title={title}>
        {text}
        {descriptor?.required && !stated && <span className="required"> *</span>}
        {!declared && <span className="hint">(not declared)</span>}
        {aside}
      </div>
      <div className="v">
        {editable
          ? <EditableScalar name={name} descriptor={descriptor!} value={value} onAmend={onAmend!}
                            ask={ask} />
          : <FieldView value={value} descriptor={descriptor} />}
      </div>
    </>
  );
}

/** One scalar input, saved on Enter or on leaving it where its VALUE changed - `2000000.00` over
 * `2000000` is no change - and with `ask` only on the confirm beside it, declining restoring the
 * value held. A save refreshes the book and the effect below re-syncs: the file is the truth. A
 * refusal renders verbatim. Exported because a quote ladder's value cells are the same input. */
export function EditableScalar({ name, descriptor, value, onAmend, ask }: {
  name: string; descriptor: Descriptor; value: unknown; onAmend: AmendField; ask?: string;
}) {
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
  const restore = () => { setRaw(incoming); setAsking(false); setRefused(null); };

  function commit(typed: string) {
    const text = numeric ? typed.trim() : typed;
    if (numeric && text !== '' && !isFinite(Number(text))) {
      setRefused([`${text} is not a number: write it with a decimal point and no separators, `
        + 'as 1234.5']);
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
      setRefused(await onAmend(name, encodeScalar(descriptor, text)));
    } catch (error) {
      setRefused([failure(error).error]);
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
              onChange={(e) => { setRaw(e.target.value); commit(e.target.value); }}>
        {(descriptor.values ?? []).map((option) => <option key={option}>{option}</option>)}
      </select>
    );
  } else if (descriptor.widget === 'Checkbox') {
    input = (
      <input type="checkbox" checked={raw === 'true'} disabled={saving}
             onChange={(e) => { setRaw(String(e.target.checked));
                                commit(String(e.target.checked)); }} />
    );
  } else {
    // a number reads as the desk reads one until it is edited, and then as the exact value
    const shown = numeric && !editing && raw === incoming && raw !== ''
      ? formatNumber(Number(raw), descriptor.sized ? 'money' : 'plain') : raw;
    input = (
      <input
        type={descriptor.widget === 'DatePicker' ? 'date' : 'text'}
        inputMode={numeric ? 'decimal' : undefined} value={shown} disabled={saving}
        onFocus={() => setEditing(true)}
        onChange={(e) => setRaw(e.target.value)}
        onBlur={() => { setEditing(false); commit(raw); }}
        onKeyDown={(e) => { if (e.key === 'Enter') commit(raw); }}
      />
    );
  }
  return (
    <span className="editcell">
      {input}{suffix && <span className="hint">{suffix}</span>}
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
