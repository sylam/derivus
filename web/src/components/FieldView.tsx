import { curveOf, formatNumber, isObject, offsetText, token } from '../tokens';
import type { Descriptor } from '../types';
import { CurveChart } from './CurveChart';
import { DataTable } from './DataTable';
import { JsonView } from './JsonView';
import { SurfaceHeatmap } from './SurfaceHeatmap';

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
    if (arity === 2) return <CurveChart data={curve.data} />;
    if (arity === 3 || arity === 4) return <SurfaceHeatmap data={curve.data} />;
    return <JsonView value={value} />;
  }
  const stamp = token(value, '.Timestamp');
  if (stamp !== undefined) return <span className="num">{String(stamp)}</span>;
  if (token(value, '.DateOffset') !== undefined) {
    return <span className="num">{offsetText(value)}</span>;
  }
  const grid = token(value, '.Grid');
  if (Array.isArray(grid)) {
    return <span className="num">{grid.map(offsetText).join(' ')}</span>;
  }
  const percent = token(value, '.Percent');
  if (percent !== undefined) return <span className="num">{formatNumber(percent as number)} %</span>;
  const basis = token(value, '.Basis');
  if (basis !== undefined) return <span className="num">{formatNumber(basis as number)} bp</span>;
  for (const list of ['.DateList', '.DateEqualList', '.CreditSupportList'] as const) {
    const rows = token(value, list);
    if (Array.isArray(rows)) {
      return <DataTable columns={descriptor?.col_names ?? []} index={[]} data={rows} />;
    }
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
      return <DataTable columns={descriptor?.col_names ?? []} index={[]} data={value} />;
    }
    return <JsonView value={value} />;
  }
  if (isObject(value)) return <JsonView value={value} />;

  // --- scalars ---
  if (value === null || value === undefined || value === '') {
    return <span className="hint">—</span>;
  }
  if (typeof value === 'boolean') return <span>{value ? '✓ true' : '✗ false'}</span>;
  if (typeof value === 'number') return <span className="num">{formatNumber(value)}</span>;
  return (
    <span>
      {String(value)}
      {descriptor?.values && descriptor.values.length > 1 && (
        <span className="hint">({descriptor.values.join(' | ')})</span>
      )}
    </span>
  );
}

/** One card of labelled fields: declared descriptors in declaration order with the document's
 * values over them, then every UNDECLARED key of the value dict - visible, marked, at the end. */
export function DescriptorPanel({ title, fields, values }: {
  title: string;
  fields?: Record<string, Descriptor>;
  values: Record<string, unknown>;
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
          return container ? (
            <div key={key} style={{ gridColumn: '1 / -1' }}>
              <DescriptorPanel
                title={descriptor?.description ?? key}
                fields={descriptor?.sub_fields}
                values={isObject(value) ? value : {}}
              />
            </div>
          ) : (
            <FieldRow key={key} name={key} descriptor={descriptor} value={value}
                      declared={!!descriptor || !undeclared.includes(key)} />
          );
        })}
      </div>
    </section>
  );
}

function FieldRow({ name, descriptor, value, declared }: {
  name: string; descriptor?: Descriptor; value: unknown; declared: boolean;
}) {
  return (
    <>
      <div className="k">
        {descriptor?.description ?? name}
        {descriptor?.required && !value && <span className="required"> *</span>}
        {!declared && <span className="hint">(not declared)</span>}
      </div>
      <div className="v"><FieldView value={value} descriptor={descriptor} /></div>
    </>
  );
}
