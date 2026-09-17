import type { ReactNode } from 'react';
import { formatNumber, label } from '../tokens';

/** Runs of equal adjacent values, for the top row of a two-level header. */
function runs(values: string[]): { text: string; span: number }[] {
  const out: { text: string; span: number }[] = [];
  for (const text of values) {
    const last = out[out.length - 1];
    if (last && last.text === text) last.span += 1;
    else out.push({ text, span: 1 });
  }
  return out;
}

function Cell({ value }: { value: unknown }) {
  if (value === null || value === undefined) return <td className="n">—</td>;
  if (typeof value === 'number') return <td className="n">{formatNumber(value)}</td>;
  return <td>{label(value)}</td>;
}

/** Any frame the wire carries: `columns` may be plain labels or same-length tuples (a MultiIndex
 * renders as a two-row header), `index` labels the leading column when present, and a row may be
 * a plain scalar (a serialised vector). `cell` replaces one cell's content by (row, column) -
 * which is what makes a quote ladder's value columns editable without a second table renderer. */
export function DataTable({ columns, index, data, cell }: {
  columns: unknown[]; index: unknown[]; data: unknown[];
  cell?: (row: number, column: number) => ReactNode | undefined;
}) {
  const twoLevel = columns.length > 0 &&
    columns.every((c) => Array.isArray(c) && c.length >= 2);
  const hasIndex = index.length > 0;
  const rows = data.map((row) => (Array.isArray(row) ? row : [row]));

  return (
    <div className="tablewrap">
      <table className="data">
        <thead>
          {twoLevel ? (
            <>
              <tr>
                {hasIndex && <th />}
                {runs(columns.map((c) => label((c as unknown[])[0]))).map((run, i) => (
                  <th key={i} colSpan={run.span}>{run.text}</th>
                ))}
              </tr>
              <tr>
                {hasIndex && <th />}
                {columns.map((c, i) => (
                  <th key={i}>{label((c as unknown[]).slice(1))}</th>
                ))}
              </tr>
            </>
          ) : (
            <tr>
              {hasIndex && <th />}
              {columns.map((c, i) => <th key={i}>{label(c)}</th>)}
            </tr>
          )}
        </thead>
        <tbody>
          {rows.map((row, r) => (
            <tr key={r}>
              {hasIndex && <td>{label(index[r])}</td>}
              {row.map((value, c) => {
                const override = cell?.(r, c);
                return override === undefined
                  ? <Cell key={c} value={value} /> : <td key={c}>{override}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
