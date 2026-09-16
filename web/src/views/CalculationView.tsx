import { Fragment, useEffect } from 'react';
import { failure, getResult, getTable, postExecute } from '../api';
import { DescriptorPanel } from '../components/FieldView';
import { DataTable } from '../components/DataTable';
import { TimeSeriesChart } from '../components/TimeSeriesChart';
import { useApp } from '../state';
import { formatNumber, isObject, token } from '../tokens';
import type { TableShape } from '../types';

const PAGE = 200;
const POLL_MS = 500;

const isScalar = (shape: TableShape) => shape.rows <= 1 && shape.columns.length === 0;

/** A page whose every index entry is a `.Timestamp` is a time series - the SHAPE rule that turns
 * an exposure profile into a chart without ever naming `exposure_profile`. */
const isDateIndexed = (index: unknown[]) =>
  index.length > 1 && index.every((entry) => token(entry, '.Timestamp') !== undefined);

export function CalculationView() {
  const { state, dispatch } = useApp();
  const { doc, schema, run } = state;

  // poll while the run is live
  useEffect(() => {
    if (!run.resultId || run.status === 'done' || run.status === 'error' || run.status === 'idle') {
      return;
    }
    const timer = setInterval(async () => {
      try {
        dispatch({ type: 'RUN_POLLED', summary: await getResult(run.resultId!) });
      } catch (error) {
        dispatch({ type: 'RUN_FAILED', error: failure(error).error });
      }
    }, POLL_MS);
    return () => clearInterval(timer);
  }, [run.resultId, run.status, dispatch]);

  // on done: fetch every scalar table for the stat strip, and open the default table
  useEffect(() => {
    if (run.status !== 'done' || !run.summary?.tables || !run.resultId) return;
    const tables = run.summary.tables;
    const scalarNames = Object.keys(tables).filter((name) => isScalar(tables[name]));
    Promise.all(scalarNames.map(async (name) =>
      [name, (await getTable(run.resultId!, name)).data[0]] as const,
    )).then((pairs) => dispatch({ type: 'SCALARS_LOADED', scalars: Object.fromEntries(pairs) }));
    if (!run.table) {
      const first = Object.keys(tables).filter((name) => !isScalar(tables[name]))
        .sort((a, b) => tables[b].columns.length === 0 ? -1 : a.localeCompare(b))[0];
      const preferred = tables['exposure_profile'] ? 'exposure_profile'
        : tables['mtm'] ? 'mtm' : first;
      if (preferred) dispatch({ type: 'TABLE_SELECTED', table: preferred });
    }
  }, [run.status, run.summary, run.resultId, run.table, dispatch]);

  // fetch the selected table's page
  useEffect(() => {
    if (!run.table || !run.resultId || run.status !== 'done') return;
    getTable(run.resultId, run.table, 0, PAGE)
      .then((page) => dispatch({ type: 'PAGE_LOADED', page }))
      .catch((error) => dispatch({ type: 'RUN_FAILED', error: failure(error).error }));
  }, [run.table, run.resultId, run.status, dispatch]);

  if (!doc || !schema) return null;
  const calc = doc.Calc.Calculation;
  const calcType = String(calc.Object ?? '');

  async function execute() {
    try {
      const submitted = await postExecute(doc!);
      dispatch({ type: 'RUN_SUBMITTED', resultId: submitted.result_id });
    } catch (error) {
      dispatch({ type: 'RUN_FAILED', error: failure(error).error });
    }
  }

  return (
    <div className="main">
      <div className="panel">
        <DescriptorPanel
          title={`Calculation — ${calcType}`}
          fields={schema.Calculation.types[calcType]}
          values={calc}
        />
        <div className="statusrow">
          <button className="primary" onClick={execute}
                  disabled={run.status === 'queued' || run.status === 'running'}>
            Execute
          </button>
          <StatusChip />
          {run.summary?.plan_hash && (
            <>
              <span className="chip" title="plan hash">
                <span className="mono">{run.summary.plan_hash.slice(0, 12)}</span></span>
              <span className="chip" title="values hash">
                <span className="mono">{run.summary.values_hash?.slice(0, 12)}</span></span>
              <span className="chip">seed {run.summary.seed}</span>
            </>
          )}
        </div>
        {run.error && <div className="error-box">{run.error}</div>}
        {run.status === 'done' && <Results />}
      </div>
    </div>
  );
}

function StatusChip() {
  const { state } = useApp();
  const { status, startedAt } = state.run;
  if (status === 'idle') return null;
  const elapsed = startedAt ? ((Date.now() - startedAt) / 1000).toFixed(1) : null;
  const kind = status === 'done' ? 'done' : status === 'error' ? 'error' : 'running';
  return (
    <span className={`chip ${kind}`}>
      {status}{elapsed && status !== 'done' ? ` · ${elapsed}s` : ''}
    </span>
  );
}

/** A stat value as one line: a list joins, a number formats, and a dict past the one level of
 * nesting a card gives it reads as itself. */
function statText(value: unknown): string {
  if (Array.isArray(value)) return value.map(statText).join(', ');
  if (typeof value === 'number') return formatNumber(value);
  if (value === null || value === undefined) return '—';
  return isObject(value) ? JSON.stringify(value) : String(value);
}

/** One stat the tile strip cannot hold: a dict as a key/value card, a dict of dicts as a card per
 * key one level down. The job-shaped verbs publish their WHOLE answer under one such key, so a
 * strip that kept only numbers left their page blank. */
function StatCard({ name, value, nest = true }: { name: string; value: unknown; nest?: boolean }) {
  const entries = Object.entries(value as Record<string, unknown>);
  const cards = nest ? entries.filter(([, inner]) => isObject(inner)) : [];
  const lines = entries.filter(([, inner]) => !nest || !isObject(inner));
  return (
    <section className="card">
      <h3>{name}</h3>
      {lines.length > 0 && (
        <div className="fields">
          {lines.map(([key, inner]) => (
            <Fragment key={key}>
              <div className="k">{key}</div>
              <div className="v">
                <span className={typeof inner === 'number' ? 'num' : undefined}>
                  {statText(inner)}</span>
              </div>
            </Fragment>
          ))}
        </div>
      )}
      {cards.map(([key, inner]) => (
        <div className="nested" key={key}><StatCard name={key} value={inner} nest={false} /></div>
      ))}
    </section>
  );
}

function Results() {
  const { state, dispatch } = useApp();
  const { run } = state;
  const tables = run.summary?.tables ?? {};
  const listed = Object.keys(tables).filter((name) => !isScalar(tables[name])).sort();
  // every key the service publishes appears: a number is a tile, a dict a card, anything else a
  // line. Four verbs answer ENTIRELY under a dict key, and Results empty beside it
  const stats = Object.entries(run.summary?.stats ?? {});
  const tiles = stats.filter(([, value]) => typeof value === 'number');
  const cards = stats.filter(([, value]) => isObject(value));
  const lines = stats.filter(([, value]) => typeof value !== 'number' && !isObject(value));

  return (
    <>
      {(Object.keys(run.scalars).length > 0 || tiles.length > 0) && (
        <div className="stats">
          {Object.entries(run.scalars).map(([name, value]) => (
            <div className="stat" key={name}>
              <div className="label">{name}</div>
              <div className="value">{statText(value)}</div>
            </div>
          ))}
          {tiles.map(([name, value]) => (
            <div className="stat" key={name}>
              <div className="label">{name}</div>
              <div className="value">{formatNumber(value as number)}</div>
            </div>
          ))}
        </div>
      )}
      {lines.map(([name, value]) => (
        <div className="pager" key={name}>
          <span>{name}</span><span className="mono">{statText(value)}</span>
        </div>
      ))}
      {cards.map(([name, value]) => <StatCard key={name} name={name} value={value} />)}
      {listed.length > 0 && (
        <div className="pager">
          <span>table</span>
          <select
            value={run.table ?? ''}
            onChange={(e) => dispatch({ type: 'TABLE_SELECTED', table: e.target.value })}
          >
            {listed.map((name) => (
              <option key={name} value={name}>
                {name} ({tables[name].rows}×{tables[name].columns.length || 1})
              </option>
            ))}
          </select>
          {run.table && run.page && tables[run.table] &&
            tables[run.table].rows > run.page.data.length && (
              <span>first {run.page.data.length} of {tables[run.table].rows} rows</span>
            )}
        </div>
      )}
      {run.page && isDateIndexed(run.page.index) && <TimeSeriesChart page={run.page} />}
      {run.page && (
        <div className="card">
          <DataTable columns={run.page.columns} index={run.page.index} data={run.page.data} />
        </div>
      )}
    </>
  );
}
