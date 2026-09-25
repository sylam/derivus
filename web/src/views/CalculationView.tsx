import { Fragment, useEffect, useState, type ReactNode } from 'react';
import {
  failure, getCalculations, getResult, getTable, postExecute, runCalculation, saveCalculation,
} from '../api';
import { DescriptorPanel } from '../components/FieldView';
import { DataTable } from '../components/DataTable';
import { Navigator, type Page } from '../components/Navigator';
import { TimeSeriesChart } from '../components/TimeSeriesChart';
import { useApp } from '../state';
import { formatNumber, isObject, token } from '../tokens';
import type { Schema, TableShape } from '../types';

const PAGE = 200;
const POLL_MS = 500;

const isScalar = (shape: TableShape) => shape.rows <= 1 && shape.columns.length === 0;

/** A page whose every index entry is a `.Timestamp` is a time series - the SHAPE rule that turns
 * an exposure profile into a chart without ever naming `exposure_profile`. */
const isDateIndexed = (index: unknown[]) =>
  index.length > 1 && index.every((entry) => token(entry, '.Timestamp') !== undefined);

type Calculations = Record<string, Record<string, unknown>>;

/** A calculation's dials: every key but `Object`, which is the type the page is filed under. */
const dials = (block: Record<string, unknown>) =>
  Object.fromEntries(Object.entries(block).filter(([key]) => key !== 'Object'));
type Save = (name: string, calculation: Record<string, unknown> | null) =>
  Promise<string[] | null>;
type Submit = (origin: string, post: () => Promise<{ result_id: string }>) => Promise<void>;

/** The calculations a desk runs: the book's own, read-only here, and the ones this workstation
 * keeps under a name - filed by type, each its own dials over the book's and its own Run, over the
 * whole book or one subtree. A named calculation lives in `DV_HOME`, never in the book or the
 * record, and runs in the curiosity lane, so nothing it does is recorded. */
export function CalculationView() {
  const { state, dispatch } = useApp();
  const { doc, schema, run } = state;
  const live = state.source?.kind === 'book';
  const [saved, setSaved] = useState<Calculations>({});
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!live) return;
    getCalculations()
      .then((answer) => { setSaved(answer.calculations); setError(null); })
      .catch((failed) => setError(failure(failed).error));
  }, [live]);

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

  /** One submission, filed under the page that asked. */
  const submit: Submit = async (origin, post) => {
    const submitted = await post();
    dispatch({ type: 'RUN_SUBMITTED', resultId: submitted.result_id, origin });
  };

  /** A save answered by the file as it now stands, or by what to fix. */
  const save: Save = async (name, calculation) => {
    const outcome = await saveCalculation(name, calculation);
    if (outcome.written && outcome.calculations) setSaved(outcome.calculations);
    return outcome.written ? null : outcome.refused ?? ['refused'];
  };

  const scopes = doc.Calc.Deals.Deals.Children.map((node, position) => {
    const deal = node.Instrument['.Deal'];
    return [String(position), `${deal.Object ?? '?'} ${deal.Reference ?? ''}`] as const;
  });
  const calculation = doc.Calc.Calculation;
  const type = String(calculation.Object ?? '');
  const pages: Page[] = [
    { id: 'book', label: "the book's own", content: (
      <>
        <DescriptorPanel title={`Calculation — ${type}`} fields={schema.Calculation.types[type]}
                         values={dials(calculation)} />
        <RunStrip origin="book" submit={() => submit('book', () => postExecute(doc))} />
      </>
    ) },
    ...Object.entries(saved).map(([name, block]) => ({
      id: name, folder: String(block.Object), label: name, content: (
        <NamedCalculation schema={schema} name={name} block={block} scopes={scopes}
                          save={save} submit={submit} />
      ),
    })),
    ...(live ? [{ id: '+', label: 'a new calculation', accent: true, content: (
      <NewCalculation types={Object.keys(schema.Calculation.types)} taken={Object.keys(saved)}
                      create={async (name, type) => {
                        const refused = await save(name, { Object: type });
                        if (!refused) dispatch({ type: 'PICK', screen: 'calculation', id: name });
                        return refused;
                      }} />
    ) }] : []),
  ];

  return (
    <Navigator screen="calculation" pages={pages}
               head={error && <div className="error-box">{error}</div>} />
  );
}

/** One named calculation: its declared dials over what it states, every edit saved whole, a Run
 * over the whole book or one top-level node, and the run it asked for beneath. */
function NamedCalculation({ schema, name, block, scopes, save, submit }: {
  schema: Schema; name: string; block: Record<string, unknown>;
  scopes: (readonly [string, string])[]; save: Save; submit: Submit;
}) {
  const [scope, setScope] = useState('');
  const type = String(block.Object);
  return (
    <>
      <DescriptorPanel
        title={`${name} — ${type}`} fields={schema.Calculation.types[type]} values={dials(block)}
        onAmend={(key, wire) => save(name, { ...block, [key]: wire })} />
      <RunStrip
        origin={name}
        submit={() => submit(name, () => runCalculation(name, scope || undefined))}
        extra={
          <>
            <select value={scope} onChange={(event) => setScope(event.target.value)}>
              <option value="">the whole book</option>
              {scopes.map(([path, label]) => <option key={path} value={path}>{label}</option>)}
            </select>
            <button className="ghost" onClick={() => void save(name, null)}>delete</button>
          </>
        } />
    </>
  );
}

/** A new calculation: a name and a type, saved as the type alone - every dial at the book's own
 * until one is changed. */
function NewCalculation({ types, taken, create }: {
  types: string[]; taken: string[];
  create: (name: string, type: string) => Promise<string[] | null>;
}) {
  const [name, setName] = useState('');
  const [type, setType] = useState(types[0] ?? '');
  const [refused, setRefused] = useState<string[] | null>(null);
  const clash = taken.includes(name.trim());
  return (
    <section className="card">
      <h3>a new calculation</h3>
      <div className="pager" style={{ padding: '0 14px' }}>
        <input type="text" value={name} placeholder="its name"
               onChange={(event) => setName(event.target.value)} />
        <select value={type} onChange={(event) => setType(event.target.value)}>
          {types.map((one) => <option key={one}>{one}</option>)}
        </select>
        <button className="primary" disabled={!name.trim() || clash}
                onClick={async () => setRefused(await create(name.trim(), type))}>create</button>
        {clash && <span className="hint">a calculation is already saved under that name</span>}
      </div>
      {refused?.map((message, i) => <div key={i} className="error-box">{message}</div>)}
    </section>
  );
}

/** The Run button, the run's own status and replay tuple, and its results - shown under the page
 * that asked, so a named run's numbers never read as the book's own. */
function RunStrip({ origin, submit, extra }: {
  origin: string; submit: () => Promise<void>; extra?: ReactNode;
}) {
  const { state } = useApp();
  const { run } = state;
  const [refused, setRefused] = useState<string | null>(null);
  const mine = run.origin === origin;
  const busy = run.status === 'queued' || run.status === 'running';
  return (
    <>
      <div className="statusrow">
        <button className="primary" disabled={busy} onClick={async () => {
          setRefused(null);
          try {
            await submit();
          } catch (error) {
            setRefused(failure(error).error);
          }
        }}>
          Run
        </button>
        {extra}
        {mine && <StatusChip />}
        {mine && run.summary?.plan_hash && (
          <>
            <span className="chip" title="plan hash">
              <span className="mono">{run.summary.plan_hash.slice(0, 12)}</span></span>
            <span className="chip" title="values hash">
              <span className="mono">{run.summary.values_hash?.slice(0, 12)}</span></span>
            <span className="chip">seed {run.summary.seed}</span>
          </>
        )}
      </div>
      {refused && <div className="error-box">{refused}</div>}
      {mine && run.error && <div className="error-box">{run.error}</div>}
      {mine && run.status === 'done' && <Results />}
    </>
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
