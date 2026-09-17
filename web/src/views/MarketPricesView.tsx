import { useEffect, useState } from 'react';
import { failure, getResult, tickBloomberg, tickMarket } from '../api';
import { DataTable } from '../components/DataTable';
import { DescriptorPanel, EditableScalar } from '../components/FieldView';
import { useApp, written } from '../state';
import { isObject } from '../tokens';
import type { Descriptor, Schema, Section } from '../types';

const POLL_MS = 500;

/** A quote row's columns: a table declares them by name, a container by its sub-fields - the two
 * `Points` families declare the same row both ways. A container sub-field is the INSTRUMENT the
 * quote is a price for, structure rather than a column, and the row's `Descriptor` names it. */
const columnsOf = (declared: Descriptor) =>
  declared.col_names ?? Object.entries(declared.sub_fields ?? {})
    .filter(([, sub]) => sub.widget !== 'Container').map(([key]) => key);

/** One quote column's descriptor: a container declares its sub-field outright, and a table
 * publishes a rendering type per column, which names the same scalar form. */
function columnDescriptor(declared: Descriptor, column: string): Descriptor {
  const sub = declared.sub_fields?.[column];
  if (sub) return sub;
  const kind = (declared.sub_types?.[columnsOf(declared).indexOf(column)] ?? {}) as
    { type?: string };
  return {
    widget: kind.type === 'date' ? 'DatePicker' : kind.type === 'numeric' ? 'Float' : 'Text',
    description: column, value: null,
  };
}

/** The block's own quote ladder: the declared field whose ROW declares every value column. That
 * predicate is the engine's own (`quote_containers`), read off the store the service publishes
 * rather than spelled as a table name here. */
function quoteField(fields: Section | undefined, values: string[]) {
  return Object.entries(fields ?? {}).find(([, declared]) =>
    values.every((name) => columnsOf(declared).includes(name)));
}

/** One `Market Prices` block: its family's declarations over the instrument, then the quote ladder
 * under its declared columns with the VALUE columns editable. An edit posts the WHOLE block back
 * through the tick verb, which moves a quoted value, its two-way and its timestamp and refuses a
 * structural change by name; a refusal renders verbatim and a success re-reads the book. */
function MarketPriceBlock({ schema, name, block, live }: {
  schema: Schema; name: string; block: unknown; live: boolean;
}) {
  const { dispatch } = useApp();
  const fields = schema.MarketPrices.types[name.split('.')[0]];
  const instrument = isObject(block) ? block['instrument'] : undefined;
  if (!fields || !isObject(instrument)) {
    return <DescriptorPanel title={name} values={isObject(block) ? block : { value: block }} />;
  }
  const quotes = quoteField(fields, schema.MarketPrices.values);
  const ladder = quotes && instrument[quotes[0]];
  const rows = Array.isArray(ladder) ? ladder as Record<string, unknown>[] : [];
  const columns = quotes ? columnsOf(quotes[1]) : [];
  const editable = new Set(live ? schema.MarketPrices.values : []);

  async function save(row: number, column: string, wire: unknown) {
    const next = JSON.parse(JSON.stringify(block)) as { instrument: Record<string, unknown> };
    (next.instrument[quotes![0]] as Record<string, unknown>[])[row][column] = wire;
    return written(dispatch, await tickMarket({ [name]: next }));
  }

  return (
    <>
      <DescriptorPanel
        title={name} values={instrument}
        fields={Object.fromEntries(
          Object.entries(fields).filter(([key]) => key !== quotes?.[0]))} />
      {rows.length > 0 && (
        <section className="card">
          <h3>{name} — {quotes![0]} ({rows.length})</h3>
          <DataTable
            columns={columns} index={[]}
            data={rows.map((row) => columns.map((column) => row[column]))}
            cell={(r, c) => (editable.has(columns[c]) ? (
              <EditableScalar
                key={`${r}.${columns[c]}`} name={columns[c]}
                descriptor={columnDescriptor(quotes![1], columns[c])}
                value={rows[r][columns[c]]}
                onAmend={(column, wire) => save(r, column, wire)} />
            ) : undefined)} />
        </section>
      )}
    </>
  );
}

/** The terminal round trip as one button: a queued job, polled the way an execute is, its outcome
 * read off the run's own stats. Nothing here names a pair - the scope defaults to the desk's. */
function BloombergTick() {
  const [run, setRun] = useState<{ id: string; status: string } | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [outcome, setOutcome] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    if (!run || run.status === 'done' || run.status === 'error') return;
    const timer = setInterval(async () => {
      try {
        const summary = await getResult(run.id);
        setNote(summary.progress?.note ?? null);
        setRun({ id: run.id, status: summary.status });
        if (summary.status === 'done') {
          setOutcome((summary.stats?.Bloomberg ?? {}) as Record<string, unknown>);
        } else if (summary.status === 'error') setOutcome({ refused: [summary.error] });
      } catch (error) {
        setRun({ id: run.id, status: 'error' });
        setOutcome({ refused: [failure(error).error] });
      }
    }, POLL_MS);
    return () => clearInterval(timer);
  }, [run?.id, run?.status]);

  async function tick() {
    setOutcome(null);
    setNote(null);
    try {
      const submitted = await tickBloomberg();
      setRun({ id: submitted.result_id, status: submitted.status });
    } catch (error) {
      setRun({ id: '', status: 'error' });
      setOutcome({ refused: [failure(error).error] });
    }
  }

  const named = (key: string) => (outcome?.[key] ?? []) as string[];
  const live = run?.status === 'queued' || run?.status === 'running';
  return (
    <>
      <div className="statusrow">
        <button className="ghost" onClick={tick} disabled={live}>tick from Bloomberg</button>
        {run && (
          <span className={`chip ${run.status === 'done' ? 'done'
            : run.status === 'error' ? 'error' : 'running'}`}>{run.status}</span>
        )}
        {note && <span className="mono">{note}</span>}
        {['installed', 'updated', 'new_factors'].map((key) => named(key).length > 0 && (
          <span key={key} className="chip on">{key}: {named(key).join(', ')}</span>
        ))}
      </div>
      {named('refused').map((message, i) => (
        <div key={i} className="error-box">{message}</div>
      ))}
    </>
  );
}

/** The quotes half of the market data: every `Market Prices` block filed by family, its values
 * editable over the live book, and one button that goes and gets the desk's own. */
export function MarketPricesView() {
  const { state } = useApp();
  const { doc, schema } = state;
  if (!doc || !schema) return null;

  const prices = doc.Calc.MergeMarketData?.ExplicitMarketData?.['Market Prices'];
  const blocks = isObject(prices) ? prices : {};
  const names = Object.keys(blocks).sort();
  const live = state.source?.kind === 'book';

  return (
    <div className="main">
      <div className="panel">
        <BloombergTick />
        {names.length === 0 && <div className="placeholder">this book quotes nothing</div>}
        {names.map((name, i) => (
          <div key={name}>
            {name.split('.')[0] !== names[i - 1]?.split('.')[0] && (
              <div className="pager"><b>{name.split('.')[0]}</b></div>
            )}
            <MarketPriceBlock schema={schema} name={name} block={blocks[name]} live={live} />
          </div>
        ))}
      </div>
    </div>
  );
}
