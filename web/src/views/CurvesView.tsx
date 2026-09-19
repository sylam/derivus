import { useEffect, useState } from 'react';
import { configureCurve, failure, getCurves } from '../api';
import { DataTable } from '../components/DataTable';
import { EmptyState } from '../components/EmptyState';
import { DescriptorPanel, EditableScalar } from '../components/FieldView';
import {
  curveFields, curveRequest, curveValues, edited, prefill, solvedFactor, withRow, type CurveForm,
} from '../curves';
import { useApp } from '../state';
import { curveOf } from '../tokens';
import type {
  CurveBlock, CurveOutcome, CurveRow, CurvesAnswer, Descriptor, Schema, Section,
} from '../types';

/** The family a curve block is - the one name this screen spells, and the one the service's own
 * answer is keyed by (`InterestRatePrices.<curve>`). Everything else here is a declaration. The
 * factor it WRITES is its stem, which is what the interpolation menu is keyed by. */
const FAMILY = 'InterestRatePrices';
const FACTOR = FAMILY.replace(/Prices$/, '');

/** The four row fields a request states, against the declarations that describe them - the block's
 * own quote row, which is where the quote's widget and the hold-out's two values come from. */
const ROW_FIELDS: [keyof CurveRow, string][] = [
  ['tenor', 'Tenor'], ['security', 'Security'], ['quote', 'Quoted_Market_Value'], ['use', 'Use'],
];

const TEXT: Descriptor = { widget: 'Text', description: '', value: '' };

const rowCells = (row: CurveRow) => ROW_FIELDS.map(([key]) => row[key]);

/** The panel's own declarations: the family's, and a menu row for a field it declares none for -
 * the curve's own scheme, which is a rule in a section and so has no column to be declared on. */
const panelFields = (declared: Section, fields: string[], menu: string[]): Section =>
  Object.fromEntries(fields.map((key) => [key, declared[key] ?? {
    widget: 'Dropdown', value: '', values: menu,
    description: "The scheme this curve alone is built under; blank takes the type's own",
  }]));

/** One cell as the row carries it: a cleared QUOTE is not a quote of zero, and travels as the null
 * that says no number was stated - which is what sends the row to the terminal, or refuses it. */
const cleared = (key: string, wire: unknown) =>
  ({ [key]: wire === '' && key === 'quote' ? null : wire }) as Partial<CurveRow>;

/** One block as the definition it is: what it was stated in, the rows it was solved from, and the
 * factor the bootstrap wrote where the market data store carries one. */
function CurveCard({ schema, name, block, panel, fields, factors, onEdit }: {
  schema: Schema; name: string; block: CurveBlock; panel: Section; fields: string[];
  factors: Record<string, unknown>; onEdit: (form: CurveForm) => void;
}) {
  const solved = solvedFactor(factors, block.curve);
  const knots = solved && Object.values(solved[1]).map(curveOf).find(Boolean)?.data.length;
  return (
    <>
      <DescriptorPanel title={name} fields={panel} values={curveValues(fields, block)} />
      {block.note && <div className="error-box">{block.note}</div>}
      <section className="card">
        <h3>{name} — rows ({block.rows.length})</h3>
        <div className="pager">
          <span>interpolation <b>{block.interpolation}</b></span>
          <button className="ghost" onClick={() => onEdit(prefill(block.curve, block))}>
            state these rows again
          </button>
        </div>
        <DataTable columns={ROW_FIELDS.map(([key]) => key)} index={[]}
                   data={block.rows.map(rowCells)} />
      </section>
      {solved && (
        <DescriptorPanel title={`${solved[0]} — ${knots} knots`} values={solved[1]}
                         fields={schema.Factor.types[solved[0].split('.')[0]]} />
      )}
    </>
  );
}

/** Setting one up: a seeded curve pre-fills the rows and conventions, or a new one is named with
 * its currency; the rows edit in place and the conventions through the same descriptor panel the
 * Bootstrapper screen uses. One button authors, solves and writes in one atomic request - the
 * answer names the knots and what the bootstrap rewrote, a refusal is the service's own words,
 * and the form stands as it was either way. */
function SetUp({ declared, panel, fields, answer, form, setForm }: {
  declared: Section; panel: Section; fields: string[]; answer: CurvesAnswer;
  form: CurveForm; setForm: (form: CurveForm) => void;
}) {
  const [outcome, setOutcome] = useState<CurveOutcome | null>(null);
  const [refused, setRefused] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const seeded = answer.seeded ?? {};
  const row = declared.Points?.sub_fields ?? {};

  async function save() {
    setSaving(true);
    setRefused(null);
    setOutcome(null);
    try {
      setOutcome(await configureCurve(curveRequest(form)));
    } catch (error) {
      setRefused(failure(error).error);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="panel">
      <div className="pager">
        <span>pre-fill from</span>
        <select value={form.curve in seeded ? form.curve : ''}
                onChange={(event) =>
                  setForm(prefill(event.target.value, seeded[event.target.value] ?? {}))}>
          <option value="">a new curve</option>
          {Object.keys(seeded).map((name) => <option key={name}>{name}</option>)}
        </select>
        <input type="text" value={form.curve} placeholder="curve"
               onChange={(event) => setForm({ ...form, curve: event.target.value })} />
        <span className="hint">{Object.keys(seeded).length} seeded here, none verified</span>
      </div>
      {seeded[form.curve]?.refused && (
        <div className="error-box">{seeded[form.curve].refused}</div>
      )}
      <DescriptorPanel
        title="what the curve is stated in" fields={panel}
        values={{ ...curveValues(fields, { ...form, conventions: form.stated }), ...form.edits }}
        onAmend={async (key, wire) => { setForm(edited(form, key, wire)); return null; }} />
      <section className="card">
        <h3>the benchmark rows ({form.rows.length})</h3>
        <DataTable
          columns={[...ROW_FIELDS.map(([key]) => key), '']} index={[]}
          data={form.rows.map((entry) => [...rowCells(entry), null])}
          cell={(r, c) => (c === ROW_FIELDS.length ? (
            <button className="ghost"
                    onClick={() => setForm({ ...form, rows: withRow(form.rows, r, null) })}>✕</button>
          ) : (
            <EditableScalar
              key={`${r}.${c}`} name={ROW_FIELDS[c][0]} value={form.rows[r][ROW_FIELDS[c][0]]}
              descriptor={row[ROW_FIELDS[c][1]] ?? TEXT}
              onAmend={async (key, wire) => {
                setForm({ ...form, rows: withRow(form.rows, r, cleared(key, wire)) });
                return null;
              }} />
          ))} />
        <button className="ghost"
                onClick={() => setForm({ ...form, rows: withRow(form.rows, form.rows.length, {}) })}>
          add a row
        </button>
      </section>
      <div className="statusrow">
        <button className="primary" disabled={saving || !form.curve.trim()}
                onClick={() => void save()}>set the curve up</button>
        {saving && <span className="chip running">solving</span>}
        {outcome && <span className="chip done">{outcome.block}</span>}
        {outcome && (
          <span className="hint">{(outcome.knots ?? []).length} knots:{' '}
            <span className="mono">{(outcome.knots ?? []).join(' ')}</span>
          </span>
        )}
        {(outcome?.rewrote ?? []).length > 0 && (
          <span className="chip on">rewrote {outcome!.rewrote!.join(', ')}</span>
        )}
      </div>
      {refused && <div className="error-box">{refused}</div>}
    </div>
  );
}

/** The curves half of the market: every curve block the book carries read back as the definition
 * it is - the rows it was solved from, the conventions they were authored under and the factor the
 * bootstrap wrote - beside the form that sets another up. THE ROWS ARE THE BENCHMARKS: a tenor
 * says what each instrument is, and the service is what reads it. */
export function CurvesView() {
  const { state } = useApp();
  const [answer, setAnswer] = useState<CurvesAnswer | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [form, setForm] = useState<CurveForm>(() => prefill('', {}));
  const source = state.source;

  // the definitions ride the etag poll, like everything else on screen: a write here, a tick or
  // another client's booking repaints them, and nothing holds a copy of what the file says
  useEffect(() => {
    if (source?.kind !== 'book') return;
    let live = true;
    getCurves()
      .then((next) => { if (live) { setAnswer(next); setError(null); } })
      .catch((failed) => { if (live) setError(failure(failed).error); });
    return () => { live = false; };
  }, [source]);

  if (source?.kind !== 'book') {
    return (
      <EmptyState
        title="A curve is set up in the live book."
        hint="This document was opened from a file. Start the service with --book <job file> and the book's curves, and the ones this workstation's seed can set up, appear here."
      />
    );
  }
  if (!state.schema || !state.doc) return null;
  if (error) {
    return <div className="main"><div className="panel"><div className="error-box">{error}</div></div></div>;
  }
  if (!answer) {
    return <EmptyState title="Reading the book's curves…" hint="Every curve block the book carries, as the definition it is." />;
  }

  const declared = state.schema.MarketPrices.types[FAMILY] ?? {};
  const fields = curveFields(declared, answer);
  const panel = panelFields(declared, fields, state.schema.Interpolation_factor_map[FACTOR] ?? []);
  const market = state.doc.Calc.MergeMarketData?.ExplicitMarketData ?? {};
  const names = Object.keys(answer.curves);

  return (
    <div className="main">
      <div className="panel">
        {names.length === 0 && (
          <div className="placeholder">this book carries no curve block yet</div>
        )}
        {names.map((name) => (
          <CurveCard key={name} schema={state.schema!} name={name} block={answer.curves[name]}
                     panel={panel} fields={fields} factors={market['Price Factors'] ?? {}}
                     onEdit={setForm} />
        ))}
      </div>
      <SetUp declared={declared} panel={panel} fields={fields} answer={answer} form={form}
             setForm={setForm} />
    </div>
  );
}
