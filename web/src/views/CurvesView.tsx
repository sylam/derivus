import { useEffect, useState } from 'react';
import { configureCurve, failure, getCurves } from '../api';
import { DataTable } from '../components/DataTable';
import { EmptyState } from '../components/EmptyState';
import { DescriptorPanel, EditableScalar, type AmendField } from '../components/FieldView';
import {
  QUIET_MS, commit, curveFields, curveRequest, due, nearPair, panelValues, prefill, solvedFactor,
  type CurveEdit, type CurveForm, type Editing,
} from '../curves';
import { useApp, written } from '../state';
import { curveOf, isObject, token } from '../tokens';
import type {
  ConfigSection, CurveBlock, CurveOutcome, CurveRow, CurvesAnswer, Descriptor, Schema,
  SeededCurve, Section,
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

const idle = (form: CurveForm): Editing => ({ form, at: 0, posted: 0, saving: false });

/** The panel's own declarations: the family's, and a menu row for a field it declares none for -
 * the curve's own scheme, which is a rule in a section and so has no column to be declared on. */
const panelFields = (declared: Section, fields: string[], menu: string[]): Section =>
  Object.fromEntries(fields.map((key) => [key, declared[key] ?? {
    widget: 'Dropdown', value: '', values: menu,
    description: "The scheme this curve alone is built under; the type's own clears its rule",
  }]));

/** One curve, edited in place: the conventions panel, the benchmark rows and the solved factor
 * beside them. EVERY COMMIT - a field's blur or Enter, a row added or removed, a use toggled -
 * folds into the form the card stands in and posts the whole curve through `POST /book/curve` a
 * beat later, so a burst of edits is one solve. A post in flight is a chip, a refusal is the
 * service's own words with the edited values standing, and the block the service wrote comes back
 * through the etag poll. `seeded` makes the card the NEW one: a pre-fill picker above the same
 * editor, blank again once the commit that completes it has created the curve. */
function CurveCard({ schema, title, block, seeded, fields, panel, declared, factors, typeScheme }: {
  schema: Schema; title: string; block?: CurveBlock; seeded?: Record<string, SeededCurve>;
  fields: string[]; panel: Section; declared: Section;
  factors: Record<string, unknown>; typeScheme: string;
}) {
  const { dispatch } = useApp();
  const [edit, setEdit] = useState<Editing>(() => idle(prefill(block?.curve ?? '', block ?? {},
                                                              fields)));
  const [outcome, setOutcome] = useState<CurveOutcome | null>(null);
  const [refused, setRefused] = useState<string | null>(null);
  const { form } = edit;

  // the block the service wrote repaints the card, exactly as a quote ladder's incoming value
  // repaints its input - unless the desk has moved on, whose edits are never thrown away
  const incoming = JSON.stringify([block ?? null, fields]);
  useEffect(() => {
    if (!block) return;
    setEdit((standing) => (standing.saving || standing.at > standing.posted ? standing
      : idle(prefill(block.curve, block, fields))));
  }, [incoming]);

  useEffect(() => {
    const timer = setTimeout(async () => {
      if (!due(edit, Date.now())) return;
      setEdit((standing) => ({ ...standing, posted: standing.at, saving: true }));
      try {
        const answer = await configureCurve(curveRequest(form, typeScheme));
        setOutcome(answer);
        await written(dispatch, answer);
        if (!block) setEdit(idle(prefill('', {}, fields)));  // the new card is a new curve again
      } catch (error) {
        setRefused(failure(error).error);
      } finally {
        setEdit((standing) => ({ ...standing, saving: false }));
      }
    }, QUIET_MS);
    return () => clearTimeout(timer);
  }, [edit]);

  /** One commit through the editor the quote ladders use: folded in, the clock moved, and what the
   * last post said cleared off the card. Nothing is refused HERE - the post is what answers. */
  const commits = (change: (key: string, wire: unknown) => CurveEdit): AmendField =>
    async (key, wire) => { fold(change(key, wire)); return null; };

  function fold(change: CurveEdit) {
    setRefused(null);
    setOutcome(null);
    setEdit((standing) => ({ ...standing, form: commit(standing.form, change), at: Date.now() }));
  }

  const near = nearPair(form, typeScheme);
  const solved = block && solvedFactor(factors, block.curve);
  const knots = solved && Object.values(solved[1]).map(curveOf).find(Boolean)?.data.length;
  const row = declared.Points?.sub_fields ?? {};
  return (
    <>
      {seeded && (
        <div className="pager">
          <span>pre-fill from</span>
          <select value={form.curve in seeded ? form.curve : ''}
                  onChange={(event) => setEdit(idle(prefill(
                    event.target.value, seeded[event.target.value] ?? {}, fields)))}>
            <option value="">a new curve</option>
            {Object.keys(seeded).map((name) => <option key={name}>{name}</option>)}
          </select>
          <EditableScalar name="Curve" value={form.curve}
                          descriptor={{ ...TEXT, description: 'curve' }}
                          onAmend={commits((key, value) => ({ field: key, value }))} />
          <span className="hint">{Object.keys(seeded).length} seeded here, none verified</span>
        </div>
      )}
      {seeded && seeded[form.curve]?.refused && (
        <div className="error-box">{seeded[form.curve].refused}</div>
      )}
      <DescriptorPanel
        title={title} fields={panel} values={panelValues(fields, form, typeScheme)}
        editable={(key) => key !== 'Near_Interpolation' || near.stated}
        // a curve that wants the scheme every curve of its type takes carries no rule of its own
        onAmend={commits((key, wire) => ({
          field: key, value: key === 'Interpolation' && wire === typeScheme ? '' : wire }))} />
      {block?.note && <div className="error-box">{block.note}</div>}
      <section className="card">
        <h3>{title} — rows ({form.rows.length})</h3>
        <div className="pager">
          <span>interpolation <b>{form.interpolation || typeScheme}</b></span>
          <span className="hint">
            {form.interpolation ? "this curve's rule" : "the type's default"}
          </span>
          {edit.saving && <span className="chip running">solving</span>}
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
        <DataTable
          columns={[...ROW_FIELDS.map(([key]) => key), '']} index={[]}
          data={form.rows.map((entry) => [...rowCells(entry), null])}
          cell={(r, c) => (c === ROW_FIELDS.length ? (
            <button className="ghost" onClick={() => fold({ row: r, patch: null })}>✕</button>
          ) : (
            <EditableScalar
              key={`${r}.${c}`} name={ROW_FIELDS[c][0]} value={form.rows[r][ROW_FIELDS[c][0]]}
              descriptor={row[ROW_FIELDS[c][1]] ?? TEXT}
              onAmend={commits((key, wire) => ({ row: r, patch: { [key]: wire } }))} />
          ))} />
        <button className="ghost" onClick={() => fold({ row: form.rows.length, patch: {} })}>
          add a row
        </button>
      </section>
      {refused && <div className="error-box">{refused}</div>}
      {solved && (
        <DescriptorPanel title={`${solved[0]} — ${knots} knots`} values={solved[1]}
                         fields={schema.Factor.types[solved[0].split('.')[0]]} />
      )}
    </>
  );
}

/** The curves half of the market: every curve block the book carries as the definition it is AND
 * the editor of it, with one card at the bottom that sets another up. THE ROWS ARE THE BENCHMARKS:
 * a tenor says what each instrument is, and the service is what reads it. */
export function CurvesView() {
  const { state } = useApp();
  const [answer, setAnswer] = useState<CurvesAnswer | null>(null);
  const [error, setError] = useState<string | null>(null);
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
  // the scheme every curve of the type is built under where no rule names one - what a card posts
  // BLANK for, so wanting what every curve has writes no rule at all
  const section: ConfigSection = state.schema.Configuration['Price Factor Interpolation'] ?? {};
  const params = token(market['Price Factor Interpolation'], '.ModelParams');
  const defaults = isObject(params) ? params[section.entry ?? ''] : undefined;
  const typeScheme = String((isObject(defaults) ? defaults[FACTOR] : '') || section.value || '');
  const names = Object.keys(answer.curves);
  const card = { schema: state.schema, fields, panel, declared, typeScheme,
                 factors: market['Price Factors'] ?? {} };

  return (
    <div className="main">
      <div className="panel">
        {names.length === 0 && (
          <div className="placeholder">this book carries no curve block yet</div>
        )}
        {names.map((name) => (
          <CurveCard key={name} {...card} title={name} block={answer.curves[name]} />
        ))}
        <CurveCard key="a new curve" {...card} title="a new curve" seeded={answer.seeded ?? {}} />
      </div>
    </div>
  );
}
