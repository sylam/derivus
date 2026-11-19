import { useEffect, useState } from 'react';
import { configureCurve, failure, getCurves, stateSurvival } from '../api';
import { DataTable } from '../components/DataTable';
import { EmptyState } from '../components/EmptyState';
import { DescriptorPanel, EditableScalar, type AmendField } from '../components/FieldView';
import { Navigator } from '../components/Navigator';
import { Whose } from '../components/Whose';
import {
  QUIET_MS, commit, curveFields, curveRequest, due, nearPair, panelValues, prefill, solvedFactor,
  type CurveEdit, type CurveForm, type Editing,
} from '../curves';
import {
  SURVIVAL, creditCounterparties, entityText, refusedAt, spreadForm, spreadRequest, spreads,
  survivalCurve, valuationDay, type SpreadForm,
} from '../desk';
import { blockHint, marketOf, useApp, written } from '../state';
import { curveOf, formatNumber, isObject, tenorLabel, token } from '../tokens';
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
      {block && <div className="pager"><Whose path={['Market Prices', title]} /></div>}
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
        <>
          <div className="pager"><Whose path={['Price Factors', solved[0]]} /></div>
          <DescriptorPanel title={`${solved[0]} — ${knots} knots`} values={solved[1]}
                           fields={schema.Factor.types[solved[0].split('.')[0]]} />
        </>
      )}
    </>
  );
}

/** A credit page's id, behind a prefix no curve block's name carries. */
const CREDIT = 'credit:';

/** What a credit curve is written under on each desk. */
const HOLDS = {
  record: 'On this desk a curve is written for an entity the record declares, by a seat holding '
    + 'mark over the book.',
  plain: 'This desk keeps no record: a curve is written for any counterparty named.',
};

/** Rows of a credit curve as the officer reads one: a tenor, its spread - a number in basis points,
 * anything else as typed - and the survival to it. */
function SpreadTable({ rows }: { rows: [string, unknown, number][] }) {
  return (
    <table className="data">
      <thead><tr><th>Tenor</th><th className="n">Spread</th><th className="n">Survival</th></tr></thead>
      <tbody>
        {rows.map(([tenor, spread, survival], at) => (
          <tr key={at}>
            <td>{tenor}</td>
            <td className="n">{typeof spread === 'number' ? formatNumber(spread, 'basis')
              : String(spread)}</td>
            <td className="n">{isFinite(survival) ? formatNumber(100 * survival, 'percent') : '—'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** One counterparty's credit curve: the curve the book holds read back as the spreads typed, and
 * the form stating it - one flat spread or a spread per tenor, at a recovery offered at the
 * factor's convention - with what it will write beside it, written on one confirm. A refusal is the
 * service's own sentence beside what it is about. With no `id` the counterparty is typed by name. */
function CreditCard({ id, title, factor, base, convention, recording }: {
  id?: string; title: string; factor?: unknown; base: string; convention: number;
  recording: boolean;
}) {
  const { state, dispatch } = useApp();
  const [name, setName] = useState(id ?? '');
  const [form, setForm] = useState(() => spreadForm(factor, convention));
  const [said, setSaid] = useState<{ at: string; text: string } | null>(null);
  const [saving, setSaving] = useState(false);
  // the curve the book holds repaints the form - after this card's own write, or another client's
  const incoming = JSON.stringify(factor ?? null);
  useEffect(() => setForm(spreadForm(factor, convention)), [incoming]);

  const request = spreadRequest(name, form);
  const knots = survivalCurve(request, base).slice(1);
  const typed = Array.isArray(request.spread) ? request.spread
    : [{ tenor: '1Y', spread: request.spread }];
  const curve = curveOf(token(factor, 'Curve'));
  const recovery = Number(token(factor, 'Recovery_Rate') ?? convention);
  const edit = (change: Partial<SpreadForm>) => { setSaid(null); setForm({ ...form, ...change }); };
  const row = (at: number, cell: Partial<SpreadForm['rows'][number]>) =>
    edit({ rows: form.rows.map((held, i) => (i === at ? { ...held, ...cell } : held)) });
  const refusal = (at: string) => said?.at === at && <span className="refusal">{said.text}</span>;

  async function write() {
    setSaving(true);
    try {
      const answer = await stateSurvival(request);
      await written(dispatch, answer);
      setSaid({ at: 'done', text: `wrote ${answer.factor}` });
      if (id) return;
      // the counterparty now has a page of its own, which is where its curve reads back
      setName('');
      setForm(spreadForm(undefined, convention));
      dispatch({ type: 'PICK', screen: 'curves', id: CREDIT + request.counterparty });
    } catch (error) {
      const text = failure(error).error;
      setSaid({ at: refusedAt(text), text });
    } finally {
      setSaving(false);
    }
  }

  return (
    <>
      <section className="card credit">
        <h3>{title} — {SURVIVAL}.{name || '…'}</h3>
        <div className="pager">
          {id && <Whose path={['Price Factors', `${SURVIVAL}.${id}`]} />}
          <span>{recording ? HOLDS.record : HOLDS.plain}</span>
        </div>
        {curve ? (
          <>
            <div className="hint">As the book holds it, at a recovery of{' '}
              {formatNumber(100 * recovery, 'percent')}:</div>
            <SpreadTable rows={spreads(curve.data, recovery).map(([tenor, spread, survival]) =>
              [tenorLabel(tenor), spread * 1e4, survival])} />
          </>
        ) : id && (
          <div className="note">No credit curve yet: a CVA against this counterparty has no default
            to read, and no trade books under a netting set naming it until it has one.</div>
        )}
      </section>
      <section className="card credit">
        <h3>{curve ? 'Restate the curve' : 'State the curve'}</h3>
        {!id && (
          <div className="statusrow">
            <input placeholder={recording ? 'an entity the record declares' : 'counterparty'}
                   value={name} onChange={(event) => { setSaid(null); setName(event.target.value); }} />
            {refusal('counterparty')}
          </div>
        )}
        <div className="statusrow">
          <span>recovery</span>
          <input value={form.recovery} inputMode="decimal"
                 onChange={(event) => edit({ recovery: event.target.value })} />
          <span className="hint">the factor's convention is {String(convention)}</span>
        </div>
        <div className="statusrow">
          <label><input type="radio" checked={form.flat} onChange={() => edit({ flat: true })} />
            {' '}one flat spread</label>
          <label><input type="radio" checked={!form.flat} onChange={() => edit({ flat: false })} />
            {' '}a spread per tenor</label>
          {form.flat && (
            <>
              <input value={form.rows[0].spread} inputMode="decimal" placeholder="spread"
                     onChange={(event) => row(0, { spread: event.target.value })} />
              <span>bp</span>
            </>
          )}
        </div>
        {!form.flat && (
          <>
            <table className="data">
              <thead><tr><th>Tenor</th><th>Spread, bp</th><th /></tr></thead>
              <tbody>
                {form.rows.map((held, at) => (
                  <tr key={at}>
                    <td><input value={held.tenor} placeholder="5Y"
                               onChange={(event) => row(at, { tenor: event.target.value })} /></td>
                    <td><input value={held.spread} inputMode="decimal"
                               onChange={(event) => row(at, { spread: event.target.value })} /></td>
                    <td><button className="ghost" onClick={() => edit({
                      rows: form.rows.filter((_, i) => i !== at) })}>✕</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
            <button className="ghost" onClick={() => edit({
              rows: [...form.rows, { tenor: '', spread: '' }] })}>add a tenor</button>
          </>
        )}
        <div className="statusrow">{refusal('curve')}</div>
        <div className="hint">What it will write: each spread a hazard of the spread over one less
          the recovery, from {base} to its tenor ACT/365, a flat spread read to a year, and past the
          last tenor the curve runs on at its spread.</div>
        <SpreadTable rows={typed.map((held, at) =>
          [held.tenor, held.spread, Math.exp(-knots[at][1])])} />
        <div className="statusrow">
          <button onClick={() => void write()} disabled={saving}>
            write {SURVIVAL}.{request.counterparty || '…'}
          </button>
          {recording && state.seat && <span className="hint">as {state.seat}</span>}
          {saving && <span className="chip running">writing</span>}
          {said?.at === 'done' && <span className="chip done">{said.text}</span>}
          {refusal('act')}
        </div>
      </section>
    </>
  );
}

/** The curves half of the market: every curve block the book prices on as the definition it is,
 * marked whose it is, AND the editor of it, with one card at the bottom that sets another up.
 * THE ROWS ARE THE BENCHMARKS: a tenor says what each instrument is, and the service reads it. */
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
  const { doc } = state;
  const market = marketOf(state);
  if (!state.schema || !doc || !market) return null;
  if (error) {
    return <div className="main"><div className="panel"><div className="error-box">{error}</div></div></div>;
  }
  if (!answer) {
    return <EmptyState title="Reading the book's curves…" hint="Every curve block the book carries, as the definition it is." />;
  }

  const declared = state.schema.MarketPrices.types[FAMILY] ?? {};
  const fields = curveFields(declared, answer);
  const panel = panelFields(declared, fields, state.schema.Interpolation_factor_map[FACTOR] ?? []);
  // the scheme every curve of the type is built under where no rule names one - what a card posts
  // BLANK for, so wanting what every curve has writes no rule at all
  const section: ConfigSection = state.schema.Configuration['Price Factor Interpolation'] ?? {};
  const params = token(market['Price Factor Interpolation'], '.ModelParams');
  const defaults = isObject(params) ? params[section.entry ?? ''] : undefined;
  const typeScheme = String((isObject(defaults) ? defaults[FACTOR] : '') || section.value || '');
  const card = { schema: state.schema, fields, panel, declared, typeScheme,
                 factors: market['Price Factors'] ?? {} };
  // a credit curve per counterparty, the ones still without one filed first
  const recording = state.record.spine !== null;
  const entities = recording ? state.paper.data?.entities ?? [] : null;
  const credit = { base: valuationDay(doc), recording,
                   convention: Number(state.schema.Factor.types[SURVIVAL]?.Recovery_Rate?.value) };
  const counterparties = creditCounterparties(entities, doc, card.factors).map((id) => ({
    id, factor: card.factors[`${SURVIVAL}.${id}`],
    title: entities ? entityText(entities, id) : id,
  }));

  // filed by currency, the one family being every curve's; every card stays mounted, since a
  // burst of edits posts a beat after its last commit and a click away must not drop it
  return (
    <Navigator screen="curves" keep head={!Object.keys(answer.curves).length && (
      <div className="note">
        This book sets up no curve from benchmark rows: a rate curve it prices on is a price
        factor stated whole, read under <b>Market Data</b>; <b>a new curve</b> sets one up from
        benchmarks.
      </div>
    )} pages={[
      ...Object.entries(answer.curves).map(([name, block]) => ({
        id: name, folder: block.currency, label: block.curve,
        hint: blockHint(doc, ['Market Prices', name]),
        content: <CurveCard {...card} title={name} block={block} />,
      })),
      { id: 'a new curve', label: 'a new curve', accent: true,
        content: <CurveCard {...card} title="a new curve" seeded={answer.seeded ?? {}} /> },
      ...[...counterparties.filter((row) => !row.factor), ...counterparties.filter((row) => row.factor)]
        .map(({ id, factor, title }) => ({
          id: CREDIT + id, folder: factor ? 'credit curves' : 'no credit curve yet', label: title,
          content: <CreditCard {...credit} id={id} title={title} factor={factor} />,
        })),
      { id: CREDIT, label: 'a credit curve for another counterparty', accent: true,
        content: <CreditCard {...credit} title="another counterparty" /> },
    ]} />
  );
}
