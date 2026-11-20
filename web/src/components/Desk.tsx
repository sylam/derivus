// The widgets more than one screen shows: the way back, a deal's side and level cells, an amount,
// the strip saying what a screen's numbers are on, a quote's ticket card, a client's credit curve,
// the agreement form and a deal's editor.

import { useEffect, useRef, useState, type ReactNode } from 'react';
import { amendDeal, declareAgreement, failure, stateSurvival } from '../api';
import { SET, cellText, dateOf, describe, fieldsOf, quoted, toRows, type Picked } from '../blotter';
import {
  SURVIVAL, entityText, instantMs, marketSource, refusedAt, spreadForm, spreadRequest, spreads,
  stampText, survivalCurve, valuationDay, type SpreadForm,
} from '../desk';
import type { Agreement, Entity } from '../positions';
import { NO_PAPER } from '../positions';
import {
  STANDS, clientText, firmSeconds, firmText, onMarketAxis, salesName, solvedNumbers, spreadText,
  standsAt, ticketLegs, verdictOf, type TicketLeg,
} from '../quote';
import { heldBack, marketOf, spineOf, useApp, written } from '../state';
import {
  COLLATERAL, curveOf, formatNumber, onPaper, statedOnly, tenorLabel, termsMoved, token,
  uncollateralised, type Kind,
} from '../tokens';
import type { DealNode, JobDoc, PendingQuote, Schema, SettlementState } from '../types';
import { DealPanels, FieldView, type AmendField } from './FieldView';
import { Whose } from './Whose';

/** The way back to the tab a page was opened from, where it was opened from one. */
export function Back() {
  const { state, dispatch, labels } = useApp();
  const back = state.back === null ? undefined : labels[state.back];
  return back ? (
    <button className="ghost" onClick={() => dispatch({ type: 'TAB', tab: state.back! })}>
      ‹ back to {back}
    </button>
  ) : null;
}

/** Whose side the Side column states. */
export const SIDE = 'the desk\'s side of the trade, as the desk booked it: Sell is the desk selling, '
  + 'whatever the client calls the same trade';

/** Buy or sell as a chip, whatever the family calls it - Buy_Sell, Payer_Receiver, a lender - and
 * always the desk's. */
export function Side({ side }: { side?: Picked }) {
  const text = side ? cellText(side.value) : undefined;
  if (!text) return null;
  const short = ['Sell', 'Pay', 'Payer', 'Lender'].includes(text) ? 'sell' : 'buy';
  return (
    <span className={`side ${short}`} title={`${SIDE} (${side!.title ?? side!.key})`}>{text}</span>
  );
}

/** One picked column. A shape a cell cannot state says so and points at the expander, which
 * renders it whole - the viewer never hides a value it cannot format. */
export function Value({ picked, object, schema, kind }: {
  picked?: Picked; object: string; schema: Schema | null; kind?: Kind;
}) {
  if (!picked) return <td className="n" />;
  const text = cellText(picked.value, kind);
  const exact = typeof picked.value === 'number' ? `: ${picked.value}` : '';
  const title = picked.title ?? describe(schema, object, picked.key) + exact;
  if (text === undefined) {
    return <td className="n"><span className="hint" title={title}>{'{…}'}</span></td>;
  }
  return <td className="n" title={title}>{text}</td>;
}

/** An amount, in whole units unless it is an adjustment, in the `currency` beside it where the
 * header does not name one. A null is an absence, and says why on hover. */
export function Amount({ value, currency, absent, kind = 'whole' }: {
  value: number | null; currency?: string | null; absent: string; kind?: 'whole' | 'money';
}) {
  return (
    <td className={`n${value !== null && value < 0 ? ' neg' : ''}`}
        title={value === null ? undefined : String(value)}>
      {value === null ? <span className="hint" title={absent}>—</span> : (
        <>{formatNumber(value, kind)}{currency && <> <span className="unit">{currency}</span></>}</>
      )}
    </td>
  );
}

/** What the numbers on a desk screen are ON: the day the book values on, when they were computed,
 * the currency they report in and where the market came from as far as the book says - and, where
 * the answer was read at an older clock than the book on screen, that in words. `doc` is the
 * document they were computed over where that is not the one on screen, and `again` what brings
 * fresh ones where nothing reads them again on its own. */
export function OnStrip({ computed, title, currency, lagging, loading, doc, again }: {
  computed: ReactNode; title?: string; currency: ReactNode; lagging: boolean; loading: boolean;
  doc?: JobDoc; again?: string;
}) {
  const { state } = useApp();
  const on = doc ?? state.doc;
  const market = marketSource(on, marketOf(state));
  return (
    <div className={`onstrip${lagging ? ' behind' : ''}`}>
      <span><span className="k">valued on</span><b>{valuationDay(on) || '—'}</b></span>
      <span title={title}><span className="k">computed</span>{computed}</span>
      <span><span className="k">reported in</span><b>{currency || '—'}</b></span>
      <span>
        <span className="k">market</span>
        {market.file ? <><b>{market.file}</b> with the book's own block over it</>
          : <>the book's own block</>}
        {market.seen && <>, quotes last seen {stampText(market.seen)}</>}
      </span>
      {lagging && (
        <span className="lag">
          These numbers are of an earlier book: it or its record has moved since they were read.
          {again ?? (loading ? ' A fresh read is on its way.'
            : ' They are read again on the next poll.')}
        </span>
      )}
    </div>
  );
}

/** The card: who it is for and what it is, its legs on the market's axis, the solved number set
 * apart, the net, the market it was struck on, the spread, the notes, who quoted it and when, and
 * how long the desk holds it firm. */
export function TicketCard({ pending }: { pending: PendingQuote }) {
  const { state } = useApp();
  const { schema } = state;
  const paper = state.paper.data ?? NO_PAPER;
  const { quote } = pending;
  const legs = ticketLegs(quote, pending.deal, schema);
  const first = legs[0]?.row.deal ?? {};
  const surface = first.FX_Volatility;
  const spot = onMarketAxis(quote.spot.value_market, quote.params.pair, surface);
  const forward = onMarketAxis(quote.forward, quote.params.pair, surface);
  const window = firmSeconds(state.doc, schema);
  const { word, until } = standsAt(pending, window, Date.now());
  // painted again the moment the window closes, which nothing else on the desk would move
  const [, lapse] = useState(0);
  useEffect(() => {
    if (word !== 'firm') return;
    const timer = setTimeout(() => lapse((count) => count + 1),
                             instantMs(until)! - Date.now() + 50);
    return () => clearTimeout(timer);
  }, [word, until]);
  const delivery = dateOf(first.Delivery_Date);
  const level = (on: { pair: string; level: number } | undefined, raw: number) =>
    (on ? `${on.pair} ${formatNumber(on.level)}` : formatNumber(raw));

  return (
    <section className="card ticket">
      <h3 title={quote.structure}>
        {salesName(schema, quote.structure)}{quote.variation && ` · ${quote.variation}`}
      </h3>
      <div className="nested">
        <div className="ticketline">
          <b>{clientText(quote.netting_set, paper.entities, paper.agreements)}</b>
          {quote.client && (
            <span>the client buys {quote.client.buys}, sells {quote.client.sells}</span>
          )}
          <span>{String(quote.params.pair)} · {String(quote.params.expiry)}
            {legs[0]?.row.expiry && ` · expires ${legs[0].row.expiry}`}
            {delivery && ` · delivers ${delivery}`}</span>
        </div>
        <div className="stats">
          <div className="stat">
            <div className="label">net, in the client's sign</div>
            <div className="value">
              {formatNumber(quote.net, 'money')} <span className="unit">{quote.currency}</span>
            </div>
          </div>
          {solvedNumbers(legs).map((solved) => (
            <div key={solved.what} className="stat solved">
              <div className="label">solved · {solved.what}</div>
              <div className="value">
                {solved.on ? `${solved.on} ` : ''}{formatNumber(solved.value)}
              </div>
            </div>
          ))}
          <div className="stat">
            <div className="label">
              spot · {quote.spot.source === 'terminal' ? 'live' : "book's"}
            </div>
            <div className="value">{level(spot, quote.spot.value_market)}</div>
          </div>
          <div className="stat">
            <div className="label">outright forward{delivery ? ` to ${delivery}` : ''}</div>
            <div className="value">{level(forward, quote.forward)}</div>
          </div>
          {until && (
            <div className={`stat${word === 'lapsed' ? ' warn' : ''}`}>
              <div className="label">{word === 'lapsed' ? STANDS.lapsed : 'firm until'}</div>
              <div className="value">{stampText(until)}</div>
            </div>
          )}
        </div>
        <TicketLegs legs={legs} currency={quote.currency} />
        <div className="note">{spreadText(quote)}</div>
        {quote.margin && (
          <div className="note">
            Sales margin {formatNumber(quote.margin.amount, 'money')} {quote.margin.currency},{' '}
            {formatNumber(quote.margin.value, 'money')} {quote.margin.pricing_currency} at the
            book's spot, charged where the spread is.
          </div>
        )}
        {[quote.spot.note, ...(quote.notes ?? []),
          ...quote.legs.map((leg) => leg.note && `${leg.role}: ${leg.note}`)]
          .filter((note): note is string => !!note)
          .map((note) => <div key={note} className="hint ticketnote">{note}</div>)}
        <div className="ticketline">
          <span>quoted by <b>{pending.quoted_by ?? 'no seat named'}</b> at{' '}
            {stampText(pending.quoted_at)}</span>
          <span className="hint">{firmText(window, state.record.spine !== null)}</span>
        </div>
      </div>
    </section>
  );
}

/** The legs, each with the client's side, its size, its levels on the market's axis - the one the
 * recipe solved marked - its terms and premium; a leg opens onto its whole deal. */
function TicketLegs({ legs, currency }: { legs: TicketLeg[]; currency: string }) {
  const { state } = useApp();
  const [opened, setOpened] = useState<string | null>(null);
  const level = (leg: TicketLeg, which: 'strike' | 'barrier') => {
    const picked = leg.read[which];
    return (
      <td className={`n${leg.solved === which ? ' solved' : ''}`} title={picked?.title}>
        {picked ? cellText(picked.value) : ''}
      </td>
    );
  };
  return (
    <div className="tablewrap">
      <table className="data blotter ticketlegs">
        <thead>
          <tr>
            <th>Leg</th><th>Client</th><th>On</th><th className="n">Amount</th><th>Ccy</th>
            <th className="n">Strike</th><th className="n">Barrier</th><th>Terms</th>
            <th className="n" title="what the leg is worth to the client at mid; the net adds the spread and the margin">
              At mid {currency}</th><th />
          </tr>
        </thead>
        <tbody>
          {legs.map((leg) => (
            <TicketLegRow key={leg.leg.reference} leg={leg} level={level}
                          opened={opened === leg.leg.reference} schema={state.schema}
                          onOpen={() => setOpened(opened === leg.leg.reference ? null
                            : leg.leg.reference)} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function TicketLegRow({ leg, level, opened, schema, onOpen }: {
  leg: TicketLeg; level: (leg: TicketLeg, which: 'strike' | 'barrier') => JSX.Element;
  opened: boolean; schema: Schema | null; onOpen: () => void;
}) {
  const side = String(leg.size.side?.value ?? '');
  return (
    <>
      <tr onClick={onOpen}>
        <td className="ref">{leg.leg.role}</td>
        <td>
          {side && <span className={`side ${side === 'Sell' ? 'sell' : 'buy'}`}>{side}</span>}
        </td>
        <td title={leg.read.on?.title}>
          {leg.read.on ? cellText(leg.read.on.value) : ''}{leg.read.sense && ` ${leg.read.sense}`}
        </td>
        <td className="n">{typeof leg.size.amount?.value === 'number'
          ? formatNumber(leg.size.amount.value, 'money') : ''}</td>
        <td>{leg.size.of ? cellText(leg.size.of.value) : ''}</td>
        {level(leg, 'strike')}
        {level(leg, 'barrier')}
        <td className="terms" title={leg.read.terms?.title}>{leg.read.terms?.parts.join(' · ')}</td>
        <td className="n">{formatNumber(leg.leg.premium, 'money')}</td>
        <td className="opener"><span>{opened ? '×' : '⋯'}</span></td>
      </tr>
      {leg.schedule && (
        <tr className="schedule">
          <td />
          <td colSpan={9}>
            <FieldView value={leg.schedule.value} descriptor={leg.schedule.declared} />
          </td>
        </tr>
      )}
      {opened && (
        <tr className="detail">
          <td colSpan={10}><DealPanels deal={leg.row.deal} schema={schema} /></td>
        </tr>
      )}
    </>
  );
}

/** A credit page's id, behind a prefix no curve block's name carries. */
export const CREDIT = 'credit:';

/** What a credit curve is written under on each desk. */
const HOLDS = {
  record: 'On this desk a curve is written for an entity the record declares, by a seat holding '
    + 'mark over the book.',
  plain: 'This desk keeps no record: a curve is written for any counterparty named.',
};

/** Rows of a credit curve as the officer reads one: a tenor, its spread - a number in basis points,
 * anything else as typed - and the survival to it. */
export function SpreadTable({ rows }: { rows: [string, unknown, number][] }) {
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
 * factor's convention - with what it will write beside it, written on one confirm, and how far the
 * write runs it on. A refusal is the service's own sentence beside the box it opens with. With no
 * `id` the counterparty is typed by name. */
export function CreditCard({ id, title, factor, base, convention, recording }: {
  id?: string; title: string; factor?: unknown; base: string; convention: number;
  recording: boolean;
}) {
  const { state, dispatch } = useApp();
  const hold = heldBack(state, 'mark');
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
      setSaid({ at: 'done', text: `wrote ${answer.factor}, its last spread run on to ${
        formatNumber(answer.horizon)} years` });
      if (id) return;
      // the counterparty now has a page of its own, which is where its curve reads back
      setName('');
      setForm(spreadForm(undefined, convention));
      dispatch({ type: 'PICK', screen: 'curves', id: CREDIT + request.counterparty });
    } catch (error) {
      const text = failure(error).error;
      setSaid({ at: refusedAt(text, request), text });
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
            to read, and the first trade under a netting set naming it is refused until it has
            one.</div>
        )}
      </section>
      <section className="card credit">
        <h3>{curve ? 'Restate the curve' : 'State the curve'}</h3>
        {hold ? <div className="note">{hold}</div> : <>
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
          {refusal('recovery')}
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
          {refusal('spread')}
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
        <div className="hint">What it will write: each spread read as a hazard rate, the spread
          divided by one less the recovery, counted ACT/365 from {base} to its tenor. A flat spread
          is stated at one year. Past the last tenor the curve runs on at its last spread.</div>
        <SpreadTable rows={typed.map((held, at) =>
          [held.tenor, held.spread, Math.exp(-knots[at][1])])} />
        <div className="statusrow">
          <button className="primary" onClick={() => void write()} disabled={saving}>
            write {SURVIVAL}.{request.counterparty || '…'}
          </button>
          {recording && state.seat && <span className="hint">as {state.seat}</span>}
          {saving && <span className="chip running">writing</span>}
          {said?.at === 'done' && <span className="chip done">{said.text}</span>}
          {refusal('act')}{id && refusal('counterparty')}
        </div>
        </>}
      </section>
    </>
  );
}

/** What a declaration answered: where it landed, or the service's refusal in its own words. */
export type Said = { refused: boolean; text: string } | null;

/** An agreement's terms as the netting set declares them: its own sections, the way a deal's
 * panel renders, each term its convention where the terms state nothing - those of a set whose
 * collateral is off left out, being collateral it does not have, and what paper never states
 * (`unstated`) - and every term, a table's cells among them, editable where `onChange` takes the
 * terms back. */
export function AgreementTerms({ schema, terms, unstated, onChange }: {
  schema: Schema; terms: Record<string, unknown>; unstated: SettlementState;
  onChange?: (terms: Record<string, unknown>) => void;
}) {
  const bare = !onChange && uncollateralised(fieldsOf(schema, SET) ?? {}, terms);
  return (
    <DealPanels deal={terms} schema={schema} own tables={!!onChange}
                cut={(fields) => onPaper(bare ? statedOnly(fields, terms, [COLLATERAL]) : fields,
                                         unstated)}
                onAmend={onChange && (async (key, wire) => {
                  onChange({ ...terms, [key]: wire });
                  return null;
                })} />
  );
}

/** Declaring an agreement with an entity: its id, the document's kind and its terms - the netting
 * set every position under it compiles into, edited off the declaration and posted as it stands.
 * Started `from` an agreement in force, it holds that one's terms - from an entity alone, a new
 * agreement with it; an id the record declares already is a restatement, which names every term it
 * moves before it is filed. What paper never states (`unstated`) is left out, and said once. */
export function AgreementForm({ schema, entities, agreements, unstated, from }: {
  schema: Schema; entities: Entity[]; agreements: Agreement[]; unstated: SettlementState;
  from?: Partial<Agreement>;
}) {
  const hold = heldBack(useApp().state, 'document', null);
  const [named, setNamed] = useState({
    agreement: from?.agreement ?? '', entity: from?.entity ?? '', kind: from?.kind ?? '' });
  const [terms, setTerms] = useState<Record<string, unknown>>(from?.terms ?? { Object: SET });
  const [said, setSaid] = useState<Said>(null);
  const card = useRef<HTMLElement>(null);
  useEffect(() => { if (from) card.current?.scrollIntoView({ block: 'start' }); }, [from]);
  const held = agreements.find((row) => row.agreement === named.agreement);
  const moved = held ? termsMoved(held.terms, terms) : [];

  async function declare() {
    try {
      const filed = await declareAgreement({ ...named, terms });
      setSaid({ refused: false, text: `declared at LSN ${filed.recorded.lsn}` });
    } catch (error) {
      setSaid({ refused: true, text: failure(error).error });
    }
  }

  const words = (keys: string[]) => keys.map((key) => key.replace(/_/g, ' ')).join(', ');
  return (
    <section className="card" ref={card}>
      <h3>{held ? `Restate ${held.agreement}` : 'Declare an agreement'}</h3>
      {hold ? <div className="note">{hold}</div> : <>
      <div className="statusrow">
        <input placeholder="agreement" value={named.agreement}
               onChange={(event) => setNamed({ ...named, agreement: event.target.value })} />
        <select value={named.entity}
                onChange={(event) => setNamed({ ...named, entity: event.target.value })}>
          <option value="">with entity…</option>
          {entities.map((row) => (
            <option key={row.entity} value={row.entity}>{entityText(entities, row.entity)}</option>
          ))}
        </select>
        <input placeholder="kind (ISDA 2002 with CSA)" value={named.kind}
               onChange={(event) => setNamed({ ...named, kind: event.target.value })} />
        <button className="ghost" onClick={declare}>
          {held ? 'restate the agreement' : 'declare the agreement'}
        </button>
      </div>
      {held && (
        <div className="banner">
          {named.agreement} is in force: filing this restates it whole, every term left out going
          back to its convention. {moved.length
            ? `Terms that differ from the version in force: ${moved.join(', ')}.`
            : 'No term differs from the version in force yet.'}
        </div>
      )}
      {said && <div className={said.refused ? 'error-box' : 'hint'}>{said.text}</div>}
      {unstated.fields.length + unstated.collateral_columns.length > 0 && (
        <div className="note">Paper states no {words(unstated.fields)}, nor a collateral row's{' '}
          {words(unstated.collateral_columns)}: those are settlement state, which the record holds
          from each collateral settlement the Back Office files.</div>
      )}
      <AgreementTerms schema={schema} terms={terms} unstated={unstated} onChange={setTerms} />
      </>}
    </section>
  );
}

/** What filing an edit does under a record, where an amendment of a booked trade is a restrike. */
const RESTRIKE = 'Filing this restrikes the trade: its new terms are filed under a new instrument '
  + 'and go through approval again.';

/** The deal at `path` in its panels, each level box saying how the market quotes it by the
 * blotter's rule, and every scalar amendable over the LIVE BOOK - a local file has nothing
 * server-side to amend - a restrike under a record asked first, its answer read above the
 * panels as a booking's is. A filed amendment re-reads the book at once rather than waiting a poll
 * tick. */
export function DealEditor({ path, node }: { path: string; node: DealNode }) {
  const { state, dispatch } = useApp();
  const [said, setSaid] = useState<string[]>([]);
  const recording = spineOf(state) !== null;
  const read = quoted(toRows([node], state.schema?.Instrument.containers ?? [], undefined)[0],
                      marketOf(state));
  const beside = (key: string) => {
    const level = [read.strike, read.barrier].find((picked) => picked?.key === key);
    if (level?.title) return <span className="readas">the market quotes {level.title}</span>;
    if (key === 'Barrier_Type' && read.terms) {
      return <span className="readas">the market reads {read.terms.parts.join(' · ')}</span>;
    }
    return key === 'Option_Type' && read.on && read.sense && (
      <span className="readas">
        a {read.sense} on {String(read.on.value)} as the market quotes it</span>);
  };
  const onAmend: AmendField | undefined = state.source?.kind === 'book'
    ? async (key, wireValue) => {
      const answer = await amendDeal(path, { [key]: wireValue });
      setSaid(answer.written ? verdictOf(answer).words : []);
      return written(dispatch, answer);
    } : undefined;
  return (
    <>
      {said.map((words) => <div key={words} className="note">{words}</div>)}
      <DealPanels deal={node.Instrument['.Deal']} schema={state.schema} onAmend={onAmend}
                  hold={heldBack(state, 'book')}
                  beside={beside} ask={recording ? RESTRIKE : undefined} />
    </>
  );
}
