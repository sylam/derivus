import { useEffect, useRef, useState } from 'react';
import { getBookActivity, type About } from '../api';
import {
  baseDateOf, cellText, quoted, rowsByPath, sizeOf, toRows, type BlotterRow, type Picked,
} from '../blotter';
import {
  CLIENT, RUNS, agreementsOf, bookedSince, bookedText, fundedAt, heldUnder, limitOf,
  readiness, ready, termsReading, touching, type Held, type Line, type Term,
} from '../client';
import { EmptyState } from '../components/EmptyState';
import { Navigator } from '../components/Navigator';
import {
  PLUS, SURVIVAL, ageOf, creditCounterparties, entityText, lastRunChip, offBook, plural,
  spreads, stampText, valuationDay,
} from '../desk';
import {
  positionName, statusText, type Agreement, type Entity, type Position,
} from '../positions';
import { pinnedAt } from '../spine';
import {
  bookRisk, marketOf, opened, readAs, riskNow, standing, unread, useApp, useRead,
} from '../state';
import { curveOf, formatNumber, tenorLabel, token } from '../tokens';
import { markOf, markShown, markText, noMark, wholeRecord } from '../trade';
import type {
  ActivityRow, BookXva, Market, Schema, SettlementState, XvaSet,
} from '../types';
import { AgreementForm } from '../components/AgreementForm';
import { CreditCard, SpreadTable } from '../components/CreditCard';
import { Amount, Back, LimitLine, Money, Side, Value } from '../components/Desk';

/** What the page reads beside the store: on a desk that keeps one the record's paper, the XVA
 * projection, and the consolidated risk for the marks. */
type Reads = {
  entities: Entity[] | null; agreements: Agreement[] | null; positions: Position[] | null;
  unstated: SettlementState | null;
  xva: { data: BookXva | null; error: string | null };
  risk: ReturnType<typeof bookRisk>;
};

/** One page per client: whether it is ready to trade, its agreements in a credit officer's words,
 * its credit curve as stated, what the desk has on with it, its XVA and what happened to it. On a
 * desk that records, a client is an entity legal declares; on one that does not, a counterparty its
 * netting sets name. Everything here is a read; the two forms it opens are Paper's and the Curves
 * tab's own, each refusing in the service's words. */
export function ClientView() {
  const { state } = useApp();
  const { doc, schema, source } = state;
  const recording = state.record.spine !== null;
  const live = source?.kind === 'book';
  const { paper, xva } = state;
  const risk = bookRisk(state);

  if (!live) {
    return <EmptyState {...offBook('a page for each counterparty the book trades with')} />;
  }
  if (!schema || !doc) return null;
  const unanswered = recording && unread(paper);
  if (unanswered) {
    return <EmptyState title={unanswered}
                       hint="The entities legal declares, their agreements and the positions under them." />;
  }
  const held = recording ? paper.data : null;
  const reads: Reads = {
    entities: held?.entities ?? null, agreements: held?.agreements ?? null,
    unstated: held?.unstated ?? null, positions: held?.positions ?? null, xva, risk,
  };
  const market = marketOf(state);
  const factors = market?.['Price Factors'] ?? {};
  const rows = rowsByPath(toRows(doc.Calc.Deals.Deals.Children, schema.Instrument.containers,
                                 baseDateOf(doc, market)));
  // every entity legal declares, every counterparty a netting set names, and the one opened
  const ids = new Set([...creditCounterparties(reads.entities, doc, {}),
                       ...creditCounterparties(null, doc, {}), state.selection.picks[CLIENT] ?? '']);
  const pages = [...ids].filter(Boolean).sort().map((id) => {
    const own = agreementsOf(id, reads.agreements, rows);
    const lines = readiness(id, reads.entities, own, doc, state.record.reconcile,
                            market?.['Price Factors'], schema);
    return { id, lines, own };
  }).sort((a, b) => Number(ready(a.lines)) - Number(ready(b.lines)));

  return (
    <Navigator screen={CLIENT} empty={recording ? 'Legal declares no entity yet: declare one on Paper.'
      : 'No netting set in the book names a counterparty.'}
               pages={pages.map(({ id, lines, own }) => ({
                 id, folder: ready(lines) ? 'ready to trade' : 'not ready to trade',
                 label: entityText(reads.entities, id),
                 content: <ClientPage id={id} lines={lines} own={own} reads={reads} rows={rows}
                                      factor={factors[`${SURVIVAL}.${id}`]} market={market}
                                      schema={schema} />,
               }))} />
  );
}

/** What a fix opens: Paper for an entity, Paper's agreement form, the Curves tab's credit form. */
const FIXES = { entity: 'declare it on Paper', agreement: 'declare an agreement',
                curve: 'state the curve' };

function ClientPage({ id, lines, own, reads, rows, factor, market, schema }: {
  id: string; lines: Line[]; own: Agreement[]; reads: Reads; rows: Map<string, BlotterRow>;
  factor: unknown; market: Market | null; schema: Schema;
}) {
  const { state, dispatch } = useApp();
  const [form, setForm] = useState<{ agreement?: Partial<Agreement>; curve?: boolean } | null>(null);
  const curveCard = useRef<HTMLDivElement>(null);
  useEffect(() => { if (form?.curve) curveCard.current?.scrollIntoView({ block: 'start' }); }, [form]);
  const { entities } = reads;
  // its history asked for by its entity and by each of its agreements, where the desk records
  const asked: About[] = [{ entity: id }, ...own.map((row) => ({ agreement: row.agreement }))];
  const history = useRead(entities ? async () => touching(await Promise.all(asked.map((about) =>
    wholeRecord((since) => getBookActivity(since, about))))) : null,
  [JSON.stringify(asked), !!entities, pinnedAt(state.record.spine)]);
  const entity = entities?.find((row) => row.entity === id);
  const below = (entities ?? []).filter((row) => row.parent === id);
  const name = entityText(entities, id);
  const todo = lines.filter((line) => line.tone === 'todo').length;
  const onPaper = () => opened(dispatch, state.doc, null, id);
  const fix = (line: Line) => (line.fix === 'entity' ? onPaper()
    : setForm(line.fix === 'curve' ? { curve: true } : { agreement: { entity: id } }));

  return (
    <div className="clientpage">
      <div className="clienthead">
        <Back />
        <h2>{entity?.name ?? id}</h2>
        {entity && <span className="mono">{id}</span>}
        {entity?.parent && <span className="hint">under {entityText(entities!, entity.parent)}</span>}
        {below.length > 0 && (
          <span className="hint">over {below.map((row) => entityText(entities!, row.entity)).join(', ')}</span>
        )}
        {entity && <button className="ghost" onClick={onPaper}>open on Paper</button>}
      </div>

      <section className="card">
        <h3>Is it ready to trade</h3>
        <div className={`verdict ${todo ? 'not' : 'ready'}`}>
          {todo ? `Not ready to trade: ${plural(todo, 'thing')} to put right.` : 'Ready to trade.'}
        </div>
        <ul className="facts">
          {lines.map((line, at) => (
            <li key={at} data-tone={line.tone}>
              {line.text}
              {line.fix && <button className="ghost" onClick={() => fix(line)}>{FIXES[line.fix]}</button>}
            </li>
          ))}
        </ul>
      </section>

      <section className="card">
        <h3>{entities ? `Agreements (${own.length})` : `Netting sets naming ${id} (${own.length})`}</h3>
        {!entities && (
          <div className="note">This desk keeps no record: a netting set's own terms are the
            agreement, and they are amended on the set in the Portfolio.</div>
        )}
        {entities && !own.length && (
          <div className="note">No agreement is in force with {name}: declaring one opens Paper's
            form here, with {id} chosen.</div>
        )}
        {own.map((row) => (
          <AgreementCard key={row.agreement} agreement={row} schema={schema}
                         amend={entities ? () => setForm({ agreement: row }) : undefined} />
        ))}
      </section>
      {form?.agreement && entities && reads.agreements && reads.unstated && (
        <>
          <div className="statusrow">
            <span className="hint">Paper's agreement form, opened here</span>
            <button className="ghost" onClick={() => setForm(null)}>close the form</button>
          </div>
          <AgreementForm key={JSON.stringify(form.agreement)} schema={schema} entities={entities}
                         agreements={reads.agreements} unstated={reads.unstated}
                         from={form.agreement} />
        </>
      )}

      <div ref={curveCard}>
        {form?.curve ? (
          <>
            <div className="statusrow">
              <span className="hint">The Curves tab's credit form, opened here</span>
              <button className="ghost" onClick={() => setForm(null)}>close the form</button>
            </div>
            <CreditCard id={id} title={name} factor={factor} base={valuationDay(state.doc)}
                        recording={!!entities}
                        convention={Number(schema.Factor.types[SURVIVAL]?.Recovery_Rate?.value)} />
          </>
        ) : <CurveRead id={id} factor={factor} restate={() => setForm({ curve: true })} />}
      </div>

      <OnBook own={own} reads={reads} rows={rows} market={market} schema={schema} />
      <XvaRows id={id} xva={reads.xva} positions={reads.positions} />
      <Happened history={entities && history} />
    </div>
  );
}

/** An agreement as a credit officer says a CSA, every term off the netting set's declaration: the
 * collateral - whether it is held, never which way it moves, which no term read here says - then
 * each term with the currency or unit its description names, a term at its convention marked so
 * and the rest of those folded away. */
function AgreementCard({ agreement, schema, amend }: {
  agreement: Agreement; schema: Schema; amend?: () => void;
}) {
  const { state, dispatch } = useApp();
  const read = termsReading(agreement.terms, schema);
  const { path, moved } = standing(state.doc, state.record.reconcile, agreement);
  return (
    <div className="agreement">
      <div className="pager">
        <b className="mono">{agreement.agreement}</b>
        {agreement.kind && <span>{agreement.kind}</span>}
        <span className={moved.length ? 'chip error' : 'chip'}>
          {path === null ? 'no trade booked under it yet'
            : moved.length ? `in the book at ${path}, off the paper on ${moved.join(', ')}`
              : `in the book at ${path}`}
        </span>
        {path !== null && (
          <button className="ghost" onClick={() => opened(dispatch, state.doc, path)}>open its set</button>
        )}
        {amend && <button className="ghost" onClick={amend}>amend</button>}
      </div>
      <div className="csa">
        {read.collateralised ? 'Collateralised' : 'Not collateralised'}
        {!read.stated && <span className="hint"> (its convention)</span>}
        {read.collateralised && read.currency && `, every collateral amount in ${read.currency}`}.
      </div>
      <Terms terms={read.terms} />
      {read.conventions.length > 0 && (
        <details className="fold">
          <summary>{plural(read.conventions.length, 'more term')} at {
            read.conventions.length === 1 ? 'its convention' : 'their conventions'}</summary>
          <Terms terms={read.conventions} />
        </details>
      )}
    </div>
  );
}

function Terms({ terms }: { terms: Term[] }) {
  return (
    <div className="fields">
      {terms.map((term) => [
        <div key={`${term.key}k`} className="k" title={term.key}>{term.words}</div>,
        <div key={`${term.key}v`} className={`v${term.convention ? ' convention' : ''}`}>
          {term.value}{term.unit && ` ${term.unit}`}
          {term.convention && <span className="hint">its convention</span>}
        </div>,
      ])}
    </div>
  );
}

/** The client's credit curve as the spreads stated, read back by the Curves tab's own rule. */
function CurveRead({ id, factor, restate }: { id: string; factor: unknown; restate: () => void }) {
  const curve = curveOf(token(factor, 'Curve'));
  const recovery = Number(token(factor, 'Recovery_Rate'));
  return (
    <section className="card credit">
      <h3>Credit curve — {SURVIVAL}.{id}</h3>
      {curve ? (
        <>
          <div className="hint">As the book holds it, at a recovery of{' '}
            {formatNumber(100 * recovery, 'percent')}:</div>
          <SpreadTable rows={spreads(curve.data, recovery).map(([tenor, spread, survival]) =>
            [tenorLabel(tenor), spread * 1e4, survival])} />
        </>
      ) : (
        <div className="note">No credit curve yet: a CVA against {id} has no default to read.</div>
      )}
      <div><button className="ghost" onClick={restate}>{curve ? 'restate it' : 'state it'}</button></div>
    </section>
  );
}

/** What the desk has on with the client, agreement by agreement: each trade in the blotter's
 * reading with its own mark, the count, and the set's mark where the consolidated risk gives one -
 * in the risk's currency, said as the desk's side. A trade the record holds nothing of reads as
 * closed. */
function OnBook({ own, reads, rows, market, schema }: {
  own: Agreement[]; reads: Reads; rows: Map<string, BlotterRow>; market: Market | null;
  schema: Schema;
}) {
  const { state } = useApp();
  const risk = reads.risk.data;
  // a position's share is of the units held when its mark was read: none while the book has moved
  const now = riskNow(state);
  return (
    <section className="card">
      <h3>What the desk has on with it</h3>
      {!own.length && <div className="note">No agreement, so nothing on: a trade books under one.</div>}
      {own.map((agreement) => {
        const { path } = standing(state.doc, state.record.reconcile, agreement);
        const held = reads.positions && reads.positions.filter(
          (position) => position.agreement === agreement.agreement);
        const trades = heldUnder(path, rows, held);
        const closed = trades.filter((trade) => trade.closed).length;
        const mark = path !== null ? markOf(reads.risk.marks, path).own : undefined;
        return (
          <div key={agreement.agreement} className="agreement">
            <div className="pager">
              <b className="mono">{agreement.agreement}</b>
              <span>{plural(trades.length - closed, 'trade')} on
                {closed ? `, ${closed} closed` : ''}</span>
              <span>
                {mark ? <>marked <b>{markText(mark.value, risk!.currency)}</b>, read {stampText(risk!.as_of)}</>
                  : `no mark${reads.risk.joiner}${noMark(reads.risk, mark, false, path === null
                    ? 'nothing is booked under it' : 'the risk names no row for its set')}`}
              </span>
            </div>
            {trades.length > 0 && (
              <div className="tablewrap">
                <table className="data desk blotter">
                  <thead>
                    <tr>
                      <th>Reference</th><th>Type</th><th>Desk side</th><th>On</th>
                      <th className="n">Amount</th><th>Ccy</th><th className="n">Strike</th>
                      <th className="n">Barrier</th><th>Terms</th><th>Expiry</th>
                      {reads.positions && <><th className="n">Qty</th><th>Status</th></>}
                      <th className="n"
                          title="the trade's own mark, positive is worth that much to the desk">
                        Desk's value{risk?.currency ? ` (${risk.currency})` : ''}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {trades.map((trade, at) => (
                      <TradeRow key={at} trade={trade} market={market} schema={schema}
                                recording={!!reads.positions}
                                mark={trade.row
                                  && markShown(now, trade.row.path, trade.position)} />
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        );
      })}
    </section>
  );
}

/** One trade in the blotter's reading, in the blotter's own cells: its side and size as the desk
 * holds it, its levels on the market's axis and its expiry; a click opens the trade's page. */
function TradeRow({ trade, market, schema, recording, mark }: {
  trade: Held; market: Market | null; schema: Schema; recording: boolean; mark?: number;
}) {
  const { state, dispatch } = useApp();
  const { row, position, closed } = trade;
  if (!row) {
    return (
      <tr className="muted">
        <td className="ref">{positionName(position!)}</td>
        <td colSpan={9} className="hint">booked under this agreement and not in the book file under
          its set, so nothing prices it</td>
        <td className="n">{formatNumber(position!.quantity)}</td><td>{statusText([position!])}</td>
        <td />
      </tr>
    );
  }
  const size = sizeOf(row.deal, row.object, schema, row.units);
  const read = quoted(row, market, schema);
  const cell = (picked?: Picked) => (picked ? cellText(picked.value) ?? '' : '');
  const value = (picked?: Picked, kind?: 'money') =>
    <Value picked={picked} object={row.object} schema={schema} kind={kind} />;
  return (
    <tr className={closed ? 'muted' : position?.pending ? 'pending' : undefined}
        onClick={() => opened(dispatch, state.doc, row.path)} title="open the trade's page">
      <td className="ref" title={row.reference}>{row.reference}
        {closed && <span className="badge">closed</span>}</td>
      <td className="type" title={row.object}>{row.object}</td>
      <td>{!closed && <Side side={size.side} />}</td>
      <td title={read.on?.title}>{cell(read.on)}{read.sense && ` ${read.sense}`}</td>
      {closed ? <td className="n" /> : value(size.amount, 'money')}
      <td>{cell(size.of)}</td>
      {value(read.strike)}
      {value(read.barrier)}
      <td className="terms" title={read.terms?.title}>{read.terms?.parts.join(' · ')}</td>
      <td className="date">{row.expiry ?? ''}</td>
      {recording && (
        <>
          <td className="n">{closed ? '0' : formatNumber(position!.quantity)}</td>
          <td title={closed ? 'the record holds nothing of it' : undefined}>
            {closed ? 'closed' : statusText([position!])}</td>
        </>
      )}
      <Amount value={mark ?? null} kind="money" />
    </tr>
  );
}

/** The client's XVA rows - each the last run of a set naming it - with when each ran and, on a
 * desk that records, the trades booked under the set since off the record's `positions`. */
function XvaRows({ id, xva, positions }: {
  id: string; xva: Reads['xva']; positions: Position[] | null;
}) {
  const sets = (xva.data?.sets ?? []).filter((set) => set.counterparty === id);
  const limit = xva.data && limitOf(id, xva.data.limits);
  return (
    <section className="card">
      <h3>XVA</h3>
      {xva.error && <div className="error-box">{xva.error}</div>}
      {!xva.data && !xva.error && <div className="hint">Reading the projection…</div>}
      {limit && <LimitLine read={limit} />}
      {xva.data && !sets.length && (
        <div className="note">No netting set names {id} in the book, and the projection holds no
          row for one: there is no XVA to read.</div>
      )}
      {sets.length > 0 && (
        <table className="data desk xva">
          <thead>
            <tr>
              <th>Netting set</th><th>CSA</th>
              <th className="n">CVA<div className="sub">{PLUS.CVA}</div></th>
              <th className="n">FVA<div className="sub">{PLUS.FVA}</div></th>
              <th>Run</th><th>Status</th>
            </tr>
          </thead>
          <tbody>
            {sets.map((set) => (
              <XvaRow key={set.reference} set={set} read={xva.data!.as_of}
                      booked={positions && bookedSince(set, positions)} />
            ))}
          </tbody>
        </table>
      )}
      {sets.length > 0 && (
        <div className="note">In this table {RUNS}. {fundedAt(xva.data!.funding)}</div>
      )}
    </section>
  );
}

function XvaRow({ set, read, booked }: { set: XvaSet; read: string; booked: number | null }) {
  const age = ageOf(set.as_of, read);
  const chip = lastRunChip(set);
  const money = (value: number | null) => (value === null ? <span className="hint">—</span>
    : <Money value={value} currency={set.currency} />);
  return (
    <>
      <tr className={set.note ? 'muted' : undefined}>
        <td><b>{set.reference}</b>{set.note && <span className="badge" title={set.note}>left the book</span>}</td>
        <td><span className={`chip${set.collateralized ? ' on' : ''}`}>
          {set.collateralized ? 'CSA' : 'no CSA'}</span></td>
        <td className="n">{money(set.cva)}</td>
        <td className="n">{money(set.fva)}</td>
        <td className={`age ${booked ? 'stale' : age.tone}`} title={age.title}>
          {set.as_of ? `${stampText(set.as_of)} · ${age.text}` : age.text}
          {!!booked && <div>{bookedText(booked)}</div>}</td>
        <td><span className={`chip ${chip.tone}`} title={chip.title}>{chip.text}</span>
          {set.recalc && <span className="chip running">{set.recalc.status}</span>}</td>
      </tr>
      {set.status === 'failed' && set.error && (
        <tr className="detail"><td colSpan={6}><div className="error-box">{set.error}</div></td></tr>
      )}
    </>
  );
}

/** The client's own rows of the record, newest first, each opening what it names - null `history`
 * where the desk keeps no record. */
function Happened({ history }: {
  history: { data: ActivityRow[] | null; error: string | null } | null;
}) {
  const { state, dispatch } = useApp();
  if (!history?.data) {
    return (
      <section className="card">
        <h3>What happened</h3>
        {!history ? <div className="note">This desk keeps no record: nothing says who did what to
          this client, or when.</div>
          : <div className={readAs(null, history).tone}>{unread(history)}</div>}
      </section>
    );
  }
  const rows = history.data;
  return (
    <section className="card">
      <h3>What happened ({rows.length})</h3>
      <table className="data desk">
        <tbody>
          {rows.map((row) => (
            <tr key={row.lsn} className="pick"
                onClick={() => opened(dispatch, state.doc, row.deal_path, row.agreement ?? row.entity)}>
              <td className="n mono">{row.lsn}</td>
              <td>{stampText(row.record_time)}</td>
              <td>{row.by}</td>
              <td className="prose">{row.summary}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
