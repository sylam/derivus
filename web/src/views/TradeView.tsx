import { Fragment, useMemo, useState } from 'react';
import { failure, fillDeal, getBookActivity } from '../api';
import {
  baseDateOf, cellText, nameOf, quoted, rowsByPath, sizeOf, toRows, type BlotterRow,
} from '../blotter';
import { EmptyState } from '../components/EmptyState';
import { FieldView } from '../components/FieldView';
import { currencyOf, lagsBook, stampText } from '../desk';
import {
  NO_PAPER, byPath, clientName, closed, netOf, nodeAt, type Position, type Ticket,
} from '../positions';
import { clientText, quoteAt, verdictOf, type Verdict } from '../quote';
import { pinnedAt } from '../spine';
import {
  bookRisk, clockOf, heldBack, marketOf, noneShown, readAs, riskNow, spineOf, unread, useApp,
  useQuoteList, useQuotes, useRead, written,
} from '../state';
import { formatNumber, ungrouped } from '../tokens';
import {
  bookedBy, datesOf, favour, historyOf, judgedAt, legsOf, markOf, movedTerms, noMark, positionMark,
  tradeAt, tradeName, tradeSize, unwindAgainst, unwindFill, unwindWords, wholeRecord, type Price,
} from '../trade';
import type { QuoteGiven } from '../types';
import { Back, DealEditor, Gated, OnStrip, Side, Stat, Value } from '../components/Desk';
import { CreditCharge, TicketCard } from '../components/TicketCard';

/** One trade's rows on the record, newest first, asked for by its path where the desk keeps one
 * and again whenever the record moves, as a read (`unread`): the service's own sentence where it
 * is refused, and no rows until the read of this path lands. */
function useHistory(path: string | undefined, recording: boolean, at: string | null) {
  const read = useRead(recording && path !== undefined ? async () => ({ path, rows: historyOf(
    await wholeRecord((since) => getBookActivity(since, { deal_path: path }))) }) : null,
  [path, recording, at]);
  return { error: read.error, data: read.data && read.data.path === path ? read.data.rows : null };
}
type History = ReturnType<typeof useHistory>;

/** One trade on one page: what it is in the market's words, where it came from, what it is
 * worth, what happened to it on the record, and the amendment and the unwind - each part saying
 * what holds on this desk where it keeps no record. A leg opens its structure's page. */
export function TradeView() {
  const { state } = useApp();
  const { doc, schema } = state;
  const market = marketOf(state);
  const children = doc?.Calc.Deals?.Deals?.Children ?? [];
  const base = useMemo(() => (doc ? baseDateOf(doc, market) : undefined), [doc, market]);
  const rows = useMemo(() => rowsByPath(toRows(children, schema?.Instrument.containers ?? [], base)),
                       [children, schema, base]);
  const picked = state.selection.picks.trade;
  const trade = picked === undefined ? undefined : tradeAt(children, picked);
  const row = trade && rows.get(trade.path);
  const recording = spineOf(state) !== null;
  const history = useHistory(trade?.path, recording, pinnedAt(state.record.spine));
  const paper = state.paper.data ?? NO_PAPER;
  const held = (trade && byPath(paper.positions).get(trade.path)) || [];
  const units = netOf(held);
  const listed = useQuoteList();

  if (!doc) return null;
  if (!trade || !row) {
    return (
      <EmptyState title="No trade is open." hint={'A trade opens from its name in the blotter, its '
                  + 'node in the portfolio or a booked quote.'} />
    );
  }
  const quote = listed.data ? quoteAt(listed.data.quotes, trade.path, row.reference) ?? null
    : undefined;

  return (
    <div className="main">
      <div className="panel tradepage">
        <div className="pager"><Back /></div>
        <h2 className="tradetitle">{row.reference || '(no reference)'}
          <span className="hint" title="where the book file holds it">{row.object} at{' '}
            <span className="mono">{trade.path}</span></span></h2>
        <Terms row={row} set={trade.set} units={recording ? units ?? null : undefined}
               quote={quote ?? undefined} />
        <Origin recording={recording} held={held} history={history} quote={quote} />
        <Worth path={trade.path} reference={row.reference} legs={row.children.length}
               shut={closed(units)} />
        <History recording={recording} history={history} deal={row.deal} />
        <Actions key={`act ${trade.path}`} row={row} held={held} recording={recording} />
      </div>
    </div>
  );
}

/** What the trade is, as the market says it: the client, what it is and on what, how much of it
 * the record holds, its dates, and each leg's side, size and levels on the market's axis. */
function Terms({ row, set, units, quote }: {
  row: BlotterRow; set: string | null; units?: number | null; quote?: QuoteGiven;
}) {
  const { state } = useApp();
  const { schema } = state;
  const paper = state.paper.data ?? NO_PAPER;
  const market = marketOf(state);
  const shut = closed(units);
  // a trade the record closed reads at its sizes as written
  const scale = shut ? undefined : units ?? undefined;
  const read = quoted(row, market, schema);
  const what = tradeName(row, quote, schema);
  const legs = legsOf(row);
  return (
    <section className="card ticket">
      <h3>What it is</h3>
      <div className="nested">
        <div className="ticketline">
          <b>{clientText(set, paper.entities, paper.agreements)}</b>
          <span title={row.object}>{what}</span>
          {read.on && (
            <span title={read.on.title ?? read.on.key}>on {cellText(read.on.value)}</span>
          )}
          {units !== undefined && (
            <span title="the net of every fill, in units of the trade as written">
              {units === null ? unread(state.paper) ?? <>{noneShown(state)}: its sizes are as written</>
                : shut ? <><b>closed</b>: netted to nothing, its sizes are as written</>
                : <>held <b>{formatNumber(units)}</b> of the trade as written</>}
            </span>
          )}
        </div>
        <div className="stats">
          {datesOf(row).map(({ key, date }) => (
            <div key={`${key} ${date}`} className="stat">
              <div className="label">{key.replace(/_/g, ' ').toLowerCase()}</div>
              <div className="value">{date}</div>
            </div>
          ))}
        </div>
        <div className="tablewrap">
          <table className="data blotter ticketlegs">
            <thead>
              <tr>
                {row.children.length > 0 && <th>Leg</th>}
                <th title="the desk's side, as the desk booked it">Desk side</th><th>On</th>
                <th className="n">Amount</th><th>Ccy</th><th className="n">Strike</th>
                <th className="n">Barrier</th><th>Terms</th><th>Expiry</th>
              </tr>
            </thead>
            <tbody>
              {legs.map((leg) => {
                const size = sizeOf(leg.deal, leg.object, schema, scale);
                const levels = quoted(leg, market, schema);
                return (
                  <tr key={leg.path}>
                    {row.children.length > 0 && <td className="ref">{nameOf(leg)}</td>}
                    <td><Side side={size.side} /></td>
                    <td title={levels.on?.title ?? levels.on?.key}>
                      {levels.on ? cellText(levels.on.value) : ''}
                      {levels.sense && ` ${levels.sense}`}
                    </td>
                    <Value picked={size.amount} object={leg.object} schema={schema} kind="money" />
                    <td title={size.of?.key}>{size.of ? cellText(size.of.value) ?? '' : ''}</td>
                    <Value picked={levels.strike} object={leg.object} schema={schema} />
                    <Value picked={levels.barrier} object={leg.object} schema={schema} />
                    <td className="terms" title={levels.terms?.title}>
                      {levels.terms?.parts.join(' · ')}</td>
                    <td className="date">{leg.expiry ?? ''}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

const CHIP: Record<string, string> = { approved: 'done', pending: 'running', rejected: 'error' };

/** Where the trade came from: the quote that landed at it, on any desk, its ticket a click away;
 * and where the desk records, who booked it and when, and every clip with what its ticket reads -
 * the tier it fell in, while pending what it waits on, and each check it fell past, a line each, in
 * the service's words. */
function Origin({ recording, held, history, quote }: {
  recording: boolean; held: Position[]; history: History; quote: QuoteGiven | null | undefined;
}) {
  const { state } = useApp();
  const [showing, setShowing] = useState(false);
  const read = useQuotes(showing && quote ? [quote.quote_id] : []);
  const pending = quote && read.data?.[quote.quote_id];
  const booked = history.data ? bookedBy(history.data) : undefined;
  const tickets: Ticket[] = held.flatMap((position) => position.tickets)
    .sort((a, b) => a.fill - b.fill);
  // what stands where no fill is read, and the quote's line, said once where the two are one
  const unbooked = unread(history) ?? noneShown(state, 'no fill on the record opens this trade');
  const unquoted = noneShown(state, 'No quote this desk gave landed here');
  return (
    <section className="card">
      <h3>Where it came from</h3>
      <div className="nested">
        {!recording ? (
          <div className="note">This desk keeps no record: nothing here files who booked this
            trade or a ticket for it.</div>
        ) : (
          <div className="ticketline">
            {booked ? (
              <span>booked by <b>{booked.by}</b> at {stampText(booked.record_time)}, record
                position {booked.lsn}</span>
            ) : <span className="hint">{unbooked}</span>}
          </div>
        )}
        {quote ? (
          <div className="statusrow">
            <span>booked from quote <span className="mono" title={quote.quote_id}>
              {quote.quote_id.slice(0, 8)}…</span>, quoted by{' '}
              <b>{quote.quoted_by ?? 'no seat named'}</b> at {stampText(quote.quoted_at)}
            </span>
            <button className="ghost" onClick={() => setShowing(!showing)}>
              {showing ? 'hide its ticket' : 'show its ticket'}
            </button>
          </div>
        ) : quote === null && (booked || !recording || unquoted !== unbooked) && (
          <div className="hint">{unquoted}.</div>
        )}
        {showing && (read.error ? <div className="error-box">{read.error}</div>
          : pending ? <TicketCard pending={pending} />
            : <div className="hint">reading its ticket…</div>)}
        {!recording ? null : tickets.length > 0 ? (
          <div className="tablewrap">
            <table className="data desk">
              <thead>
                <tr>
                  <th className="n" title="where on the record the fill sits">Fill</th>
                  <th className="n">Qty</th><th>Execution</th><th>Ticket by</th><th>At</th>
                  <th>Ticket</th>
                </tr>
              </thead>
              <tbody>
                {tickets.map((ticket) => (
                  <tr key={ticket.ticket}>
                    <td className="n">{ticket.fill}</td>
                    <td className="n">{formatNumber(ticket.quantity)}</td>
                    <td className="mono" title={ticket.execution_reference}>
                      {ticket.execution_reference === quote?.quote_id ? `quote ${
                        ticket.execution_reference.slice(0, 8)}…` : ticket.execution_reference}
                    </td>
                    <td>{ticket.actor}{ticket.lsn !== ticket.fill && (
                      <span className="hint"> · restruck at {ticket.lsn}</span>)}</td>
                    <td>{stampText(ticket.at)}</td>
                    <td className="prose">
                      <span className={`chip ${CHIP[ticket.status] ?? ''}`}>{ticket.status}</span>
                      {ticket.tier && <span className="hint"> tier <b>{ticket.tier}</b></span>}
                      {ticket.waits_on && <span className="hint">: {ticket.waits_on}</span>}
                      {(ticket.fell_past ?? []).map((sentence) => (
                        <div key={sentence} className="hint">fell past: {sentence}</div>
                      ))}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : booked && <div className="hint">{unread(state.paper) ?? `${noneShown(state)}.`}</div>}
      </div>
    </section>
  );
}

/** What the trade is worth to the desk, and what the number is on: its own mark where the risk
 * answer holds one, else the top-level node it sits inside, said as that - and, asked for, what
 * it adds to its client's credit numbers. Both are a what-if, so a seat held back from one is
 * told so once. */
function Worth({ path, reference, legs, shut }: {
  path: string; reference: string; legs: number; shut: boolean;
}) {
  const { state } = useApp();
  const risk = bookRisk(state);
  const body = () => {
    if (state.source?.kind !== 'book') {
      return <div className="note">A file opened in this browser is not marked: the risk is a
        reading of the live book.</div>;
    }
    if (risk.error) {
      return <div className={risk.tone}>{risk.error}</div>;
    }
    if (!risk.data) return <div className="hint">{noMark(risk)}</div>;
    const { own, inside } = markOf(risk.marks, path);
    const currency = risk.data.currency ?? '';
    return (
      <>
        <OnStrip computed={stampText(risk.data.as_of)} title={`risk etag ${risk.data.etag}`}
                 currency={currency} lagging={lagsBook(risk.clock, clockOf(state))}
                 loading={risk.loading} />
        <div className="stats">
          {own && <Stat label={`worth to the desk, ${favour(own.value)}`} value={own.value}
                        currency={currency} title="positive is worth that much to the desk" />}
          {inside && <Stat label={`the net of ${inside.reference}, which holds it`}
                           value={inside.value} currency={currency}
                           title="the netting set's value to the desk, this trade inside it" />}
        </div>
        {!own && <div className="note">{noMark(risk, own, shut)}</div>}
        {legs > 0 && (
          <div className="hint">Its {legs} legs have no mark of their own: the book marks the
            structure whole.</div>
        )}
      </>
    );
  };
  return (
    <section className="card">
      <h3>What it is worth</h3>
      <div className="nested">
        {body()}
        {state.source?.kind === 'book' && !risk.held && (
          <CreditCharge key={path} ask={{ deal_path: path, reference: reference || undefined }} />
        )}
      </div>
    </section>
  );
}

/** What happened to the trade, newest first, as the record tells it: who did what and when. */
function History({ recording, history, deal }: {
  recording: boolean; history: History; deal: Record<string, unknown>;
}) {
  const { state } = useApp();
  const { schema } = state;
  return (
    <section className="card">
      <h3>What happened to it</h3>
      <div className="nested">
        {!recording ? (
          <div className="note">This desk keeps no record: nothing says who booked or changed
            this trade.</div>
        ) : !history.data ? <div className={readAs(null, history).tone}>{unread(history)}</div> : (
          <div className="tablewrap">
            <table className="data desk">
              <tbody>
                {history.data.map((row) => (
                  <tr key={row.lsn}>
                    <td className="n mono">{row.lsn}</td>
                    <td>{stampText(row.record_time)}</td>
                    <td>{row.by}</td>
                    <td className="prose">{row.summary}
                      {row.changed && (
                        <dl className="signfacts moved">
                          {movedTerms(row.changed, deal, schema).map((moved) => (
                            <Fragment key={moved.field}>
                              <dt title={moved.field}>{moved.words}</dt>
                              <dd><FieldView value={moved.was} descriptor={moved.declared} />
                                {' → '}
                                <FieldView value={moved.became} descriptor={moved.declared} />
                                {moved.unit && <span className="hint"> {moved.unit}</span>}</dd>
                            </Fragment>
                          ))}
                        </dl>
                      )}
                    </td>
                  </tr>
                ))}
                {!history.data.length && (
                  <tr><td className="hint">
                    {noneShown(state, 'nothing on the record opens this trade')}.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  );
}

/** What can be done to the trade: its terms amended through the editor the portfolio uses, and
 * on a desk that records, the position unwound at a price - the service's answer kept in view
 * once the position it moved has gone. */
function Actions({ row, held, recording }: {
  row: BlotterRow; held: Position[]; recording: boolean;
}) {
  const { state } = useApp();
  const [editing, setEditing] = useState(false);
  const [verdict, setVerdict] = useState<Verdict | null>(null);
  const children = state.doc?.Calc.Deals.Deals.Children ?? [];
  const live = state.source?.kind === 'book';
  // each position is judged where it sits; a trade this seat is shown none of, at the book
  const hold = heldBack(state, 'fill');
  const open = held.filter((position) => !closed(position.quantity));
  return (
    <section className="card">
      <h3>What can be done</h3>
      <div className="nested">
        <div className="statusrow">
          <button className="ghost" onClick={() => setEditing(!editing)}>
            {editing ? 'close the terms' : live ? 'amend the terms' : 'the terms as written'}
          </button>
          <span className="hint">{!live ? 'a file opened here amends nothing on the service'
            : recording ? 'an amendment restrikes the trade, and goes through approval again'
              : 'an amendment rewrites the trade in the book file'}</span>
        </div>
        {editing && [row, ...row.children].map((node, at) => {
          const deal = nodeAt(children, node.path);
          return deal && (at === 0 ? <DealEditor key={node.path} path={node.path} node={deal} /> : (
            <details key={node.path} className="fold">
              <summary>{nameOf(node)} · {node.object}</summary>
              <DealEditor path={node.path} node={deal} />
            </details>
          ));
        })}
        {!recording ? (
          <div className="note">This desk keeps no record, so nothing is unwound at a price here:
            an unwind is a fill, and a fill is filed on a desk that records.</div>
        ) : open.length ? open.map((position) => {
          const key = `${position.agreement}/${position.portfolio}`;
          const kept = heldBack(state, 'fill', judgedAt([position]));
          return <Gated key={key} hold={kept}><Unwind row={row} position={position}
            onVerdict={setVerdict} /></Gated>;
        }) : (
          <Gated hold={hold}><div className="note">{unread(state.paper) ?? `${held.length
            ? 'Closed: the record holds none of it' : noneShown(state)}, so there is nothing to unwind.`}
          </div></Gated>
        )}
        {verdict && (verdict.booked ? (
          <>
            <div className="statusrow">
              <span className="chip done">filed</span>
              <span>the unwind is on the record at position {verdict.booked.lsn ?? '—'}</span>
            </div>
            {verdict.words.filter(Boolean).map((words) => (
              <div key={words} className="note">{words}</div>
            ))}
          </>
        ) : verdict.words.map((words) => <div key={words} className="error-box">{words}</div>))}
      </div>
    </section>
  );
}

/** One position unwound: how much of it - all of it unless typed - at what price, in which
 * currency and, where the price moves cash, settling on which day, asked once more in words saying
 * who pays whom what and when and what is left, then filed as a fill toward nothing under an
 * execution reference minted for that confirm. */
function Unwind({ row, position, onVerdict }: {
  row: BlotterRow; position: Position; onVerdict: (said: Verdict | null) => void;
}) {
  const { state, dispatch } = useApp();
  const held = position.quantity;
  const [units, setUnits] = useState(String(Math.abs(held)));
  const [price, setPrice] = useState('');
  const [currency, setCurrency] = useState(currencyOf(state.doc));
  const [day, setDay] = useState('');
  const [asking, setAsking] = useState<string | null>(null);
  const [filing, setFiling] = useState(false);
  const count = Number(ungrouped(units.trim()));
  const at = Number(ungrouped(price.trim()));
  const cost: Price = { amount: at, currency: currency.trim(), day: day || undefined };
  const typed = units.trim() !== '' && price.trim() !== '' && isFinite(count) && isFinite(at)
    && (at === 0 || !!day);
  const size = tradeSize(row, state.schema);
  const client = clientName(state.paper.data?.entities ?? [], position.counterparty)
    || 'the client';
  // the reference the price is typed beside: the desk's mark of these units, or why none is read
  const risk = riskNow(state);
  const unit = positionMark(risk, row.path, position)?.unit;
  const against = unit == null ? `the desk's mark of these units: ${noMark(risk)}`
    : unwindAgainst(held, count, unit, risk.data!.currency ?? '',
                    price.trim() !== '' && isFinite(at) ? cost : undefined);

  async function file(execution: string) {
    setFiling(true);
    try {
      const answer = await fillDeal(position.deal_paths[0], {
        reference: row.reference, execution_reference: execution, portfolio: position.portfolio,
        ...unwindFill(held, count, cost) });
      onVerdict(verdictOf(answer));
      if (answer.written) await written(dispatch, answer);
      setAsking(null);
    } catch (error) {
      onVerdict({ words: [failure(error).error], requote: false });
    } finally {
      setFiling(false);
    }
  }

  return (
    <div className="unwind">
      <div className="statusrow">
        <span className="hint">unwind</span>
        <input className="amount" type="text" inputMode="decimal" value={units}
               onChange={(event) => { setUnits(event.target.value); setAsking(null); }} />
        <button className={`ghost${count === Math.abs(held) ? ' on' : ''}`}
                onClick={() => { setUnits(String(Math.abs(held))); setAsking(null); }}>
          all of it
        </button>
        <span className="hint">of the {formatNumber(held)} held{position.portfolio
          ? ` in ${position.portfolio}` : ''}, at</span>
        <input className="amount" type="text" inputMode="decimal" placeholder="price a unit"
               value={price}
               onChange={(event) => { setPrice(event.target.value); setAsking(null); }} />
        <input className="ccy" type="text" value={currency}
               onChange={(event) => { setCurrency(event.target.value); setAsking(null); }} />
        {at !== 0 && isFinite(at) && (
          <>
            <span className="hint">settling on</span>
            <input type="date" value={day}
                   onChange={(event) => { setDay(event.target.value); setAsking(null); }} />
          </>
        )}
        <button className="primary" disabled={!typed || filing}
                onClick={() => {
                  onVerdict(null);
                  setAsking(`unwind ${new Date().toISOString()}`);
                }}>
          unwind
        </button>
        <span className="hint">a unit is the trade as written, and the price is what one unit
          changes hands at</span>
      </div>
      {against && <div className="hint">{against}</div>}
      {asking && typed && (
        <div className="banner">
          {unwindWords(held, count, size, cost, client)}{' '}
          <button className="primary" disabled={filing} onClick={() => file(asking)}>
            {filing ? 'filing…' : 'file the unwind'}
          </button>{' '}
          <button className="ghost" onClick={() => setAsking(null)}>not yet</button>
          <div className="hint">filed as execution <span className="mono">{asking}</span></div>
        </div>
      )}
    </div>
  );
}
