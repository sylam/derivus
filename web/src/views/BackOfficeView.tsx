import { useEffect, useMemo, useState } from 'react';
import {
  approveTicket, failure, getCloseCheck, postClose, postTransition, rejectTicket,
} from '../api';
import { baseDateOf, rowsByPath, toRows, type BlotterRow } from '../blotter';
import { paperLine } from '../client';
import { EmptyState } from '../components/EmptyState';
import { stampText } from '../desk';
import { NO_PAPER } from '../positions';
import { clientText, quoteAt, spreadText } from '../quote';
import { noWork } from '../spine';
import {
  type Act, clearsNone, grantsRead, heldBack, marketOf, opened, useApp, useQuoteList, useQuotes,
  useRead,
} from '../state';
import { formatNumber, ungrouped } from '../tokens';
import { ticketWords, tradeAt, tradeName, waitedText } from '../trade';
import type { CloseVerdict, Decided, PendingQuote, WorkRow, Worklist } from '../types';
import { TicketCard } from '../components/TicketCard';

/** The three lists this seat clears, each by the act that clears a row of it. */
type List = 'payments' | 'unconfirmed' | 'calls';
const LISTS: { list: List; title: string; act: Act }[] = [
  { list: 'payments', title: 'Payments due', act: 'status_transition' },
  { list: 'unconfirmed', title: 'Trades to confirm', act: 'confirmed' },
  { list: 'calls', title: 'Collateral called', act: 'collateral' },
];

/** What waits on this seat: the tickets it signs, the payments due, the trades unconfirmed and the
 * collateral the day's marks call, each row beside the fact that clears it, and the book's day's
 * close behind the check's own verdict. Read off the ONE worklist the banner reads, so the two
 * never count apart, and nothing kept here - a row leaves when its fact lands. A refusal is the
 * service's own sentence. */
export function BackOfficeView() {
  const { state } = useApp();
  const { spine, worklist, worklistError } = state.source?.kind === 'book'
    ? state.record : { spine: null, worklist: null, worklistError: null };
  // the day's close is checked again at every read of what waits, which is every head move
  const read = useRead(worklist ? () => getCloseCheck(worklist.date) : null, [worklist]);

  const none = noWork(spine !== null, state.source?.kind === 'book');
  if (none) return <EmptyState {...none} />;
  const check = read.data;
  const error = worklistError ?? read.error;
  if (!worklist) {
    return error ? (
      <div className="main"><div className="panel"><div className="error-box">{error}</div></div></div>
    ) : <EmptyState title="Reading what waits…" hint="The tickets, the diary's payments, the fills, the day's calls." />;
  }
  return (
    <div className="main">
      <div className="panel">
        {error && <div className="error-box">{error}</div>}
        {check && <CloseCheck check={check} />}
        <Tickets key={worklist.seat ?? ''} worklist={worklist} />
        {LISTS.map(({ list, title, act }) => (
          <section className="card" key={list}>
            <h3>{title} ({worklist.counts[list]}){' '}
              <span className="hint">{grantsRead(state)?.read?.demands[act]}</span></h3>
            {worklist[list].length ? (
              <div className="tablewrap">
                <table className="data">
                  <tbody>
                    {worklist[list].map((row, position) => (
                      <Clearing key={`${row.key}/${row.agreement ?? ''}/${position}`} list={list}
                                row={row} day={worklist.date} />
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <div className="nested">
              {nothing(clearsNone(state, act), 'nothing waits on this seat')}
            </div>}
          </section>
        ))}
      </div>
    </div>
  );
}

/** An empty list: that this seat clears none of it where it does not (`clearsNone`), else that
 * none waits - the worklist lists only the rows a seat clears. */
const nothing = (hold: string | null, none: string) =>
  <div className={hold ? 'note' : 'hint signing'}>{hold ?? none}</div>;

/** What a verdict answered, kept on the card once its ticket has left the list. */
type Said = { status: string | null; text: string };

/** The tickets waiting for a second seat: those this seat signs, each with its two verdicts, then
 * those it is shown waiting on another seat, each with the record's sentence why - once on the
 * card where every one says the same - which it reads and never clears. What a verdict answered
 * stays after its row leaves, which it does when the record's head moves past it. */
function Tickets({ worklist }: { worklist: Worklist }) {
  const { state } = useApp();
  const { doc, schema } = state;
  const market = marketOf(state);
  const children = doc?.Calc.Deals?.Deals?.Children ?? [];
  const rows = useMemo(() => rowsByPath(toRows(children, schema?.Instrument.containers ?? [],
                                               doc ? baseDateOf(doc, market) : undefined)),
                       [children, schema, doc, market]);
  const [said, setSaid] = useState<Said | null>(null);
  // how long each has waited is read against this browser's clock, again every half minute
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 30_000);
    return () => clearInterval(timer);
  }, []);
  const { elsewhere } = worklist;
  const positions = state.paper.data?.positions ?? [];
  const executions = new Map(positions.flatMap((position) => position.tickets)
    .map((ticket) => [ticket.ticket, ticket.execution_reference]));
  // each ticket's quote is the one that booked its trade, read whole for its margin and spread
  const desk = useQuoteList();
  const given = (row: WorkRow) => (desk.data && row.deal_path
    ? quoteAt(desk.data.quotes, row.deal_path, row.reference) : undefined);
  const quotes = useQuotes([...new Set([...worklist.pending, ...elsewhere]
    .flatMap((row) => given(row)?.quote_id ?? []))], desk.data?.quotes);
  const signing = (row: WorkRow, hold: string | null) => {
    const at = row.deal_path ? tradeAt(children, row.deal_path) : undefined;
    const quote = given(row);
    return (
      <Signing key={row.key} row={row} trade={at && rows.get(at.path)} now={now} hold={hold}
               execution={executions.get(row.key)}
               quote={!desk.data ? undefined : quote ? quotes.data?.[quote.quote_id] : null}
               onSaid={setSaid} />
    );
  };
  // what holds this seat from signing: once on a card where it holds every row, else per row
  const hold = heldBack(state, 'approval');
  const holds = elsewhere.map((row) => row.may ?? null);
  const once = hold ?? (new Set(holds).size === 1 ? holds[0] : null);
  return (
    <>
      <section className="card">
        <h3>Tickets to sign ({worklist.counts.pending}) <span className="hint">approve, or refuse
          with a reason</span></h3>
        <div className="nested">
          {(desk.error ?? quotes.error) && (
            <div className="error-box">
              the desk's quotes did not answer: {desk.error ?? quotes.error}
            </div>
          )}
          {said && (
            <div className="statusrow">
              <span className={`chip ${said.status === 'approved' ? 'done' : 'error'}`}>
                {said.status ?? 'filed'}</span>
              <span>{said.text}</span>
            </div>
          )}
          {worklist.pending.map((row) => signing(row, hold && ''))}
          {(hold || !worklist.pending.length)
            && nothing(hold, 'nothing waits on this seat to sign')}
        </div>
      </section>
      {elsewhere.length > 0 && (
        <section className="card">
          <h3>Waiting on another seat ({worklist.counts.elsewhere}){' '}
            <span className="hint">read only</span></h3>
          <div className="nested">
            {once && <div className="note">{once}</div>}
            {elsewhere.map((row, at) => signing(row, hold || holds[at] === once ? '' : holds[at]))}
          </div>
        </section>
      )}
    </>
  );
}

/** One pending ticket as its signer reads it: the client and the trade in one line, its reference
 * opening its page; who booked it, when and how long it has waited; why it waits, in the tier's own
 * words, and each check that sent it past an earlier tier, a line each; what it was quoted at, its
 * ticket a click away; the paper it books under - then its two verdicts, or what holds for this
 * seat, blank where its card says it. Either refusal is the service's sentence, verbatim. */
function Signing({ row, trade, now, hold, execution, quote, onSaid }: {
  row: WorkRow; trade?: BlotterRow; now: number; hold: string | null; execution?: string;
  quote?: PendingQuote | null; onSaid: (said: Said) => void;
}) {
  const { state, dispatch } = useApp();
  const { schema } = state;
  const paper = state.paper.data ?? NO_PAPER;
  const [reason, setReason] = useState('');
  const [refused, setRefused] = useState<string | null>(null);
  const [filing, setFiling] = useState(false);
  const [showing, setShowing] = useState(false);
  const reference = row.reference ?? row.key.slice(0, 12);
  const client = clientText(row.agreement ?? null, paper.entities, paper.agreements);

  async function decide(verdict: () => Promise<Decided>, why?: string) {
    setFiling(true);
    try {
      const answer = await verdict();
      setRefused(null);
      onSaid({ status: answer.status, text: `${reference} for ${client}: ${answer.status}${
        why ? `, "${why}"` : ''}, on the record at position ${answer.recorded.lsn}.` });
    } catch (error) {
      setRefused(failure(error).error);
    } finally {
      setFiling(false);
    }
  }

  const quoted = () => {
    if (quote === undefined) return 'reading…';
    if (quote === null) return <>booked from no quote this desk gave{execution !== undefined && <>:
      its execution reference is <span className="mono">{execution}</span></>}</>;
    const { net, currency, margin } = quote.quote;
    return (
      <>
        net <b>{formatNumber(net, 'money')} {currency}</b>
        {margin && <>, sales margin <b>{formatNumber(margin.amount, 'money')} {margin.currency}</b></>}
        , quoted by <b>{quote.quoted_by ?? 'no seat named'}</b> at {stampText(quote.quoted_at)}{' '}
        <button className="ghost" onClick={() => setShowing(!showing)}>
          {showing ? 'hide its ticket' : 'show its ticket'}
        </button>
        <div className="hint">{spreadText(quote.quote)}</div>
      </>
    );
  };

  return (
    <div className="signing">
      <div className="ticketline">
        <b>{client}</b>
        <span>{trade ? ticketWords(trade, tradeName(trade, quote?.quote, schema),
                                   marketOf(state), schema)
          : `the book file holds no trade at ${row.deal_path ?? 'any path'}`}</span>
        <button className="tradelink ref" title="open the trade's page"
                onClick={() => opened(dispatch, state.doc, row.deal_path)}>{reference} ›</button>
      </div>
      <dl className="signfacts">
        <dt>booked</dt>
        <dd>by <b>{row.booker}</b> at {stampText(row.since)}
          {typeof row.price === 'number' && `, done at ${formatNumber(row.price, 'money')} ${
            row.currency ?? ''}`} · <b>waiting {waitedText(row.since, now)}</b></dd>
        <dt>waits</dt>
        <dd>{row.tier ? <>tier <b>{row.tier}</b>: {row.waits_on}</> : 'on a second seat'}</dd>
        {!!row.fell_past?.length && (
          <>
            <dt>fell past</dt>
            <dd>{row.fell_past.map((sentence) => <div key={sentence}>{sentence}</div>)}</dd>
          </>
        )}
        <dt>quoted</dt>
        <dd>{quoted()}</dd>
        <dt>paper</dt>
        <dd>{row.agreement ? paperLine(row.agreement, paper.agreements) : ''}</dd>
      </dl>
      {showing && quote && <TicketCard pending={quote} />}
      {hold !== null ? hold && <div className="note">{hold}</div> : (
        <div className="statusrow">
          <span className="editcell">
            <button className="ghost" disabled={filing}
                    onClick={() => decide(() => approveTicket(row.key))}>approve</button>
            <input type="text" className="reason" placeholder="reason"
                   title="why it is refused - the record keeps it on the row" value={reason}
                   onChange={(event) => setReason(event.target.value)} />
            <button className="ghost" disabled={filing || !reason.trim()}
                    onClick={() => decide(() => rejectTicket(row.key, reason.trim()),
                                          reason.trim())}>refuse</button>
            {refused && <span className="refusal">{refused}</span>}
          </span>
        </div>
      )}
    </div>
  );
}

/** One row and the fact that clears it: a payment `settled` with its money, under the agreement
 * the row carries; a fill `confirmed`; a call `settled` as a kind the row says it settles as, its
 * agreement the subject. The amount and the value date start at the row's own, the reference is
 * the settlement system's to state. */
function Clearing({ list, row, day }: { list: List; row: WorkRow; day: string }) {
  const [amount, setAmount] = useState(row.amount === null || row.amount === undefined
    ? '' : String(row.amount));
  const [valueDate, setValueDate] = useState(list === 'payments' ? row.since ?? day : day);
  const [reference, setReference] = useState('');
  const [kind, setKind] = useState(row.settles_as?.[0] ?? '');
  const [refused, setRefused] = useState<string | null>(null);
  const [filing, setFiling] = useState(false);
  const moves = list !== 'unconfirmed';

  async function file() {
    setFiling(true);
    try {
      await postTransition(moves ? {
        subject: row.key, status: 'settled', amount: Number(ungrouped(amount.trim())),
        asset: row.currency,
        kind: list === 'payments' ? row.kind : kind, reference, value_date: valueDate,
        ...(row.agreement ? { agreement: row.agreement } : {}),
      } : { subject: row.key, status: 'confirmed' });
      setRefused(null);
    } catch (error) {
      setRefused(failure(error).error);
    } finally {
      setFiling(false);
    }
  }

  return (
    <tr>
      <td>
        {row.what}
        {row.unknown?.map((named) => <div key={named.what} className="hint">{named.what}</div>)}
      </td>
      <td className="mono" title={row.key}>{row.key.slice(0, 12)}</td>
      <td>{row.agreement ?? ''}</td>
      {moves ? (
        <>
          <td>
            <input type="text" inputMode="decimal" placeholder="amount" value={amount}
                   onChange={(event) => setAmount(event.target.value)} /> {row.currency ?? ''}
          </td>
          <td>
            <input type="date" value={valueDate}
                   onChange={(event) => setValueDate(event.target.value)} />
          </td>
          <td>
            <input placeholder="reference" value={reference}
                   onChange={(event) => setReference(event.target.value)} />
          </td>
          <td>
            {row.settles_as && (
              <select value={kind} onChange={(event) => setKind(event.target.value)}>
                {row.settles_as.map((held) => <option key={held}>{held}</option>)}
              </select>
            )}
          </td>
        </>
      ) : <td colSpan={4} />}
      <td>
        <span className="editcell">
          <button className="ghost" disabled={filing} onClick={file}>
            {moves ? 'settled' : 'confirmed'}
          </button>
          {refused && <span className="refusal">{refused}</span>}
        </span>
      </td>
    </tr>
  );
}

/** The book's day's close behind the check's own verdict: legal, and declaring it is one act; not,
 * and what it waits on is listed. The close refuses an illegal day in its own words besides. */
function CloseCheck({ check }: { check: CloseVerdict }) {
  const hold = heldBack(useApp().state, 'official_close_declared', null);
  const [refused, setRefused] = useState<string | null>(null);
  const [closing, setClosing] = useState(false);

  async function close() {
    setClosing(true);
    try {
      await postClose(check.date);
      setRefused(null);
    } catch (error) {
      setRefused(failure(error).error);
    } finally {
      setClosing(false);
    }
  }

  return (
    <section className="card">
      <h3>The close of {check.date}</h3>
      <div className="statusrow">
        <span className={`chip ${check.legal ? 'done' : 'error'}`}>
          {check.legal ? 'legal' : `waits on ${check.outstanding.length}`}
        </span>
        <span className="hint">{check.due} due by the day</span>
        {!hold && (
          <button className="ghost" disabled={!check.legal || closing} onClick={close}>
            declare the close
          </button>
        )}
      </div>
      {hold && <div className="note">{hold}</div>}
      {refused && <div className="error-box">{refused}</div>}
      {!check.legal && (
        <div className="tablewrap">
          <table className="data">
            <tbody>
              {check.outstanding.map((row, position) => (
                <tr key={position}>
                  <td>{row.kind}</td>
                  <td>{row.leg}</td>
                  <td>{row.due_date ?? 'every day'}</td>
                  <td className="mono" title={row.instrument ?? ''}>
                    {row.instrument?.slice(0, 12) ?? ''}
                  </td>
                  <td>{row.reason ?? ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
