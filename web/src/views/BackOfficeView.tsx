import { useEffect, useMemo, useState } from 'react';
import {
  approveTicket, failure, getCloseCheck, postClose, postTransition, rejectTicket,
} from '../api';
import { baseDateOf, rowsByPath, toRows, type BlotterRow } from '../blotter';
import { EmptyState } from '../components/EmptyState';
import { stampText } from '../desk';
import { NO_PAPER } from '../positions';
import { clientText, quoteAt, spreadText } from '../quote';
import { marketOf, opened, useApp } from '../state';
import { formatNumber, ungrouped } from '../tokens';
import {
  holdsFor, paperText, ticketWords, tradeAt, tradeName, waitedText, waitingElsewhere,
} from '../trade';
import type { CloseVerdict, Decided, PendingQuote, WorkRow, Worklist } from '../types';
import { TicketCard, useQuoteList, useQuotes } from './QuoteView';

/** The three lists this seat clears, each by its own verb. */
type List = 'payments' | 'unconfirmed' | 'calls';
const LISTS: { list: List; title: string; verb: string }[] = [
  { list: 'payments', title: 'Payments due', verb: 'settle' },
  { list: 'unconfirmed', title: 'Trades to confirm', verb: 'confirm' },
  { list: 'calls', title: 'Collateral called', verb: 'collateral' },
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
  const [read, setRead] = useState<{ check: CloseVerdict | null; error: string | null }>(
    { check: null, error: null });

  // the day's close is checked again at every read of what waits, which is every head move
  useEffect(() => {
    if (!worklist) return;
    let live = true;
    getCloseCheck(worklist.date)
      .then((check) => { if (live) setRead({ check, error: null }); })
      .catch((error) => { if (live) setRead({ check: null, error: failure(error).error }); });
    return () => { live = false; };
  }, [worklist]);

  if (spine === null) {
    return (
      <EmptyState
        title="This desk keeps no record, so nothing waits on a second seat."
        hint="A booking here writes the book file and stands as written: no ticket waits to be signed, and no payment, confirmation or collateral call is tracked. A desk started with --book and a spine home keeps them, and lists them here."
      />
    );
  }
  const { check } = read;
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
        {LISTS.map(({ list, title, verb }) => (
          <section className="card" key={list}>
            <h3>{title} ({worklist.counts[list]}) <span className="hint">{verb}</span></h3>
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
            ) : <div className="hint">nothing waits on this seat</div>}
          </section>
        ))}
      </div>
    </div>
  );
}

/** What a verdict answered, kept on the card once its ticket has left the list. */
type Said = { status: string | null; text: string };

/** The tickets waiting for a second seat: those this seat's worklist lists, each with its two
 * verdicts or what holds for this seat, then those it does not sign, which it reads and never
 * clears. What a verdict answered stays after its row leaves, which it does when the record's head
 * moves past it. */
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
  const positions = state.paper.data?.positions ?? [];
  const elsewhere = waitingElsewhere(positions, worklist.pending);
  const executions = new Map(positions.flatMap((position) => position.tickets)
    .map((ticket) => [ticket.ticket, ticket.execution_reference]));
  // each ticket's quote is the one that landed at its trade, read whole for its margin and spread
  const desk = useQuoteList();
  const given = (row: WorkRow) => (desk.data && row.deal_path
    ? quoteAt(desk.data.quotes, row.deal_path) : undefined);
  const quotes = useQuotes([...new Set([...worklist.pending, ...elsewhere]
    .flatMap((row) => given(row)?.quote_id ?? []))]);
  const own = worklist.pending.filter((row) => row.booker === worklist.seat).length;
  const signing = (row: WorkRow, listed: boolean) => {
    const at = row.deal_path ? tradeAt(children, row.deal_path) : undefined;
    const quote = given(row);
    return (
      <Signing key={row.key} row={row} trade={at && rows.get(at.path)} now={now}
               hold={holdsFor(row, worklist.seat, listed)} execution={executions.get(row.key)}
               quote={!desk.data ? undefined : quote ? quotes.data?.[quote.quote_id] : null}
               onSaid={setSaid} />
    );
  };
  return (
    <>
      <section className="card">
        <h3>Tickets to sign ({worklist.counts.pending}) <span className="hint">{own
          ? `${own} booked by this seat, which another seat signs`
          : 'approve, or refuse with a reason'}</span></h3>
        <div className="nested">
          {(desk.error ?? quotes.error) && (
            <div className="error-box">the desk's quotes did not answer: {desk.error ?? quotes.error}</div>
          )}
          {said && (
            <div className="statusrow">
              <span className={`chip ${said.status === 'approved' ? 'done' : 'error'}`}>
                {said.status ?? 'filed'}</span>
              <span>{said.text}</span>
            </div>
          )}
          {worklist.pending.map((row) => signing(row, true))}
          {!worklist.pending.length && (
            <div className="hint signing">nothing waits on this seat to sign</div>
          )}
          {worklist.pending.length > 0 && (
            <div className="hint signing">
              Which check sent a ticket past the tiers above its own - its size, its tenor, where it
              books - is not on the record's reads: a ticket names only the tier it fell in.
            </div>
          )}
        </div>
      </section>
      {elsewhere.length > 0 && (
        <section className="card">
          <h3>Waiting on another seat ({elsewhere.length}) <span className="hint">read only</span>
          </h3>
          <div className="nested">{elsewhere.map((row) => signing(row, false))}</div>
        </section>
      )}
    </>
  );
}

/** One pending ticket as its signer reads it: the client and the trade in one line, its reference
 * opening its page; who booked it, when and how long it has waited; why it waits, in the tier's own
 * words; what it was quoted at, its ticket a click away; the paper it books under - then its two
 * verdicts, or what holds for this seat. Either refusal is the service's sentence, verbatim. */
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
        <dt>quoted</dt>
        <dd>{quoted()}</dd>
        <dt>paper</dt>
        <dd>{row.agreement ? paperText(row.agreement, paper.agreements, schema) : ''}</dd>
      </dl>
      {showing && quote && <TicketCard pending={quote} />}
      {hold ? <div className="note">{hold}</div> : (
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
 * the row carries; a fill `confirmed`; a call `settled` as collateral or margin, its agreement the
 * subject. The amount and the value date start at the row's own, the reference is the settlement
 * system's to state. */
function Clearing({ list, row, day }: { list: List; row: WorkRow; day: string }) {
  const [amount, setAmount] = useState(row.amount === null || row.amount === undefined
    ? '' : String(row.amount));
  const [valueDate, setValueDate] = useState(list === 'payments' ? row.since ?? day : day);
  const [reference, setReference] = useState('');
  const [kind, setKind] = useState('collateral');
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
            {list === 'calls' && (
              <select value={kind} onChange={(event) => setKind(event.target.value)}>
                <option>collateral</option>
                <option>margin</option>
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
        <button className="ghost" disabled={!check.legal || closing} onClick={close}>
          declare the close
        </button>
      </div>
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
