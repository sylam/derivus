import { useEffect, useState } from 'react';
import {
  approveTicket, failure, getCloseCheck, postClose, postTransition, rejectTicket,
} from '../api';
import { cellText, sizeOf } from '../blotter';
import { EmptyState } from '../components/EmptyState';
import { stampText } from '../desk';
import { nodeAt } from '../positions';
import { opened, useApp } from '../state';
import { ungrouped } from '../tokens';
import type { CloseVerdict, Decided, WorkRow } from '../types';

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
        title="The back office works the live book's record."
        hint="Start the service with --book and a spine home: the tickets to sign, the payments due, the trades to confirm and the collateral called appear here, each with the fact that clears it."
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
        <Tickets rows={worklist.pending} count={worklist.counts.pending} />
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

/** The tickets waiting for this seat's signature, each a trade as the desk reads it - the client,
 * the deal, how much, what it was booked at, who booked it, since when and why it waits, in the
 * tier's own words - beside the two verdicts that clear it. What each verdict answered stays
 * here after its row leaves, which it does when the record's head moves past it. */
function Tickets({ rows, count }: { rows: WorkRow[]; count: number }) {
  const [said, setSaid] = useState<string | null>(null);
  return (
    <section className="card">
      <h3>Tickets to sign ({count}) <span className="hint">approve</span></h3>
      {said && <div className="hint">{said}</div>}
      {rows.length ? (
        <div className="tablewrap">
          <table className="data">
            <thead>
              <tr>
                <th>Trade</th><th>Client</th><th>Deal</th><th className="n">Amount</th>
                <th>Ccy</th><th className="n">Booked at</th><th>Booker</th><th>Waiting since</th>
                <th>Why it waits</th><th />
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => <Ticket key={row.key} row={row} onSaid={setSaid} />)}
            </tbody>
          </table>
        </div>
      ) : <div className="hint">nothing waits on this seat</div>}
      {rows.length > 0 && (
        <div className="hint">
          Not on the record yet, so not shown: the margin, the spread charged and the limit used.
        </div>
      )}
    </section>
  );
}

/** One ticket and its two verdicts: signed, or refused with a reason the row keeps. The refusal of
 * either is the service's sentence, verbatim. */
function Ticket({ row, onSaid }: { row: WorkRow; onSaid: (said: string) => void }) {
  const { state, dispatch } = useApp();
  const [reason, setReason] = useState('');
  const [refused, setRefused] = useState<string | null>(null);
  const [filing, setFiling] = useState(false);
  const deal = row.deal_path && state.doc ? nodeAt(state.doc.Calc.Deals.Deals.Children,
                                                   row.deal_path)?.Instrument['.Deal'] : undefined;
  const object = String(deal?.Object ?? '');
  const size = deal ? sizeOf(deal, object, state.schema) : {};
  const trade = row.reference ?? row.key.slice(0, 12);

  async function decide(verdict: () => Promise<Decided>) {
    setFiling(true);
    try {
      const answer = await verdict();
      setRefused(null);
      onSaid(`${trade} for ${row.client}: ${answer.status}, recorded at LSN ${
        answer.recorded.lsn}`);
    } catch (error) {
      setRefused(failure(error).error);
    } finally {
      setFiling(false);
    }
  }

  return (
    <tr>
      <td>
        <button className="ghost" onClick={() => opened(dispatch, state.doc, row.deal_path)}>
          {trade}
        </button>
      </td>
      <td>{row.client ?? ''}<div className="hint">{row.agreement ?? ''}</div></td>
      <td>{object}{size.side ? <div className="hint">{String(size.side.value)}</div> : null}</td>
      <td className="n">{size.amount ? cellText(size.amount.value) : ''}</td>
      <td>{size.of ? String(size.of.value) : ''}</td>
      <td className="n">
        {row.price === null || row.price === undefined ? '' : `${cellText(row.price)} ${
          row.currency ?? ''}`}
      </td>
      <td>{row.booker ?? ''}</td>
      <td>{stampText(row.since)}</td>
      <td className="prose">{row.waits_on ?? ''}</td>
      <td>
        <span className="editcell">
          <button className="ghost" disabled={filing}
                  onClick={() => decide(() => approveTicket(row.key))}>approve</button>
          <input placeholder="reason" title="why it is refused - the record keeps it on the row"
                 value={reason}
                 onChange={(event) => setReason(event.target.value)} />
          <button className="ghost" disabled={filing || !reason.trim()}
                  onClick={() => decide(() => rejectTicket(row.key, reason.trim()))}>refuse</button>
          {refused && <span className="refusal">{refused}</span>}
        </span>
      </td>
    </tr>
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
