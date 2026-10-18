import { useEffect, useState } from 'react';
import { failure, getCloseCheck, getWorklist, postClose, postTransition } from '../api';
import { EmptyState } from '../components/EmptyState';
import { worklistAt } from '../spine';
import { useApp } from '../state';
import type { CloseVerdict, WorkRow, Worklist } from '../types';

/** The three lists this seat clears, each by its own verb. */
type List = 'payments' | 'unconfirmed' | 'calls';
const LISTS: { list: List; title: string; verb: string }[] = [
  { list: 'payments', title: 'Payments due', verb: 'settle' },
  { list: 'unconfirmed', title: 'Trades to confirm', verb: 'confirm' },
  { list: 'calls', title: 'Collateral called', verb: 'collateral' },
];

/** The back office's seat: the payments due, the trades unconfirmed and the collateral the day's
 * marks call, each row beside the fact that clears it, and the book's day's close behind the
 * check's own verdict. Read off the worklist at the record's head, again at every move of it, and
 * nothing kept here - a row leaves when its fact lands. A refusal is the service's own sentence. */
export function BackOfficeView() {
  const { state } = useApp();
  const spine = state.source?.kind === 'book' ? state.record.spine : null;
  const at = worklistAt(spine);
  const [read, setRead] = useState<{
    worklist: Worklist | null; check: CloseVerdict | null; error: string | null }>(
    { worklist: null, check: null, error: null });

  useEffect(() => {
    if (at === null) return;
    let live = true;
    getWorklist()
      .then(async (worklist) => {
        const check = await getCloseCheck(worklist.date);
        if (live) setRead({ worklist, check, error: null });
      })
      .catch((error) => {
        if (live) setRead((held) => ({ ...held, error: failure(error).error }));
      });
    return () => { live = false; };
  }, [at]);

  if (at === null) {
    return (
      <EmptyState
        title="The back office works the live book's record."
        hint="Start the service with --book and a spine home: the payments due, the trades to confirm and the collateral called appear here, each with the fact that clears it."
      />
    );
  }
  const { worklist, check, error } = read;
  if (!worklist) {
    return error ? (
      <div className="main"><div className="panel"><div className="error-box">{error}</div></div></div>
    ) : <EmptyState title="Reading what waits…" hint="The diary's payments, the fills, the day's calls." />;
  }
  return (
    <div className="main">
      <div className="panel">
        {error && <div className="error-box">{error}</div>}
        {check && <CloseCheck check={check} />}
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
        subject: row.key, status: 'settled', amount: Number(amount), asset: row.currency,
        kind: list === 'payments' ? 'payment' : kind, reference, value_date: valueDate,
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
            <input type="number" step="any" placeholder="amount" value={amount}
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
