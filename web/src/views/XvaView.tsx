import { useState, type ReactNode } from 'react';
import { getBookCollateral, getXvaProfile } from '../api';
import { EXPORTS, SET, fieldsOf, rowsByPath, saveCsv, toRows } from '../blotter';
import {
  LIMIT_USAGE, RUNS, WRONG, actionFirst, actsOf, actsText, agreementsOf, bookedText, closeRead,
  creditExport, creditLines, exposureAfter, fundedAt, limitOf, openClient, paperText, runSum, statedHolding,
  summed, unmarked, wrongWith, type CreditLine, type LimitRead, type Wrong,
} from '../client';
import { EmptyState } from '../components/EmptyState';
import { TimeSeriesChart } from '../components/TimeSeriesChart';
import {
  NO_BOOK, PLUS, SURVIVAL, ageOf, reportCurrency, entityText, heldSpreads, lagsBook, offBook, peaks,
  plural, replayRows, stampText, tableReading, valuationDay, xvaTotals, type Peak,
} from '../desk';
import {
  bookRisk, clockOf, marketOf, nothingWrong, spineOf, unread, useApp, useRead,
} from '../state';
import { formatNumber, tenorLabel } from '../tokens';
import { Amount, LimitLine, Money, OnStrip } from '../components/Desk';

/** What a cell says of a done run whose row keeps no peaks. */
const UNKEPT = 'this row was filed before rows kept a run\'s peaks: a recalc keeps them';

/** A client as the screen holds it: its lines, what is wrong under them, its sums, and the credit
 * limit declared on it. */
type Client = {
  id: string; name: string; spread?: [number, number]; lines: CreditLine[]; wrong: Wrong[];
  value: number | null; after: number | null; limit?: LimitRead;
};

/** Every counterparty as a credit officer reads it, the ones to act on first: what the desk is
 * owed, the collateral against it, the exposure after collateral, how large it can get, its CVA
 * and FVA - each its own set's last run, with its own age - and what is wrong, in business words.
 * A set's row opens its profile and leads to the client's page.
 *
 * READ-ONLY BY DESIGN: a recalc is asked for through the chat, never from here, and the screen
 * says so. A run's peaks are read off its row, its profile one set at a time, and on a desk that
 * records the trades booked under its set since off the positions. One that keeps no record reads
 * no collateral call, no second seat and no count, and says so.
 */
export function XvaView() {
  const { state } = useApp();
  const { xva, source, doc } = state;
  const live = source?.kind === 'book';
  const recording = state.record.spine !== null;
  const clock = clockOf(state);
  const credit = useCredit();

  if (!live) return <EmptyState {...offBook("the last credit run of each of the desk's sets")} />;
  if (xva.status === 404) return <EmptyState {...NO_BOOK} />;
  // a corrupt projection file names itself in the refusal; the rows are still on disk
  if (xva.error) {
    return (
      <div className="main">
        <div className="panel">
          <div className="error-box">{xva.error}</div>
          <div className="hint">
            The projection file could not be read. Nothing was lost — the rows are on disk, and
            this screen shows them again as soon as the file parses.
          </div>
        </div>
      </div>
    );
  }
  if (!doc || !credit) {
    return (
      <EmptyState
        title={(recording && unread(state.paper))
          || (xva.loading ? 'Reading the projection…' : 'No projection has been read yet.')}
        hint="The last run per netting set, joined with the book's own set list and the clients it trades with."
      />
    );
  }

  const { view, risk, collateral, close, marked, reads } = credit;
  const { currency } = reads;
  const clients = actionFirst(credit.ids.map(credit.client));
  const acting = clients.filter((client) => actsOf(client).length);

  return (
    <div className="main">
      <div className="panel">
        <OnStrip
          computed={<>{RUNS}; the value to the desk
            {risk.data ? <> read {stampText(risk.data.as_of)}</> : ' unread'}
            {recording && <>; collateral{collateral.data ? ' on the ' : ': '}{marked()}</>}</>}
          currency={currency} lagging={lagsBook(xva.clock, clock)} loading={xva.loading}
        />
        {risk.error && (
          <div className={risk.tone}>
            The value to the desk is not read{risk.joiner}{risk.error}</div>
        )}
        {close && <div className="note">The collateral is not read. {close.title} {close.hint}</div>}
        {collateral.error && (
          <div className="error-box">The collateral is not read: {collateral.error}</div>)}
        <div className={`blotterbar wrongs${acting.length ? ' act' : ''}`}>
          {acting.length ? (
            <span className="hint">
              <b>{plural(acting.length, 'client')}</b> of {clients.length} to act on:{' '}
              {actsText(clients)}
            </span>
          ) : (
            <span className="hint">
              <b>Nothing to act on.</b> {nothingWrong(state)}.
            </span>
          )}
          <span className="spacer" />
          <span className="hint">
            {plural(clients.length, 'client')} · value to the desk <b>{amount(summed(
              clients.map((client) => client.value)))}</b> {currency} · after collateral{' '}
            <b>{amount(summed(clients.map((client) => client.after)))}</b> {currency}
            {xvaTotals(view).sums.map((sum) => (
              <span key={sum.currency ?? ''}> · CVA over the {sum.counted} run{' '}
                <b>{formatNumber(sum.cva, 'money')}</b>{' '}
                {sum.currency ?? '(currency unknown)'}</span>
            ))}
          </span>
          <button className="ghost" title={EXPORTS} onClick={() => saveCsv(
            `xva ${valuationDay(doc) || 'undated'}`, creditExport(clients, currency))}>
            export CSV
          </button>
        </div>

        <div className="note">
          Each row is a client; a client trading under several netting sets has each beneath it,
          and a set's row opens its profile.{' '}
          {recording ? 'Collateral is read on the close marked for the book\'s day; trades '
            + 'waiting on a second seat, and those booked under a set since its run, off the record.'
            : "This desk keeps no record: collateral is what each set's terms state it holds, no "
              + 'call is worked and no trade waits on a second seat.'}
          <div>{fundedAt(view.funding)}</div>
          {view.limits.length > 0 && <div>{LIMIT_USAGE}</div>}
        </div>

        <CreditTable clients={clients} reads={reads} />
      </div>
    </div>
  );
}

/** What the rows read beside their own lines: what the collateral column is read on (`heldOn`). */
type Reads = {
  currency: string; recording: boolean; noClose: boolean; collateralError: string | null;
  riskError: string | null; read: string; heldOn: string;
};

/** The credit table's reads - the projection, the risk's marks, the book's rows and, on a desk
 * that records, its paper and the calls on the close marked for the book's day - the clients it
 * reads (`ids`: the paper's entities, then each a set names) and one client as its rows read it.
 * Null until the projection, the book and the paper are read. */
export function useCredit() {
  const { state } = useApp();
  const { xva, doc, schema } = state;
  const recording = state.record.spine !== null;
  const risk = bookRisk(state);
  // a collateral call is read on the book's day marked: what is missing is known before asking
  const spine = spineOf(state);
  const close = unmarked(spine, true);
  const collateral = useRead(spine && !close ? getBookCollateral : null,
                             [clockOf(state), state.seat, close?.title]);
  const paper = recording ? state.paper.data : null;
  if (!xva.data || !doc || !schema || (recording && !paper)) return null;

  const view = xva.data;
  const currency = risk.data?.currency ?? reportCurrency(doc);
  const market = marketOf(state);
  const factors = market?.['Price Factors'] ?? {};
  const rows = rowsByPath(toRows(doc.Calc.Deals.Deals.Children, [], undefined));
  const client = (id: string): Client => {
    const agreements = agreementsOf(id, paper?.agreements ?? null, rows, view.sets);
    const lines = creditLines(id, {
      sets: view.sets, agreements, rows, positions: paper?.positions ?? null,
      marks: risk.marks, calls: recording ? collateral.data?.calls ?? null
        : agreements.map((row) => statedHolding(row.terms, fieldsOf(schema, SET) ?? {}, market)),
    });
    const own = lines.filter((line) => !line.run?.note);
    const [first] = heldSpreads(factors[`${SURVIVAL}.${id}`], schema)?.rows ?? [];
    return {
      id, name: entityText(paper?.entities ?? null, id), lines, wrong: wrongWith(lines),
      spread: first && [first[0], first[1]], value: summed(own.map((line) => line.value)),
      after: summed(own.map((line) => exposureAfter(line, currency))),
      limit: limitOf(id, view.limits),
    };
  };
  const marked = (unread?: string) => closeRead(collateral.data?.date, !!close, unread);
  return {
    view, risk, collateral, close, marked, client,
    ids: [...new Set([...(paper?.entities ?? []).map((row) => row.entity),
                      ...view.sets.flatMap((set) => (set.counterparty ? [set.counterparty] : []))])],
    reads: { currency, recording, noClose: !!close, collateralError: collateral.error,
             riskError: risk.error, read: view.as_of,
             heldOn: recording ? `the call · ${marked('close unread')}` : "as each set's terms state" },
  };
}

/** The credit table: a row per client, a client trading under several netting sets each beneath it,
 * and a set's row opening its profile. */
export function CreditTable({ clients, reads }: { clients: Client[]; reads: Reads }) {
  const [opened, setOpened] = useState<string | null>(null);
  return (
    <div className="tablewrap">
      <table className="data desk xva credit">
        <thead>
          <tr>
            <th>Client<div className="sub">its sets beneath it</div></th>
            <th className="n">Spread<div className="sub">bp, its curve's first tenor</div></th>
            <th>Paper<div className="sub">by the agreement's terms</div></th>
            <th className="n">Trades<div className="sub">on · waiting on a second seat</div></th>
            <th className="n">Value to the desk<div className="sub">{reads.currency} now · + the client owes</div></th>
            <th className="n">Collateral held<div className="sub">{reads.heldOn}</div></th>
            <th className="n">After collateral<div className="sub">{reads.currency} · nought if owed</div></th>
            <th className="n">Peak EE<div className="sub">run currency · its day</div></th>
            <th className="n">Peak PFE<div className="sub">its percentile · its day</div></th>
            <th className="n">CVA<div className="sub">{PLUS.CVA}</div></th>
            <th className="n">FVA<div className="sub">{PLUS.FVA}</div></th>
            <th>Run<div className="sub">its age</div></th>
            <th />
          </tr>
        </thead>
        <tbody>
          {clients.map((client) => (
            <ClientRows key={client.id} client={client} reads={reads} opened={opened}
                        onOpen={(set) => setOpened(opened === set ? null : set)} />
          ))}
          {clients.length === 0 && (
            <tr><td colSpan={13} className="hint">no client is declared and no netting set
              names one: there is no credit to read.</td></tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

const amount = (value: number | null) => (value === null ? '—' : formatNumber(value, 'whole'));

function ClientRows({ client, reads, opened, onOpen }: {
  client: Client; reads: Reads; opened: string | null; onOpen: (set: string) => void;
}) {
  const { dispatch } = useApp();
  const { lines, spread } = client;
  const own = lines.filter((line) => !line.run?.note);
  const one = lines.length === 1 ? lines[0] : null;
  const shown = lines.find((line) => line.reference === opened);
  const cva = runSum(lines, 'cva');
  const fva = runSum(lines, 'fva');
  const by = <td className="hint n">{lines.length ? 'by set' : ''}</td>;
  const opener = (line: CreditLine) => (
    <td className="opener"><span>{line === shown ? '×' : '⋯'}</span></td>);
  return (
    <>
      <tr className={`${one ? 'pick' : ''}${actsOf(client).length ? ' acting' : ''}`}
          onClick={() => one && onOpen(one.reference)}>
        <td>
          <button className="ghost" title="open this client's page"
                  onClick={(event) => { event.stopPropagation(); openClient(dispatch, client.id); }}>
            {client.name}
          </button>
          {one && <div className="hint mono">{one.reference}</div>}
        </td>
        <td className="n">{spread ? <>{formatNumber(spread[1] * 1e4, 'whole')}
          <div className="hint">{tenorLabel(spread[0])}</div></> : <span className="hint">no curve</span>}</td>
        {one ? <LineCells line={one} reads={reads} /> : (
          <>
            <td>{lines.length ? plural(lines.length, 'set')
              : <span className="hint">no agreement, nothing on</span>}</td>
            <td className="n">{own.reduce((sum, line) => sum + line.trades, 0)}
              {own.some((line) => line.pending) && <span className="badge pending">
                {own.reduce((sum, line) => sum + line.pending, 0)} waiting</span>}</td>
            <Amount value={client.value} absent={reads.riskError ?? 'a set is unread'} />
            {by}
            <Amount value={client.after} absent="a set's value or collateral is unread" />
            {by}{by}
            <Amount kind="money" value={cva.value} currency={cva.currency} absent="a set has no CVA to add" />
            <Amount kind="money" value={fva.value} currency={fva.currency} absent="a set has no FVA to add" />
            {by}
          </>
        )}
        {one ? opener(one) : <td />}
      </tr>
      {client.limit && (
        <tr className="limit"><td colSpan={13}><LimitLine read={client.limit} /></td></tr>
      )}
      {!one && lines.map((line) => (
        <tr key={line.reference} className={`node pick${line.run?.note ? ' muted' : ''}`}
            onClick={() => onOpen(line.reference)}>
          <td className="mono">{line.reference}
            {line.run?.note && <span className="badge" title={line.run.note}>left the book</span>}</td>
          <td />
          <LineCells line={line} reads={reads} />
          {opener(line)}
        </tr>
      ))}
      {client.wrong.length > 0 && (
        <tr className="wrongs">
          <td colSpan={13}>
            <ul className="facts">
              {client.wrong.map((wrong, at) => (
                <li key={at} data-tone={wrong.kind === 'failed' || wrong.kind === 'unpapered' ? 'todo' : 'warn'}>
                  <b>{WRONG[wrong.kind]}</b>{!one && <> under <span className="mono">{wrong.reference}</span></>}:
                  {' '}{wrong.text} <span className="hint">— {wrong.who}</span>
                </li>
              ))}
            </ul>
          </td>
        </tr>
      )}
      {shown && (
        <tr className="detail">
          <td colSpan={13}>
            <Profiled line={shown} />
            <button className="ghost" onClick={() => openClient(dispatch, client.id)}>
              open {client.name}'s page: its paper, its trades and what happened to it
            </button>
          </td>
        </tr>
      )}
    </>
  );
}

/** One set's ten cells, from its paper to its run's age - its peaks off its own row, each PFE
 * column's a line of the cell. */
function LineCells({ line, reads }: { line: CreditLine; reads: Reads }) {
  const { run, call } = line;
  const age = run && ageOf(run.as_of, reads.read);
  const top = peaks(run?.peaks ?? null);
  const why = !run || run.status === 'never run' ? 'never run' : run.status === 'failed'
    ? 'run failed' : !run.peaks ? 'not kept' : 'no column';
  const amount = (peak: Peak, day: string): ReactNode => (
    <div key={day}><Money value={peak.value} currency={run!.currency} kind="whole" />
      <div className="hint">{day}</div></div>);
  const peak = (values: ReactNode[]) => (
    <td className="n">{values.length ? values : (
      <span className="hint" title={why === 'not kept' ? UNKEPT : undefined}>{why}</span>
    )}</td>);
  return (
    <>
      <td title={line.paper ?? undefined}>
        <span className={`chip${line.paper === null ? ' error' : line.collateralised ? ' on' : ''}`}>
          {paperText(line)}</span></td>
      <td className="n">{line.trades}
        {line.pending > 0 && <span className="badge pending">{line.pending} waiting</span>}</td>
      <Amount value={line.value} absent={reads.riskError ?? 'the risk marks top-level sets only'} />
      <td className="n">
        {!line.collateralised ? <span className="hint">no CSA</span>
          : call?.balance !== null && call?.balance !== undefined ? (
              <><Money value={call.balance} currency={call.currency} kind="whole" />
                <div className={call.direction ? 'call' : 'hint'}>
                  {!reads.recording ? 'as its terms state'
                    : call.direction === 'call' ? `calls ${formatNumber(call.call!, 'whole')}`
                      : call.direction === 'post' ? `posts ${formatNumber(-call.call!, 'whole')}`
                        : 'no call'}</div></>
            ) : <span className="hint" title={reads.collateralError ?? call?.unknown
              .map((row) => row.what).join('; ')}>{closeRead(undefined, reads.noClose)}</span>}
      </td>
      <Amount value={exposureAfter(line, reads.currency)}
              absent={line.collateralised ? 'its collateral is unread' : 'its value is unread'} />
      {peak(top.ee ? [amount(top.ee, top.ee.day)] : [])}
      {peak(top.pfe.map((pfe) => amount(pfe, `${formatNumber(pfe.quantile)}th · ${pfe.day}`)))}
      <Amount kind="money" value={run?.cva ?? null} currency={run?.currency ?? null}
              absent={run?.status === 'failed' ? 'its run failed' : 'no completed run'} />
      <Amount kind="money" value={run?.fva ?? null} currency={run?.currency ?? null}
              absent={run?.status === 'failed' ? 'its run failed' : 'no completed run'} />
      <td className={`age ${line.booked ? 'stale' : age?.tone ?? 'none'}`} title={age?.title}>
        {age ? age.text : 'no set booked'}
        {!!line.booked && <div>{bookedText(line.booked)}</div>}
        {run?.recalc && <span className="chip running">{run.recalc.status}</span>}
      </td>
    </>
  );
}

/** One set's profile against time, read for it alone when its row opens: its expected exposure
 * and PFE as the run read them, net of the collateral its terms hold - or the service's sentence
 * for why there is none. */
function Profiled({ line }: { line: CreditLine }) {
  const { state } = useApp();
  const { run } = line;
  const read = useRead(run ? () => getXvaProfile(line.reference) : null,
                       [line.reference, run?.result_id, run?.as_of]);
  const page = read.data?.profile;
  return (
    <section className="card profile">
      <h3>{line.reference} — {line.collateralised
        ? 'collateralised: the exposure is net of the collateral its terms hold'
        : 'uncollateralised'}{run?.as_of && <span className="hint"> · run {stampText(run.as_of)}</span>}</h3>
      {run?.status === 'failed' && <div className="error-box">{run.error}</div>}
      {page ? (
        <>
          <div className="tablenote">
            <span>amounts in <b>{run?.currency}</b>, the run's report currency</span>
            {tableReading('exposure_profile', page.columns, true, state.doc!, [],
                          marketOf(state)).columns.map(([column, words]) => (
              <span key={column}><b>{column}</b> {words}</span>
            ))}
          </div>
          <TimeSeriesChart page={page} />
        </>
      ) : (
        <div className="hint">{!run ? 'Nothing is booked under it, so it has no run and no profile.'
          : read.error ?? 'Reading its profile…'}</div>
      )}
      {run?.as_of && (
        <details className="fold">
          <summary>the run behind the numbers, to reproduce them</summary>
          <div className="fields">
            {replayRows(run).map(([name, value]) => [
              <div key={`${name}k`} className="k">{name}</div>,
              <div key={`${name}v`} className="v mono">{value ?? '—'}</div>,
            ])}
          </div>
        </details>
      )}
    </section>
  );
}
