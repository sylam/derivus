import { useState, type ReactNode } from 'react';
import { getBookActivity, getBookCollateral, getBookRisk, getXvaProfile } from '../api';
import { SET, fieldsOf, rowsByPath, toRows } from '../blotter';
import {
  RUNS, WRONG, actionFirst, agreementsOf, creditLines, exposureAfter, openClient, runSum,
  statedHolding, summed, unmarked, wrongWith, type CreditLine, type Wrong,
} from '../client';
import { EmptyState } from '../components/EmptyState';
import { TimeSeriesChart } from '../components/TimeSeriesChart';
import {
  SURVIVAL, ageOf, currencyOf, entityText, lagsBook, peaks, plural, replayRows, spreads, stampText,
  tableReading, xvaTotals, type Peak,
} from '../desk';
import { pinnedAt } from '../spine';
import { clockOf, marketOf, useApp } from '../state';
import { curveOf, formatNumber, tenorLabel, token } from '../tokens';
import { wholeRecord } from '../trade';
import { useRead } from './ClientView';
import { OnStrip } from './RiskView';

/** What a cell says of a done run whose row keeps no peaks. */
const UNKEPT = 'this row was filed before rows kept a run\'s peaks: a recalc keeps them';

/** A client as the screen holds it: its lines, what is wrong under them, and its sums. */
type Client = {
  id: string; name: string; spread?: [number, number]; lines: CreditLine[]; wrong: Wrong[];
  value: number | null; after: number | null;
};

/** Every counterparty as a credit officer reads it, the ones to act on first: what the desk is
 * owed, the collateral against it, the exposure after collateral, how large it can get, its CVA
 * and FVA - each its own set's last run, with its own age - and what is wrong, in business words.
 * A set's row opens its profile and leads to the client's page.
 *
 * READ-ONLY BY DESIGN: a recalc is asked for through the chat, never from here, and the screen
 * says so. A run's peaks are read off its row, its profile one set at a time; on a desk that
 * records, the events filed under each run set since its run are read by the set's agreement. One
 * that keeps no record reads no collateral call, no second seat and no events, and says so.
 */
export function XvaView() {
  const { state } = useApp();
  const { xva, source, doc, schema } = state;
  const live = source?.kind === 'book';
  const recording = state.record.spine !== null;
  const clock = clockOf(state);
  const [opened, setOpened] = useState<string | null>(null);
  const risk = useRead(live ? getBookRisk : null, [clock, state.seat]);
  const collateral = useRead(live && recording ? getBookCollateral : null, [clock, state.seat]);
  const ran = (xva.data?.sets ?? []).filter((set) => set.as_of !== null).map((set) => set.reference);
  const record = useRead(live && recording && ran.length ? async () => Object.fromEntries(
    await Promise.all(ran.map(async (reference) => [reference, await wholeRecord((since) =>
      getBookActivity(since, { agreement: reference }))] as const))) : null,
  [ran.join('\n'), pinnedAt(state.record.spine)]);

  if (!live) {
    return (
      <EmptyState
        title="The XVA projection belongs to the live book."
        hint="This document was opened from a file. Start the service with --book <job file> and the desk's projection appears here."
      />
    );
  }
  if (xva.status === 404) {
    return (
      <EmptyState
        title="The service is serving no book."
        hint="It was started without --book, or the book has been closed since this page loaded."
      />
    );
  }
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
  if (!xva.data || !doc || !schema || (recording && !state.paper.data)) {
    return (
      <EmptyState
        title={state.paper.error ?? (xva.loading || state.paper.loading ? 'Reading the projection and the paper…'
          : 'No projection has been read yet.')}
        hint="The last run per netting set, joined with the book's own set list and the clients it trades with."
      />
    );
  }

  const view = xva.data;
  const currency = risk.data?.currency ?? currencyOf(doc);
  const paper = recording ? state.paper.data : null;
  const market = marketOf(state);
  const factors = market?.['Price Factors'] ?? {};
  const rows = rowsByPath(toRows(doc.Calc.Deals.Deals.Children, [], undefined));
  const ids = new Set([...(paper?.entities ?? []).map((row) => row.entity),
                       ...view.sets.flatMap((set) => (set.counterparty ? [set.counterparty] : []))]);
  const clients: Client[] = actionFirst([...ids].map((id) => {
    const agreements = agreementsOf(id, paper?.agreements ?? null, doc);
    const lines = creditLines(id, {
      sets: view.sets, agreements, rows, positions: paper?.positions ?? null,
      record: recording ? record.data : null,
      marks: risk.data?.per_deal ?? null, calls: recording ? collateral.data?.calls ?? null
        : agreements.map((row) => statedHolding(row.terms, fieldsOf(schema, SET) ?? {}, market)),
    }, schema);
    const own = lines.filter((line) => !line.run?.note);
    const factor = factors[`${SURVIVAL}.${id}`];
    const curve = curveOf(token(factor, 'Curve'));
    const [first] = curve ? spreads(curve.data, Number(token(factor, 'Recovery_Rate'))) : [];
    return {
      id, name: paper ? entityText(paper.entities, id) : id, lines, wrong: wrongWith(lines),
      spread: first && [first[0], first[1]], value: summed(own.map((line) => line.value)),
      after: summed(own.map((line) => exposureAfter(line, currency))),
    };
  }));
  const acting = clients.filter((client) => client.wrong.length);
  const counts = (Object.keys(WRONG) as (keyof typeof WRONG)[]).map((kind) =>
    [kind, clients.flatMap((client) => client.wrong).filter((wrong) => wrong.kind === kind).length] as const)
    .filter(([, count]) => count);
  const marked = collateral.data ? `close of ${collateral.data.date}`
    : unmarked(collateral.error) ? 'no close marked' : null;
  const reads = { currency, recording, marked, collateralError: collateral.error,
                  riskError: risk.error, read: view.as_of };

  return (
    <div className="main">
      <div className="panel">
        <OnStrip
          computed={<>{RUNS}; the value to the desk
            {risk.data ? <> read {stampText(risk.data.as_of)}</> : ' unread'}
            {recording && <>; collateral{collateral.data ? ` on the ${marked}` : `: ${marked ?? 'unread'}`}</>}</>}
          currency={currency} lagging={lagsBook(xva.clock, clock)} loading={xva.loading}
        />
        {[risk.error && `The value to the desk is not read: ${risk.error}`,
          collateral.error && `The collateral is not read: ${collateral.error}`]
          .map((refused) => refused && <div key={refused} className="error-box">{refused}</div>)}
        <div className={`blotterbar wrongs${acting.length ? ' act' : ''}`}>
          {acting.length ? (
            <span className="hint">
              <b>{plural(acting.length, 'client')}</b> of {clients.length} to act on:{' '}
              {counts.map(([kind, count]) => `${count} ${WRONG[kind]}`).join(' · ')}
            </span>
          ) : (
            <span className="hint">
              <b>Nothing to act on.</b> Every client's sets have a run that is done{recording
                ? ', nothing filed under them since, and no trade waits on a second seat' : ''}.
            </span>
          )}
          <span className="spacer" />
          <span className="hint">
            {plural(clients.length, 'client')} · value to the desk <b>{amount(summed(
              clients.map((client) => client.value)))}</b> {currency} · after collateral{' '}
            <b>{amount(summed(clients.map((client) => client.after)))}</b> {currency}
            {xvaTotals(view).sums.map((sum) => (
              <span key={sum.currency ?? ''}> · CVA over the {sum.counted} run in{' '}
                {sum.currency ?? '(currency unknown)'} <b>{formatNumber(sum.cva, 'money')}</b></span>
            ))}
          </span>
        </div>

        <div className="note">
          Each row is a client; a client trading under several netting sets has each beneath it,
          and a set's row opens its profile.{' '}
          {recording ? 'Collateral is read on the close marked for the book\'s day; trades '
            + 'waiting on a second seat and the events filed under a set since its run off the '
            + 'record.'
            : "This desk keeps no record: collateral is what each set's terms state it holds, no "
              + 'call is worked, no trade waits on a second seat and no event is filed under a set.'}
        </div>

        <div className="tablewrap">
          <table className="data desk xva credit">
            <thead>
              <tr>
                <th>Client<div className="sub">its sets beneath it</div></th>
                <th className="n">Spread<div className="sub">bp, its curve's first tenor</div></th>
                <th>Paper<div className="sub">by the agreement's terms</div></th>
                <th className="n">Trades<div className="sub">on · waiting on a second seat</div></th>
                <th className="n">Value to the desk<div className="sub">{currency} now · + the client owes</div></th>
                <th className="n">Collateral held<div className="sub">{recording
                  ? `the call · ${marked ?? 'close unread'}` : "as each set's terms state"}</div></th>
                <th className="n">After collateral<div className="sub">{currency} · nought if owed</div></th>
                <th className="n">Peak EE<div className="sub">run currency · its day</div></th>
                <th className="n">Peak PFE<div className="sub">its percentile · its day</div></th>
                <th className="n">CVA<div className="sub">+ an expected loss</div></th>
                <th className="n">FVA<div className="sub">+ a funding cost</div></th>
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
      </div>
    </div>
  );
}

/** What the rows read beside their own lines. */
type Reads = {
  currency: string; recording: boolean; marked: string | null; collateralError: string | null;
  riskError: string | null; read: string;
};

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
      <tr className={`${one ? 'pick' : ''}${client.wrong.length ? ' acting' : ''}`}
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
    <div key={day}>{formatNumber(peak.value, 'whole')} <span className="unit">{run!.currency}</span>
      <div className="hint">{day}</div></div>);
  const peak = (values: ReactNode[]) => (
    <td className="n">{values.length ? values : (
      <span className="hint" title={why === 'not kept' ? UNKEPT : undefined}>{why}</span>
    )}</td>);
  return (
    <>
      <td title={line.paper ?? undefined}>{line.paper === null
        ? <span className="chip error">no agreement</span>
        : <span className={`chip${line.collateralised ? ' on' : ''}`}>
          {line.collateralised ? 'collateralised' : 'uncollateralised'}</span>}</td>
      <td className="n">{line.trades}
        {line.pending > 0 && <span className="badge pending">{line.pending} waiting</span>}</td>
      <Amount value={line.value} absent={reads.riskError ?? 'the risk marks top-level sets only'} />
      <td className="n">
        {!line.collateralised ? <span className="hint">no CSA</span>
          : call?.balance !== null && call?.balance !== undefined ? (
              <>{formatNumber(call.balance, 'whole')} <span className="unit">{call.currency}</span>
                <div className={call.direction ? 'call' : 'hint'}>
                  {!reads.recording ? 'as its terms state'
                    : call.direction === 'call' ? `calls ${formatNumber(call.call!, 'whole')}`
                      : call.direction === 'post' ? `posts ${formatNumber(-call.call!, 'whole')}`
                        : 'no call'}</div></>
            ) : <span className="hint" title={reads.collateralError ?? call?.unknown
              .map((row) => row.what).join('; ')}>{reads.marked === 'no close marked'
                ? 'no close marked' : 'unread'}</span>}
      </td>
      <Amount value={exposureAfter(line, reads.currency)}
              absent={line.collateralised ? 'its collateral is unread' : 'its value is unread'} />
      {peak(top.ee ? [amount(top.ee, top.ee.day)] : [])}
      {peak(top.pfe.map((pfe) => amount(pfe, `${formatNumber(pfe.quantile)}th · ${pfe.day}`)))}
      <Amount kind="money" value={run?.cva ?? null} currency={run?.currency ?? null}
              absent={run?.status === 'failed' ? 'its run failed' : 'no completed run'} />
      <Amount kind="money" value={run?.fva ?? null} currency={run?.currency ?? null}
              absent={run?.status === 'failed' ? 'its run failed' : 'no completed run'} />
      <td className={`age ${line.moved ? 'stale' : age?.tone ?? 'none'}`} title={age?.title}>
        {age ? age.text : 'no set booked'}
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
            {tableReading('exposure_profile', page.columns, true, state.doc!, state.schema,
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

/** An amount, in whole units unless it is an adjustment, in the `currency` beside it where the
 * header does not name one. A null is an absence, and says why on hover. */
function Amount({ value, currency, absent, kind = 'whole' }: {
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
