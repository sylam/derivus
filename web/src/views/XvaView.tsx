import { useState, type ReactNode } from 'react';
import {
  failure, getBookActivity, getBookCollateral, getBookRisk, getResult, getTable,
} from '../api';
import { SET, fieldsOf, rowsByPath, toRows } from '../blotter';
import {
  WRONG, actionFirst, agreementsOf, creditLines, exposureAfter, openClient, runSum, statedHolding,
  summed, wrongWith, type CreditLine, type Wrong,
} from '../client';
import { EmptyState } from '../components/EmptyState';
import { TimeSeriesChart } from '../components/TimeSeriesChart';
import {
  SURVIVAL, ageOf, currencyOf, entityText, lagsBook, peaks, plural, replayRows, spreads, stampText,
  tableReading, xvaTotals,
} from '../desk';
import { clockOf, marketOf, useApp } from '../state';
import { curveOf, formatNumber, tenorLabel, token } from '../tokens';
import { wholeRecord } from '../trade';
import type { TablePage } from '../types';
import { useRead } from './ClientView';
import { OnStrip } from './RiskView';

/** A run's exposure profile, or the word for why it cannot be read. */
type Profile = { page: TablePage } | { missing: string };

/** What a cell says of a profile the service does not hold: it keeps a run's tables in memory. */
const NOT_HELD = 'the service keeps a run\'s profile in memory only, and it has restarted since '
  + 'this run: a recalc reads it again';

const profileOf = (id: string): Promise<Profile> => getResult(id)
  .then((result): Profile | Promise<Profile> => (result.tables?.exposure_profile
    ? getTable(id, 'exposure_profile').then((page) => ({ page })) : { missing: 'no profile' }))
  .catch((error) => ({ missing: failure(error).status === 404 ? 'not held' : failure(error).error }));

/** A client as the screen holds it: its lines, what is wrong under them, and its sums. */
type Client = {
  id: string; name: string; spread?: [number, number]; lines: CreditLine[]; wrong: Wrong[];
  value: number | null; after: number | null;
};

/** Every counterparty as a credit officer reads it, the ones to act on first: what the desk is
 * owed, the collateral against it, the exposure after collateral, how large it can get, its CVA
 * and FVA - each its own set's last run, with its own age - and what is wrong, in business words.
 * A row opens its runs' profiles and leads to the client's page.
 *
 * READ-ONLY BY DESIGN: a recalc is asked for through the MCP verbs, never from here, and the
 * screen says so. On a desk that keeps no record, collateral, a second seat and a run older than
 * the book are not read, and the screen says that too.
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
  const record = useRead(live && recording ? () => wholeRecord(getBookActivity) : null, [clock]);
  const ran = (xva.data?.sets ?? []).filter((set) => set.status === 'done' && set.result_id)
    .map((set) => set.result_id!);
  const profiles = useRead(ran.length ? async () => Object.fromEntries(await Promise.all(
    ran.map(async (id) => [id, await profileOf(id)] as const))) : null, [ran.join()]);

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
      sets: view.sets, agreements, rows, positions: paper?.positions ?? null, record: record.data,
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
    : /^no marks stand/.test(collateral.error ?? '') ? 'no close marked' : null;
  const reads = { currency, recording, marked, collateralError: collateral.error,
                  riskError: risk.error, profiles: profiles.data, read: view.as_of };

  return (
    <div className="main">
      <div className="panel">
        <OnStrip
          computed={<>each row its own set's last run, with its own age; the value to the desk
            {risk.data ? <> read {stampText(risk.data.as_of)}</> : ' unread'}
            {recording && <>; collateral on the {marked ?? 'close: unread'}</>}</>}
          currency={currency} lagging={lagsBook(xva.clock, clock)} loading={xva.loading}
        />
        {[risk.error && `The value to the desk is not read: ${risk.error}`,
          !marked && collateral.error && `The collateral is not read: ${collateral.error}`]
          .map((refused) => refused && <div key={refused} className="error-box">{refused}</div>)}
        <div className={`blotterbar wrongs${acting.length ? ' act' : ''}`}>
          {acting.length ? (
            <span className="hint">
              <b>{plural(acting.length, 'client')}</b> of {clients.length} to act on:{' '}
              {counts.map(([kind, count]) => `${count} ${WRONG[kind]}`).join(' · ')}
            </span>
          ) : (
            <span className="hint">
              <b>Nothing to act on.</b> Every client's sets have a run that is done
              {recording ? ', nothing is booked under them since, and no trade waits on a second seat'
                : ''}.
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
          Each row is a client; a client trading under several netting sets has each beneath it.
          A recalc is asked for through the MCP verbs — <code>recalc_xva</code> — and never from
          here: this screen reads the desk's projection, the last run per set.{' '}
          {recording ? 'Collateral is read on the close marked for the book\'s day; trades waiting '
            + 'on a second seat and a run older than the book are read off the record.'
            : "This desk keeps no record: collateral is what each set's terms state it holds and "
              + 'no call is worked, no trade waits on a second seat, and whether a run is older '
              + 'than the book is not read here.'}{' '}
          The service holds a run's profile in memory: once it restarts, the profile reads "not
          held" until a recalc.
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
                <ClientRows key={client.id} client={client} reads={reads}
                            opened={opened === client.id}
                            onOpen={() => setOpened(opened === client.id ? null : client.id)} />
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
  riskError: string | null; profiles: Record<string, Profile> | null; read: string;
};

const amount = (value: number | null) => (value === null ? '—' : formatNumber(value, 'whole'));

function ClientRows({ client, reads, opened, onOpen }: {
  client: Client; reads: Reads; opened: boolean; onOpen: () => void;
}) {
  const { dispatch } = useApp();
  const { lines, spread } = client;
  const own = lines.filter((line) => !line.run?.note);
  const one = lines.length === 1 ? lines[0] : null;
  const cva = runSum(lines, 'cva');
  const fva = runSum(lines, 'fva');
  const by = <td className="hint n">{lines.length ? 'by set' : ''}</td>;
  return (
    <>
      <tr className={`pick${client.wrong.length ? ' acting' : ''}`} onClick={onOpen}>
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
        <td className="opener"><span>{opened ? '×' : '⋯'}</span></td>
      </tr>
      {!one && lines.map((line) => (
        <tr key={line.reference} className={`node${line.run?.note ? ' muted' : ''}`}>
          <td className="mono">{line.reference}
            {line.run?.note && <span className="badge" title={line.run.note}>left the book</span>}</td>
          <td />
          <LineCells line={line} reads={reads} />
          <td />
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
      {opened && (
        <tr className="detail">
          <td colSpan={13}>
            {lines.map((line) => <Profiled key={line.reference} line={line} reads={reads} />)}
            <button className="ghost" onClick={() => openClient(dispatch, client.id)}>
              open {client.name}'s page: its paper, its trades and what happened to it
            </button>
          </td>
        </tr>
      )}
    </>
  );
}

/** One set's ten cells, from its paper to its run's age. */
function LineCells({ line, reads }: { line: CreditLine; reads: Reads }) {
  const { run, call } = line;
  const age = run && ageOf(run.as_of, reads.read);
  const profile = run?.result_id ? reads.profiles?.[run.result_id] : undefined;
  const top = profile && 'page' in profile ? peaks(profile.page) : undefined;
  const why = !run || run.status === 'never run' ? 'never run' : run.status === 'failed'
    ? 'run failed' : !profile ? '…' : 'missing' in profile ? profile.missing : 'no column';
  const peak = (value: ReactNode, day?: string) => (
    <td className="n">{value}{day && <div className="hint">{day}</div>}</td>);
  const short = (word: string) => peak(<span className="hint" title={word === 'not held'
    ? NOT_HELD : undefined}>{word}</span>);
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
      {top?.ee ? peak(<>{formatNumber(top.ee.value, 'whole')} <span className="unit">{run!.currency}</span></>,
                      top.ee.day) : short(why)}
      {top?.pfe ? peak(<>{formatNumber(top.pfe.value, 'whole')} <span className="unit">{run!.currency}</span></>,
                       `${formatNumber(top.pfe.quantile)}th · ${top.pfe.day}`) : short(why)}
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

/** One set's profile against time: its expected exposure and PFE as the run read them, net of
 * the collateral its terms hold - or why it cannot be read. */
function Profiled({ line, reads }: { line: CreditLine; reads: Reads }) {
  const { state } = useApp();
  const { run } = line;
  const profile = run?.result_id ? reads.profiles?.[run.result_id] : undefined;
  const page = profile && 'page' in profile ? profile.page : null;
  return (
    <section className="card profile">
      <h3>{line.reference} — {line.collateralised
        ? 'collateralised: the exposure is net of the collateral its terms hold'
        : 'uncollateralised'}{run?.as_of && <span className="hint"> · run {stampText(run.as_of)}</span>}</h3>
      {run?.status === 'failed' && <div className="error-box">{run.error}</div>}
      {page ? (
        <>
          <div className="tablenote">
            <span>amounts in <b>{run!.currency}</b>, the run's report currency</span>
            {tableReading('exposure_profile', page.columns, true, state.doc!, state.schema,
                          marketOf(state)).columns.map(([column, words]) => (
              <span key={column}><b>{column}</b> {words}</span>
            ))}
          </div>
          <TimeSeriesChart page={page} />
        </>
      ) : run?.status === 'done' && (
        <div className="hint">{profile && 'missing' in profile && profile.missing === 'not held'
          ? `No profile: ${NOT_HELD}.` : 'Reading the profile…'}</div>
      )}
      {(!run || run.status === 'never run') && (
        <div className="hint">No run yet, so no profile: {line.trades
          ? 'a recalc through recalc_xva runs one.' : 'nothing is booked under it.'}</div>
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
