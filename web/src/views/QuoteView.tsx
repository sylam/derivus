import { useEffect, useRef, useState } from 'react';
import { acceptQuote, failure, getQuoteSheet, getResult, quoteStructure } from '../api';
import { EmptyState } from '../components/EmptyState';
import { DescriptorPanel } from '../components/FieldView';
import { Navigator } from '../components/Navigator';
import { stampText } from '../desk';
import { NO_PAPER, nodeAt } from '../positions';
import {
  clients, clientText, directions, type Draft, draftOf, firmSeconds, firmText, ticketFields, keep,
  type Kept, landedAt, requestOf, salesName, salesNames, solving, STANDS, standsAt, type Verdict,
  verdictOf,
} from '../quote';
import { heldBack, openTrade, useApp, useQuoteList, useQuotes } from '../state';
import { formatNumber } from '../tokens';
import type { PendingQuote, Quote, QuoteGiven, StructureType } from '../types';
import { TicketCard } from '../components/TicketCard';

const POLL_MS = 500;

/** The page listing the desk's quotes, and the one a new ticket is written on. */
const LIST = 'list';
const NEW = 'new';

/** Where this browser keeps the re-quotes it made, each with the quote it re-quotes, which the
 * service does not link - a convenience, so a storage that refuses costs the links alone. */
const KEPT = 'derivus.quotes';

function useKept() {
  const [list, setList] = useState<Kept[]>(() => {
    try {
      return JSON.parse(localStorage.getItem(KEPT) ?? '[]');
    } catch {
      return [];
    }
  });
  useEffect(() => {
    try {
      localStorage.setItem(KEPT, JSON.stringify(list));
    } catch { /* kept for the session alone */ }
  }, [list]);
  return [list, setList] as const;
}

/** A quote id as a desk reads one on a screen: its first eight characters, whole on hover. */
const short = (id: string) => `${id.slice(0, 8)}…`;

/** The sales desk's ticket: a structure quoted off what `/schema` declares for it, the answer read
 * as one card a person reads to a client, and the acceptance that books it - beside every quote
 * the desk has given, newest first, a quote the chat gave opening by its id as the same card.
 * Every level reads on the market's axis by the blotter's rule. A quote is an act: changing one
 * quotes again, and the old one stands beside the new. */
export function QuoteView() {
  const { state, dispatch } = useApp();
  const { doc, schema } = state;
  const [links, setLinks] = useKept();
  // a quote made or typed here, until the desk's list holds it
  const [opened, setOpened] = useState<string[]>([]);
  const [again, setAgain] = useState(0);
  const listed = useQuoteList(again);
  // where each quote stands is read against this browser's clock, again every half minute
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 30_000);
    return () => clearInterval(timer);
  }, []);

  if (!doc || !schema) return null;
  const structures = schema.Structure?.types ?? {};
  if (state.source?.kind !== 'book' || !Object.keys(structures).length) {
    return (
      <EmptyState
        title="A quote prices against the desk's own book."
        hint={state.source?.kind !== 'book'
          ? 'A file opened in this browser has no service book behind it: start the service '
            + 'with --book to quote.'
          : 'This service publishes no structures to quote.'} />
    );
  }

  const rows = listed.data?.quotes ?? [];
  const window = firmSeconds(doc, schema);
  const open = (id: string, from?: string) => {
    if (from) setLinks((standing) => keep(standing, id, from));
    setOpened((standing) => [id, ...standing]);
    setAgain((count) => count + 1);
    dispatch({ type: 'PICK', screen: 'quote', id });
  };
  const ticket = (id: string) => (
    <Ticket id={id} from={links.find((entry) => entry.id === id)?.from}
            requotes={links.filter((entry) => entry.from === id).map((entry) => entry.id)}
            onMade={open} />
  );

  return (
    <Navigator
      screen="quote" side={<OpenById onOpen={(id) => open(id)} />}
      pages={[
        { id: LIST, label: "the desk's quotes",
          content: <QuoteList listed={listed} window={window} now={now} /> },
        { id: NEW, label: 'new quote', accent: true,
          content: <QuoteForm onQuoted={(id) => open(id)} /> },
        ...opened.filter((id) => !rows.some((row) => row.quote_id === id))
          .map((id) => ({ id, label: short(id), hint: 'opened by id', content: ticket(id) })),
        ...rows.map((row) => ({
          id: row.quote_id, label: `${salesName(schema, row.structure)} · ${
            row.netting_set ?? 'root'}`,
          hint: standsAt(row, window, now).word, content: ticket(row.quote_id),
        })),
      ]}
    />
  );
}

const CHIP: Record<string, string> = { booked: 'done', firm: 'running', lapsed: 'error' };

/** Where a quote stands as its row says it: a word, the time its window closes or closed, and
 * the trade a booked one booked - where the file holds it now, opening its page. */
function Stands({ row, window, now }: { row: QuoteGiven; window: number | null; now: number }) {
  const { state, dispatch } = useApp();
  const { word, until } = standsAt(row, window, now);
  const children = state.doc?.Calc.Deals?.Deals?.Children ?? [];
  const at = row.booked ? landedAt(children, row.booked) : undefined;
  const named = row.booked && (row.booked.reference ?? (at === undefined
    ? row.booked.deal_path : nodeAt(children, at)?.Instrument['.Deal'].Reference));
  return (
    <>
      <span className={`chip ${CHIP[word] ?? ''}`}>{STANDS[word]}</span>
      {until && (
        <span className="hint"> {word === 'firm' ? 'until' : 'since'} {stampText(until)}</span>
      )}
      {row.booked && (at === undefined ? (
        <span className="hint"> {String(named)}, which the book file holds no more</span>
      ) : (
        <button className="tradelink" title="open the trade's page"
                onClick={(event) => {
                  event.stopPropagation();
                  openTrade(dispatch, at);
                }}>
          {' '}{String(named ?? at)} ›
        </button>
      ))}
    </>
  );
}

/** The desk's quotes, newest first: when and by whom, what it is and for whom, its net with its
 * currency, and where it stands - a row opening its ticket. */
function QuoteList({ listed, window, now }: {
  listed: { data: { quotes: QuoteGiven[] } | null; error: string | null }; window: number | null;
  now: number;
}) {
  const { state, dispatch } = useApp();
  const paper = state.paper.data ?? NO_PAPER;
  const rows = listed.data?.quotes ?? [];
  return (
    <section className="card">
      <h3>The desk's quotes <span className="hint">newest first</span></h3>
      <div className="nested">
        {listed.error && <div className="error-box">{listed.error}</div>}
        {!listed.data && !listed.error && <div className="hint">reading the desk's quotes…</div>}
        {listed.data && !rows.length && (
          <div className="hint">This desk has given no quote yet.</div>
        )}
        {rows.length > 0 && (
          <div className="tablewrap">
            <table className="data desk quotes">
              <thead>
                <tr>
                  <th>Quoted</th><th>By</th><th>What</th><th>For</th><th className="n">Net</th>
                  <th>Ccy</th><th>Where it stands</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.quote_id} className="pick" title={`quote ${row.quote_id}`}
                      onClick={() => dispatch({ type: 'PICK', screen: 'quote', id: row.quote_id })}>
                    <td>{stampText(row.quoted_at)}</td>
                    <td>{row.quoted_by ?? <span className="hint">no seat named</span>}</td>
                    <td title={row.structure}>{salesName(state.schema, row.structure)}
                      {row.variation && ` · ${row.variation}`}</td>
                    <td>{clientText(row.netting_set, paper.entities, paper.agreements)}</td>
                    <td className="n">{formatNumber(row.net, 'money')}</td>
                    <td>{row.currency}</td>
                    <td><Stands row={row} window={window} now={now} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <div className="hint">{firmText(window, state.record.spine !== null)}</div>
      </div>
    </section>
  );
}

/** A quote given elsewhere - the chat, another browser - opened by the id it was filed under. */
function OpenById({ onOpen }: { onOpen: (id: string) => void }) {
  const [text, setText] = useState('');
  const open = () => { if (text.trim()) onOpen(text.trim()); setText(''); };
  return (
    <div className="quoteopen">
      <input type="text" value={text} placeholder="open a quote by its id"
             onChange={(event) => setText(event.target.value)}
             onKeyDown={(event) => { if (event.key === 'Enter') open(); }} />
      <button className="ghost" onClick={open} disabled={!text.trim()}>open</button>
    </div>
  );
}

/** A quote asked of the service: submitted under the acting seat, then polled until its run
 * answers - the id of the quote it filed, or the service's own sentence. One at a time. */
function useQuoting(onQuoted: (id: string) => void) {
  const [run, setRun] = useState<{ id: string; status: string; since: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const busy = useRef(false);
  const finish = (said: string | null) => { busy.current = false; setRun(null); setError(said); };

  useEffect(() => {
    if (!run?.id) return;
    const timer = setInterval(async () => {
      try {
        const summary = await getResult(run.id);
        if (summary.status === 'done') {
          finish(null);
          onQuoted(String((summary.stats?.Quote as Quote | undefined)?.quote_id));
        } else if (summary.status === 'error') {
          finish(summary.error ?? 'the quote did not price');
        } else setRun((standing) => standing && { ...standing, status: summary.status });
      } catch (thrown) {
        finish(failure(thrown).error);
      }
    }, POLL_MS);
    return () => clearInterval(timer);
  }, [run?.id]);

  async function start(request: Record<string, unknown>) {
    if (busy.current) return;
    busy.current = true;
    setError(null);
    setRun({ id: '', status: 'sending', since: Date.now() });
    try {
      const submitted = await quoteStructure(request);
      setRun({ id: submitted.result_id, status: submitted.status, since: Date.now() });
    } catch (thrown) {
      finish(failure(thrown).error);
    }
  }
  return { run, error, start };
}

/** The wait, and what ended it where it was a refusal. */
function Waiting({ run, error }: ReturnType<typeof useQuoting>) {
  return (
    <>
      {run && (
        <span className="chip running">
          pricing - {run.status}, {Math.floor((Date.now() - run.since) / 1000)} s
        </span>
      )}
      {error && <div className="error-box">{error}</div>}
    </>
  );
}

const EMPTY: Draft = {
  structure: '', variation: null, values: {}, nettingSet: '', margin: { amount: '', currency: '' },
};

/** A ticket written off what `/schema` declares for the structure picked: its forms as the
 * currency the client buys, its parameters in market terms - a blank the recipe solves offered as
 * one, and said - the client's netting set and the margin. `from` is the quote it changes. */
function QuoteForm({ draft: initial = EMPTY, from, onQuoted }: {
  draft?: Draft; from?: string; onQuoted: (id: string) => void;
}) {
  const { state } = useApp();
  const types = state.schema?.Structure?.types ?? {};
  const paper = state.paper.data ?? NO_PAPER;
  const [draft, setDraft] = useState<Draft>(initial);
  const quoting = useQuoting(onQuoted);
  const declared: StructureType | undefined = types[draft.structure];
  const edit = (change: Partial<Draft>) => setDraft((standing) => ({ ...standing, ...change }));
  const sets = clients(state.doc, paper.agreements);

  return (
    <section className="card quoteform">
      <h3>{from ? `Change quote ${short(from)} and quote again` : 'A new quote'}</h3>
      <div className="nested">
        <div className="statusrow">
          <span className="hint">structure</span>
          <select value={draft.structure}
                  onChange={(event) => edit({ structure: event.target.value, variation: null })}>
            <option value="">pick what the client asked for</option>
            {Object.entries(types).map(([name, entry]) => {
              const [said, ...also] = salesNames(entry);
              return (
                <option key={name} value={name}>
                  {said}{also.length ? ` (${also.join(', ')})` : ''}
                </option>
              );
            })}
          </select>
        </div>
        {declared && (
          <>
            {declared.variations && (
              <div className="statusrow">
                <span className="hint">the client buys</span>
                {directions(declared, draft.values.pair).map(({ word, buys }) => (
                  <button key={word} className={`ghost${draft.variation === word ? ' on' : ''}`}
                          onClick={() => edit({ variation: word })}>
                    {buys} <span className="hint">· {word}</span>
                  </button>
                ))}
              </div>
            )}
            <DescriptorPanel
              title="Parameters, in market terms" tables
              fields={ticketFields(declared, draft.variation)} values={draft.values}
              onAmend={async (key, wire) => {
                edit({ values: { ...draft.values, [key]: wire } });
                return null;
              }} />
            {solving(declared).map((said) => <div key={said} className="note">{said}</div>)}
            <div className="statusrow">
              <span className="hint">client</span>
              <select value={draft.nettingSet}
                      onChange={(event) => edit({ nettingSet: event.target.value })}>
                <option value="">no netting set - books at the root</option>
                {sets.map((set) => (
                  <option key={set} value={set}>
                    {clientText(set, paper.entities, paper.agreements)}
                  </option>
                ))}
              </select>
              <span className="hint">sales margin</span>
              <input className="amount" type="text" inputMode="decimal" placeholder="amount"
                     value={draft.margin.amount}
                     onChange={(event) => edit({ margin: { ...draft.margin,
                                                           amount: event.target.value } })} />
              <input className="ccy" type="text" placeholder="currency"
                     value={draft.margin.currency}
                     onChange={(event) => edit({ margin: { ...draft.margin,
                                                           currency: event.target.value } })} />
            </div>
          </>
        )}
        <div className="statusrow">
          <button className="primary" disabled={!declared || !!quoting.run}
                  onClick={() => declared && quoting.start(requestOf(draft, declared))}>
            {quoting.run ? 'pricing…' : from ? 'quote again' : 'quote'}
          </button>
          <Waiting {...quoting} />
        </div>
      </div>
    </section>
  );
}

/** One quote as the card a salesperson reads to the client, and what can be done with it here:
 * the sheet, the acceptance, a re-quote. */
function Ticket({ id, from, requotes, onMade }: {
  id: string; from?: string; requotes: string[]; onMade: (id: string, from?: string) => void;
}) {
  const { state, dispatch } = useApp();
  const [changing, setChanging] = useState(false);
  const [verdict, setVerdict] = useState<Verdict | null>(null);
  const quoting = useQuoting((made) => onMade(made, id));
  const read = useQuotes([id]);
  const pending = read.data?.[id];
  const pick = (other: string) => dispatch({ type: 'PICK', screen: 'quote', id: other });

  if (read.error) return <div className="error-box">{read.error}</div>;
  if (pending === undefined) return <div className="placeholder">reading quote {short(id)}…</div>;

  const { quote } = pending;
  const declared = state.schema?.Structure?.types[quote.structure];
  const again = declared && draftOf(quote, declared);
  return (
    <>
      <div className="pager">
        <span className="mono" title={id}>quote {short(id)}</span>
        {from && (
          <span>
            re-quotes <button className="ghost" onClick={() => pick(from)}>{short(from)}</button>,
            which stands as it was
          </span>
        )}
        {requotes.map((other) => (
          <span key={other}>quoted again as{' '}
            <button className="ghost" onClick={() => pick(other)}>{short(other)}</button></span>
        ))}
      </div>
      <TicketCard pending={pending} />
      <Acceptance id={id} pending={pending} verdict={verdict} quoting={quoting}
                  onVerdict={setVerdict}
                  onRequote={() => again && declared
                    && quoting.start(requestOf(again, declared))} />
      <div className="statusrow">
        <Sheet id={id} quote={quote} />
        {again && (
          <button className="ghost" onClick={() => setChanging(!changing)}>
            {changing ? 'leave it as quoted' : 'change and quote again'}
          </button>
        )}
      </div>
      {changing && again && (
        <QuoteForm draft={again} from={id} onQuoted={(made) => onMade(made, id)} />
      )}
    </>
  );
}

/** The spreadsheet a client is sent, saved by this browser - or, where the quote wrote none, the
 * reason it gave. */
function Sheet({ id, quote }: { id: string; quote: Quote }) {
  const [said, setSaid] = useState<string | null>(null);
  if (quote.files && !quote.files.sheet) {
    return <span className="hint">no quote sheet: {quote.files.sheet_note}</span>;
  }
  async function save() {
    setSaid(null);
    try {
      const link = document.createElement('a');
      link.href = URL.createObjectURL(await getQuoteSheet(id));
      link.download = `quote ${id.slice(0, 8)}.xlsx`;
      link.click();
      URL.revokeObjectURL(link.href);
    } catch (error) {
      setSaid(failure(error).error);
    }
  }
  return (
    <>
      <button className="ghost" onClick={save}>download the quote sheet</button>
      {said && <span className="refusal">{said}</span>}
    </>
  );
}

/** The acceptance, asked for twice - the confirm naming the client, the structure and the net -
 * and what it answered: where the trade landed with its row in the blotter, or the service's own
 * words and the one thing to do next. A quote booked reads as booked, and books no more. */
function Acceptance({ id, pending, verdict, quoting, onVerdict, onRequote }: {
  id: string; pending: PendingQuote; verdict: Verdict | null;
  quoting: ReturnType<typeof useQuoting>; onVerdict: (said: Verdict) => void;
  onRequote: () => void;
}) {
  const { state, dispatch } = useApp();
  const [asking, setAsking] = useState(false);
  const [booking, setBooking] = useState(false);
  const paper = state.paper.data ?? NO_PAPER;
  const recording = state.record.spine !== null;
  const hold = heldBack(state, 'fill');
  const { quote } = pending;
  const booked = pending.booked ?? verdict?.booked;

  async function accept() {
    setAsking(false);
    setBooking(true);
    try {
      onVerdict(verdictOf(await acceptQuote(id)));
    } catch (error) {
      onVerdict({ words: [failure(error).error], requote: true });
    } finally {
      setBooking(false);
    }
  }
  const blotter = (path: string) => {
    dispatch({ type: 'SELECT_DEAL', path });
    dispatch({ type: 'TAB', tab: 'blotter' });
  };

  return (
    <section className={`card acceptance${booked ? ' booked' : ''}`}>
      <div className="nested">
        {booked ? (
          <div className="statusrow">
            <span className="chip done">booked</span>
            <span>at <span className="mono">{booked.deal_path}</span>
              {booked.lsn !== null ? `, record position ${booked.lsn}` : ', in the book file'}
            </span>
            <button className="ghost" onClick={() => blotter(booked.deal_path)}>
              open its row in the blotter
            </button>
            <button className="ghost" onClick={() => openTrade(dispatch, booked.deal_path)}>
              open the trade page ›
            </button>
          </div>
        ) : pending.accepted ? (
          <div className="statusrow">
            <span className="chip error">accepted, not booked</span>
            <span>The client's price stands on the record at position {pending.accepted.lsn}; this
              desk's tiers did not book it, and quoting again does not change that.</span>
          </div>
        ) : hold ? <div className="note">{hold}</div>
        : asking ? (
          <div className="banner">
            Book the {salesName(state.schema, quote.structure)} for{' '}
            <b>{clientText(quote.netting_set, paper.entities, paper.agreements)}</b> at net{' '}
            <b>{formatNumber(quote.net, 'money')} {quote.currency}</b>?{' '}
            <button className="primary" onClick={accept}>book it</button>{' '}
            <button className="ghost" onClick={() => setAsking(false)}>not yet</button>
          </div>
        ) : (
          <div className="statusrow">
            <button className="primary" disabled={booking} onClick={() => setAsking(true)}>
              {booking ? 'booking…' : 'accept'}
            </button>
            <span className="hint">
              {recording ? `This desk keeps a record: the acceptance files the quote and its fill${
                state.seat ? ` under ${state.seat}` : ', under the seat at the top right'}.`
                : 'This desk keeps no record: the acceptance writes the trade into the book file.'}
            </span>
          </div>
        )}
        {verdict?.words.filter(Boolean).map((words) => (
          <div key={words} className={verdict.booked ? 'note' : 'error-box'}>{words}</div>
        ))}
        {verdict && !verdict.booked && verdict.requote && (
          <div className="statusrow">
            <button className="primary" disabled={!!quoting.run} onClick={onRequote}>
              re-quote from the same parameters
            </button>
            <Waiting {...quoting} />
          </div>
        )}
      </div>
    </section>
  );
}
