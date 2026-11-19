import { useEffect, useRef, useState } from 'react';
import { acceptQuote, failure, getQuote, getQuoteSheet, getResult, quoteStructure } from '../api';
import { cellText, dateOf } from '../blotter';
import { EmptyState } from '../components/EmptyState';
import { DealPanels, DescriptorPanel, FieldView } from '../components/FieldView';
import { Navigator } from '../components/Navigator';
import { stampText } from '../desk';
import { NO_PAPER } from '../positions';
import {
  clientText, clients, directions, draftOf, firmSeconds, firmText, firmUntil, keep, onMarketAxis,
  requestOf, salesNames, solvedNumbers, solving, spreadText, ticketFields, ticketLegs, verdictOf,
  type Draft, type Kept, type TicketLeg, type Verdict,
} from '../quote';
import { useApp } from '../state';
import { formatNumber } from '../tokens';
import type { PendingQuote, Quote, Schema, StructureType } from '../types';

const POLL_MS = 500;

/** The page a new ticket is written on. */
const NEW = 'new';

/** Where this browser keeps the quotes it opened or made - a convenience, so a storage that
 * refuses (a private window) costs the list and nothing else. */
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

const salesName = (schema: Schema | null, structure: string) => {
  const declared = schema?.Structure?.types[structure];
  return declared ? salesNames(declared)[0] : structure;
};

/** The sales desk's ticket: a structure quoted off what `/schema` declares for it, the answer read
 * as one card a person reads to a client, and the acceptance that books it - beside every quote
 * this browser opened or made, a quote the chat gave opening by its id as the same card. Every
 * level reads on the market's axis by the blotter's rule. A quote is an act: changing one quotes
 * again, and the old one stands beside the new. */
export function QuoteView() {
  const { state, dispatch } = useApp();
  const { doc, schema } = state;
  const [list, setList] = useKept();
  const [held, setHeld] = useState<Record<string, PendingQuote | string>>({});
  const etag = state.source?.kind === 'book' ? state.source.etag : null;

  // every kept quote read again where the book moves: an acceptance anywhere writes onto its file
  useEffect(() => {
    let live = true;
    Promise.all(list.map(({ id }) => getQuote(id).then((pending) => [id, pending] as const,
      (error) => [id, failure(error).error] as const)))
      .then((read) => { if (live) setHeld(Object.fromEntries(read)); });
    return () => { live = false; };
  }, [list, etag]);

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

  const made = (id: string, from?: string) => {
    setList((standing) => keep(standing, id, from));
    dispatch({ type: 'PICK', screen: 'quote', id });
  };
  const read = (id: string, pending: PendingQuote) =>
    setHeld((standing) => ({ ...standing, [id]: pending }));

  return (
    <Navigator
      screen="quote" side={<OpenById onOpen={(id) => made(id)} />}
      pages={[
        { id: NEW, label: 'new quote', accent: true,
          content: <QuoteForm onQuoted={(id) => made(id)} /> },
        ...list.map(({ id, from }) => {
          const pending = held[id];
          const quote = typeof pending === 'object' ? pending.quote : null;
          return {
            id, label: quote ? `${salesName(schema, quote.structure)} · ${
              quote.netting_set ?? 'root'}` : short(id),
            hint: typeof pending === 'string' ? 'unread' : pending?.booked ? 'booked' : short(id),
            content: (
              <Ticket id={id} pending={pending} from={from}
                      requotes={list.filter((entry) => entry.from === id).map((entry) => entry.id)}
                      onMade={made} onRead={(fresh) => read(id, fresh)} />
            ),
          };
        }),
      ]}
    />
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
function Ticket({ id, pending, from, requotes, onMade, onRead }: {
  id: string; pending?: PendingQuote | string; from?: string; requotes: string[];
  onMade: (id: string, from?: string) => void; onRead: (pending: PendingQuote) => void;
}) {
  const { state, dispatch } = useApp();
  const [changing, setChanging] = useState(false);
  const [verdict, setVerdict] = useState<Verdict | null>(null);
  const quoting = useQuoting((made) => onMade(made, id));
  const pick = (other: string) => dispatch({ type: 'PICK', screen: 'quote', id: other });

  if (pending === undefined) return <div className="placeholder">reading quote {short(id)}…</div>;
  if (typeof pending === 'string') return <div className="error-box">{pending}</div>;

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
                  onVerdict={(said) => {
                    setVerdict(said);
                    getQuote(id).then(onRead, () => undefined);
                  }}
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

/** The card: who it is for and what it is, its legs on the market's axis, the solved number set
 * apart, the net, the market it was struck on, the spread, the notes, who quoted it and when, and
 * how long the desk holds it firm. */
function TicketCard({ pending }: { pending: PendingQuote }) {
  const { state } = useApp();
  const { schema } = state;
  const paper = state.paper.data ?? NO_PAPER;
  const { quote } = pending;
  const legs = ticketLegs(quote, pending.deal, schema);
  const first = legs[0]?.row.deal ?? {};
  const surface = first.FX_Volatility;
  const spot = onMarketAxis(quote.spot.value_market, quote.params.pair, surface);
  const forward = onMarketAxis(quote.forward, quote.params.pair, surface);
  const window = firmSeconds(state.doc, schema);
  const until = window === null ? undefined : firmUntil(pending.quoted_at, window);
  const lapsed = until !== undefined && !pending.booked && Date.parse(until) < Date.now();
  // painted again the moment the window closes, which nothing else on the desk would move
  const [, lapse] = useState(0);
  useEffect(() => {
    if (until === undefined || lapsed || pending.booked) return;
    const timer = setTimeout(() => lapse((count) => count + 1),
                             Date.parse(until) - Date.now() + 50);
    return () => clearTimeout(timer);
  }, [until, lapsed, pending.booked]);
  const delivery = dateOf(first.Delivery_Date);
  const level = (on: { pair: string; level: number } | undefined, raw: number) =>
    (on ? `${on.pair} ${formatNumber(on.level)}` : formatNumber(raw));

  return (
    <section className="card ticket">
      <h3 title={quote.structure}>
        {salesName(schema, quote.structure)}{quote.variation && ` · ${quote.variation}`}
      </h3>
      <div className="nested">
        <div className="ticketline">
          <b>{clientText(quote.netting_set, paper.entities, paper.agreements)}</b>
          {quote.client && (
            <span>the client buys {quote.client.buys}, sells {quote.client.sells}</span>
          )}
          <span>{String(quote.params.pair)} · {String(quote.params.expiry)}
            {legs[0]?.row.expiry && ` · expires ${legs[0].row.expiry}`}
            {delivery && ` · delivers ${delivery}`}</span>
        </div>
        <div className="stats">
          <div className="stat">
            <div className="label">net, in the client's sign</div>
            <div className="value">
              {formatNumber(quote.net, 'money')} <span className="unit">{quote.currency}</span>
            </div>
          </div>
          {solvedNumbers(legs).map((solved) => (
            <div key={solved.what} className="stat solved">
              <div className="label">solved · {solved.what}</div>
              <div className="value">
                {solved.on ? `${solved.on} ` : ''}{formatNumber(solved.value)}
              </div>
            </div>
          ))}
          <div className="stat">
            <div className="label">
              spot · {quote.spot.source === 'terminal' ? 'live' : "book's"}
            </div>
            <div className="value">{level(spot, quote.spot.value_market)}</div>
          </div>
          <div className="stat">
            <div className="label">outright forward{delivery ? ` to ${delivery}` : ''}</div>
            <div className="value">{level(forward, quote.forward)}</div>
          </div>
          {until && !pending.booked && (
            <div className={`stat${lapsed ? ' warn' : ''}`}>
              <div className="label">{lapsed ? 'past its firm window' : 'firm until'}</div>
              <div className="value">{stampText(until)}</div>
            </div>
          )}
        </div>
        <TicketLegs legs={legs} currency={quote.currency} />
        <div className="note">{spreadText(quote)}</div>
        {quote.margin && (
          <div className="note">
            Sales margin {formatNumber(quote.margin.amount, 'money')} {quote.margin.currency},{' '}
            {formatNumber(quote.margin.value, 'money')} {quote.margin.pricing_currency} at the
            book's spot, charged where the spread is.
          </div>
        )}
        {[quote.spot.note, ...(quote.notes ?? []),
          ...quote.legs.map((leg) => leg.note && `${leg.role}: ${leg.note}`)]
          .filter((note): note is string => !!note)
          .map((note) => <div key={note} className="hint ticketnote">{note}</div>)}
        <div className="ticketline">
          <span>quoted by <b>{pending.quoted_by ?? 'no seat named'}</b> at{' '}
            {stampText(pending.quoted_at)}</span>
          <span className="hint">{firmText(window, state.record.spine !== null)}</span>
        </div>
      </div>
    </section>
  );
}

/** The legs, each with the client's side, its size, its levels on the market's axis - the one the
 * recipe solved marked - its terms and premium; a leg opens onto its whole deal. */
function TicketLegs({ legs, currency }: { legs: TicketLeg[]; currency: string }) {
  const { state } = useApp();
  const [opened, setOpened] = useState<string | null>(null);
  const level = (leg: TicketLeg, which: 'strike' | 'barrier') => {
    const picked = leg.read[which];
    return (
      <td className={`n${leg.solved === which ? ' solved' : ''}`} title={picked?.title}>
        {picked ? cellText(picked.value) : ''}
      </td>
    );
  };
  return (
    <div className="tablewrap">
      <table className="data blotter ticketlegs">
        <thead>
          <tr>
            <th>Leg</th><th>Client</th><th>On</th><th className="n">Amount</th><th>Ccy</th>
            <th className="n">Strike</th><th className="n">Barrier</th><th>Terms</th>
            <th className="n">Premium {currency}</th><th />
          </tr>
        </thead>
        <tbody>
          {legs.map((leg) => (
            <TicketLegRow key={leg.leg.reference} leg={leg} level={level}
                          opened={opened === leg.leg.reference} schema={state.schema}
                          onOpen={() => setOpened(opened === leg.leg.reference ? null
                            : leg.leg.reference)} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function TicketLegRow({ leg, level, opened, schema, onOpen }: {
  leg: TicketLeg; level: (leg: TicketLeg, which: 'strike' | 'barrier') => JSX.Element;
  opened: boolean; schema: Schema | null; onOpen: () => void;
}) {
  const side = String(leg.size.side?.value ?? '');
  return (
    <>
      <tr onClick={onOpen}>
        <td className="ref">{leg.leg.role}</td>
        <td>
          {side && <span className={`side ${side === 'Sell' ? 'sell' : 'buy'}`}>{side}</span>}
        </td>
        <td title={leg.read.on?.title}>
          {leg.read.on ? cellText(leg.read.on.value) : ''}{leg.read.sense && ` ${leg.read.sense}`}
        </td>
        <td className="n">{typeof leg.size.amount?.value === 'number'
          ? formatNumber(leg.size.amount.value, 'money') : ''}</td>
        <td>{leg.size.of ? cellText(leg.size.of.value) : ''}</td>
        {level(leg, 'strike')}
        {level(leg, 'barrier')}
        <td className="terms" title={leg.read.terms?.title}>{leg.read.terms?.parts.join(' · ')}</td>
        <td className="n">{formatNumber(leg.leg.premium, 'money')}</td>
        <td className="opener"><span>{opened ? '×' : '⋯'}</span></td>
      </tr>
      {leg.schedule && (
        <tr className="schedule">
          <td />
          <td colSpan={9}>
            <FieldView value={leg.schedule.value} descriptor={leg.schedule.declared} />
          </td>
        </tr>
      )}
      {opened && (
        <tr className="detail">
          <td colSpan={10}><DealPanels deal={leg.row.deal} schema={schema} /></td>
        </tr>
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
          </div>
        ) : pending.accepted ? (
          <div className="statusrow">
            <span className="chip error">accepted, not booked</span>
            <span>The client's price stands on the record at position {pending.accepted.lsn}; this
              desk's tiers did not book it, and quoting again does not change that.</span>
          </div>
        ) : asking ? (
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
