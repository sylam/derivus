// One store, one reducer. Viewing is a pure function of (schema, doc); the service is touched on
// user action (describe/validate/execute) and by the book poll, which only re-fetches when the
// etag moved - a booking made by any other client appears within a tick.

import { createContext, useContext, useEffect, useRef, useState, type Dispatch } from 'react';
import {
  SEAT, failure, getBook, getBookStatus, getQuote, getQuotes, getResult, stored,
  type BookDealOutcome,
} from './api';
import { bookClock } from './desk';
import { NO_PAPER, type Grouping, type Paper } from './positions';
import { ranAs } from './quote';
import { pinnedAt, readRecord, reconciledRecord, type RecordHeld } from './spine';
import { isObject } from './tokens';
import { tradeAt } from './trade';
import type {
  ActivityPage, BookMarket, BookResponse, BookRisk, BookXva, DealNode, DescribeResult, JobDoc,
  Market, PathsAnswer, Reconcile, ResultSummary, Schema, SeatGrants, SpineBlock, TablePage,
  ValidateResult, Worklist,
} from './types';

/** `opened` names the job file the served book was opened from, where it was opened from one. */
export type Source =
  | { kind: 'book'; etag: string; path: string; opened?: string }
  | { kind: 'local'; name: string };

export type RunState = {
  status: 'idle' | 'submitting' | 'queued' | 'running' | 'done' | 'error';
  resultId: string | null;
  /** The page that asked for the run - the book's own calculation, or a named one. */
  origin: string | null;
  summary: ResultSummary | null;
  error: string | null;
  table: string | null;
  page: TablePage | null;
  scalars: Record<string, unknown>;
  startedAt: number | null;
  /** The book's clock (`clockOf`) when the run was asked for, and the document it ran over - its
   * calculation the one the run states - so its numbers say what they are on and whether the book
   * has moved since, as the risk's do. */
  clock: string | null;
  doc: JobDoc | null;
};

/** One fetched desk view: the answer, the failure that stands in its place, and whether a fetch
 * is in flight. A failure keeps its STATUS, because the three a book verb gives - 404 no book,
 * 422 a named cause, anything else - are three different screens. */
export type Fetched<T> = {
  data: T | null;
  error: string | null;
  status: number | null;
  loading: boolean;
};

/** A desk view read off the live book, plus the CLOCK it was fetched at (`desk.bookClock`): the
 * book's etag and where the record stood. That is what makes "the numbers on screen are behind
 * the book" answerable - the risk's own `etag` is a hash of a different thing, and comparing the
 * two would always read stale. */
export type Clocked<T> = Fetched<T> & { clock: string | null };
export type RiskState = Clocked<BookRisk>;

/** What the RECORD answers this client, off the one poll the app already runs. The slice and both
 * its transitions live in `web/src/spine.ts`, where they are arithmetic a node check drives:
 * `spine` null is a desk that records nothing - not an empty record - and every reader of this
 * renders nothing at all under it. */
export type RecordState = RecordHeld;

export type AppState = {
  schema: Schema | null;
  schemaError: string | null;
  doc: JobDoc | null;
  source: Source | null;
  docError: string | null;
  tab: string;
  /** The tab the reader was on before this one - where a page opened from it goes back to. */
  back: string | null;
  /** The seat this browser acts as where no token names one - what the record's reads wait on. */
  seat: string;
  /** The deal the portfolio shows, and the page each navigated screen shows, by screen. */
  selection: { deal: string | null; picks: Record<string, string> };
  describe: DescribeResult | null;
  validate: ValidateResult | null;
  run: RunState;
  risk: RiskState;
  xva: Clocked<BookXva>;
  /** The market the live book prices on, read again whenever the book moves. */
  market: Fetched<BookMarket>;
  /** Where this workstation reads the files the served book names, read with the market. */
  paths: Fetched<PathsAnswer>;
  record: RecordState;
  /** The positions standing and the paper they sit under, for the screens that group by them. */
  paper: Fetched<Paper>;
  /** How the Portfolio screen and the blotter group the book - one choice for both. */
  grouping: Grouping;
  /** What `GET /book/seat` answered asked as the seat `asked`, or its refusal (`heldBack`). */
  grants: { asked: string; read: SeatGrants | null; refused: string | null } | null;
};

export const IDLE_RUN: RunState = {
  status: 'idle', resultId: null, origin: null, summary: null, error: null,
  table: null, page: null, scalars: {}, startedAt: null, clock: null, doc: null,
};

const NOTHING_FETCHED = { data: null, error: null, status: null, loading: false };

export const IDLE_RISK: RiskState = { ...NOTHING_FETCHED, clock: null };
export const IDLE_XVA: Clocked<BookXva> = { ...NOTHING_FETCHED, clock: null };
export const IDLE_MARKET: Fetched<BookMarket> = NOTHING_FETCHED;
export const IDLE_PATHS: Fetched<PathsAnswer> = NOTHING_FETCHED;
export const IDLE_PAPER: Fetched<Paper> = NOTHING_FETCHED;
export const NO_RECORD: RecordState = {
  spine: null, rows: [], reconcile: null, reconciledAt: null, worklist: null, worklistAt: null,
  worklistError: null,
};

export const INITIAL: AppState = {
  schema: null, schemaError: null, doc: null, source: null, docError: null,
  tab: 'portfolio', back: null, seat: stored(SEAT), selection: { deal: null, picks: {} },
  describe: null, validate: null, run: IDLE_RUN, risk: IDLE_RISK, xva: IDLE_XVA,
  market: IDLE_MARKET, paths: IDLE_PATHS, record: NO_RECORD, paper: IDLE_PAPER, grouping: 'book',
  grants: null,
};

export type Action =
  | { type: 'SCHEMA_LOADED'; schema: Schema }
  | { type: 'SCHEMA_FAILED'; error: string }
  // `refresh` is a book poll: the document moved underneath the user, so keep their place
  | { type: 'DOC_LOADED'; doc: JobDoc; source: Source; refresh?: boolean }
  | { type: 'DOC_FAILED'; error: string }
  | { type: 'TAB'; tab: string }
  | { type: 'SEAT'; seat: string }
  | { type: 'SELECT_DEAL'; path: string | null }
  | { type: 'PICK'; screen: string; id: string }
  | { type: 'DESCRIBED'; describe: DescribeResult }
  | { type: 'VALIDATED'; validate: ValidateResult }
  | { type: 'RUN_SUBMITTED'; resultId: string; origin: string; clock: string | null; doc: JobDoc }
  | { type: 'RUN_POLLED'; summary: ResultSummary }
  | { type: 'RUN_FAILED'; error: string }
  | { type: 'TABLE_SELECTED'; table: string }
  | { type: 'PAGE_LOADED'; page: TablePage }
  | { type: 'SCALARS_LOADED'; scalars: Record<string, unknown> }
  // the desk views: one fetch each, off the etag the book poll already moves
  | { type: 'RISK_FETCHING' }
  | { type: 'RISK_LOADED'; risk: BookRisk; clock: string }
  | { type: 'RISK_FAILED'; error: string; status: number | null }
  | { type: 'XVA_FETCHING' }
  | { type: 'XVA_LOADED'; xva: BookXva; clock: string }
  | { type: 'XVA_FAILED'; error: string; status: number | null }
  | { type: 'MARKET_LOADED'; market: BookMarket }
  | { type: 'MARKET_FAILED'; error: string; status: number | null }
  | { type: 'PATHS_LOADED'; paths: PathsAnswer }
  | { type: 'PATHS_FAILED'; error: string; status: number | null }
  | { type: 'PAPER_FETCHING' }
  // the parts of the paper read (`paperOwed`), over the rest held
  | { type: 'PAPER_LOADED'; paper: Partial<Paper> }
  | { type: 'PAPER_FAILED'; error: string; status: number | null }
  | { type: 'GROUP'; grouping: Grouping }
  // one beat, the book and the record in ONE step - the document where its stamp moved, where
  // the file stands at `etag` and the strip's next page - so a booking moves the clock once; then
  // the reconcile answer, under the head and file it was read at (`reconcileAt`)
  | { type: 'BEAT'; etag: string; live: BookResponse | null; spine: SpineBlock | null;
      page: ActivityPage | null }
  | { type: 'RECONCILED'; reconcile: Reconcile; at: string | null }
  // the one worklist every screen counts from, read at a head and seat (`worklistAt`)
  | { type: 'WORKLISTED'; at: string | null; worklist: Worklist | null; error: string | null }
  | { type: 'GRANTS'; grants: AppState['grants'] };

export function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'SCHEMA_LOADED':
      return { ...state, schema: action.schema, schemaError: null };
    case 'SCHEMA_FAILED':
      return { ...state, schemaError: action.error };
    // a REFRESH keeps the desk views standing: the numbers on screen were true of the book they
    // were computed over, and the honest thing is to keep showing them while they are refetched,
    // marked as behind. A different DOCUMENT is a different book, and they go.
    case 'DOC_LOADED':
      return action.refresh
        ? { ...state, doc: action.doc, source: action.source, docError: null, validate: null }
        : {
            ...state, doc: action.doc, source: action.source, docError: null,
            selection: { deal: null, picks: {} }, describe: null, validate: null, run: IDLE_RUN,
            risk: IDLE_RISK, xva: IDLE_XVA, market: IDLE_MARKET, paths: IDLE_PATHS,
            paper: IDLE_PAPER,
          };
    case 'DOC_FAILED':
      return { ...state, docError: action.error };
    case 'TAB':
      return action.tab === state.tab ? state : { ...state, tab: action.tab, back: state.tab };
    case 'SEAT':
      return { ...state, seat: action.seat };
    case 'SELECT_DEAL':
      return { ...state, selection: { ...state.selection, deal: action.path } };
    case 'PICK':
      return { ...state, selection: {
        ...state.selection, picks: { ...state.selection.picks, [action.screen]: action.id } } };
    case 'DESCRIBED':
      return { ...state, describe: action.describe };
    case 'VALIDATED':
      return { ...state, validate: action.validate };
    case 'RUN_SUBMITTED':
      return {
        ...state,
        run: { ...IDLE_RUN, status: 'queued', resultId: action.resultId, origin: action.origin,
               startedAt: Date.now(), clock: action.clock, doc: action.doc },
      };
    case 'RUN_POLLED': {
      const status = action.summary.status as RunState['status'];
      return {
        ...state,
        run: {
          ...state.run, summary: action.summary,
          status: status === 'done' || status === 'error' ? status : 'running',
          error: action.summary.error ?? null,
        },
      };
    }
    case 'RUN_FAILED':
      return { ...state, run: { ...state.run, status: 'error', error: action.error } };
    case 'TABLE_SELECTED':
      return { ...state, run: { ...state.run, table: action.table, page: null } };
    case 'PAGE_LOADED':
      return { ...state, run: { ...state.run, page: action.page } };
    case 'SCALARS_LOADED':
      return { ...state, run: { ...state.run, scalars: action.scalars } };
    case 'RISK_FETCHING':
      return { ...state, risk: { ...state.risk, loading: true } };
    case 'RISK_LOADED':
      return {
        ...state,
        risk: { data: action.risk, clock: action.clock, error: null, status: null, loading: false },
      };
    // a refusal REPLACES the numbers: a 422 says the book will not price, and a stale mark left
    // on screen under a red box is a number a desk might still read
    case 'RISK_FAILED':
      return {
        ...state,
        risk: { data: null, clock: null, error: action.error, status: action.status, loading: false },
      };
    case 'XVA_FETCHING':
      return { ...state, xva: { ...state.xva, loading: true } };
    case 'XVA_LOADED':
      return {
        ...state,
        xva: { data: action.xva, clock: action.clock, error: null, status: null, loading: false },
      };
    case 'XVA_FAILED':
      return {
        ...state,
        xva: { data: null, clock: null, error: action.error, status: action.status, loading: false },
      };
    // no market is the service saying the one held is still current: its etag and stamp move
    case 'MARKET_LOADED': {
      const market = action.market.market ?? state.market.data?.market ?? null;
      return {
        ...state,
        market: { data: { ...action.market, market }, error: null, status: null, loading: false },
      };
    }
    case 'MARKET_FAILED':
      return {
        ...state,
        market: { data: null, error: action.error, status: action.status, loading: false },
      };
    case 'PATHS_LOADED':
      return { ...state, paths: { data: action.paths, error: null, status: null, loading: false } };
    case 'PATHS_FAILED':
      return {
        ...state,
        paths: { data: null, error: action.error, status: action.status, loading: false },
      };
    case 'PAPER_FETCHING':
      return { ...state, paper: { ...state.paper, loading: true } };
    case 'PAPER_LOADED':
      return { ...state, paper: { data: { ...(state.paper.data ?? NO_PAPER), ...action.paper },
                                  error: null, status: null, loading: false } };
    case 'PAPER_FAILED':
      return {
        ...state,
        paper: { data: null, error: action.error, status: action.status, loading: false },
      };
    case 'GROUP':
      return { ...state, grouping: action.grouping };
    // both record transitions are `web/src/spine.ts`'s, and both answer the record they were
    // handed where nothing moved - the beat is two seconds and most beats of it bring nothing.
    // A book already held loads nothing - a write's read-back and a beat can both carry it - and
    // a pin read against a book other than the one now held is behind it, and waits a beat.
    case 'BEAT': {
      const { live, etag, page } = action;
      const held = state.source?.kind === 'book' ? state.source : null;
      const same = held?.path === live?.path;
      const moved = live && !(same && held?.etag === live.etag)
        ? reducer(state, bookLoaded(live, same)) : state;
      const spine = moved.source?.kind === 'book' && moved.source.etag !== etag
        ? moved.record.spine : action.spine;
      const record = readRecord(moved.record, spine, page);
      return record === moved.record ? moved : { ...moved, record };
    }
    case 'RECONCILED': {
      const record = reconciledRecord(state.record, action.reconcile, action.at);
      return record === state.record ? state : { ...state, record };
    }
    case 'WORKLISTED':
      return { ...state, record: { ...state.record, worklist: action.worklist,
                                   worklistAt: action.at, worklistError: action.error } };
    case 'GRANTS':
      return { ...state, grants: action.grants };
  }
}

/** Open what a row of the record names: a trade of `doc` - or a leg of one - on its page, any
 * other node or a position on the Portfolio, else an agreement or an entity on the Paper. */
export function opened(dispatch: Dispatch<Action>, doc: JobDoc | null,
                       deal: string | null | undefined, paper?: string | null) {
  const trade = deal ? tradeAt(doc?.Calc.Deals?.Deals?.Children ?? [], deal) : undefined;
  if (trade) openTrade(dispatch, trade.path);
  else if (deal) {
    dispatch({ type: 'SELECT_DEAL', path: deal });
    dispatch({ type: 'TAB', tab: 'portfolio' });
  } else if (paper) {
    dispatch({ type: 'PICK', screen: 'paper', id: paper });
    dispatch({ type: 'TAB', tab: 'paper' });
  }
}

/** Open the page of the trade a deal path names - a leg's opening its structure's. */
export function openTrade(dispatch: Dispatch<Action>, path: string) {
  dispatch({ type: 'PICK', screen: 'trade', id: path });
  dispatch({ type: 'TAB', tab: 'trade' });
}

/** A `GET /book` answer as the document it loads - `refresh` where it is the one on screen moved,
 * which keeps the user's place. */
export function bookLoaded(live: BookResponse, refresh = false): Action {
  return {
    type: 'DOC_LOADED', doc: live.document, refresh,
    source: { kind: 'book', etag: live.etag, path: live.path, opened: live.opened },
  };
}

/** One book WRITE as the answer an editing panel reads: the service's refusal verbatim, or null
 * once the book and where the record stands are read back in one step. There is no client-side
 * edit state anywhere - the file is the truth, so a write is answered by reading it back. */
export async function written(dispatch: Dispatch<Action>, outcome: BookDealOutcome) {
  if (!outcome.written) return outcome.refused ?? ['refused'];
  const [live, status] = await Promise.all([getBook(), getBookStatus()]);
  dispatch({ type: 'BEAT', etag: status.etag, live, spine: status.spine, page: null });
  return null;
}

/** The screens showing the record's paper - the entities legal declares and their agreements -
 * and, of those, the ones showing the holdings under it: the positions and the portfolio nodes. */
export const PAPER_TABS = ['portfolio', 'blotter', 'quote', 'trade', 'risk', 'xva', 'curves',
                           'backoffice', 'client', 'paper', 'pnl'];
const HOLDING_TABS = ['portfolio', 'blotter', 'quote', 'trade', 'risk', 'xva', 'backoffice',
                      'client'];

/** Which parts of the paper `tab` owes a read: all of it where none is held, else each part it
 * shows whose answer in hand (`answered`) was read elsewhere than where that part now stands
 * (`at`, `paperAt`). */
export function paperOwed(tab: string, held: Paper | null, answered: Record<string, string>,
                          at: { paper: string; holdings: string }):
  { paper: boolean; holdings: boolean } {
  const shown = PAPER_TABS.includes(tab);
  const holdings = HOLDING_TABS.includes(tab) && answered.holdings !== at.holdings;
  return { paper: shown && (!held || answered.paper !== at.paper),
           holdings: shown && (!held || holdings) };
}

/** The record's pin on the live book, or null where the desk keeps no record or shows a local
 * file. */
export const spineOf = (state: AppState): SpineBlock | null =>
  (state.source?.kind === 'book' ? state.record.spine : null);

/** The clock every answer read off the live book is held against (`desk.bookClock`): the book's
 * etag and where the record stands; null for a document opened from a file, which never moves. */
export function clockOf(state: AppState): string | null {
  return state.source?.kind === 'book'
    ? bookClock(state.source.etag, pinnedAt(state.record.spine)) : null;
}

/** The clock an answer read AS the seat on screen is held against - the risk, the positions: the
 * book's clock and that seat, so another seat named reads again. Null off the live book. */
export function seatClock(state: AppState): string | null {
  const clock = clockOf(state);
  return clock === null ? null : `${clock}|${state.seat}`;
}

/** Where each part of the paper stands, which `paperOwed` holds the answers in hand against: the
 * paper at the record's last declaration of it, and the holdings, of which a seat is shown its
 * own, on the seat's clock. Null off the live book. */
export function paperAt(state: AppState): { paper: string; holdings: string } | null {
  const holdings = seatClock(state);
  return holdings === null ? null : { paper: String(spineOf(state)?.paper), holdings };
}

/** The market a screen shows: for the live book what `GET /book/market` answered, the file it
 * names with its block merged over it, and null until that lands; for a document opened from a
 * file its own block, which is all a browser holds of it. */
export function marketOf(state: AppState): Market | null {
  if (state.source?.kind === 'book') return state.market.data?.market ?? null;
  return state.doc?.Calc.MergeMarketData?.ExplicitMarketData ?? {};
}

/** Whether `record` holds a key at every step of `path` - a section, an entry and any key below
 * it. Asked of the document's explicit block, it is what makes an entry the block's. */
export function holds(record: unknown, path: string[]): boolean {
  let held = record;
  for (const key of path) {
    if (!isObject(held) || !(key in held)) return false;
    held = held[key];
  }
  return true;
}

/** Where an agreement stands on the desk: the path of the set the book file holds for it - one of
 * its terms' type under its id, none before the first booking under it - and the fields that set
 * reads off the paper, as the record's reconcile names them. */
export function standing(doc: JobDoc | null, reconcile: Reconcile | null,
                         agreement: { agreement: string; terms: Record<string, unknown> }):
  { path: string | null; moved: string[] } {
  const find = (nodes: DealNode[], at: string): string | null => {
    for (const [position, node] of nodes.entries()) {
      const path = `${at}${position}`;
      const deal = node.Instrument['.Deal'];
      if (deal.Reference === agreement.agreement && deal.Object === agreement.terms.Object) {
        return path;
      }
      const below = find(node.Children ?? [], `${path}/`);
      if (below !== null) return below;
    }
    return null;
  };
  return { path: find(doc?.Calc.Deals.Deals.Children ?? [], ''),
           moved: reconcile?.terms_mismatch?.find((row) => row.agreement === agreement.agreement)
             ?.fields ?? [] };
}

/** A tree item's word for an entry the block states over the file the book names; nothing on a
 * book naming none, every entry of which is the block's. */
export function blockHint(doc: JobDoc, path: string[]): string | undefined {
  const merge = doc.Calc.MergeMarketData;
  return merge?.MarketDataFile && holds(merge.ExplicitMarketData, path) ? 'block' : undefined;
}

// ---- what the seat may do -----------------------------------------------------------------------

/** What a control does: an act the record files, as the map in force keys it - a transition's a
 * payment's `status_transition`, a confirmation's `confirmed` or a collateral movement's
 * `collateral` - or a verb the service asks itself where nothing is filed: `mark`, of a write of
 * the market or of this box's paths, and `validate`, which its queue asks of a what-if. */
export type Act = 'fill' | 'amendment' | 'quote_filed' | 'approval' | 'entity_declared'
  | 'agreement_declared' | 'official_close_declared' | 'status_transition' | 'confirmed'
  | 'collateral' | 'mark' | 'validate';

/** Why the seat may not `act` at `node` - the book by default, null for the firm's own facts, which
 * only a grant over every node reaches - in a line naming the verb the act demands under the map
 * in force (`demands`), a verb the service asks being itself; null where it may: off a desk that
 * records, until its grants are read for this seat, and where it holds that verb at `node` or at
 * a node above it - a what-if at any node under it too, the queue admitting a seat that works one
 * node of a book to price the book. */
export function heldBack(state: AppState, act: Act,
                         node: string | null = String(state.doc?.Calc.Deals.Reference ?? '')):
  string | null {
  const { grants } = state;
  if (!spineOf(state) || grants?.asked !== state.seat) {
    return null;
  }
  if (grants.refused !== null) return grants.refused;
  const { read } = grants;
  const verb = read?.demands[act] ?? act;
  const held = read?.verbs[verb];
  if (!read || !held || (node !== null && held.some((at) => node === at
    || node.startsWith(`${at}/`) || (act === 'validate' && at.startsWith(`${node}/`))))) {
    return null;
  }
  return `Needs ${verb}${held.length ? ` over ${node ?? 'the whole firm'}` : ''}: ${read.seat} `
    + `holds it ${held.length ? `only over ${held.join(', ')}` : 'nowhere'}.`;
}

/** Why the seat may not accept a quote that books at `node`, the book where it states none: an
 * acceptance files the quote and its fill, so it is held where either act is. */
export const acceptHeld = (state: AppState, node?: string): string | null =>
  heldBack(state, 'quote_filed', node) ?? heldBack(state, 'fill', node);

/** A read as its screen shows it where the seat is held back from the act it asks - `hold`, the
 * line `heldBack` says, known before asking, so nothing is asked: that line in the read's place,
 * nothing read before it shown, and `held` saying so. Else the read as it stands, a refusal of it
 * one only the service could give. */
export function readAs<T extends { data: unknown; error: string | null }>(
  hold: string | null, read: T): T & { held: boolean } {
  return hold === null ? { ...read, held: false }
    : { ...read, data: null, error: hold, held: true };
}

/** The consolidated risk as a screen shows it, its read a what-if the queue admits (`readAs`). */
export const bookRisk = (state: AppState) => readAs(heldBack(state, 'validate'), state.risk);

/** The seat this client acts as, however it was set: the one the service answers its acts are
 * filed under (`GET /book/seat`), a sign-in token's or the header box's, and the box's own where
 * that is not read for it. */
export const actingSeat = (state: AppState): string =>
  (state.grants?.asked === state.seat && state.grants.read?.seat) || state.seat;

/** What a seat shown nothing of a thing on the record is told - no position in a trade, no row of
 * its history or of the record's strip - the record showing a seat what sits under the nodes it
 * holds a grant at: `whole`, that the record holds none, where it is shown the whole book - a node
 * at or above it, or no grants read - and else that the seat is shown none, with what it sees. */
export function noneShown(state: AppState, whole = 'The record holds none of it now'): string {
  const { grants } = state;
  const book = String(state.doc?.Calc.Deals.Reference ?? '');
  const held = grants?.asked === state.seat ? Object.values(grants.read?.verbs ?? {}) : [];
  const sees = [...new Set(held.flatMap((at) => at ?? [book]))].sort();
  if (!held.length || sees.some((node) => book === node || book.startsWith(`${node}/`))) {
    return whole;
  }
  return `${grants!.read!.seat} is shown none of it, ${sees.length
    ? `seeing only what sits under ${sees.join(', ')}` : 'holding no grant at any node'}`;
}

/** The store, its dispatch, and each tab's label by its id. */
export const AppContext = createContext<{
  state: AppState; dispatch: Dispatch<Action>; labels: Record<string, string>;
} | null>(null);

export function useApp() {
  const app = useContext(AppContext);
  if (!app) throw new Error('useApp outside AppContext');
  return app;
}

/** One read of the live book while a screen is up, again whenever `keys` move - the answer held
 * on screen while the next is read - or the service's refusal in its own words. */
export function useRead<T>(read: (() => Promise<T>) | null, keys: unknown[]) {
  const [held, setHeld] = useState<{ data: T | null; error: string | null; loading: boolean }>(
    { data: null, error: null, loading: false });
  useEffect(() => {
    if (!read) return;
    let live = true;
    setHeld((standing) => ({ ...standing, loading: true }));
    read().then((data) => { if (live) setHeld({ data, error: null, loading: false }); })
      .catch((error) => {
        if (live) setHeld({ data: null, error: failure(error).error, loading: false });
      });
    return () => { live = false; };
  }, keys);
  return held;
}

const QUEUE_POLL_MS = 500;

/** A queued run read every half second while `id` names one: each summary handed to `read`, and
 * a read that fails to `failed` as the service's sentence. */
export function usePolled(id: string | null | undefined, read: (summary: ResultSummary) => void,
                          failed: (said: string) => void) {
  useEffect(() => {
    if (!id) return;
    const timer = setInterval(async () => {
      try {
        read(await getResult(id));
      } catch (thrown) {
        failed(failure(thrown).error);
      }
    }, QUEUE_POLL_MS);
    return () => clearInterval(timer);
  }, [id]);
}

/** A job asked of the service's queue: submitted, then polled (`usePolled`) until its run answers
 * under its `stats[stat]` - handed to `landed` - or with the service's own sentence, a refused
 * ask's or a failed run's (`ranAs`); `run` is the wait, with what the job says of its progress.
 * One at a time. */
export function useQueued<R, T>(
  submit: (request: R) => Promise<{ result_id: string; status: string }>, stat: string,
  landed: (answer: T) => void) {
  const [run, setRun] = useState<{ id: string; status: string; since: number;
                                   progress?: ResultSummary['progress'] } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const busy = useRef(false);
  const finish = (said: string | null) => { busy.current = false; setRun(null); setError(said); };

  usePolled(run?.id, (summary) => {
    const read = ranAs<T>(summary, stat);
    if (!read) {
      setRun((standing) => standing
        && { ...standing, status: summary.status, progress: summary.progress });
    } else if ('refused' in read) finish(read.refused);
    else {
      finish(null);
      landed(read.answer);
    }
  }, finish);

  async function start(request: R) {
    if (busy.current) return;
    busy.current = true;
    setError(null);
    setRun({ id: '', status: 'sending', since: Date.now() });
    try {
      const submitted = await submit(request);
      setRun({ id: submitted.result_id, status: submitted.status, since: Date.now() });
    } catch (thrown) {
      finish(failure(thrown).error);
    }
  }
  return { run, error, start };
}

/** The desk's quotes, newest first, read again where the book or the record moves and on `again`;
 * nothing where no service book stands behind the page. */
export function useQuoteList(again?: number) {
  const { state } = useApp();
  return useRead(state.source?.kind === 'book' ? getQuotes : null, [clockOf(state), again]);
}

/** The pending trade each quote id names, read again where the book or the record moves. */
export function useQuotes(ids: string[]) {
  const { state } = useApp();
  return useRead(ids.length ? async () => Object.fromEntries(await Promise.all(ids.map(
    async (id) => [id, await getQuote(id)] as const))) : null, [ids.join(' '), clockOf(state)]);
}
