// One function per service endpoint, same-origin: vite proxies in dev, the service serves the
// build at /ui in production. No client class - the endpoints are the vocabulary.

import type { SpreadRequest } from './desk';
import type { MarkedDays, Pnl } from './pnl';
import type { Agreement, Entity, Position } from './positions';
import type {
  Acceptance, ActivityPage, BookCollateral, BookMarket, BookMarkets, BookResponse, BookRisk,
  BookStatus, BookXva,
  CalculationOutcome, PendingQuote, QuoteGiven,
  CalculationsAnswer, CloseVerdict, CurveOutcome, CurvesAnswer, Decided, DescribeResult, JobDoc,
  PathMaps,
  PathsAnswer, Reconcile, Recorded, ResultSummary, Schema, SeatGrants, SecuritiesAnswer,
  SeedOutcome, SettlementState, TablePage, ValidateResult, Worklist,
} from './types';

/** Where this browser keeps who it acts as: the ID token a deployment's sign-in stored, else the
 * seat the settings control names - which the record files a write under, and reads by. */
export const TOKEN = 'derivus.token';
export const SEAT = 'derivus.seat';

/** One of the two, or nothing where storage is blocked or holds none. */
export function stored(key: string): string {
  try {
    return localStorage.getItem(key) ?? '';
  } catch {
    return '';
  }
}

/** The seat a request names: the settings control's, where no token proves one. */
const seat = () => (stored(TOKEN) ? '' : stored(SEAT));

/** A write's body naming the seat, and a read's path asking as it - which `call` does. */
const signed = <T extends object>(body: T) => (seat() ? { ...body, actor: seat() } : body);
const seated = (path: string) => (seat()
  ? `${path}${path.includes('?') ? '&' : '?'}actor=${encodeURIComponent(seat())}` : path);

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

/** What a desk is told where an act names no seat and the deployment wants one. */
export const NO_SEAT = 'No seat is set, and this desk files every act under the seat that asks: '
  + 'type yours in the seat box at the top right.';

/** A thrown error as the pair a view renders: the status, and the service's OWN wording. A book
 * verb refusing carries its cause in `detail` (`the book will not price a consolidated risk run:
 * ...`), and printing the JSON envelope at a reader would put a second author between the engine
 * and the desk. Anything that is not an `ApiError` - the network being down - reads as itself. The
 * one rewording is a seat named by nobody, which the desk fixes in its own box. */
export function failure(error: unknown): { status: number | null; error: string } {
  if (!(error instanceof ApiError)) return { status: null, error: String(error) };
  try {
    const detail = (JSON.parse(error.message) as { detail?: unknown }).detail;
    if (typeof detail === 'string' && detail.startsWith('no actor for this act')) {
      return { status: error.status, error: NO_SEAT };
    }
    if (typeof detail === 'string') return { status: error.status, error: detail };
  } catch { /* not JSON: the body is already the message */ }
  return { status: error.status, error: error.message || `HTTP ${error.status}` };
}

/** One request, a refusal thrown as the `ApiError` `failure` reads. */
async function send(method: string, path: string, body?: unknown): Promise<Response> {
  const token = stored(TOKEN);
  const response = await fetch(method === 'GET' ? seated(path) : path, {
    method,
    headers: {
      ...(body === undefined ? {} : { 'content-type': 'application/json' }),
      ...(token ? { authorization: `Bearer ${token}` } : {}),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new ApiError(response.status, text.slice(0, 500) || response.statusText);
  }
  return response;
}

const call = async <T>(method: string, path: string, body?: unknown): Promise<T> =>
  (await send(method, path, body)).json() as Promise<T>;

export type BookDealOutcome = {
  written: boolean;
  deal_path?: string;
  refused?: string[];
  etag?: string;
};

export const getSchema = () => call<Schema>('GET', '/schema');
export const getBook = () => call<BookResponse>('GET', '/book');
// a job file filed and served as the book, or with neither argument the desk's own served again:
// an undefined key stays off the wire, so that body is `{}`
export const openBook = (name?: string, document?: JobDoc) =>
  call<{ path: string; etag: string }>('POST', '/book/open', { name, document });
// one endpoint and a deal's vocabularies, each signed: fields merged into the deal at a path, and
// a fill of the position the node at a path is held in
const postDeal = <T>(body: Record<string, unknown>) =>
  call<T>('POST', '/book/deals', signed(body));
export const amendDeal = (dealPath: string, fields: Record<string, unknown>) =>
  postDeal<BookDealOutcome>({ action: 'amend', deal_path: dealPath, fields });
// the market the book prices on: the file it names with its explicit block merged over it, named
// by a stamp - asked with the one held, it answers no market while that one is current
export const getBookMarket = (stamp = '') =>
  call<BookMarket>('GET', `/book/market?stamp=${stamp}`);
// the market tick, one endpoint and three vocabularies: a price factor's values, whole quote
// blocks - which the verb value-updates and bootstraps in the same atomic write - or the block's
// own entries retired, each by its path from the section down, so the file's stands again. Every
// write of the market or its set-up below is signed: a recording desk takes it from `mark` alone
const postMarket = (body: Record<string, unknown>) =>
  call<BookDealOutcome>('POST', '/book/market', signed(body));
export const patchMarket = (factor: string, fields: Record<string, unknown>) =>
  postMarket({ patch: { [factor]: fields } });
export const tickMarket = (quotes: Record<string, unknown>) => postMarket({ quotes });
export const retireMarket = (paths: string[][]) => postMarket({ retire: paths });
// the desk's own scope, fetched and installed as a queued job - polled like an execute
export const tickBloomberg = () =>
  call<{ result_id: string; status: string }>('POST', '/book/bloomberg', signed({}));
// the bootstrap's own dials: merged into one declared entry, then the market re-bootstrapped
export const configureBook = (section: string, entry: string, fields: Record<string, unknown>) =>
  call<BookDealOutcome>('POST', '/book/configure', signed({ section, entry, fields }));
// the curves: the book's blocks read back as the definitions they are beside the ones this
// workstation's seed could set up, and one request that authors a curve from its benchmark rows,
// solves it and bootstraps - or refuses in its own words with the file untouched
export const getCurves = () => call<CurvesAnswer>('GET', '/book/curve');
export const configureCurve = (request: Record<string, unknown>) =>
  call<CurveOutcome>('POST', '/book/curve', signed(request));
// a counterparty's credit curve from its spread, written under the acting seat, and the years its
// last spread is run on to
export const stateSurvival = (request: SpreadRequest) =>
  call<BookDealOutcome & { factor: string; horizon: number }>('POST', '/book/survival',
                                                              signed(request));
// the ticker vocabulary: the candidates this desk could quote, what a terminal verified about
// them and every knot's own print in one read; one entry of the desk's own seed merged; and the
// terminal round trip that rewrites the map, queued and polled like an execute
export const getSecurities = () => call<SecuritiesAnswer>('GET', '/book/securities');
export const configureSecurities = (request: Record<string, unknown>) =>
  call<SeedOutcome>('POST', '/book/securities', signed(request));
export const verifySecurities = (scope: Record<string, unknown>) =>
  call<{ result_id: string; status: string }>('POST', '/book/securities/verify', signed(scope));
// the desk's two data views. Both are GETs: the risk verb runs the book on a cache miss and
// answers from the cache afterwards, and the XVA verb never runs anything at all - a recalc is
// asked for through the MCP verbs, and this client does not have that vocabulary on purpose.
export const getBookRisk = () => call<BookRisk>('GET', '/book/risk');
export const getBookXva = () => call<BookXva>('GET', '/book/xva');
// the record's four readings. `status` carries the pin and how far the record has moved since,
// which is a walk; `reconcile` says what the two disagree ABOUT and folds at the head, so it is
// asked for when the counts say there is something and never on the poll's own beat
export const getBookStatus = () => call<BookStatus>('GET', '/book/status');
// an EMPTY cursor is no `since` at all - the newest page, which is a strip's first paint - and
// the `lsn` a page answers is the cursor the next one is asked with; a deal path asks for one
// trade's rows alone, through its restrikes
export const getBookActivity = (since: number | '', dealPath?: string) =>
  call<ActivityPage>('GET', `/book/activity?since=${since}${
    dealPath === undefined ? '' : `&deal_path=${encodeURIComponent(dealPath)}`}`);
// the doorbell is the one endpoint that is a STREAM, so it is a path an `EventSource` opens
// rather than a call: a beat per head move, carrying a position and nothing else
export const DOORBELL = '/spine/doorbell';
export const getBookMarkets = () => call<BookMarkets>('GET', '/book/markets');
export const getBookReconcile = () => call<Reconcile>('GET', '/book/reconcile');
// what waits on somebody, asked where the head or the file moved and never on the beat
export const getWorklist = () => call<Worklist>('GET', '/book/worklist');
// the back office: the fact that clears a row, whether a day's close is legal, and the close
// itself - which the service declares only behind that same verdict
export const postTransition = (fact: Record<string, unknown>) =>
  call<Recorded>('POST', '/book/transition', signed(fact));
// the collateral each agreement calls on the book's own day, read on the close marked for it
export const getBookCollateral = () => call<BookCollateral>('GET', '/book/collateral');
export const getCloseCheck = (date: string) =>
  call<CloseVerdict>('GET', `/book/close/check?date=${date}`);
export const postClose = (date: string) => call<Recorded>('POST', '/book/close', signed({ date }));
// a verdict on a ticket under the acting seat: signed, or refused with the reason on the row
export const approveTicket = (ticket: string) =>
  call<Decided>('POST', '/book/approve', signed({ ticket }));
export const rejectTicket = (ticket: string, reason: string) =>
  call<Decided>('POST', '/book/reject', signed({ ticket, reason }));
// the positions standing, each where the file holds it with the status its ticket reads, the paper
// they sit under and the declared tree - the keys the grouped views file the book by
export const getBookPositions = () => call<{ positions: Position[] }>('GET', '/book/positions');
export const getBookPortfolios = () =>
  call<{ portfolios: { path: string }[] }>('GET', '/book/portfolios');
// the P&L: the days it can run between, and what the book made between two of them
export const getMarkedDays = () => call<MarkedDays>('GET', '/book/marks');
export const getPnl = (query: string) => call<Pnl>('GET', `/book/pnl?${query}`);
export const getBookEntities = () => call<{ entities: Entity[] }>('GET', '/book/entities');
export const getBookAgreements = () =>
  call<{ agreements: Agreement[]; settlement_state: SettlementState }>('GET', '/book/agreements');
// the paper, declared under the acting seat: an entity, and an agreement whose terms are its set
export const declareEntity = (entity: Record<string, unknown>) =>
  call<Recorded>('POST', '/book/entities', signed(entity));
export const declareAgreement = (agreement: Record<string, unknown>) =>
  call<Recorded>('POST', '/book/agreements', signed(agreement));
// the desk's own named calculations: kept on this workstation beside the book, saved one at a time
// (a null removes one), and run over the live book - whole or one subtree - in the curiosity lane
export const getCalculations = () => call<CalculationsAnswer>('GET', '/calculations');
export const saveCalculation = (name: string, calculation: Record<string, unknown> | null) =>
  call<CalculationOutcome>('POST', '/calculations', { name, calculation });
export const runCalculation = (name: string, dealPath?: string) =>
  call<{ result_id: string; status: string }>('POST', '/calculations/run',
    signed(dealPath === undefined ? { name } : { name, deal_path: dealPath }));
// where this workstation reads the files a job names: the two maps the service keeps on this box,
// never in a job, beside a row per file the served book names; a save replaces both maps whole
export const getPaths = () => call<PathsAnswer>('GET', '/paths');
export const savePaths = (maps: PathMaps) => call<PathsAnswer>('POST', '/paths', maps);
// a quote: a structure priced against the book under the acting seat - queued and polled like an
// execute, the quote landing under the run's `stats.Quote` - the pending trade its id names, the
// spreadsheet a client is sent, and the acceptance that books it
export const quoteStructure = (request: Record<string, unknown>) =>
  call<{ result_id: string; status: string }>('POST', '/book/structure', signed(request));
export const getQuote = (id: string) =>
  call<PendingQuote>('GET', `/book/quote/${encodeURIComponent(id)}`);
export const QUOTE_SHEET = (id: string) => `/book/quote/${encodeURIComponent(id)}/sheet`;
export const getQuoteSheet = async (id: string) => (await send('GET', QUOTE_SHEET(id))).blob();
export const acceptQuote = (id: string) =>
  call<Acceptance>('POST', '/book/quote', signed({ quote_id: id }));
// ---- the desk's quotes ----
// every quote the desk has given, newest first, with where each stands
export const getQuotes = () => call<{ quotes: QuoteGiven[] }>('GET', '/book/quotes');
// ---- a trade's own page ----
// a signed change of the position the file's node at `dealPath` is held in, at a price - an unwind
// is a fill toward nothing - answered with the booking fields an acceptance answers
export const fillDeal = (dealPath: string, fill: Record<string, unknown>) =>
  postDeal<Acceptance>({ action: 'fill', deal_path: dealPath, ...fill });
export const postDescribe = (doc: JobDoc) => call<DescribeResult>('POST', '/describe', doc);
export const postValidate = (doc: JobDoc) => call<ValidateResult>('POST', '/validate', doc);
export const postExecute = (doc: JobDoc) =>
  call<{ result_id: string; status: string }>('POST', '/execute', signed(doc));
export const getResult = (id: string) => call<ResultSummary>('GET', `/results/${id}`);
export const getTable = (id: string, table: string, offset = 0, limit?: number,
                         width?: number) =>
  call<TablePage>('GET', `/results/${id}/${table}?offset=${offset}${
    limit === undefined ? '' : `&limit=${limit}`}${
    width === undefined ? '' : `&width=${width}`}`);
// ---- what the seat asking may do, and over which nodes ----
export const getSeat = () => call<SeatGrants>('GET', '/book/seat');
// two paths a link opens rather than a call: the service answers each as the file a browser
// saves, a table whole at full precision and the served book byte for byte
export const tableCsv = (id: string, table: string) => `/results/${id}/${table}?format=csv`;
export const BOOK_DOCUMENT = '/book/document';
