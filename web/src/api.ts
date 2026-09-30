// One function per service endpoint, same-origin: vite proxies in dev, the service serves the
// build at /ui in production. No client class - the endpoints are the vocabulary.

import type { MarkedDays, Pnl } from './pnl';
import type { Agreement, Entity, Position } from './positions';
import type {
  ActivityPage, BookMarkets, BookResponse, BookRisk, BookStatus, BookXva, CalculationOutcome,
  CalculationsAnswer, CurveOutcome, CurvesAnswer, DescribeResult, JobDoc, Reconcile,
  ResultSummary, Schema, SecuritiesAnswer, SeedOutcome, TablePage, ValidateResult, Worklist,
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

/** A thrown error as the pair a view renders: the status, and the service's OWN wording. A book
 * verb refusing carries its cause in `detail` (`the book will not price a consolidated risk run:
 * ...`), and printing the JSON envelope at a reader would put a second author between the engine
 * and the desk. Anything that is not an `ApiError` - the network being down - reads as itself. */
export function failure(error: unknown): { status: number | null; error: string } {
  if (!(error instanceof ApiError)) return { status: null, error: String(error) };
  try {
    const detail = (JSON.parse(error.message) as { detail?: unknown }).detail;
    if (typeof detail === 'string') return { status: error.status, error: detail };
  } catch { /* not JSON: the body is already the message */ }
  return { status: error.status, error: error.message || `HTTP ${error.status}` };
}

async function call<T>(method: string, path: string, body?: unknown): Promise<T> {
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
  return response.json() as Promise<T>;
}

export type BookDealOutcome = {
  written: boolean;
  deal_path?: string;
  refused?: string[];
  etag?: string;
};

export const getSchema = () => call<Schema>('GET', '/schema');
export const getBook = () => call<BookResponse>('GET', '/book');
export const amendDeal = (dealPath: string, fields: Record<string, unknown>) =>
  call<BookDealOutcome>('POST', '/book/deals',
    signed({ action: 'amend', deal_path: dealPath, fields }));
// the market tick, one endpoint and two vocabularies: a price factor's values, or whole quote
// blocks - which the verb value-updates and bootstraps in the same atomic write
const postMarket = (body: Record<string, unknown>) =>
  call<BookDealOutcome>('POST', '/book/market', body);
export const patchMarket = (factor: string, fields: Record<string, unknown>) =>
  postMarket({ patch: { [factor]: fields } });
export const tickMarket = (quotes: Record<string, unknown>) => postMarket({ quotes });
// the desk's own scope, fetched and installed as a queued job - polled like an execute
export const tickBloomberg = () =>
  call<{ result_id: string; status: string }>('POST', '/book/bloomberg', {});
// the bootstrap's own dials: merged into one declared entry, then the market re-bootstrapped
export const configureBook = (section: string, entry: string, fields: Record<string, unknown>) =>
  call<BookDealOutcome>('POST', '/book/configure', { section, entry, fields });
// the curves: the book's blocks read back as the definitions they are beside the ones this
// workstation's seed could set up, and one request that authors a curve from its benchmark rows,
// solves it and bootstraps - or refuses in its own words with the file untouched
export const getCurves = () => call<CurvesAnswer>('GET', '/book/curve');
export const configureCurve = (request: Record<string, unknown>) =>
  call<CurveOutcome>('POST', '/book/curve', request);
// the ticker vocabulary: the candidates this desk could quote, what a terminal verified about
// them and every knot's own print in one read; one entry of the desk's own seed merged; and the
// terminal round trip that rewrites the map, queued and polled like an execute
export const getSecurities = () => call<SecuritiesAnswer>('GET', '/book/securities');
export const configureSecurities = (request: Record<string, unknown>) =>
  call<SeedOutcome>('POST', '/book/securities', request);
export const verifySecurities = (scope: Record<string, unknown>) =>
  call<{ result_id: string; status: string }>('POST', '/book/securities/verify', scope);
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
// the `lsn` a page answers is the cursor the next one is asked with
export const getBookActivity = (since: number | '') =>
  call<ActivityPage>('GET', `/book/activity?since=${since}`);
// the doorbell is the one endpoint that is a STREAM, so it is a path an `EventSource` opens
// rather than a call: a beat per head move, carrying a position and nothing else
export const DOORBELL = '/spine/doorbell';
export const getBookMarkets = () => call<BookMarkets>('GET', '/book/markets');
export const getBookReconcile = () => call<Reconcile>('GET', '/book/reconcile');
// what waits on somebody, asked where the head or the file moved and never on the beat
export const getWorklist = () => call<Worklist>('GET', '/book/worklist');
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
  call<{ agreements: Agreement[] }>('GET', '/book/agreements');
// the desk's own named calculations: kept on this workstation beside the book, saved one at a time
// (a null removes one), and run over the live book - whole or one subtree - in the curiosity lane
export const getCalculations = () => call<CalculationsAnswer>('GET', '/calculations');
export const saveCalculation = (name: string, calculation: Record<string, unknown> | null) =>
  call<CalculationOutcome>('POST', '/calculations', { name, calculation });
export const runCalculation = (name: string, dealPath?: string) =>
  call<{ result_id: string; status: string }>('POST', '/calculations/run',
    signed(dealPath === undefined ? { name } : { name, deal_path: dealPath }));
export const postDescribe = (doc: JobDoc) => call<DescribeResult>('POST', '/describe', doc);
export const postValidate = (doc: JobDoc) => call<ValidateResult>('POST', '/validate', doc);
export const postExecute = (doc: JobDoc) =>
  call<{ result_id: string; status: string }>('POST', '/execute', signed(doc));
export const getResult = (id: string) => call<ResultSummary>('GET', `/results/${id}`);
export const getTable = (id: string, table: string, offset = 0, limit?: number) =>
  call<TablePage>('GET', `/results/${id}/${table}?offset=${offset}${
    limit === undefined ? '' : `&limit=${limit}`}`);
