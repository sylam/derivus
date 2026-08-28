// One function per service endpoint, same-origin: vite proxies in dev, the service serves the
// build at /ui in production. No client class - the endpoints are the vocabulary.

import type {
  BookResponse, BookRisk, BookXva, DescribeResult, JobDoc, ResultSummary, Schema, TablePage,
  ValidateResult,
} from './types';

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
  const response = await fetch(path, {
    method,
    headers: body === undefined ? undefined : { 'content-type': 'application/json' },
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
    { action: 'amend', deal_path: dealPath, fields });
export const patchMarket = (factor: string, fields: Record<string, unknown>) =>
  call<BookDealOutcome>('POST', '/book/market', { patch: { [factor]: fields } });
// the desk's two data views. Both are GETs: the risk verb runs the book on a cache miss and
// answers from the cache afterwards, and the XVA verb never runs anything at all - a recalc is
// asked for through the MCP verbs, and this client does not have that vocabulary on purpose.
export const getBookRisk = () => call<BookRisk>('GET', '/book/risk');
export const getBookXva = () => call<BookXva>('GET', '/book/xva');
export const postDescribe = (doc: JobDoc) => call<DescribeResult>('POST', '/describe', doc);
export const postValidate = (doc: JobDoc) => call<ValidateResult>('POST', '/validate', doc);
export const postExecute = (doc: JobDoc) =>
  call<{ result_id: string; status: string }>('POST', '/execute', doc);
export const getResult = (id: string) => call<ResultSummary>('GET', `/results/${id}`);
export const getTable = (id: string, table: string, offset = 0, limit?: number) =>
  call<TablePage>('GET', `/results/${id}/${table}?offset=${offset}${
    limit === undefined ? '' : `&limit=${limit}`}`);
