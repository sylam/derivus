// One function per service endpoint, same-origin: vite proxies in dev, the service serves the
// build at /ui in production. No client class - the endpoints are the vocabulary.

import type {
  BookResponse, DescribeResult, JobDoc, ResultSummary, Schema, TablePage, ValidateResult,
} from './types';

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
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
export const postDescribe = (doc: JobDoc) => call<DescribeResult>('POST', '/describe', doc);
export const postValidate = (doc: JobDoc) => call<ValidateResult>('POST', '/validate', doc);
export const postExecute = (doc: JobDoc) =>
  call<{ result_id: string; status: string }>('POST', '/execute', doc);
export const getResult = (id: string) => call<ResultSummary>('GET', `/results/${id}`);
export const getTable = (id: string, table: string, offset = 0, limit?: number) =>
  call<TablePage>('GET', `/results/${id}/${table}?offset=${offset}${
    limit === undefined ? '' : `&limit=${limit}`}`);
