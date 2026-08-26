// One store, one reducer. Viewing is a pure function of (schema, doc); the service is touched on
// user action (describe/validate/execute) and by the book poll, which only re-fetches when the
// etag moved - a booking made by any other client appears within a tick.

import { createContext, useContext, type Dispatch } from 'react';
import type {
  DescribeResult, JobDoc, ResultSummary, Schema, TablePage, ValidateResult,
} from './types';

export type Source =
  | { kind: 'book'; etag: string; path: string }
  | { kind: 'local'; name: string };

export type RunState = {
  status: 'idle' | 'submitting' | 'queued' | 'running' | 'done' | 'error';
  resultId: string | null;
  summary: ResultSummary | null;
  error: string | null;
  table: string | null;
  page: TablePage | null;
  scalars: Record<string, unknown>;
  startedAt: number | null;
};

export type AppState = {
  schema: Schema | null;
  schemaError: string | null;
  doc: JobDoc | null;
  source: Source | null;
  docError: string | null;
  tab: string;
  selection: { deal: string | null; factor: string | null };
  describe: DescribeResult | null;
  validate: ValidateResult | null;
  run: RunState;
};

export const IDLE_RUN: RunState = {
  status: 'idle', resultId: null, summary: null, error: null,
  table: null, page: null, scalars: {}, startedAt: null,
};

export const INITIAL: AppState = {
  schema: null, schemaError: null, doc: null, source: null, docError: null,
  tab: 'portfolio', selection: { deal: null, factor: null },
  describe: null, validate: null, run: IDLE_RUN,
};

export type Action =
  | { type: 'SCHEMA_LOADED'; schema: Schema }
  | { type: 'SCHEMA_FAILED'; error: string }
  // `refresh` is a book poll: the document moved underneath the user, so keep their place
  | { type: 'DOC_LOADED'; doc: JobDoc; source: Source; refresh?: boolean }
  | { type: 'DOC_FAILED'; error: string }
  | { type: 'TAB'; tab: string }
  | { type: 'SELECT_DEAL'; path: string | null }
  | { type: 'SELECT_FACTOR'; name: string | null }
  | { type: 'DESCRIBED'; describe: DescribeResult }
  | { type: 'VALIDATED'; validate: ValidateResult }
  | { type: 'RUN_SUBMITTED'; resultId: string }
  | { type: 'RUN_POLLED'; summary: ResultSummary }
  | { type: 'RUN_FAILED'; error: string }
  | { type: 'TABLE_SELECTED'; table: string }
  | { type: 'PAGE_LOADED'; page: TablePage }
  | { type: 'SCALARS_LOADED'; scalars: Record<string, unknown> };

export function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'SCHEMA_LOADED':
      return { ...state, schema: action.schema, schemaError: null };
    case 'SCHEMA_FAILED':
      return { ...state, schemaError: action.error };
    case 'DOC_LOADED':
      return action.refresh
        ? { ...state, doc: action.doc, source: action.source, docError: null, validate: null }
        : {
            ...state, doc: action.doc, source: action.source, docError: null,
            selection: { deal: null, factor: null }, describe: null, validate: null, run: IDLE_RUN,
          };
    case 'DOC_FAILED':
      return { ...state, docError: action.error };
    case 'TAB':
      return { ...state, tab: action.tab };
    case 'SELECT_DEAL':
      return { ...state, selection: { ...state.selection, deal: action.path } };
    case 'SELECT_FACTOR':
      return { ...state, selection: { ...state.selection, factor: action.name } };
    case 'DESCRIBED':
      return { ...state, describe: action.describe };
    case 'VALIDATED':
      return { ...state, validate: action.validate };
    case 'RUN_SUBMITTED':
      return {
        ...state,
        run: { ...IDLE_RUN, status: 'queued', resultId: action.resultId, startedAt: Date.now() },
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
  }
}

export const AppContext = createContext<{ state: AppState; dispatch: Dispatch<Action> } | null>(null);

export function useApp() {
  const app = useContext(AppContext);
  if (!app) throw new Error('useApp outside AppContext');
  return app;
}
