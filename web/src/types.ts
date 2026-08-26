// The wire contract, as the service publishes it. Descriptors are a tagged union on `widget`;
// values are the job document's own JSON, wire tokens (`.Curve`, `.Timestamp`, ...) included.

export type Descriptor = {
  widget: string;
  description: string;
  value: unknown;
  required?: boolean;
  values?: string[];
  min?: number;
  max?: number;
  bind?: string;
  obj?: string | string[];
  col_names?: string[];
  sub_types?: unknown[];
  sub_fields?: Record<string, Descriptor>;
};

export type Section = Record<string, Descriptor>;

export type Schema = {
  Instrument: {
    groups: Record<string, string[]>;
    sections: Record<string, Section>;
    types: Record<string, string[]>;
    containers: string[];
  };
  Factor: { types: Record<string, Section> };
  Process: { types: Record<string, Section> };
  Calculation: { types: Record<string, Section> };
  Calibration: { types: Record<string, Section> };
  MarketPrices: { types: Record<string, Section> };
  Process_factor_map: Record<string, string[]>;
  Interpolation_factor_map: Record<string, string[]>;
  System: { fields: Section; types: Record<string, string[]> };
  engine_version: string;
};

export type DealNode = {
  Instrument: { '.Deal': Record<string, unknown> };
  Children?: DealNode[];
  Ignore?: string;
};

export type JobDoc = {
  Calc: {
    Calculation: Record<string, unknown>;
    Deals: { Tag_Titles?: unknown; Reference?: string; Deals: { Children: DealNode[] } };
    MergeMarketData?: {
      MarketDataFile?: string;
      ExplicitMarketData?: Record<string, Record<string, unknown>>;
    };
    [key: string]: unknown;
  };
};

export type TableShape = { rows: number; columns: unknown[] };

export type ResultSummary = {
  status: string;
  error?: string;
  stats?: Record<string, unknown>;
  plan_hash?: string;
  values_hash?: string;
  engine_version?: string;
  seed?: number;
  tables?: Record<string, TableShape>;
};

export type TablePage = {
  name: string;
  rows: number;
  columns: unknown[];
  offset: number;
  index: unknown[];
  data: unknown[];
};

export type DescribeResult = {
  deals: Record<string, number>;
  factors: { resolved: string[]; missing: string[] };
  calculation: Record<string, unknown>;
  cost?: { class: number; estimate: number; basis: string };
};

export type ValidateResult = { deals: Record<string, string[]>; factors: string[] };

export type BookResponse = { document: JobDoc; etag: string; path: string };
