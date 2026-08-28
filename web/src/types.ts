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

// ---- the desk's two data views: /book/risk and /book/xva ----------------------------------------
// Both are READS of the live book. The risk answer is computed on a cache miss and served from the
// cache afterwards; the XVA answer is the projection file joined with the book's own set list, and
// nothing on the web side ever asks for a recalc.

export type GreekRow = {
  factor: string;
  /** Present ONLY where the factor has coordinates: absent on a scalar like an FxRate spot, one
   * entry on a curve point, two on a vol-surface node. The key's presence is per FACTOR, so every
   * row of one curve carries a tenor including the 0.0 one. */
  tenor?: number[];
  /** The report-currency derivative per unit of the factor, aggregated over the WHOLE book. */
  value: number;
};

/** One TOP-LEVEL trade's mark. A structure or a netting set appears once, with its net - its legs
 * are inside the row their container reports, which is why `mtm` is exactly the sum of these. */
export type DealMark = {
  reference: string;
  /** The positional identity the other book verbs take, or null where the reference is not
   * unique in the book. */
  deal_path: string | null;
  value: number;
};

export type BookRisk = {
  /** When the RUN happened - a warm hit returns the cached stamp unchanged. */
  as_of: string;
  /** The risk cache key. A DIFFERENT hash from the book's etag, and never compared to it. */
  etag: string;
  currency: string | null;
  mtm: number;
  per_deal: DealMark[];
  greeks: GreekRow[];
};

/** One netting set: what the BOOK says it is now over what the last RUN said about it. The two
 * halves can disagree - that is staleness, and the service reports it on purpose. */
export type XvaSet = {
  reference: string;
  deal_path: string | null;
  counterparty: string | null;
  /** A real JSON boolean - the book stores the CSA field as the text "True"/"False" and the
   * service reads it, so a browser never treats "False" as truthy. */
  collateralized: boolean;
  /** Null except on an ORPHAN row - a row whose set has left the book, which keeps its last
   * numbers and says so here. */
  note: string | null;
  /** 'done' | 'failed' | 'never run'. Typed as the wire's own string: the view branches on the
   * three it knows and shows anything else as itself rather than swallowing it. */
  status: string;
  /** The book's report currency, and null unless `status` is 'done'. */
  cva: number | null;
  /** When THAT row's run happened - not when the view was read. */
  as_of: string | null;
  result_id: string | null;
  plan_hash: string | null;
  values_hash: string | null;
  seed: number | null;
  /** The engine's own wording, and a string only where `status` is 'failed'. */
  error: string | null;
  /** A recalc in flight for this set: {result_id, status: 'queued' | 'running'}. */
  recalc: { result_id: string; status: string } | null;
};

export type BookXva = {
  /** When THIS read happened. Each row's own `as_of` is when that row's run happened, and they
   * will differ by hours. */
  as_of: string;
  path: string;
  sets: XvaSet[];
};
