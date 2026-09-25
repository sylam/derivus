// The wire contract, as the service publishes it. Descriptors are a tagged union on `widget`;
// values are the job document's own JSON, wire tokens (`.Curve`, `.Timestamp`, ...) included.

export type Descriptor = {
  widget: string;
  description: string;
  value: unknown;
  /** Inside the `Instrument` store this is MUST BE STATED - a term, whose `value` is only what a
   * blank panel shows; on every other store it is still `default=REQUIRED`. */
  required?: boolean;
  /** Instrument only: leaving this field out MEANS `value`, so an author may say nothing. */
  convention?: boolean;
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

/** One market-data section the book states its bootstrap in, rendered by SHAPE: `types` is an
 * entry per declared key with the spellings an older book files it under, `menu` NAMES a sibling
 * store holding one list of values per key and `value` is what the engine uses where the book
 * says nothing. `entry` and `rules` are the two halves that shape writes - one value per key, and
 * a rule per factor named by `key` - and the configure verb takes the key itself. */
export type ConfigSection = {
  types?: Record<string, { aliases: string[]; fields: Section }>;
  entry?: string;
  rules?: string;
  key?: string;
  menu?: string;
  value?: string;
};

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
  /** `values` is the quote-row plane the tick verb moves - the only columns a client may edit. */
  MarketPrices: { types: Record<string, Section>; values: string[] };
  Configuration: Record<string, ConfigSection>;
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
  /** A job that publishes its own, while it waits or runs. */
  progress?: { done: number; total: number; note: string };
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
  /** A QUOTE's row as the quote view reads it: `factor` is the block the quote came off. */
  quote?: string;
};

/** One quote: the report-currency derivative per unit of the quote AS QUOTED - a par rate in
 * percent reads per 1% - taken through the calibration that builds a factor from it. */
export type QuoteRow = { block: string; quote: string; value: number };

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
  /** The risk in QUOTE space, every quote of every block the book's factors are built from. */
  quotes: QuoteRow[];
  /** The factors those quotes stand for - read on their quotes, and in `greeks` beside them. */
  quoted: string[];
  /** Why a book's quotes did not reach its risk, where they did not; null where they all did. */
  quote_note: string | null;
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
  /** The funding adjustment, a column of the SAME run - one CMC per set, so it shares this row's
   * `as_of` and replay tuple. Null on a done row means that row was filed before the column
   * existed, not that funding cost nothing: a set declaring no `Funding_Rate` reads exactly 0.0. */
  fva: number | null;
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

// ---- the record's readings: the spine block, /book/activity, /book/markets, /book/reconcile -----
// Every one is a READING: it refuses nothing, and a desk that records nothing answers `spine: null`
// and 404s the three verbs - which is a state of the screen rather than an error on it.

/** Where the book FILE stands against the record: the LSN it was last hydrated at, and how far the
 * record has moved since - in events, and in the fills and amendments among them. Null throughout
 * on a file nothing has pinned, which is a file with nothing to compare. */
export type SpineBlock = {
  lsn: number | null;
  head: string | null;
  events_behind: number | null;
  positions_behind: number | null;
};

/** `GET /book/status`. It carries the desk's whole orientation; this client renders the rest off
 * the document it already holds and reads only the record's block here. */
export type BookStatus = { etag: string; spine: SpineBlock | null };

/** One line of the strip, envelope only - which is why it renders on a replica holding no key.
 * `record_time` is the writer's clock; `effective_time` is when the fact is TRUE and is null where
 * the fact carries no truth-time of its own. `summary` is the declared sentence for the type, or
 * the type's own name where this hub's table has none. */
export type ActivityRow = {
  lsn: number;
  record_time: string;
  effective_time: string | null;
  actor: string;
  event_type: string;
  book: string | null;
  summary: string;
};

/** `GET /book/activity`: the rows newest last, and the head the fold reached - the `since` a
 * client polls with next. */
export type ActivityPage = { lsn: number; rows: ActivityRow[] };

/** One beat of `GET /spine/doorbell` - a POSITION and never a fact, so a client learns from it
 * only that there is something to read. Nothing of an event's body is on this wire. */
export type Doorbell = { lsn: number; head: string };

/** The official close standing on a market. `supersedes_lsn` is the close this one restated, null
 * on the first - a close is superseded by a NEW close rather than corrected in place. */
export type MarketClose = {
  market: string;
  values_hash: string;
  supersedes_lsn: number | null;
  effective_time: string | null;
  lsn: number;
};

export type MarketName = {
  name: string;
  values_hash: string;
  actor: string;
  effective_time: string | null;
  lsn: number;
};

export type MarketSnapshot = { blob: string; book: string | null; lsn: number };

export type BookMarkets = {
  lsn: number;
  names: MarketName[];
  closes: MarketClose[];
  snapshots: MarketSnapshot[];
};

/** `GET /book/reconcile` - where the file and the record disagree, the record folded AT ITS HEAD.
 * The file carries terms and never a signed quantity, so what the two can disagree about is how
 * many CLIPS stand, with the record's own quantity beside them. */
export type Reconcile = {
  lsn: number | null;
  events_behind: number | null;
  positions_behind: number | null;
  in_record_not_in_file: {
    instrument: string; netting_set: string | null; quantity: number; last_lsn: number;
  }[];
  in_file_not_in_record: { instrument: string; deal_path: string; reference: string | null }[];
  quantity_mismatch: {
    instrument: string; record_clips: number; file_nodes: number; record_quantity: number;
  }[];
};

// ---- the curves half of the market: /book/curve -------------------------------------------------
// The book's curve BLOCKS read back as the definitions they are, and the curves this workstation's
// seed could set one up from. The conventions travel in the verb's own spelling - the declared
// field name in lower case - and none of the seeded rows is verified against a terminal.

/** One benchmark: the tenor says which instrument, the security where the number came from, the
 * quote is the number a desk states and `use` whether it enters the solve. */
export type CurveRow = { tenor: string; security: string; quote: number | null; use: string };

export type CurveBlock = {
  curve: string;
  currency: string;
  discount_rate: string;
  /** The scheme the solved factor carries - the `Price Factor Interpolation` rule naming this
   * curve, else the routed type's own method, else the engine's fallback. Not a field of the
   * block: `interpolation_source` says whether a rule named it (`curve`) or not (`default`). */
  interpolation: string;
  interpolation_source: string;
  /** Absent on a block authored before it carried its definition, which `note` then explains. */
  conventions?: Record<string, unknown>;
  note?: string;
  rows: CurveRow[];
};

/** A curve the seed declares. An entry it states too little of carries the emitter's refusal in
 * place of the rows, which is the answer to what is missing. */
export type SeededCurve = {
  currency?: string;
  conventions?: Record<string, unknown>;
  rows?: { tenor: string; security: string }[];
  refused?: string;
};

/** `seeded` is absent when the answer was narrowed to one curve. */
export type CurvesAnswer = {
  etag: string;
  curves: Record<string, CurveBlock>;
  seeded?: Record<string, SeededCurve>;
};

/** What a write answered: the block it installed, the knots it solved on and the price factors
 * the bootstrap rewrote. A refusal is an `ApiError`, never a shape. */
export type CurveOutcome = {
  written: boolean;
  block?: string;
  knots?: string[];
  rewrote?: string[];
  installed?: string[];
};

// ---- the ticker vocabulary and its evidence: /book/securities -----------------------------------
// The seed is the CANDIDATES a desk could quote, the map is what a terminal VERIFIED, and `used` is
// the join between the two and the book. None of the three is declared by `/schema` - the seed is
// the desk's own JSON file - so this screen renders by SHAPE, and a ticker is a string throughout.

/** What the map ever said about one security: an entry's evidence (the name a terminal answered
 * and when it was verified), a rejection's `verdict`, or the `{verdict: 'unmapped'}` the service
 * answers where the map has never heard of the ticker. */
export type Evidence = {
  name?: string | null;
  last_update?: string | null;
  verified?: string;
  verdict?: string;
  error?: string | null;
};

/** One quote row the book carries, with that evidence beside it - the IPV join. */
export type UsedRow = {
  curve: string;
  tenor: string;
  security: string;
  quote: number | null;
  /** The print's OWN timestamp, off the quote row - not when the map verified the ticker. */
  timestamp: string;
  use: string;
  evidence: Evidence;
};

/** One verified leaf of the map. `security` is what MAKES a leaf; everything above it is the path
 * a drift is reported by. */
export type MapEntry = {
  security: string;
  name: string;
  last_update: string | null;
  verified: string;
};

/** `GET /calculations` - the desk's own named calculations, each a `Calculation` block: `Object`
 * names the type and every other key is one of its declared fields. */
export type CalculationsAnswer = { calculations: Record<string, Record<string, unknown>> };

/** What a save answered: the file as it now stands, or the judgement naming what to fix. */
export type CalculationOutcome = {
  written: boolean;
  name: string;
  refused?: string[];
  calculations?: Record<string, Record<string, unknown>>;
};

export type SecuritiesAnswer = {
  etag: string;
  home: string;
  /** Whether this home carries a map at all. False reads as an EMPTY map, never a refusal. */
  provisioned: boolean;
  seed: Record<string, Record<string, unknown>>;
  used: UsedRow[];
  map: {
    generated: string | null;
    blocks: Record<string, unknown>;
    /** Keyed by TICKER: a candidate that never became an entry has no path in the map. */
    rejected: Record<string, Evidence>;
  };
};

/** What a merge answered: the tickers that block now spells, and the file the write kept. */
export type SeedOutcome = {
  written: boolean;
  block: string;
  key: string;
  candidates: string[];
  backup: string | null;
  seed: string;
};

/** What a verification answered, under its run's own `stats.Securities`. */
export type VerifyOutcome = {
  written?: boolean;
  map?: string;
  verified?: string[];
  /** Keyed by the entry's path in the map, which is what a drift is named by. */
  drifted?: Record<string, { security: string; drift: string }>;
  unknown?: string[];
  /** The names the map had never heard of, each with the verdict it was probed to. */
  added?: Record<string, string>;
  seconds?: number;
};
