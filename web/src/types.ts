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
  /** Instrument only: an amount of the instrument, which a position of q units scales q times. */
  sized?: boolean | string;
  /** Instrument only: what a date settles, each payment naming the field stating its currency and
   * the one stating its amount; a table's keyed by its column. */
  settles?: { currency: string; amount: string | null }[] | Record<string, unknown>;
  values?: string[];
  bind?: string;
  obj?: string | string[];
  col_names?: string[];
  sub_types?: unknown[];
  sub_fields?: Record<string, Descriptor>;
  /** Structure only: which way a structure is dealt, chosen rather than filled - `required` where
   * no level tells its variations apart. */
  selector?: 'required' | 'optional';
  /** The price-factor types the field's value names; a table's keyed by its column. */
  factors?: string[] | Record<string, string[]>;
};

export type Section = Record<string, Descriptor>;

/** One market-data section the book states its bootstrap in, rendered by SHAPE: `types` is an
 * entry per declared key, `menu` NAMES a sibling store holding one list of values per key and
 * `value` is what the engine uses where the book says nothing. `entry` and `rules` are the two
 * halves that shape writes - one value per key, and a rule per factor named by `key` - and the
 * configure verb takes the key itself. */
export type ConfigSection = {
  types?: Record<string, { fields: Section }>;
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
    titles: Record<string, string>;
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
  /** What a sales desk quotes, and the mandate a book may declare beside its `Calculation`. */
  Structure?: { types: Record<string, StructureType> };
  'Quote Policy'?: { fields: Section };
  engine_version: string;
};

// ---- a quote: /book/structure, /book/quote ------------------------------------------------------

/** One leg a structure books: a declared instrument type, what the structure pins on it and the
 * parameter each remaining field is filled from. */
type StructureLeg = {
  deal_type: string; pinned: Record<string, unknown>; slots: Record<string, string>;
};

/** One way a structure is dealt: the side of the pair its client buys, its own parameters and
 * legs. */
type Variation = {
  buys: 'base' | 'quote'; fields: Section; legs: Record<string, StructureLeg>;
};

/** A structure as `/schema` declares it: its sales names, the parameters every form shares, the
 * recipe's steps in words, and either its legs or its variations. */
export type StructureType = {
  vernacular: string; fields: Section; recipe: string[];
  legs?: Record<string, StructureLeg>; variations?: Record<string, Variation>;
};

/** One priced leg, on the client's side: its market-terms strike and barrier, its premium in the
 * quote's currency, and the deal field the recipe solved on it where it solved one. */
export type QuotedLeg = {
  reference: string; role: string; deal_type: string; buy_sell: string | null;
  strike_market: number | null; barrier_market: number | null; premium: number;
  solved: Record<string, number> | null; note: string | null; spread_charge: number | null;
};

/** The quote a structure answers with, every money in `currency`. */
export type Quote = {
  quote_id: string; structure: string; params: Record<string, unknown>;
  variation: string | null; client: { buys: string; sells: string } | null;
  netting_set: string | null; legs: QuotedLeg[]; net: number; net_mid: number;
  charged_on: string; edge: number; currency: string;
  spot: { source: string; note: string | null; value_market: number };
  forward: number; spread_note: string | null; notes: string[] | null;
  risk?: { note?: string | null; scale?: number | null };
  margin?: { amount: number; currency: string; value: number; pricing_currency: string };
  files?: { sheet: string | null; sheet_note?: string; quote: string };
};

/** `GET /book/quote/{id}`: the pending trade as filed - and, once accepted, where it was filed and
 * where it landed (`booked.lsn` null on a desk keeping no record). */
export type PendingQuote = {
  quote: Quote; deal: Record<string, unknown>; quoted_at: string | null;
  quoted_by: string | null; portfolio?: string;
  accepted?: { lsn: number; ticket: string } | null;
  booked?: { lsn: number | null; deal_path: string } | null;
};

/** A number of a netting set's run with a candidate trade, of its run without, and the first less
 * the second. */
export type Stepped<T> = { with: T; without: T; increment: number };

/** `stats.Increment` of a what-if asked for its XVA increment: what the candidate does to the
 * netting set it goes into, in the run's `currency` - its CVA and FVA, and under each exposure
 * column (`EE`, every `PFE_<percentile>`) a `Stepped` peak, its `value` and `day` as an XVA
 * row's (`XvaSet.peaks`) - on the `paths` both
 * runs drew. */
export type Increment = {
  netting_set: string; counterparty: string | null; currency: string | null;
  paths: { count: number }; cva: Stepped<number>; fva: Stepped<number>;
  /** Each credit limit the set draws on, held against the usage with the trade; absent for none. */
  limits?: CreditLimit[];
  [column: string]: unknown;
};

/** One credit limit held against the peak exposure of the netting sets drawn on it - the entity's
 * and those of every entity under it, `clients` naming them all, itself included: `sets` summed
 * into `usage` and `unrun` left out for want of a run, `headroom` stated only where none is and
 * negative where `breached`, `usage` and `breached` null where no set has run, and all three null
 * beside the `not_comparable` sentence. On a credit
 * charge `usage`, `headroom` and `breached` are with the trade, `usage_without` is the usage
 * without it and `breached_without` whether that is over the limit, null where not comparable. */
export type CreditLimit = {
  entity: string; name: string | null; clients: string[];
  limit: { amount: number; currency: string };
  sets: number; unrun: number; usage: number | null; headroom: number | null;
  breached: boolean | null; not_comparable: string | null; usage_without?: number | null;
  breached_without?: boolean | null;
};

/** Whose XVA increment is asked: a quote's pending trade, priced into the book as its acceptance
 * would book it, or a trade the book already holds, by its path and the reference read there. */
export type ChargeAsk = { quote_id: string } | { deal_path: string; reference?: string };

/** One row of `GET /book/quotes`, newest first: what was quoted, for whom, its net, who quoted it
 * and when, and - once the acceptance wrote them - `accepted` and where it was `booked`, beside the
 * `reference` of the trade it booked. */
export type QuoteGiven = Pick<Quote, 'quote_id' | 'structure' | 'variation' | 'netting_set' | 'net'
  | 'currency'> & Pick<PendingQuote, 'quoted_at' | 'quoted_by' | 'accepted'> & {
  booked?: { lsn: number | null; deal_path: string; reference?: string | null } | null;
};

/** What `POST /book/quote` answers short of a 422 - and a booking, a restrike or an unwind through
 * `POST /book/deals`: a booking, a refusal in words, or the quote already booked; on a desk keeping
 * a record, the tier it was routed through and the sentence of each check it `fell_past` first. */
export type Acceptance = {
  written: boolean; deal_path?: string; refused?: string[]; note?: string;
  accepted?: { lsn: number; ticket: string } | null;
  booked?: { lsn: number | null; deal_path: string };
  recorded?: { lsn: number };
  tier?: { name: string; four_eyes: boolean; fell_past: string[] }; waits_on?: string;
  firmness?: { plan?: { pinned: string; current: string; moved: boolean } };
};

export type DealNode = {
  Instrument: { '.Deal': Record<string, unknown> };
  Children?: DealNode[];
  Ignore?: string;
};

/** Market data by section, in the wire form a market data file and an explicit block both hold. */
export type Market = Record<string, Record<string, unknown>>;

export type JobDoc = {
  Calc: {
    Calculation: Record<string, unknown>;
    Deals: { Reference?: string; Deals: { Children: DealNode[] } };
    MergeMarketData?: {
      MarketDataFile?: string;
      ExplicitMarketData?: Market;
    };
    [key: string]: unknown;
  };
};

/** `GET /book/market`: the market the book prices on - the file it names with its explicit block
 * merged over it - the etag of the document it was read off, and the stamp the market is named by.
 * `market` is null where the stamp asked with is that one: the market held is still current. */
export type BookMarket = { etag: string; stamp: string; market: Market | null };

export type TableShape = { rows: number; columns: unknown[] };

export type ResultSummary = {
  status: string;
  error?: string;
  stats?: Record<string, unknown>;
  plan_hash?: string;
  values_hash?: string;
  engine_version?: string;
  seed?: number | null;
  quasi_seed?: number;
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

/** `collateralized_sets` are the netting sets a run of the job collateralises, by reference. */
export type DescribeResult = {
  deals: Record<string, number>;
  collateralized_sets: string[];
  factors: { resolved: string[]; missing: string[] };
  /** Each price factor the engine simulates, by the process it simulates it with. */
  processes: Record<string, string>;
  /** Each interpolated factor type's scheme where no rule names one. */
  interpolation: Record<string, string>;
  calculation: Record<string, unknown>;
  cost?: { class: number; estimate: number; basis: string };
};

export type ValidateResult = {
  deals: Record<string, string[]>; factors: string[]; calculation: string[];
};

/** `opened` is the file name a document opened through `POST /book/open` was posted under, and
 * present only while that document is the one served. */
export type BookResponse = { document: JobDoc; etag: string; path: string; opened?: string };

// ---- where this workstation reads the files a job names: /paths ---------------------------------
// The service's own fact, kept on this box and never written into a job: a job names its files as
// the system that wrote it saw them, and the two maps say what this box reads in their place.

/** One file the served book names: the path as the job writes it, the keys the two maps are looked
 * up by, where this box reads it, and its digest - or the operating system's refusal. */
export type NamedFile = {
  key: 'MarketDataFile' | 'CalendDataFile';
  name: string;
  folder: string;
  file: string;
  path: string;
  digest: string | null;
  unread: string | null;
};

/** A folder as a job writes it to a folder on this box, and a file name to a file name. */
export type PathMaps = {
  path_transform: Record<string, string>;
  file_transform: Record<string, string>;
};

/** `GET /paths`: both maps, and one row per file the served book names. */
export type PathsAnswer = PathMaps & { named: NamedFile[] };

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
  /** The factor's own value at the row's coordinates: a spot, a rate, a vol. */
  level?: number | null;
  /** The same derivative per the move a desk quotes the factor by, `per` its words - `per 1%`,
   * `per bp`, `per vol point`, `per unit` where its class declares none. */
  risk?: number | null;
  per?: string;
  /** A QUOTE's row as the quote view reads it: `factor` is the block the quote came off. */
  quote?: string;
};

/** One quote: the report-currency derivative per unit of the quote AS QUOTED - a par rate in
 * percent reads per 1% - taken through the calibration that builds a factor from it, and read per
 * its natural move where it has one. */
type QuoteRow = {
  block: string; quote: string; value: number; level?: number | null; risk?: number | null;
  per?: string;
};

/** One trade's mark: a TOP-LEVEL node's in `per_deal` - a structure or a netting set once, with
 * its net, which is why `mtm` is exactly their sum - or one a top-level set holds in `in_sets`. */
export type DealMark = {
  reference: string;
  /** The positional identity the other book verbs take, or null where the reference is not
   * unique in the book. */
  deal_path: string | null;
  value: number;
};

/** One position the record holds on a priced deal, at its own share of the deal's mark by the
 * record's share rule - `unit` a unit's, `value` its `quantity`'s, the positions on a deal adding
 * to its mark - both none where the deal has no mark or nets to nothing. */
export type PositionMark = {
  deal_path: string; portfolio: string; agreement: string | null; quantity: number;
  unit: number | null; value: number | null;
};

export type BookRisk = {
  /** When the RUN happened - a warm hit returns the cached stamp unchanged. */
  as_of: string;
  /** The risk cache key. A DIFFERENT hash from the book's etag, and never compared to it. */
  etag: string;
  currency: string | null;
  mtm: number;
  per_deal: DealMark[];
  /** Each trade a top-level netting set holds, inside its set's value and never summed. */
  in_sets: DealMark[];
  /** Each position the seat is shown on a priced deal. */
  per_position: PositionMark[];
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
  /** Null unless `status` is 'done'; in `currency`. */
  cva: number | null;
  /** The currency THIS row's run reported in - not the open book's - and null on a row filed
   * before the run kept it. */
  currency: string | null;
  /** The funding adjustment, a column of the SAME run - one CMC per set, so it shares this row's
   * `as_of`, replay tuple and currency. Null on a done row means that row was filed before the
   * column existed, not that funding cost nothing: a book whose calculation states no funding
   * curve reads 0.0. */
  fva: number | null;
  /** When THAT row's run happened - not when the view was read. */
  as_of: string | null;
  result_id: string | null;
  plan_hash: string | null;
  values_hash: string | null;
  seed: number | null;
  /** Null on a row filed before the replay tuple carried it. */
  quasi_seed: number | null;
  /** The engine's own wording, and a string only where `status` is 'failed'. */
  error: string | null;
  /** A recalc in flight for this set: {result_id, status: 'queued' | 'running'}. */
  recalc: { result_id: string; status: string } | null;
  /** The run's largest `EE` and each `PFE_<percentile>`, each with the day it first falls on -
   * null on a row filed before rows kept them. */
  peaks: Record<string, { value: number; day: string }> | null;
};

/** The two curves a credit run asked now funds its FVA at, null for one its calculation does not
 * state; `refused` the record's sentence, beside two nulls, where that calculation cannot be read. */
export type Funding = { cost: string | null; benefit: string | null; refused?: string[] };

export type BookXva = {
  /** When THIS read happened. Each row's own `as_of` is when that row's run happened, and they
   * will differ by hours. */
  as_of: string;
  sets: XvaSet[];
  /** What the NEXT run funds at: a row's own `fva` is what its run cost. */
  funding: Funding;
  /** The credit limits the record declares, by entity; none where it declares none. */
  limits: CreditLimit[];
};

/** `GET /book/xva/{reference}/profile`: one set's exposure profile off its row, under the run it
 * came from - a 404 and its sentence where the set has no completed run or its row keeps none. */
export type XvaProfile = {
  reference: string; result_id: string; as_of: string;
  profile: { columns: string[]; index: string[]; data: number[][] };
};

// ---- the record's readings: the spine block, /book/activity, /book/markets, /book/reconcile -----
// Every one is a READING: it refuses nothing, and a desk that records nothing answers `spine: null`
// and 404s the three verbs - which is a state of the screen rather than an error on it.

/** Where the book FILE stands against the record: the LSN it was last hydrated at, and how far the
 * record has moved since - in events, and in the fills and amendments among them. Null throughout
 * on a file nothing has pinned, which is a file with nothing to compare. `paper` is the LSN of the
 * last entity or agreement declaration standing, which moves when the paper does and only then.
 * `close` is the market the record designates for the P&L, null where it designates none, whether
 * a close is declared on it, and whether marks stand on it for the book's own day. */
export type SpineBlock = {
  lsn: number | null;
  head: string | null;
  events_behind: number | null;
  positions_behind: number | null;
  paper: number;
  close: { market: string | null; stands: boolean; marked: boolean };
};

/** `GET /book/status`. It carries the desk's whole orientation; this client renders the rest off
 * the document it already holds and reads only the record's block here. */
export type BookStatus = { etag: string; spine: SpineBlock | null };

/** One line of the strip: the envelope - `record_time` the writer's clock, `effective_time` when
 * the fact is TRUE, null where it carries no truth-time of its own - and what the service tells of
 * it: `summary` naming what it is about, `by` who did it, and the deal, agreement and entity it
 * opens, each null where it names none; a restrike's `changed`, each field it moved. */
export type ActivityRow = {
  lsn: number;
  record_time: string;
  effective_time: string | null;
  actor: string;
  event_type: string;
  book: string | null;
  summary: string;
  by: string;
  deal_path: string | null;
  agreement: string | null;
  entity: string | null;
  changed?: Moved[] | null;
};

/** One field a restrike moved, from what to what, in wire form. */
export type Moved = { field: string; was: unknown; became: unknown };

/** `GET /book/activity`: the rows newest last, and the head the fold reached - the `since` a
 * client polls with next. */
export type ActivityPage = { lsn: number; rows: ActivityRow[] };

/** One beat of `GET /spine/doorbell` - a POSITION and never a fact, so a client learns from it
 * only that there is something to read. Nothing of an event's body is on this wire. */
export type Doorbell = { lsn: number; head: string };

/** The official close standing on a market - its latest by `date`. `supersedes_lsn` is the close of
 * its day this one restated, null on a day's first - a close is superseded by a NEW close rather
 * than corrected in place. */
export type MarketClose = {
  market: string;
  values_hash: string;
  date: string;
  supersedes_lsn: number | null;
  effective_time: string | null;
  lsn: number;
};

type MarketName = {
  name: string;
  values_hash: string;
  actor: string;
  effective_time: string | null;
  lsn: number;
};

type MarketSnapshot = { blob: string; book: string | null; lsn: number };

export type BookMarkets = {
  lsn: number;
  names: MarketName[];
  closes: MarketClose[];
  snapshots: MarketSnapshot[];
};

/** `GET /book/reconcile` - where the file and the record disagree, the record folded AT ITS HEAD,
 * the two meeting where the book prices: a position no node carries, and a node no position
 * carries, each named by its set, the trade's reference and the client. */
export type Reconcile = {
  lsn: number | null;
  events_behind: number | null;
  positions_behind: number | null;
  in_record_not_in_file: {
    instrument: string; netting_set: string; quantity: number; last_lsn: number;
    reference: string | null; client: string | null;
  }[];
  in_file_not_in_record: {
    instrument: string; deal_path: string; reference: string | null; netting_set: string | null;
    client: string | null;
  }[];
  /** A client's netting set whose paper is not what legal declared, with the fields that moved. */
  terms_mismatch: { agreement: string; deal_path: string; fields: string[] }[];
};

/** One row of `GET /book/worklist`: what waits, and `key` the fact that clears it is filed
 * against - a ticket, a diary key, a fill's own key, an agreement, a day; a payment carries its
 * amount, currency and the agreement it is due under where it is one, a call the amount and
 * currency it calls and the kinds of money it settles as, and a call nobody can work out what is
 * not known. A ticket's row names its trade, and a pending one its tier, what it fell past first
 * as its ticket reads it, what it was booked at, why it waits and `may`: null where the seat may
 * sign it, else the record's sentence why not. */
export type WorkRow = {
  kind: string; what: string; key: string; lsn: number | null; since: string | null;
  amount?: number | null; currency?: string | null; agreement?: string | null;
  unknown?: { instrument: string | null; what: string }[]; settles_as?: string[];
  reference?: string | null; deal_path?: string | null; client?: string | null;
  booker?: string; tier?: string | null; fell_past?: string[] | null; price?: number | null;
  waits_on?: string | null; may?: string | null;
};

/** What an agreement's terms never state, `GET /book/agreements` names: the netting set's `fields`
 * and the `collateral_columns` of its collateral tables - settlement state, not paper. */
export type SettlementState = { fields: string[]; collateral_columns: string[] };

/** The stable name of a refusal's kind, beside its sentence on every refusal the service words: a
 * credit curve's names the field of the request it refuses. */
export type RefusalCode = 'no_actor' | 'not_granted' | 'unpriced' | 'recorded' | 'refused'
  | 'counterparty' | 'spread' | 'recovery';

/** `GET /book/seat` - the seat a request's acts are filed under, per verb the nodes it holds it
 * at, each reaching every path under it and null where nothing narrows it, `demands`, the verb
 * each act the record files demands under the map in force, and `may`, per act and per node the
 * service gates it at, null where the seat may and else the sentence the record refuses it with. */
export type SeatGrants = {
  seat: string; verbs: Record<string, string[] | null>; demands: Record<string, string>;
  may: Record<string, Record<string, string | null>>;
  /** The nodes the seat is shown, sorted - none where it holds nothing - null for the whole book. */
  sees: string[] | null;
};

/** `GET /book/close/check` - whether a close on `date` is legal, and the diary rows it waits on. */
export type CloseVerdict = {
  date: string; legal: boolean; due: number;
  outstanding: { kind: string; leg: string; due_date: string | null; instrument: string | null;
                 reason: string | null }[];
};

/** One agreement's call on a marked close, `GET /book/collateral`, every amount in its `currency`:
 * the exposure its CSA reads, what is held per asset and what that is worth (`balance`, what the
 * bank posted negative), the support required and the call - received positive - with whether the
 * bank calls or posts; `fx` the close's spots in that currency, and the worked fields null over
 * anything `unknown`. */
export type CollateralCall = {
  agreement: string; entity: string; currency: string; exposure: number | null;
  fx: Record<string, number>; held: Record<string, number>; margin: Record<string, number>;
  required: number | null; balance: number | null; call: number | null;
  direction: 'call' | 'post' | null; minimum_transfer: number | null;
  unknown: { instrument: string | null; what: string }[];
};

/** `GET /book/collateral` - the calls on the close marked for `date`, the record read at `lsn`. */
export type BookCollateral = { date: string; marks: number; lsn: number; calls: CollateralCall[] };

/** What a write to the record answers: the position it landed at, beside its own fields. */
export type Recorded = { recorded: { lsn: number } };

/** What a verdict on a ticket answers: where it landed and what the trade reads now. */
export type Decided = Recorded & { ticket: string; portfolio: string; status: string | null };

/** `GET /book/worklist` - six lists of what waits on the seat asking, read off what stands, so a
 * row leaves when its fact lands, and `elsewhere`, the tickets it is shown waiting on another
 * seat: how many each holds, and the newest of them. `seat` is the seat they are read for, null
 * where none was named and they are what waits on anyone. */
export type Worklist = {
  date: string;
  seat: string | null;
  counts: { pending: number; payments: number; unconfirmed: number; calls: number;
            unmarked: number; rejected: number; elsewhere: number };
  pending: WorkRow[];
  payments: WorkRow[];
  unconfirmed: WorkRow[];
  calls: WorkRow[];
  unmarked: WorkRow[];
  rejected: WorkRow[];
  elsewhere: WorkRow[];
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
