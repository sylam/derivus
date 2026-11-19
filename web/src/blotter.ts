// The blotter's arithmetic, kept out of the component: the deal TREE flattened to rows keyed by
// the same positional `deal_path` every other client uses, and each row's roll date measured
// against the book's own Base_Date. Nothing here touches React, so the whole reading of a book -
// which deal rolls when - is one pure function of (document, schema).
//
// The column choices are VALUE-FIRST, the same discipline `FieldView` holds: a candidate field
// wins because the deal carries it, and the schema's declaration supplies the label, the details
// block and what an amount is a size of (`sizeOf`). A type nobody anticipated therefore still
// fills what columns it can and renders the rest in its expander, rather than raising.

import { closed, netOf, type Position } from './positions';
import { formatNumber, isObject, token, type Kind } from './tokens';
import type { TreeNode } from './tree';
import type { DealNode, Descriptor, JobDoc, Market, Schema } from './types';

// ---- the candidate lists: one per column, most specific first ----------------------------------
// Read against the deal's own keys. Every name below is declared by at least one instrument in
// `mapping['Instrument']`; the order is the desk's reading order, not the schema's.

const SIDE_FIELDS = ['Buy_Sell', 'Payer_Receiver', 'Borrower_Lender', 'MtM_Side'];

/** The other of a word a side, a sense or a barrier's direction is written in. */
const OTHER: Record<string, string> = {
  Buy: 'Sell', Sell: 'Buy', Call: 'Put', Put: 'Call', Up: 'Down', Down: 'Up',
};

/** Amounts counting the underlying in its own units: no currency is ever beside them. */
const UNIT_FIELDS = ['Units', 'Volume'];

const AMOUNT_FIELDS = [
  'Principal', 'Underlying_Amount', 'Amount', 'Buy_Amount', 'Near_Buy_Amount', 'Sell_Amount',
  'Units', 'Volume', 'Settlement_Amount', 'Cash_Payoff', 'Payoff', 'LeverageNotional',
  'Opening_Balance',
];

const STRIKE_FIELDS = [
  'Strike_Price', 'Strike', 'Fixed_Price', 'Forward_Price', 'Swap_Rate', 'FRA_Rate', 'Cap_Rate',
  'Floor_Rate', 'Interest_Rate', 'Extension_Strike',
];

const BARRIER_FIELDS = ['Barrier_Price', 'Barrier'];

/** What a deal naming no pair is on: a commodity, its reference price, an equity, an index. */
const UNDERLYING_FIELDS = ['Commodity', 'Reference_Type', 'Equity', 'Index'];

/** A schedule each row of which is a fixing paying the deal's amount; one the amount averages. */
export const STRIP_FIELDS = [
  'TARF_ExpiryDates', 'Accumulator_ExpiryDates', 'Extendable_ExpiryDates', 'Observation_Dates',
];
const AVERAGE_FIELDS = ['Sampling_Data'];

/** The term-end date: what "rolls off" means for this deal. */
const EXPIRY_FIELDS = [
  'Expiry_Date', 'Option_Expiry_Date', 'Maturity_Date', 'Swap_Maturity_Date', 'Investment_Horizon',
  'Far_Settlement_Date', 'Settlement_Date', 'Delivery_Date', 'Payment_Date', 'Extension_Date',
  'Barrier_Limit_Date', 'Forward_Date',
];

// ---- dates --------------------------------------------------------------------------------------

const ISO = /^(\d{4})-(\d{2})-(\d{2})/;

/** A wire value read as an ISO date, or undefined. `.Timestamp` is the token the encoder writes;
 * a bare ISO string is what a hand-edited book carries. A `.DateOffset` is a PERIOD, not a date,
 * and is deliberately not read as one. */
export function dateOf(value: unknown): string | undefined {
  const stamp = token(value, '.Timestamp');
  const text = stamp !== undefined ? stamp : value;
  if (typeof text !== 'string') return undefined;
  const match = ISO.exec(text);
  return match ? match[0] : undefined;
}

/** Whole days from `from` to `to`, both ISO dates - the blotter's days-to-roll. UTC midnight on
 * both sides, so no timezone the browser happens to sit in can move a roll by a day. */
export function daysBetween(from: string, to: string): number | undefined {
  const a = ISO.exec(from);
  const b = ISO.exec(to);
  if (!a || !b) return undefined;
  const ms = Date.UTC(+b[1], +b[2] - 1, +b[3]) - Date.UTC(+a[1], +a[2] - 1, +a[3]);
  return Math.round(ms / 86400000);
}

/** The book's as-of date: the calculation's own, else the System Parameters of the market it
 * prices on - `quote_sheet`'s resolution order, because a book stamped in either place is a book
 * stamped. */
export function baseDateOf(doc: JobDoc, market: Market | null): string | undefined {
  const calc = doc.Calc ?? ({} as JobDoc['Calc']);
  const own = dateOf((calc.Calculation ?? {})['Base_Date']);
  if (own) return own;
  const system = market?.['System Parameters'];
  return isObject(system) ? dateOf(system['Base_Date']) : undefined;
}

/** Every ISO date anywhere inside a value: a scalar, a `.DateList`-shaped grid of rows, a list of
 * expiry rows (`Accumulator_ExpiryDates`, `TARF_ExpiryDates`, `Barrier_Dates`). Depth-bounded, so
 * a shape nobody anticipated costs a shallow walk rather than a stack. */
export function datesIn(value: unknown, depth = 0): string[] {
  const one = dateOf(value);
  if (one) return [one];
  if (depth >= 3) return [];
  if (Array.isArray(value)) return value.flatMap((entry) => datesIn(entry, depth + 1));
  if (isObject(value)) {
    return Object.values(value).flatMap((entry) => datesIn(entry, depth + 1));
  }
  return [];
}

// ---- field picking --------------------------------------------------------------------------------

/** A field a column shows, and - where the cell is not the field as written - a hover saying how
 * it was read. */
export type Picked = { key: string; value: unknown; title?: string };

/** The first candidate the DEAL carries with a value worth showing. Value-first: the deal decides,
 * the declaration only labels (see `describe`). */
export function pick(deal: Record<string, unknown>, candidates: string[]): Picked | undefined {
  for (const key of candidates) {
    const value = deal[key];
    if (value === undefined || value === null || value === '') continue;
    if (Array.isArray(value) && value.length === 0) continue;
    return { key, value };
  }
  return undefined;
}

/** A type's declared fields, its sections merged; undefined where the schema declares no type. */
export function fieldsOf(schema: Schema | null, object: string):
  Record<string, Descriptor> | undefined {
  const sections = schema?.Instrument.types[object];
  return sections
    && Object.assign({}, ...sections.map((name) => schema!.Instrument.sections[name]));
}

/** The declared description for a field of this type, for the cell's tooltip - so a reader can
 * always see WHICH field the blotter chose to show. Falls back to the raw key. */
export function describe(schema: Schema | null, object: string, key: string): string {
  return fieldsOf(schema, object)?.[key]?.description || key;
}

/** A row's side, amount, and what the amount is OF: a currency by the field stating it, or the
 * unit word of an amount counting the underlying. */
export type Size = { side?: Picked; amount?: Picked; of?: Picked };

/** The other of a two-sided word: of a field's two declared values, else the word with its first
 * part turned - `Up_And_In` seen from the other side of the pair is `Down_And_In`. */
const other = (value: string, values?: string[]): string => (values?.length === 2
  ? values[Number(values[0] === value)]
  : value.replace(/^[A-Za-z]+/, (word) => OTHER[word] ?? word));

/** The amount is the first candidate the type declares `sized` - any the deal carries where the
 * type is undeclared - so a netting set's balance is no trade size. It is OF the first currency
 * the deal states among: the one a `settles` declaration pays that amount in, `<Word>_Currency`
 * for a word of its name, the deal's own `Currency`; `Units` and `Volume` are of none. A type
 * marking no side holds it in its sizes: a `Buy_` or `Sell_` amount, the other way if negative,
 * the amount beside that word then its size, unsigned. Held at `units` of the record, the amount
 * is the written one times their size, and the side the other one where they are short. */
export function sizeOf(deal: Record<string, unknown>, object: string, schema: Schema | null,
                       units?: number): Size {
  const fields = fieldsOf(schema, object);
  const stated = pick(deal, SIDE_FIELDS);
  let amount = pick(deal, AMOUNT_FIELDS.filter((key) => !fields || fields[key]?.sized));
  const word = amount && /(Buy|Sell)_Amount$/.exec(amount.key)?.[1];
  let side: Picked | undefined = stated ?? (amount && word
    ? { key: amount.key, value: Number(amount.value) < 0 ? OTHER[word] : word } : undefined);
  if (!stated && amount && word) amount = { ...amount, value: Math.abs(Number(amount.value)) };
  if (units !== undefined) {
    const held = `${formatNumber(units)} held`;
    if (side && units < 0) {
      side = { ...side, value: other(String(side.value), fields?.[side.key]?.values),
               title: `${side.value} written, ${held}` };
    }
    if (amount) {
      const written = Number(amount.value);
      amount = { ...amount, value: written * Math.abs(units),
                 title: `${formatNumber(written, 'money')} written, ${held}` };
    }
  }
  // a row stating no amount still says the currency the deal is in
  if (!amount) return { side, of: deal.Currency ? { key: 'Currency', value: deal.Currency } : undefined };
  if (UNIT_FIELDS.includes(amount.key)) {
    return { side, amount, of: { key: amount.key, value: amount.key.toLowerCase() } };
  }
  const paid = Object.values(fields ?? {}).flatMap((field) => (
    Array.isArray(field.settles) ? field.settles : [])).find((cash) => cash.amount === amount.key);
  const key = [paid?.currency, ...amount.key.split('_').map((part) => `${part}_Currency`),
    'Currency'].find((name) => name && deal[name]);
  return { side, amount, of: key ? { key, value: deal[key] } : undefined };
}

/** The value behind a scalar cell: a number - a Percent's or a Basis's its own - a date, a word,
 * or undefined where the value is a shape a cell cannot state. */
export function rawOf(value: unknown): number | string | undefined {
  if (typeof value === 'number') return value;
  const scaled = token(value, '.Percent') ?? token(value, '.Basis');
  if (typeof scaled === 'number') return scaled;
  const stamp = dateOf(value);
  if (stamp) return stamp;
  if (typeof value === 'string') return value;
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  return undefined;
}

/** One cell of a scalar column, a number as the `kind` it is - a Percent or a Basis with its
 * unit - or undefined where the value is a shape the details expander carries whole. */
export function cellText(value: unknown, kind: Kind = 'plain'): string | undefined {
  const raw = rawOf(value);
  if (typeof raw !== 'number') return raw;
  return formatNumber(raw, token(value, '.Percent') !== undefined ? 'percent'
    : token(value, '.Basis') !== undefined ? 'basis' : kind);
}

// ---- the market's axis --------------------------------------------------------------------------
// The engine states an FX deal on its `Underlying_Currency` priced in its `Currency`, every level in
// `Currency` per 1 `Underlying_Currency`; the market quotes a pair, base then quote. The deal names
// its pair, so the turn between the two is read off the deal and never off a table of currencies.

/** A currency pair as the market writes it, base then quote: EURZAR is ZAR per 1 EUR. `through`
 * is the chain of surfaces a pair the market holds none of is quoted along. */
type Pair = { base: string; quote: string; through?: string[] };

/** One side of an exchange: a currency, how much of it, and the field stating that amount. */
type Leg = { currency: string; amount: number; key: string };

/** What a deal exchanges, an amount of one currency for one of another: each date its type
 * declares settling two payments in two currencies - a forward's one, a swap's near then far -
 * else its `Buy_` and `Sell_` amounts, as a type declaring no such date states them. */
function exchanges(deal: Record<string, unknown>, fields?: Record<string, Descriptor>): Leg[][] {
  const declared = Object.values(fields ?? {}).flatMap((field) => (
    Array.isArray(field.settles) && field.settles.length === 2 ? [field.settles] : []));
  return (declared.length ? declared : [['Buy', 'Sell'].map(
    (word) => ({ currency: `${word}_Currency`, amount: `${word}_Amount` }))])
    .map((cash) => cash.map(({ currency, amount }) => ({
      currency: String(deal[currency] ?? ''), amount: Math.abs(Number(deal[amount ?? ''])),
      key: amount ?? '' })))
    .filter(([one, other]) => one.currency && other.currency && one.currency !== other.currency);
}

/** The shortest chain of the market's `FXVol.BASE.QUOTE` surfaces from `base` to `quote`, each
 * the quote of the one before - a surface of the pair itself is the chain of two. */
function chain(market: Market | null, base: string, quote: string): string[] | undefined {
  const surfaces = Object.keys(market?.['Price Factors'] ?? {}).map((name) => name.split('.'))
    .filter((name) => name.length === 3 && name[0] === 'FXVol');
  const reached = new Map([[base, [base]]]);
  for (const [at, path] of reached) {
    if (at === quote) return path;
    surfaces.filter((name) => name[1] === at && !reached.has(name[2]))
      .forEach((name) => reached.set(name[2], [...path, name[2]]));
  }
  return undefined;
}

/** The pair a deal is on: the `BASE.QUOTE` its `FX_Volatility` names, else - a forward names none
 * - its two currencies in the order the market's surfaces put them: one of the pair, else a chain
 * through a third, a cross being quoted as the product of its legs - GBPUSD and USDZAR make
 * GBPZAR. Two currencies no chain orders are on no pair the market can say. */
function pairOf(deal: Record<string, unknown>, market: Market | null,
                fields?: Record<string, Descriptor>): Pair | undefined {
  const named = String(deal.FX_Volatility ?? '').split('.');
  if (named.length === 2) return { base: named[0], quote: named[1] };
  const [one, other] = exchanges(deal, fields)[0]?.map((leg) => leg.currency) ?? [];
  for (const [base, quote] of one ? [[one, other], [other, one]] : []) {
    const through = chain(market, base, quote);
    if (through) return { base, quote, through };
  }
  return undefined;
}

/** Whether a deal states its levels on its pair's QUOTE currency, so the market reads each as
 * `1 / level` and its sense the other way round - the engine's own `Invert_Moneyness`. */
const inverts = (deal: Record<string, unknown>, pair?: Pair): boolean =>
  !!pair && deal.Underlying_Currency === pair.quote;

/** Every level column's one road: a number a deal on a pair states, on the pair's axis - its
 * reciprocal where the deal `inverts` - with a hover saying what is shown and which field it was
 * read off; anything else as written. */
function onAxis(deal: Record<string, unknown>, picked: Picked | undefined,
                pair?: Pair): Picked | undefined {
  if (!picked || !pair || typeof picked.value !== 'number') return picked;
  const value = inverts(deal, pair) ? 1 / picked.value : picked.value;
  return { key: picked.key, value, title: `${pair.base}${pair.quote} ${formatNumber(value)}, `
    + `written as ${picked.key} ${formatNumber(picked.value)} ${deal.Currency} per `
    + `${deal.Underlying_Currency}` };
}

/** A forward's outright on its pair, one per exchange - a swap's near then far: the quote
 * currency's amount per the base's. */
function outrights(deal: Record<string, unknown>, pair: Pair | undefined,
                   fields?: Record<string, Descriptor>): Picked[] {
  return !pair ? [] : exchanges(deal, fields).flatMap((legs) => {
    const [over, under] = [pair.quote, pair.base].map(
      (currency) => legs.find((leg) => leg.currency === currency));
    const value = over && under ? over.amount / under.amount : NaN;
    return over && under && isFinite(value) ? [{
      key: over.key, value, title: `${pair.base}${pair.quote} ${formatNumber(value)}, the `
        + `outright: ${formatNumber(over.amount, 'money')} ${pair.quote} for `
        + `${formatNumber(under.amount, 'money')} ${pair.base}` }] : [];
  });
}

/** A row's terms, each a few words, and the hover saying them whole. */
export type Terms = { parts: string[]; title: string };

/** The terms a row's other columns cannot hold, its hover adding the levels as written: a strip's
 * fixings and that its amount is per fixing, an average's fixings, a target in its unit, a
 * leverage, a range's two levels and a barrier's direction - the last two on the pair's axis - and
 * the outright of every `near` exchange before the one the strike column shows. A type nobody
 * anticipated says nothing rather than something wrong. */
function termsOf(deal: Record<string, unknown>, pair?: Pair, near: Picked[] = []): Terms | undefined {
  const count = (keys: string[]) => {
    const rows = pick(deal, keys)?.value;
    return Array.isArray(rows) ? rows.length : undefined;
  };
  const [strip, average] = [count(STRIP_FIELDS), count(AVERAGE_FIELDS)];
  const [per, of] = deal.InvertedTarget ? [deal.Underlying_Currency, deal.Currency]
    : [deal.Currency, deal.Underlying_Currency];
  const leverage = Number(deal.LeverageNotional) / Number(deal.Underlying_Amount);
  const range = ['Lower_Barrier', 'Upper_Barrier'].map(
    (key) => onAxis(deal, pick(deal, [key]), pair));
  const [low, high] = range.map((level) => level?.value).sort((a, b) => Number(a) - Number(b));
  const parts = [
    strip !== undefined && `${strip} fixings, amount per fixing`,
    average !== undefined && `average of ${average} fixings`,
    typeof deal.TargetLevel === 'number'
      && `target ${formatNumber(deal.TargetLevel)} ${per} per ${of}`,
    leverage > 0 && isFinite(leverage) && `leverage ${formatNumber(leverage)}×`,
    typeof low === 'number' && typeof high === 'number'
      && `range ${formatNumber(low)}\u00a0–\u00a0${formatNumber(high)}`,
    typeof deal.Barrier_Type === 'string' && (inverts(deal, pair) ? other(deal.Barrier_Type)
      : deal.Barrier_Type).replace(/_/g, ' ').toLowerCase(),
    ...near.map((rate) => `near ${formatNumber(Number(rate.value))}`),
  ].filter((part): part is string => !!part);
  const written = [...range, ...near].map((level) => level?.title).filter(Boolean);
  return parts.length ? { parts, title: [parts.join(' · '), ...written].join('\n') } : undefined;
}

/** A row the way the market states it: what it is on - the pair as the market writes it, else the
 * commodity, equity or index it names, else the one thing every row under it is on - an option's
 * sense on that pair, its two level columns - a swap's strike its far outright - and its terms. */
export type Quoted = {
  on?: Picked; sense?: string; strike?: Picked; barrier?: Picked; terms?: Terms;
};

/** The pair's cell: the surface it was read off, and where a chain ordered it, the legs it is
 * quoted as. */
const pairCell = (deal: Record<string, unknown>, pair: Pair): Picked => {
  const legs = (pair.through ?? []).slice(1).map((quote, at) => [pair.through![at], quote]);
  return {
    key: deal.FX_Volatility ? 'FX_Volatility'
      : legs.map(([base, quote]) => `FXVol.${base}.${quote}`).join(' × '),
    value: pair.base + pair.quote,
    title: legs.length > 1 ? `the market holds no ${pair.base}${pair.quote} surface, so the pair `
      + `reads as its legs are quoted: ${legs.map((leg) => leg.join('')).join(' × ')}` : undefined,
  };
};

export function quoted(row: BlotterRow, market: Market | null, schema: Schema | null = null): Quoted {
  const { deal } = row;
  const fields = fieldsOf(schema, row.object);
  const pair = pairOf(deal, market, fields);
  const legs = row.children.map((child) => quoted(child, market, schema).on);
  const named = pick(deal, UNDERLYING_FIELDS);
  const sense = typeof deal.Option_Type === 'string' ? deal.Option_Type : undefined;
  const rates = outrights(deal, pair, fields);
  return {
    on: pair ? pairCell(deal, pair)
      : named ?? (new Set(legs.map((leg) => leg?.value)).size === 1 ? legs[0] : undefined),
    sense: sense && inverts(deal, pair) ? other(sense) : sense,
    strike: onAxis(deal, pick(deal, STRIKE_FIELDS), pair) ?? rates[rates.length - 1],
    // a barrier of nought is no barrier
    barrier: onAxis(deal, pick(deal, BARRIER_FIELDS.filter((key) => deal[key] !== 0)), pair),
    terms: termsOf(deal, pair, rates.slice(0, -1)),
  };
}

// ---- the rows ---------------------------------------------------------------------------------

export type BlotterRow = {
  /** The row's key: the positional `deal_path` ('0/2/1') - the identity the service, the tree
   * and the MCP booking verbs all use - or, grouped by the record, a folder's or a position's. */
  path: string;
  /** What a click on the row picks: the deal path, the position it belongs to, or nothing for a
   * folder. */
  target: string | null;
  /** The position the row IS, where the book is grouped by the record. */
  position?: Position;
  depth: number;
  object: string;
  reference: string;
  container: boolean;
  ignored: boolean;
  /** The units the record holds of the position the row sits under - its own, or the structure's
   * it is a leg of - where the desk keeps a record holding one. */
  units?: number;
  /** The deal's own term end, where it declares one. */
  expiry?: string;
  /** The first date on or after Base_Date the deal carries anywhere - the next fixing, payment or
   * expiry, whichever comes first. */
  next?: string;
  /** What the row rolls on: its own expiry, else its next date, else - for a container carrying no
   * dates of its own - the earliest roll of anything underneath it. */
  roll?: string;
  days?: number;
  deal: Record<string, unknown>;
  children: BlotterRow[];
};

/** The deal tree as blotter rows, containers still parents of their children. The roll date is
 * resolved bottom-up so a StructuredDeal that carries no dates of its own still sorts and
 * highlights by the earliest thing inside it. */
export function toRows(
  nodes: DealNode[], containers: string[], base: string | undefined, path = '', depth = 0,
  parentIgnored = false,
): BlotterRow[] {
  return nodes.map((node, position) => {
    const id = path ? `${path}/${position}` : String(position);
    const deal = (node.Instrument?.['.Deal'] ?? {}) as Record<string, unknown>;
    const object = String(deal.Object ?? '?');
    // the engine skips an Ignore node WHOLE (walk_groups never descends), so a leg under an
    // ignored container is off the book however live its own flag looks
    const ignored = parentIgnored || node.Ignore === 'True';
    const children = toRows(node.Children ?? [], containers, base, id, depth + 1, ignored);

    const expiry = pick(deal, EXPIRY_FIELDS);
    const expiryDate = expiry ? dateOf(expiry.value) : undefined;
    const upcoming = base
      ? datesIn(deal).filter((d) => d >= base).sort()
      : datesIn(deal).sort();
    const next = upcoming[0];

    const own = expiryDate ?? next;
    const inherited = children
      .map((child) => child.roll)
      .filter((d): d is string => !!d)
      .sort()[0];
    const roll = own ?? inherited;

    return {
      path: id,
      target: id,
      depth,
      object,
      reference: String(deal.Reference ?? ''),
      container: containers.includes(object) || children.length > 0,
      ignored,
      expiry: expiryDate,
      next,
      roll,
      days: base && roll ? daysBetween(base, roll) : undefined,
      deal,
      children,
    };
  });
}

/** One deal as a blotter row of its own, nothing above or under it. */
export const dealRow = (deal: Record<string, unknown>): BlotterRow =>
  toRows([{ Instrument: { '.Deal': deal } }], [], undefined)[0];

/** A row's name as it reads under its structure's row: what follows the `Structure_Reference` a
 * leg's own reference begins with - its role - and any other reference whole. */
export function nameOf(row: BlotterRow): string {
  const structure = row.deal.Structure_Reference;
  const role = typeof structure === 'string' && structure && row.reference.startsWith(structure)
    ? row.reference.slice(structure.length).replace(/^[\W_]+/, '') : '';
  return role || row.reference;
}

/** Every row by its deal path, legs included - what a position's row is read off. */
export function rowsByPath(rows: BlotterRow[],
                           index = new Map<string, BlotterRow>()): Map<string, BlotterRow> {
  for (const row of rows) {
    index.set(row.path, row);
    rowsByPath(row.children, index);
  }
  return index;
}

/** The file's tree holding what the record does: a node carrying positions at their net, as the
 * book prices it, every leg under it the same, and a node carrying none as written. */
export function heldRows(rows: BlotterRow[], held: Map<string, Position[]>,
                         units?: number): BlotterRow[] {
  return rows.map((row) => {
    const at = netOf(held.get(row.path) ?? []) ?? units;
    return { ...row, units: at, children: heldRows(row.children, held, at) };
  });
}

/** The earliest roll among `rows` - what a row holding them inherits. */
const earliest = (rows: BlotterRow[]): string | undefined =>
  rows.map((row) => row.roll).filter((d): d is string => !!d).sort()[0];

/** A tree of the record's grouping as blotter rows. A folder is a row carrying the earliest roll
 * beneath it; a position is the file's own row at the first node holding it, RE-KEYED under the
 * position - two positions can hold one node - with every row of it, legs included, picking the
 * position; and a position the file does not hold is a row saying so. */
export function groupedRows(nodes: TreeNode[], rows: Map<string, BlotterRow>,
                            positions: Map<string, Position>, base: string | undefined,
                            depth = 0): BlotterRow[] {
  return nodes.map((node) => {
    if (node.group) {
      const children = groupedRows(node.children ?? [], rows, positions, base, depth + 1);
      const roll = earliest(children);
      return {
        path: node.id, target: null, depth, object: '', reference: node.label, container: true,
        ignored: false, roll, days: base && roll ? daysBetween(base, roll) : undefined, deal: {},
        children,
      };
    }
    const position = positions.get(node.id);
    const held = position?.deal_paths.length ? rows.get(position.deal_paths[0]) : undefined;
    if (!held) {
      return {
        path: node.id, target: node.id, depth, object: 'not in the file', reference: node.label,
        container: false, ignored: false, deal: {}, children: [], position,
      };
    }
    const keyed = (row: BlotterRow, at: number): BlotterRow => ({
      ...row, path: node.id + row.path.slice(held.path.length), target: node.id, depth: at,
      units: position?.quantity, children: row.children.map((child) => keyed(child, at + 1)),
    });
    return { ...keyed(held, depth), position };
  });
}

export type Window = { id: string; label: string; days: number | null };

export const WINDOWS: Window[] = [
  { id: 'week', label: 'This week', days: 7 },
  { id: 'month', label: '30 days', days: 30 },
  { id: 'all', label: 'All', days: null },
];

/** Inside the roll window: dated, not already past, and within the horizon. A row with no date is
 * never "rolling off" - it is simply unknown, and says so by staying quiet. */
export function inWindow(row: BlotterRow, horizon: number | null): boolean {
  if (horizon === null) return true;
  return row.days !== undefined && row.days >= 0 && row.days <= horizon;
}

/** Whether a row rolls inside the window on its OWN account - its own expiry or next date. A row
 * holding others with no date of its own - a netting set, a folder of the record's grouping -
 * rolls only through what it holds, and an ignored row never rolls: the engine does not price it. */
const rollsOwn = (row: BlotterRow, horizon: number | null): boolean =>
  !row.ignored && (row.expiry ?? row.next) !== undefined && inWindow(row, horizon);

/** The window filter over the tree: a row rolling on its own account survives whole - a structure
 * never loses the leg that is rolling - and a row holding others survives with only those of them
 * that do, so a netting set with one trade rolling this week shows that trade. */
export function filterRows(rows: BlotterRow[], horizon: number | null): BlotterRow[] {
  if (horizon === null) return rows;
  return rows.flatMap((row) => {
    const children = filterRows(row.children, horizon);
    const rolling = rollsOwn(row, horizon);
    if (!rolling && children.length === 0) return [];
    return [{ ...row, children: rolling ? row.children : children }];
  });
}

/** The type that holds a client's trades rather than being one. */
export const SET = 'NettingCollateralSet';

/** Whether a row holds trades rather than being one: a netting set, or a folder of the record's
 * grouping, which picks nothing. */
export const holdsTrades = (row: BlotterRow): boolean => row.target === null || row.object === SET;

const rollsWithin = (row: BlotterRow, horizon: number | null): boolean =>
  rollsOwn(row, horizon) || row.children.some((child) => rollsWithin(child, horizon));

/** How many trades roll inside the window: a trade anything in which rolls is one - a structure is
 * one trade and its legs its parts - a closed one none, and a row holding trades counts what rolls
 * among them and never itself. */
export function rollingCount(rows: BlotterRow[], horizon: number | null): number {
  return rows.reduce((sum, row) => sum + (holdsTrades(row) ? rollingCount(row.children, horizon)
    : Number(!closed(row.units) && rollsWithin(row, horizon))), 0);
}

/** Sorted WITHIN the tree, never across it: siblings order by days-to-roll (undated last), and a
 * container keeps its own children. The blotter is a tree read as a list, and a sort that broke
 * the grouping would be a different screen. */
export function sortRows(rows: BlotterRow[], key: 'days' | 'reference'): BlotterRow[] {
  const sorted = [...rows].sort((a, b) => {
    if (key === 'reference') return a.reference.localeCompare(b.reference);
    if (a.days === undefined) return b.days === undefined ? 0 : 1;
    if (b.days === undefined) return -1;
    return a.days - b.days;
  });
  return sorted.map((row) => ({ ...row, children: sortRows(row.children, key) }));
}

/** The tree as the flat run of rows the table paints, collapsed subtrees omitted. */
export function flatten(rows: BlotterRow[], collapsed: Set<string>): BlotterRow[] {
  return rows.flatMap((row) => (
    collapsed.has(row.path) ? [row] : [row, ...flatten(row.children, collapsed)]
  ));
}

// ---- the export ---------------------------------------------------------------------------------

/** The blotter's columns in the order the table shows them, and the record's four after them. */
export const HEADS = ['Reference', 'Type', 'Desk side', 'On', 'Ccy', 'Amount', 'Strike', 'Barrier',
  'Terms', 'Expiry', 'Next date', 'Days'];
export const RECORD_HEADS = ['Qty', 'Portfolio', 'Client', 'Status'];

/** A row as the export writes it, a value per column of `HEADS`: every number the one behind its
 * cell - an amount under a position the one held, a level on the pair's axis - never the text the
 * cell formats it to, and the reference whole, a spreadsheet having no tree to read a role under. */
export function exported(row: BlotterRow, market: Market | null, schema: Schema | null): unknown[] {
  const size = sizeOf(row.deal, row.object, schema, row.units);
  const read = quoted(row, market, schema);
  return [row.reference, row.object, size.side?.value,
    [read.on?.value, read.sense].filter(Boolean).join(' '), size.of?.value, size.amount?.value,
    read.strike?.value, read.barrier?.value, read.terms?.parts.join(' · '), row.expiry,
    row.next !== row.expiry ? row.next : undefined, row.days].map(rawOf);
}

/** Rows of values as CSV: a number as itself, anything else as its text - quoted where it holds a
 * comma, a quote or a line break - and nothing as an empty field. */
export function csvText(rows: unknown[][]): string {
  const field = (value: unknown) => {
    const text = value === undefined || value === null ? '' : String(value);
    return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  };
  return rows.map((row) => row.map(field).join(',')).join('\r\n') + '\r\n';
}
