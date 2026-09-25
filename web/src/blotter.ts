// The blotter's arithmetic, kept out of the component: the deal TREE flattened to rows keyed by
// the same positional `deal_path` every other client uses, and each row's roll date measured
// against the book's own Base_Date. Nothing here touches React, so the whole reading of a book -
// which deal rolls when - is one pure function of (document, schema).
//
// The column choices are VALUE-FIRST, the same discipline `FieldView` holds: a candidate field
// wins because the deal carries it, and the schema's declaration supplies the label and the
// details block. A type nobody anticipated therefore still fills what columns it can and renders
// the rest in its expander, rather than raising.

import type { Position } from './positions';
import { formatNumber, isObject, token } from './tokens';
import type { TreeNode } from './tree';
import type { DealNode, JobDoc, Schema } from './types';

// ---- the candidate lists: one per column, most specific first ----------------------------------
// Read against the deal's own keys. Every name below is declared by at least one instrument in
// `mapping['Instrument']`; the order is the desk's reading order, not the schema's.

const SIDE_FIELDS = ['Buy_Sell', 'Payer_Receiver', 'Borrower_Lender', 'MtM_Side'];

const CURRENCY_FIELDS = [
  'Currency', 'Payoff_Currency', 'Settlement_Currency', 'Underlying_Currency', 'Buy_Currency',
  'Pay_Currency', 'Agreement_Currency', 'Balance_Currency', 'Equity_Currency', 'Rate_Currency',
];

const AMOUNT_FIELDS = [
  'Principal', 'Underlying_Amount', 'Amount', 'Buy_Amount', 'Near_Buy_Amount', 'Sell_Amount',
  'Units', 'Volume', 'Settlement_Amount', 'Cash_Payoff', 'LeverageNotional', 'Opening_Balance',
];

const STRIKE_FIELDS = [
  'Strike_Price', 'Strike', 'Fixed_Price', 'Forward_Price', 'Swap_Rate', 'FRA_Rate', 'Cap_Rate',
  'Floor_Rate', 'Interest_Rate', 'Extension_Strike', 'TargetLevel',
];

const BARRIER_FIELDS = ['Barrier_Price', 'Barrier'];

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

/** The book's as-of date: the calculation's own, else System Parameters' - `quote_sheet`'s
 * resolution order, because a book stamped in either place is a book stamped. */
export function baseDateOf(doc: JobDoc): string | undefined {
  const calc = doc.Calc ?? ({} as JobDoc['Calc']);
  const own = dateOf((calc.Calculation ?? {})['Base_Date']);
  if (own) return own;
  const system = (calc.MergeMarketData?.ExplicitMarketData ?? {})['System Parameters'];
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

export type Picked = { key: string; value: unknown };

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

/** The declared description for a field of this type, for the cell's tooltip - so a reader can
 * always see WHICH field the blotter chose to show. Falls back to the raw key. */
export function describe(schema: Schema | null, object: string, key: string): string {
  for (const section of schema?.Instrument.types[object] ?? []) {
    const descriptor = schema?.Instrument.sections[section]?.[key];
    if (descriptor) return descriptor.description || key;
  }
  return key;
}

/** One cell of a scalar column, or undefined where the value is a shape a cell cannot state -
 * the details expander carries those whole. */
export function cellText(value: unknown): string | undefined {
  if (typeof value === 'number') return formatNumber(value);
  const percent = token(value, '.Percent');
  if (typeof percent === 'number') return `${formatNumber(percent)} %`;
  const basis = token(value, '.Basis');
  if (typeof basis === 'number') return `${formatNumber(basis)} bp`;
  const stamp = dateOf(value);
  if (stamp) return stamp;
  if (typeof value === 'string') return value;
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  return undefined;
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
  side?: Picked;
  currency?: Picked;
  amount?: Picked;
  strike?: Picked;
  barrier?: Picked;
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
      side: pick(deal, SIDE_FIELDS),
      currency: pick(deal, CURRENCY_FIELDS),
      amount: pick(deal, AMOUNT_FIELDS),
      strike: pick(deal, STRIKE_FIELDS),
      barrier: pick(deal, BARRIER_FIELDS),
      expiry: expiryDate,
      next,
      roll,
      days: base && roll ? daysBetween(base, roll) : undefined,
      deal,
      children,
    };
  });
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
      children: row.children.map((child) => keyed(child, at + 1)),
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

/** Whether a row holds trades rather than being one: a netting set, or a folder of the record's
 * grouping, which picks nothing. */
const holdsTrades = (row: BlotterRow): boolean =>
  row.target === null || row.object === 'NettingCollateralSet';

const rollsWithin = (row: BlotterRow, horizon: number | null): boolean =>
  rollsOwn(row, horizon) || row.children.some((child) => rollsWithin(child, horizon));

/** How many trades roll inside the window: a trade anything in which rolls is one - a structure is
 * one trade and its legs its parts - and a row holding trades counts what rolls among them and
 * never itself. */
export function rollingCount(rows: BlotterRow[], horizon: number | null): number {
  return rows.reduce((sum, row) => sum + (holdsTrades(row) ? rollingCount(row.children, horizon)
    : Number(rollsWithin(row, horizon))), 0);
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
