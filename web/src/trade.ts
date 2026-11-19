// One trade as a desk reads it - pure, free of React and the network, the way `blotter.ts` and
// `quote.ts` are: which node of the file a path's trade is, what it is worth, what happened to it
// on the record, and what an unwind of it files and leaves.

import {
  SET, cellText, currencyFor, dateOf, fieldsOf, quoted, sizeOf, type BlotterRow, type Picked,
  type Size,
} from './blotter';
import { plural } from './desk';
import type { Agreement, Position } from './positions';
import { salesNames } from './quote';
import { COLLATERAL, currencyOf, formatNumber, uncollateralised } from './tokens';
import type {
  ActivityPage, ActivityRow, DealMark, DealNode, Descriptor, Market, Moved, QuoteGiven, Schema,
  WorkRow,
} from './types';

/** The trade a deal path names: the outermost node on the path that is no netting set - a leg
 * naming its structure - beside the reference of the set it sits in, null at the book's root.
 * Undefined where the path is a set or names nothing. */
export function tradeAt(nodes: DealNode[], path: string):
  { path: string; set: string | null } | undefined {
  let level = nodes;
  let at = '';
  let set: string | null = null;
  for (const step of path.split('/')) {
    const node = level[Number(step)];
    if (!node) return undefined;
    at = at ? `${at}/${step}` : step;
    const deal = node.Instrument['.Deal'];
    if (deal.Object !== SET) return { path: at, set };
    set = String(deal.Reference ?? '');
    level = node.Children ?? [];
  }
  return undefined;
}

/** A trade's mark among the risk answer's rows: its own where they hold one - a top-level node's,
 * or a trade's a top-level set holds - and, below the top, the top-level row it sits inside, whose
 * net holds it. */
export function markOf(marks: DealMark[], path: string): { own?: DealMark; inside?: DealMark } {
  const top = path.split('/')[0];
  return { own: marks.find((row) => row.deal_path === path),
           inside: top === path ? undefined : marks.find((row) => row.deal_path === top) };
}

/** Every row of the record, read a page at a time from its first event: each page asked from the
 * cursor the one before answered, until one comes back short of `size`. */
export async function wholeRecord(read: (since: number) => Promise<ActivityPage>,
                                  size = 200): Promise<ActivityRow[]> {
  const rows: ActivityRow[] = [];
  for (let since = 0; ;) {
    const page = await read(since);
    rows.push(...page.rows);
    if (page.rows.length < size) return rows;
    since = page.lsn;
  }
}

/** What happened to the trade at `path`, newest first: every row of the record that opens it. */
export const historyOf = (rows: ActivityRow[], path: string): ActivityRow[] =>
  rows.filter((row) => row.deal_path === path).sort((a, b) => b.lsn - a.lsn);

/** A restrike's moved fields as the deal's terms read them: each in its declared words - its name
 * where the description is a paragraph, as the terms panel labels it - beside its declaration, and
 * the currency it is in: an amount's by the size's rule, else the one its description names. */
export function movedTerms(changed: Moved[], deal: Record<string, unknown>,
                           schema: Schema | null):
  (Moved & { words: string; declared?: Descriptor; unit?: string })[] {
  const fields = fieldsOf(schema, String(deal.Object ?? '')) ?? {};
  return changed.map((moved) => {
    const declared = fields[moved.field];
    const description = declared?.description ?? moved.field;
    const unit = declared?.sized ? currencyFor(deal, moved.field, fields)?.value
      : currencyOf(description, Object.keys(fields), (key) => deal[key]);
    return { ...moved, declared, unit: unit === undefined ? undefined : String(unit),
             words: description.length > 64 ? moved.field.replace(/_/g, ' ') : description };
  });
}

/** Who booked the trade and when: its first fill on the record. A restrike signs the position's
 * ticket again, so the ticket names the last seat to touch it and never the first. */
export const bookedBy = (history: ActivityRow[]): ActivityRow | undefined =>
  history.filter((row) => row.event_type === 'fill').sort((a, b) => a.lsn - b.lsn)[0];

/** The rows a trade's terms read as: its legs, or the deal itself where it has none. */
export const legsOf = (row: BlotterRow): BlotterRow[] => (row.children.length ? row.children
  : [row]);

/** The trade's size as written: its own, else the one size every leg states - a collar's two
 * options on one notional - else none. */
export function tradeSize(row: BlotterRow, schema: Schema | null): Size {
  const own = sizeOf(row.deal, row.object, schema);
  if (own.amount || !row.children.length) return own;
  const legs = row.children.map((leg) => sizeOf(leg.deal, leg.object, schema));
  const said = (size: Size) => `${size.amount?.value} ${size.of?.value}`;
  return legs.every((leg) => leg.amount && said(leg) === said(legs[0]))
    ? { amount: legs[0].amount, of: legs[0].of } : {};
}

/** The trade's dates: every date a field of it or a leg states whole, each field once where
 * every leg agrees on it, in date order. */
export function datesOf(row: BlotterRow): { key: string; date: string }[] {
  const seen = new Map<string, { key: string; date: string }>();
  for (const deal of [row, ...row.children].map((one) => one.deal)) {
    for (const [key, value] of Object.entries(deal)) {
      const date = dateOf(value);
      if (date) seen.set(`${key} ${date}`, { key, date });
    }
  }
  return [...seen.values()].sort((a, b) => a.date.localeCompare(b.date)
    || a.key.localeCompare(b.key));
}

/** What one unit of a trade changes hands at: an amount in a currency, and the day it settles. */
export type Price = { amount: number; currency: string; day?: string };

/** What an unwind of `units` of a position holding `held` does: the fill's signed quantity,
 * taking the position toward nothing, what is left, and the cash the desk receives at `price`
 * per unit - the consideration from its own side, negative where it pays. */
export function unwindOf(held: number, units: number, price: number):
  { quantity: number; left: number; receives: number } {
  const quantity = -Math.sign(held) * units;
  return { quantity, left: held + quantity, receives: -quantity * price };
}

/** The fill an unwind posts: its signed quantity and its price a unit, settling on the price's day
 * - a price of nothing moves no cash and names no day. */
export function unwindFill(held: number, units: number, price: Price): Record<string, unknown> {
  return { quantity: unwindOf(held, units, price.amount).quantity, price: price.amount,
           price_currency: price.currency, ...(price.amount ? { value_date: price.day } : {}) };
}

/** The confirm an unwind asks: how much of what is held, at what price, who pays whom what on
 * which day, and what is left - in the trade's own size where it states one. */
export function unwindWords(held: number, units: number, size: Size, price: Price,
                            client: string): string {
  const { left, receives } = unwindOf(held, units, price.amount);
  const amount = size.amount?.value;
  const told = (count: number, short = count < 0) => `${formatNumber(Math.abs(count))}${
    typeof amount === 'number' ? ` (${[formatNumber(Math.abs(count) * amount, 'money'),
      size.of?.value].filter(Boolean).join(' ')})` : ''}${short ? ' short' : ''}`;
  const cash = `${formatNumber(Math.abs(receives), 'money')} ${price.currency}${
    price.day ? ` on ${price.day}` : ''}`;
  return `Unwind ${told(units, false)} of the ${told(held)} held, at `
    + `${formatNumber(price.amount, 'money')} ${price.currency} a unit: ${receives === 0
      ? 'no cash moves' : receives > 0 ? `${client} pays the desk ${cash}`
        : `the desk pays ${client} ${cash}`}, and ${left === 0 ? 'nothing is left'
      : `${told(left)} is left`}.`;
}

// ---- a ticket to sign ---------------------------------------------------------------------------

/** What a trade is in the market's words: a structure booked from a quote by its sales name and the
 * form it was dealt in, else its type - a structure's with how many legs it has. */
export function tradeName(row: BlotterRow, quote: Pick<QuoteGiven, 'structure' | 'variation'>
                          | undefined, schema: Schema | null): string {
  if (!quote) return row.children.length ? `${row.object} of ${row.children.length} legs` : row.object;
  const declared = schema?.Structure?.types[quote.structure];
  return `${declared ? salesNames(declared)[0] : quote.structure}${
    quote.variation ? ` · ${quote.variation}` : ''}`;
}

/** A side as the desk says it of itself: it buys, it sells, else it is the side its field names. */
const sideText = (side?: Picked): string | undefined => side && `desk ${
  { Buy: 'buys', Sell: 'sells' }[String(side.value)] ?? `is ${String(side.value).toLowerCase()}`}`;

/** A ticket in one line a desk reads aloud: what it is, the desk's side and its size in the
 * currency it is of, what it is on, its levels on the market's axis - a structure's leg by leg with
 * the desk's side of each - its terms, and the day it ends. */
export function ticketWords(row: BlotterRow, name: string, market: Market | null,
                            schema: Schema | null): string {
  const levels = (one: BlotterRow, legged: boolean) => {
    const read = quoted(one, market, schema);
    return [legged && sideText(sizeOf(one.deal, one.object, schema).side),
      legged && read.sense?.toLowerCase(),
      read.strike && `${legged ? '' : read.sense ? 'strike ' : 'at '}${cellText(read.strike.value)}`,
      read.barrier && `barrier ${cellText(read.barrier.value)}`].filter(Boolean).join(' ');
  };
  const read = quoted(row, market, schema);
  const size = tradeSize(row, schema);
  const end = row.expiry ?? row.roll;
  return [name,
    [sideText(size.side),
      size.amount && `${cellText(size.amount.value, 'money')} ${size.of?.value ?? ''}`.trim()]
      .filter(Boolean).join(' '),
    read.on && `on ${read.on.value}${read.sense ? ` ${read.sense.toLowerCase()}` : ''}`,
    row.children.length ? row.children.map((leg) => levels(leg, true)).join(', ')
      : levels(row, false),
    read.terms?.parts.join(', '), end && `expires ${end}`].filter(Boolean).join(' · ');
}

/** How long a ticket has waited for its signature at `now`, in words: minutes within the hour, then
 * hours on the day it was booked, then since yesterday, then days - every count floored, and the
 * days this browser's calendar ones. */
export function waitedText(since: string | null, now: number): string {
  const at = Date.parse(since ?? '');
  if (!isFinite(at)) return since ?? '';
  const minutes = Math.floor((now - at) / 60_000);
  const day = (ms: number) => {
    const date = new Date(ms);
    return Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()) / 86_400_000;
  };
  const days = Math.round(day(now) - day(at));
  if (minutes < 1) return 'under a minute';
  if (minutes < 60) return plural(minutes, 'minute');
  if (days < 1) return plural(Math.floor(minutes / 60), 'hour');
  return days === 1 ? 'since yesterday' : plural(days, 'day');
}

/** The paper a ticket books under in a line: the agreement and the document it is, and whether its
 * own terms hold collateral - the set's flag, at its declared convention where they state none - in
 * the currency its balance is kept in; an agreement the record never declared says so. */
export function paperText(set: string, agreements: Agreement[], schema: Schema | null): string {
  const paper = agreements.find((row) => row.agreement === set);
  if (!paper) return `${set}, which the record declares no agreement for`;
  const fields = fieldsOf(schema, SET) ?? {};
  const currency = [paper.terms.Balance_Currency, paper.terms.Agreement_Currency].find(Boolean);
  return `${set} (${paper.kind})${!(COLLATERAL in fields) ? ''
    : uncollateralised(fields, paper.terms) ? ', uncollateralised'
      : `, collateralised${currency ? ` in ${currency}` : ''}`}`;
}

/** The tickets awaiting a second seat that `listed` - the seat's own worklist - leaves out, oldest
 * first and each once, though a restrike books several positions under one, as rows of the
 * worklist's shape - the tier each fell in and what it waits on among them: what a seat that cannot
 * sign them reads of what waits. */
export function waitingElsewhere(positions: Position[], listed: WorkRow[]): WorkRow[] {
  const seen = new Set(listed.map((row) => row.key));
  const rows: WorkRow[] = [];
  for (const position of positions) {
    for (const ticket of position.tickets) {
      if (ticket.status !== 'pending' || seen.has(ticket.ticket)) continue;
      seen.add(ticket.ticket);
      rows.push({ kind: 'pending', what: '', key: ticket.ticket, lsn: ticket.lsn, since: ticket.at,
                  booker: ticket.actor, reference: position.reference, agreement: position.agreement,
                  deal_path: position.deal_paths[0] ?? null, client: position.counterparty,
                  tier: ticket.tier, waits_on: ticket.waits_on });
    }
  }
  return rows.sort((a, b) => a.lsn! - b.lsn!);
}

/** What holds for the seat reading a pending ticket, or null where it signs it: no seat named; the
 * booker's own, whose verdict four eyes do not count; one the seat's worklist does not list. */
export function holdsFor(row: WorkRow, seat: string | null, listed: boolean): string | null {
  if (seat === null) return 'No seat is set: type yours in the seat box at the top right to sign it.';
  if (row.booker === seat) {
    return 'Booked by this seat: under four eyes its own verdict counts for nothing, so another '
      + 'seat signs it.';
  }
  return listed ? null : 'This seat does not sign it: a seat its tier wants signs it from its own '
    + 'Back Office.';
}
