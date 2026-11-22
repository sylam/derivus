// One trade as a desk reads it - pure, free of React and the network, the way `blotter.ts` and
// `quote.ts` are: which node of the file a path's trade is, what it is worth, what happened to it
// on the record, and what an unwind of it files and leaves.

import {
  SET, cellText, currencyFor, dateOf, fieldsOf, quoted, sizeOf, type BlotterRow, type Picked,
  type Size,
} from './blotter';
import { instantMs, plural } from './desk';
import { type Agreement, type Position } from './positions';
import { salesNames } from './quote';
import {
  COLLATERAL, currencyOf, fieldLabel, formatNumber, setCurrency, uncollateralised,
} from './tokens';
import type {
  ActivityPage, ActivityRow, BookRisk, DealMark, DealNode, Descriptor, Market, Moved, PositionMark,
  QuoteGiven, Schema,
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

/** In whose favour a mark of `value` to the desk stands. */
export const favour = (value: number): string =>
  (value > 0 ? "in the desk's favour" : value < 0 ? "in the client's favour" : 'flat');

/** A mark as the desk says it: the amount in its currency, and in whose favour it stands. */
export const markText = (value: number, currency: string | null): string =>
  `${formatNumber(value, 'money')}${currency ? ` ${currency}` : ''}, ${favour(value)}`;

/** A `position`'s own row of the risk answer's `per_position` at the deal `path`. */
export const positionMark = (risk: { data: BookRisk | null }, path: string, position: Position):
  PositionMark | undefined => risk.data?.per_position.find((row) => row.deal_path === path
    && row.portfolio === position.portfolio && row.agreement === position.agreement);

/** The mark a row shows off the risk as its screen shows it (`bookRisk`): the deal's own at `path`
 * (`markOf`), and for one `position` held there its own (`positionMark`). */
export const markShown = (risk: { marks: DealMark[]; data: BookRisk | null }, path: string,
                          position?: Position): number | undefined => (position
  ? positionMark(risk, path, position)?.value ?? undefined : markOf(risk.marks, path).own?.value);

/** Why a trade's page reads no mark of the trade's `own` off the risk as its screen shows it
 * (`bookRisk`), null where it reads one: the line holding the seat from the read or the service's
 * refusal, that the book is pricing where nothing has answered, and else `none`, that the answer
 * holds no row of its own - the book marking a trade the record `shut` at nothing. */
export function noMark(risk: { data: unknown; error: string | null }, own?: DealMark, shut = false,
                       none = 'The risk answer holds no mark of this trade\'s own.'): string | null {
  if (risk.error) return risk.error;
  if (!risk.data) return 'pricing the book…';
  return own ? null : shut ? 'Closed: the record holds none of it, so the book marks it at nothing.'
    : none;
}

/** Every row a read of the record answers - the whole record, or one trade's rows - a page at a
 * time from its first event: each page asked from the cursor the one before answered, until one
 * comes back short of `size`. */
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

/** What happened to a trade, newest first, off the rows the record answers for it. */
export const historyOf = (rows: ActivityRow[]): ActivityRow[] =>
  [...rows].sort((a, b) => b.lsn - a.lsn);

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
             words: fieldLabel(moved.field, declared).text };
  });
}

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

/** What an unwind's price is set against, in a line: the desk's mark of the `units` unwound - the
 * position's `unit` mark (`positionMark`) times them, from the side `held` - in its `currency` and
 * in whose favour it stands (`markText`) - and, once a `price` is typed in that currency, what the
 * desk makes at it over that mark - the cash it receives less the worth it parts with - or gives
 * up against it. Undefined where no units are. */
export function unwindAgainst(held: number, units: number, unit: number, currency: string,
                              price?: Price): string | undefined {
  if (!units) return undefined;
  const worth = unit * Math.sign(held) * units;
  const money = (value: number) => `${formatNumber(Math.abs(value), 'money')} ${currency}`.trim();
  const makes = price && unwindOf(held, units, price.amount).receives - worth;
  return `The desk marks the ${formatNumber(units)} unwound at ${markText(worth, currency)}${
    makes === undefined ? '.' : price!.currency !== currency
      ? `: a price in ${price!.currency || 'no currency'} is not set against it.`
      : makes === 0 ? ', and this price is that mark.'
        : `, so at this price it ${makes > 0 ? 'makes' : 'gives up'} ${money(makes)} against `
          + 'its mark.'}`;
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
  const at = instantMs(since);
  if (at === undefined) return since ?? '';
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
  const currency = setCurrency(paper.terms);
  return `${set} (${paper.kind})${!(COLLATERAL in fields) ? ''
    : uncollateralised(fields, paper.terms) ? ', uncollateralised'
      : `, collateralised${currency ? ` in ${currency}` : ''}`}`;
}
