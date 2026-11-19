// One trade as a desk reads it - pure, free of React and the network, the way `blotter.ts` and
// `quote.ts` are: which node of the file a path's trade is, what it is worth, what happened to it
// on the record, and what an unwind of it files and leaves.

import { SET, UNIT_FIELDS, dateOf, sizeOf, type BlotterRow, type Size } from './blotter';
import { formatNumber } from './tokens';
import type { ActivityPage, ActivityRow, DealMark, DealNode, Schema } from './types';

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

/** A trade's mark in the risk answer: its own row where it is a top-level node, else the
 * top-level row it sits inside - the only mark that holds it, as part of a net. */
export function markOf(marks: DealMark[], path: string): { own?: DealMark; inside?: DealMark } {
  const own = marks.find((row) => row.deal_path === path);
  return own ? { own } : { inside: marks.find((row) => row.deal_path === path.split('/')[0]) };
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

/** What an unwind of `units` of a position holding `held` files: the fill's signed quantity,
 * taking the position toward nothing, what is left, and the cash the desk receives at `price`
 * per unit - the consideration from its own side, negative where it pays. */
export function unwindOf(held: number, units: number, price: number):
  { quantity: number; left: number; receives: number } {
  const quantity = -Math.sign(held) * units;
  return { quantity, left: held + quantity, receives: -quantity * price };
}

/** What `units` of a trade of `size` deal, for the desk's tiers to route its ticket on: that many
 * of its size, unsigned, in the currency it is of - nothing where the size counts the underlying
 * in its own units, or the trade states none. */
export function notionalOf(size: Size, units: number):
  { notional?: number; notional_currency?: unknown } {
  const { amount, of } = size;
  return amount && typeof amount.value === 'number' && of && !UNIT_FIELDS.includes(amount.key)
    ? { notional: Math.abs(units * amount.value), notional_currency: of.value } : {};
}

/** The confirm an unwind asks: how much of what is held, at what price, the cash that moves and
 * what is left - in the trade's own size where it states one. */
export function unwindWords(held: number, units: number, price: number, currency: string,
                            size: Size): string {
  const { left, receives } = unwindOf(held, units, price);
  const amount = size.amount?.value;
  const told = (count: number, short = count < 0) => `${formatNumber(Math.abs(count))}${
    typeof amount === 'number' ? ` (${[formatNumber(Math.abs(count) * amount, 'money'),
      size.of?.value].filter(Boolean).join(' ')})` : ''}${short ? ' short' : ''}`;
  const cash = receives === 0 ? 'no cash moves'
    : `the desk ${receives > 0 ? 'receives' : 'pays'} ${formatNumber(Math.abs(receives), 'money')}`
      + ` ${currency}`;
  return `Unwind ${told(units, false)} of the ${told(held)} held, at `
    + `${formatNumber(price, 'money')} ${currency} a unit: ${cash}, and ${left === 0
      ? 'nothing is left' : `${told(left)} is left`}.`;
}
