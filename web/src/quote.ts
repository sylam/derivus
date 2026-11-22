// A quote as a desk reads one - pure, free of React and the network, the way `blotter.ts` is for
// the blotter: what a structure asks for, read off its `/schema` declaration and never off a list
// of structure names; the ticket a quote answers, its levels on the market's axis by the blotter's
// own rule; what its trade does to the client's credit numbers; and what an acceptance says.

import {
  SET, STRIP_FIELDS, fieldsOf, levelOnAxis, pick, quoted, rowsByPath, sizeOf, toRows, tradesOf,
  type BlotterRow, type Quoted, type Size,
} from './blotter';
import { PLUS, entityText, instantMs, peaks, type Peak } from './desk';
import { under, type Agreement, type Entity } from './positions';
import { formatNumber, isBlank, isObject } from './tokens';
import type {
  Acceptance, ChargeAsk, DealNode, Descriptor, Increment, JobDoc, PendingQuote, Quote, QuoteGiven,
  QuoteRead, QuotedLeg, ResultSummary, Schema, Section, Standing, Stepped, StructureType,
} from './types';

// ---- the ticket asked for -----------------------------------------------------------------------

/** A structure's sales names, the one a desk says first. */
export const salesNames = (declared: StructureType): string[] =>
  declared.vernacular.split(',').map((name) => name.trim()).filter(Boolean);

/** The structure a schema declares by the sales name a desk says first, else by its own name. */
export const salesName = (schema: Schema | null, structure: string): string => {
  const declared = schema?.Structure?.types[structure];
  return declared ? salesNames(declared)[0] : structure;
};

/** A pair as a desk types it - USDZAR, USD/ZAR, USD.ZAR - as its base and quote currencies. */
export function pairSides(pair: unknown): [string, string] | undefined {
  const found = /^\s*([a-z]{3})[/.]?([a-z]{3})\s*$/i.exec(String(pair ?? ''));
  return found ? [found[1].toUpperCase(), found[2].toUpperCase()] : undefined;
}

/** The parameters a ticket fills: every one the forms share bar a selector, which chooses a form
 * rather than filling a leg, then the chosen form's own. */
export function ticketFields(declared: StructureType, variation: string | null): Section {
  return Object.fromEntries([
    ...Object.entries(declared.fields).filter(([, field]) => !field.selector),
    ...Object.entries(declared.variations?.[variation ?? '']?.fields ?? {})]);
}

/** Each way a structure is dealt: its word, and the currency of the pair its client buys - the
 * side, where no pair is typed yet. */
export function directions(declared: StructureType, pair: unknown):
  { word: string; buys: string }[] {
  const sides = pairSides(pair);
  return Object.entries(declared.variations ?? {}).map(([word, form]) => ({
    word, buys: sides ? sides[form.buys === 'base' ? 0 : 1] : `the ${form.buys} currency` }));
}

/** What a ticket says before it is priced: the structure, the form chosen, the parameters as
 * typed, the client's netting set, the margin - an amount and its currency, as typed - and the
 * node of the book's tree it books at, blank for the book itself. */
export type Draft = {
  structure: string; variation: string | null; values: Record<string, unknown>;
  nettingSet: string; margin: { amount: string; currency: string }; portfolio: string;
};

/** Where a quote may say it books on a desk whose record declares a tree: the book, then every
 * node under it - none where it declares no node under the book, a ticket then stating nothing. */
export function bookingNodes(nodes: string[], book: string): string[] {
  const below = nodes.filter((node) => node !== book && under(node, book)).sort();
  return below.length ? [book, ...below] : [];
}

/** The ticket as `POST /book/structure` takes it: every parameter of the form stated, a blank
 * being no statement; on a form chosen over a pair, the currency its client buys, the buying
 * selector saying it; the client, the margin and the node it books at where they are named. */
export function requestOf(draft: Draft, declared: StructureType): Record<string, unknown> {
  const fields = ticketFields(declared, draft.variation);
  const params = Object.fromEntries(Object.entries(draft.values)
    .filter(([key, value]) => key in fields && !isBlank(value)));
  const form = declared.variations?.[draft.variation ?? ''];
  const sides = pairSides(draft.values.pair);
  const buying = Object.keys(declared.fields).find(
    (key) => declared.fields[key].selector && /^buy/.test(key));
  if (form && sides && buying) params[buying] = sides[form.buys === 'base' ? 0 : 1];
  const margin = draft.margin.amount.trim();
  return {
    structure: draft.structure, params,
    ...(draft.nettingSet ? { netting_set: draft.nettingSet } : {}),
    ...(margin ? { margin: { amount: Number(margin), currency: draft.margin.currency.trim() } }
      : {}),
    ...(draft.portfolio ? { portfolio: draft.portfolio } : {}),
  };
}

/** A parameter path the recipe solved, `key` or a table's `key[row].cell`. */
const PATH = /^(\w+)(?:\[(\d+)\]\.(\w+))?$/;

/** A quote's ticket to quote again: the structure, form, client and margin it was quoted on and
 * the node it said it books at (`portfolio`), the parameters a ticket fills as priced - a selector
 * is the form's to state, a table's rows go by their declared columns - with every parameter the
 * recipe SOLVED blank again, so the same solve runs. */
export function draftOf(quote: Quote, declared: StructureType, portfolio = ''): Draft {
  const fields = ticketFields(declared, quote.variation);
  const values: Record<string, unknown> = Object.fromEntries(Object.entries(quote.params)
    .filter(([key]) => key in fields).map(([key, value]) => {
      const columns = fields[key]?.col_names;
      return [key, columns && Array.isArray(value) ? value.map((row) => (Array.isArray(row)
        ? Object.fromEntries(columns.map((column, at) => [column, row[at]])) : { ...row }))
        : value];
    }));
  for (const path of quote.legs.flatMap((leg) => Object.keys(leg.solved ?? {}))) {
    const [, key, row, cell] = PATH.exec(path) ?? [];
    if (!key || !(key in values)) continue;
    if (cell === undefined) values[key] = '';
    else (values[key] as Record<string, unknown>[])[Number(row)][cell] = '';
  }
  return {
    structure: quote.structure, variation: quote.variation, values,
    nettingSet: quote.netting_set ?? '',
    margin: quote.margin ? { amount: String(quote.margin.amount), currency: quote.margin.currency }
      : { amount: '', currency: '' },
    portfolio,
  };
}

const words = (key: string) => key.replace(/_/g, ' ').toLowerCase();

/** A parameter path in words: `strike_price` as it reads, a table's `tiers[0].coupon` as its cell
 * and row, counted from one. */
export function pathWords(path: string): string {
  const [, key, row, cell] = PATH.exec(path) ?? [];
  return cell === undefined ? words(key ?? path)
    : `${words(cell)}, ${words(key)} row ${Number(row) + 1}`;
}

/** What the recipe solves rather than the ticket stating, in words: a leg's field off its declared
 * step `Solve <leg>.<field> to <target>`, the target's premiums named; any other solve as it is
 * declared. */
export function solving(declared: StructureType): string[] {
  return declared.recipe.filter((step) => step.startsWith('Solve ')).map((step) => {
    const found = /^Solve (\w+)\.(\w+) to (.+)$/.exec(step);
    if (!found) return `The quote solves ${step.slice('Solve '.length)}.`;
    const target = found[3].trim() === '0' ? 'nothing' : found[3]
      .replace(/([-+]?)\s*Premium\((\w+)\)/g, (_, sign: string, leg: string) =>
        `${sign === '-' ? 'minus ' : sign === '+' ? 'plus ' : ''}the ${leg} premium `)
      .replace(/\s+/g, ' ').trim();
    return `The quote solves the ${found[1]} leg's ${words(found[2])}, so that leg is worth `
      + `${target}.`;
  });
}

/** The clients a quote may be for: every netting set the book holds, and every one the record's
 * paper declares, which the acceptance splices in where the book holds none yet. */
export function clients(doc: JobDoc | null, agreements: Agreement[]): string[] {
  const held = [...rowsByPath(toRows(doc?.Calc.Deals?.Deals?.Children ?? [], [], undefined))
    .values()].filter((row) => row.object === SET).map((row) => row.reference);
  const declared = agreements.filter((row) => row.terms.Object === SET)
    .map((row) => row.agreement);
  return [...new Set([...held, ...declared])].sort();
}

/** A client as a salesperson names one: the entity the record's paper declares behind the set,
 * beside the set itself. */
export function clientText(set: string | null, entities: Entity[],
                           agreements: Agreement[]): string {
  if (!set) return 'no netting set - books at the root of the book';
  const paper = agreements.find((row) => row.agreement === set);
  return paper ? `${entityText(entities, paper.entity)} · ${set}` : set;
}

// ---- the ticket answered ------------------------------------------------------------------------

/** One leg of a ticket: the blotter's reading of the deal it books - its pair and sense, size,
 * strike and barrier on the market's axis, its terms, a strip's fixings as declared - beside the
 * quote's own account of it, and which of its two levels the recipe solved. */
export type TicketLeg = {
  leg: QuotedLeg; row: BlotterRow; size: Size; read: Quoted; solved?: 'strike' | 'barrier';
  schedule?: { value: unknown; declared?: Descriptor };
};

/** The legs of a quote, each against the deal of its own position in the structure it composes. */
export function ticketLegs(quote: Quote, deal: Record<string, unknown>,
                           schema: Schema | null): TicketLeg[] {
  const children = toRows((deal.Children ?? []) as DealNode[], [], undefined);
  return quote.legs.map((leg, at) => {
    const row = children[at];
    const read = quoted(row, null, schema);
    const field = Object.keys(leg.solved ?? {})[0];
    const strip = pick(row.deal, STRIP_FIELDS);
    return { leg, row, read, size: sizeOf(row.deal, row.object, schema),
             solved: !field ? undefined : read.strike?.key === field ? 'strike'
               : read.barrier?.key === field ? 'barrier' : undefined,
             schedule: strip && { value: strip.value,
                                  declared: fieldsOf(schema, row.object)?.[strip.key] } };
  });
}

/** The numbers the recipe solved, set apart: a leg's level on the market's axis, else the ticket
 * parameter it filled, as the quote states it - each once. */
export function solvedNumbers(legs: TicketLeg[]): { what: string; value: number; on?: string }[] {
  const seen = new Set<string>();
  return legs.flatMap(({ leg, read, solved }) => Object.entries(leg.solved ?? {}).flatMap(
    ([key, value]) => {
      if (seen.has(key)) return [];
      seen.add(key);
      const level = solved && read[solved];
      return level && typeof level.value === 'number'
        ? [{ what: `${leg.role} ${solved}`, value: level.value, on: String(read.on?.value ?? '') }]
        : [{ what: pathWords(key), value }];
    }));
}

/** A level of the quote's pair - its spot, its forward - as the market quotes the pair, by the
 * blotter's own rule: a deal on `surface` stating it, its reciprocal where the ticket names the
 * pair the other way round. */
export function onMarketAxis(level: number, pair: unknown, surface: unknown):
  { pair: string; level: number } | undefined {
  const sides = pairSides(pair);
  if (!sides) return undefined;
  const read = levelOnAxis(sides[0], sides[1], level, null, surface);
  return read && { pair: read.pair, level: read.level };
}

/** What the market's two-way cost the client and where it was charged, in one sentence - or the
 * quote's own word that there was none to charge. */
export function spreadText(quote: Quote): string {
  if (quote.spread_note) return quote.spread_note;
  const on = quote.charged_on === 'premium' ? 'added to the premium'
    : `charged on the ${pathWords(quote.charged_on)}`;
  return `The market's two-way spread costs the client ${formatNumber(quote.edge, 'money')} `
    + `${quote.currency}, ${on}; at mid the legs are worth ${formatNumber(quote.net_mid, 'money')} `
    + `${quote.currency}.`;
}

// ---- what its trade does to the client's credit numbers -----------------------------------------

/** What a polled run reads as: the answer it filed under `stats[stat]` once done, the service's
 * own sentence where it failed, and nothing while it waits or runs. */
export function ranAs<T>(summary: ResultSummary, stat: string):
  { answer: T } | { refused: string } | null {
  if (summary.status === 'done') return { answer: summary.stats?.[stat] as T };
  return summary.status === 'error'
    ? { refused: summary.error ?? `the run failed and answered no ${stat}` } : null;
}

/** The ask a quote's ticket makes for its credit charge: of the quote until it books, the what-if
 * pricing its trade into the book; from then of the trade the book holds, where the file holds
 * it now under the reference it booked as - and none where the file holds it no more, nor on
 * that trade's `own` page, which asks it itself. */
export function chargeAsk(pending: Pick<PendingQuote, 'quote' | 'deal' | 'booked'>,
                          nodes: DealNode[], own?: string): ChargeAsk | undefined {
  if (!pending.booked) return { quote_id: pending.quote.quote_id };
  const reference = String(pending.deal.Reference);
  const path = landedAt(nodes, { ...pending.booked, reference });
  return path === undefined || path === own ? undefined : { deal_path: path, reference };
}

/** One line of a credit charge: a number of the client's netting set with the trade and without
 * it - a peak with the day each falls on - and what the trade does to it, as a word and a size. */
type ChargeLine = {
  what: string; plus?: string; kind: 'money' | 'whole'; with: number; without: number;
  days?: { with: string; without: string }; moves: 'adds' | 'takes off' | 'no change'; by: number;
};

/** A credit charge as the lines a ticket reads, off a run's `stats.Increment`: its CVA and FVA,
 * then the peak of each exposure column - the expected exposure's and every PFE's by its
 * percentile (`desk.peaks`) - a trade that lowers a number taking that much off it. */
export function chargeLines(answer: Increment): ChargeLine[] {
  const moved = (increment: number): Pick<ChargeLine, 'moves' | 'by'> => ({
    moves: increment > 0 ? 'adds' : increment < 0 ? 'takes off' : 'no change',
    by: Math.abs(increment) });
  const cost = (what: keyof typeof PLUS, step: Stepped<number>): ChargeLine => ({
    what, plus: PLUS[what], kind: 'money', with: step.with, without: step.without,
    ...moved(step.increment) });
  return [
    cost('CVA', answer.cva), cost('FVA', answer.fva),
    ...Object.entries(answer).flatMap(([column, held]): ChargeLine[] => {
      const step = held as Stepped<Peak>;
      if (!isObject(step) || !isObject(step.with)) return [];
      const read = peaks({ [column]: step.with });
      const what = read.ee ? 'Peak expected exposure (EE)' : read.pfe.length
        ? `Peak exposure at the ${formatNumber(read.pfe[0].quantile)}th percentile (PFE)` : column;
      return [{ what, kind: 'whole', with: step.with.value, without: step.without.value,
                days: { with: step.with.day, without: step.without.day },
                ...moved(step.increment) }];
    }),
  ];
}

// ---- how long it stands, and what an acceptance says -------------------------------------------

/** How long a quote a read found firm stays firm after that read, in ms: from the read's own stamp
 * to the close of its window, two stamps of the service's clock - none where it was not firm. */
export function firmFor(row: Standing, asOf: string | undefined): number | undefined {
  const [until, at] = [instantMs(row.firm_until), instantMs(asOf)];
  return row.stands === 'firm' && until !== undefined && at !== undefined ? until - at : undefined;
}

/** What holds on this desk between a quote and its acceptance, as a read says it: the declared
 * window, else - where the desk records - the book not moving under it, else nothing at all; or
 * why the desk's quote policy will not read, in the service's words. */
export function firmText(read: QuoteRead, recording: boolean): string {
  if (read.policy_refused) return read.policy_refused;
  if (read.firm_seconds !== null) {
    return `This desk holds a quote firm for ${formatNumber(read.firm_seconds, 'whole')} seconds; `
      + 'an older one is refused, and is quoted again.';
  }
  return recording ? 'This desk declares no firm window: a quote books while the book has not '
    + 'moved under it.' : 'This desk declares no firm window: a quote stays bookable until it is '
    + 'booked.';
}

/** An acceptance as a desk reads it - a booking, a restrike or an unwind alike: where the trade
 * landed, the service's own words, and whether quoting again is the thing to do - never where the
 * client's price already stands on the record and the desk's own tiers declined it. */
export type Verdict = {
  booked?: { deal_path: string; lsn: number | null }; words: string[]; requote: boolean;
};

/** A written answer's words: on whom it waits, then each check an earlier tier failed it on, a
 * line each - none where it fell in the first tier the desk reads. */
export function verdictOf(answer: Acceptance): Verdict {
  if (answer.written) {
    return { booked: { deal_path: answer.deal_path!, lsn: answer.recorded?.lsn ?? null },
             words: [...answer.waits_on ? [`Booked pending a second seat: ${answer.waits_on}`] : [],
                     ...(answer.tier?.fell_past ?? []).map((sentence) => `Fell past: ${sentence}`)],
             requote: false };
  }
  if (answer.booked) return { booked: answer.booked, words: [answer.note ?? ''], requote: false };
  return { words: answer.refused ?? [], requote: !answer.accepted };
}

// ---- the desk's quotes --------------------------------------------------------------------------

/** Where a quote stands, in words. */
export const STANDS: Record<Standing['stands'], string> = {
  booked: 'booked as', accepted: 'accepted, not booked', firm: 'firm',
  lapsed: 'past its firm window', quoted: 'quoted, with no firm window',
  refused: "not acceptable while the desk's quote policy will not read",
};

/** Whether a quote's booking is the trade at `path` under `reference`: by the reference the booking
 * names, which a node deleted ahead of the trade does not move - the trade now at its path under
 * another reference being none of its - and by the path only where it names none. */
const tied = (booked: NonNullable<QuoteGiven['booked']>, path: string, reference: unknown) =>
  (booked.reference ? booked.reference === reference : booked.deal_path === path);

/** The quote a trade was booked from: the one whose booking is the trade (`tied`). */
export const quoteAt = (rows: QuoteGiven[], path: string, reference: unknown) =>
  rows.find((row) => row.booked && tied(row.booked, path, reference));

/** Where a booked quote's trade stands in the file now: the first trade `tied` to it in the file's
 * order - none where the file holds it no more. */
export const landedAt = (nodes: DealNode[], booked: NonNullable<QuoteGiven['booked']>):
  string | undefined => tradesOf(toRows(nodes, [], undefined))
  .find((row) => tied(booked, row.path, row.reference))?.path;

/** What a read of the pending trades `ids` name is held against: their own rows on the desk's
 * list, which show the acceptance - the one act that writes a pending file - and nothing else. */
export const pendingKey = (ids: string[], rows: QuoteGiven[]): string =>
  JSON.stringify([ids, rows.filter((row) => ids.includes(row.quote_id))]);

/** A re-quote made here and the quote it re-quotes, which the service does not link. */
export type Kept = { id: string; from: string };

/** `list` with the re-quote `id` of `from` first. */
export const keep = (list: Kept[], id: string, from: string): Kept[] =>
  [{ id, from }, ...list.filter((entry) => entry.id !== id)];
