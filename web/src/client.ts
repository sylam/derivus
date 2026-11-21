// One client as an onboarding officer and a salesperson read it - pure, free of React and of the
// network, the way `blotter.ts` is for the blotter: whether it is ready to trade, its agreements'
// terms in words off the netting set's declaration, what the desk has on with it, and the record's
// rows and XVA rows that are its own - on a desk that records and on one that does not.

import type { Dispatch } from 'react';
import { fieldsOf, type BlotterRow } from './blotter';
import {
  SURVIVAL, entityText, factorFamily, instantMs, peaks, plural, type Peak,
} from './desk';
import { byPath, closed, type Agreement, type Entity, type Position } from './positions';
import { standing, type Action } from './state';
import {
  COLLATERAL, currencyOf, formatNumber, isBlank, label, onScreen, setCurrency, tenorLabel, token,
  uncollateralised,
} from './tokens';
import { markOf } from './trade';
import type {
  ActivityRow, CollateralCall, DealMark, Descriptor, JobDoc, Market, Reconcile, Schema, XvaSet,
} from './types';

/** The screen a client's page is, and the store's pick of which client it shows. */
export const CLIENT = 'client';

/** Open one client's page. */
export function openClient(dispatch: Dispatch<Action>, id: string) {
  dispatch({ type: 'PICK', screen: CLIENT, id });
  dispatch({ type: 'TAB', tab: CLIENT });
}

/** A client's agreements: on a desk that records, the ones its record declares with the entity; on
 * one that does not, every row of the book (`rowsByPath`, the caller's) whose credit support names
 * it - a netting set's terms being its agreement. */
export function agreementsOf(id: string, declared: Agreement[] | null,
                             rows: Map<string, BlotterRow>): Agreement[] {
  if (declared) return declared.filter((row) => row.entity === id);
  return [...rows.values()]
    .filter((row) => token(row.deal.Credit_Support_Amounts, 'Counterparty') === id)
    .map((row) => ({ agreement: row.reference, entity: id, kind: '', terms: row.deal }));
}

// ---- is it ready to trade ----------------------------------------------------------------------

/** A fact the page leads with: `ok` stands, `todo` stops a booking until it is put right, `warn` is
 * wrong and books all the same, `note` is how the desk is; `fix` names where it is put right. */
export type Line = {
  tone: 'ok' | 'todo' | 'warn' | 'note'; text: string; fix?: 'entity' | 'agreement' | 'curve';
};

/** The price factors an agreement's `terms` name that the market's `factors` do not hold, sorted:
 * every type a field's declaration names under the name the terms state, a collateral table's by
 * its column - less its counterparty's credit curve, which has a line of its own. */
export function lacked(terms: Record<string, unknown>, fields: Record<string, Descriptor>,
                       factors: Record<string, unknown>): string[] {
  const named = (declared: Descriptor['factors'], stated: unknown): string[] => (
    Array.isArray(declared)
      ? (isBlank(stated) ? [] : declared.map((type) => `${type}.${label(stated)}`))
      : Object.entries(declared ?? {}).flatMap(([column, types]) =>
        (Array.isArray(stated) ? stated : []).flatMap((row) => named(types, token(row, column)))));
  const walk = (level: Record<string, Descriptor>, stated: unknown): string[] =>
    Object.entries(level).flatMap(([key, field]) => [...named(field.factors, token(stated, key)),
      ...walk(field.sub_fields ?? {}, token(stated, key))]);
  return [...new Set(walk(fields, terms))]
    .filter((name) => !(name in factors) && factorFamily(name) !== SURVIVAL).sort();
}

/** Whether a client is ready to trade, line by line: its entity declared, an agreement in force
 * and where the book holds its netting set - each said on a desk that records, the netting sets
 * naming it on one that does not - its credit curve, without which no CVA reads and its first
 * trade is refused, and each factor its paper names that the market's `factors` lack, none said of
 * a market not read yet; a booking is refused for a factor the book newly lacks, so a set the
 * book already prices without one still takes a trade. */
export function readiness(id: string, entities: Entity[] | null, agreements: Agreement[],
                          doc: JobDoc | null, reconcile: Reconcile | null,
                          factors: Record<string, unknown> | undefined,
                          schema: Schema | null): Line[] {
  const credit = `${SURVIVAL}.${id}`;
  const lines: Line[] = entities ? paperLines(id, entities, agreements, doc, reconcile) : [{
    tone: 'note', text: `This desk keeps no record: ${id} is the counterparty ${
      agreements.map((row) => row.agreement).join(', ')} name${agreements.length === 1 ? 's' : ''}.`,
  }];
  const booked = (row: Agreement) => standing(doc, reconcile, row).path !== null;
  const wanting = !factors ? [] : agreements.flatMap((row) => lacked(
    row.terms, fieldsOf(schema, String(row.terms.Object)) ?? {}, factors).map((factor): Line => ({
    tone: 'todo', text: `${row.agreement} names ${factor}, which the market does not hold: ${
      booked(row) ? 'a trade still books under the set the book already holds, and a run that '
        + 'reads it fails' : 'the first trade under it is refused until the market holds it or '
        + 'the agreement is amended'}.`,
  })));
  return [...lines, factors?.[credit] ? { tone: 'ok', text: `Its credit curve ${credit} is stated.` } : {
    tone: 'todo', fix: 'curve', text: `No credit curve ${credit}: ${agreements.some(booked)
      ? 'its CVA cannot be read, though a trade still books under a set the book already prices '
        + 'without one' : 'its first trade is refused for want of it, and its CVA cannot be read'}.`,
  }, ...wanting];
}

function paperLines(id: string, entities: Entity[], agreements: Agreement[], doc: JobDoc | null,
                    reconcile: Reconcile | null): Line[] {
  const entity = entities.find((row) => row.entity === id);
  const named = entity ? { tone: 'ok' as const, text: `Legal declares ${entityText(entities, id)}${
    entity.parent ? `, under ${entityText(entities, entity.parent)}` : ''}.` }
    : { tone: 'todo' as const, fix: 'entity' as const,
        text: `Legal declares no entity ${id}: nothing books for it until one is declared.` };
  if (!agreements.length) {
    return [named, { tone: 'todo', fix: 'agreement',
                     text: 'No agreement is in force: nothing books for it until one is declared.' }];
  }
  return [named, ...agreements.map((row): Line => {
    const { path, moved } = standing(doc, reconcile, row);
    const force = `${row.agreement} (${row.kind}) is in force`;
    return path === null ? { tone: 'note', text: `${force}; no trade is booked under it yet, and the `
      + 'first booking adds its netting set to the book.' }
      : moved.length ? { tone: 'warn', text: `${force}, but its netting set in the book at ${path} `
        + `is off the paper on ${moved.join(', ')}: runs read the declared terms.` }
        : { tone: 'ok', text: `${force}, and the book holds its netting set at ${path} as declared.` };
  })];
}

/** Ready where nothing stops a booking. */
export const ready = (lines: Line[]): boolean => !lines.some((line) => line.tone === 'todo');

// ---- an agreement as a credit officer says it --------------------------------------------------

/** One term as a credit officer reads it: the words its declaration says it in, its value - a
 * credit support list's first amount, which a run reads - with the currency or unit its description
 * names, whether it stands at its convention, and the way collateral moves where it says one. */
export type Term = {
  key: string; words: string; value: string; unit: string; convention: boolean; side?: string;
};

/** Whether the set is collateralised and whether its terms say so, the ways collateral moves under
 * it, the currency its collateral amounts are in, its terms and those at their conventions apart. */
export type TermsReading = {
  collateralised: boolean; stated: boolean; sides: string[]; currency?: string; terms: Term[];
  conventions: Term[];
};

/** The words a declaration names the way collateral moves in. */
const SIDE = /\bthe bank (receives|posts)\b/;

function term(key: string, field: Descriptor, value: unknown,
              currency: (description: string) => string | undefined): Term {
  const held = !isBlank(value) ? value : field.value;
  const list = token(held, '.CreditSupportList');
  const amount = Array.isArray(list) ? (list[0] as unknown[])?.[1] : held;
  const phrase = /,\s*in ([^,:;]+)/.exec(field.description)?.[1] ?? '';
  return {
    key, words: field.description.split(/[,:]/)[0].trim() || key,
    value: typeof amount === 'number' ? formatNumber(amount, list ? 'money' : 'plain')
      : field.widget !== 'Table' ? label(amount) || 'none'
        : Array.isArray(held) ? plural(held.length, 'row') : 'none',
    unit: currency(field.description) ?? (phrase.startsWith('the ') ? '' : phrase),
    convention: isBlank(value), side: SIDE.exec(field.description)?.[1],
  };
}

/** An agreement's terms in words, off its type's own sections - a container's terms each a term
 * of its own: a collateralised set's every collateral term first, at its convention or not, then
 * each other term stated, the rest at their conventions apart; a set whose collateral is off only
 * what it states, collateral it does not have meaning nothing. The flag itself is the reading's. */
export function termsReading(terms: Record<string, unknown>, schema: Schema): TermsReading {
  const { fields } = onScreen(fieldsOf(schema, String(terms.Object), true) ?? {}, terms);
  const currency = (description: string) =>
    currencyOf(description, Object.keys(fields), (key) => terms[key]);
  const all = Object.entries(fields).filter(([key]) => key !== COLLATERAL)
    .flatMap(([key, field]) => (field.sub_fields
      ? Object.entries(field.sub_fields).map(([inner, sub]) =>
        term(`${key}.${inner}`, sub, token(terms[key], inner), currency))
      : [term(key, field, terms[key], currency)]));
  const off = uncollateralised(fields, terms);
  const moving = all.filter((row) => row.side);
  const rest = all.filter((row) => !row.side);
  return {
    collateralised: !off, stated: !isBlank(terms[COLLATERAL]),
    sides: [...new Set(moving.map((row) => row.side!))],
    currency: moving.map((row) => row.unit).find(Boolean),
    terms: off ? all.filter((row) => !row.convention)
      : [...moving, ...rest.filter((row) => !row.convention)],
    conventions: off ? [] : rest.filter((row) => row.convention),
  };
}

// ---- what the desk has on with it -------------------------------------------------------------

/** One trade under a client's agreement: the file's row under its set, held at the position the
 * record files it as - CLOSED where that is netted to nothing, or where a desk that records holds no
 * position of it - or a position the file does not hold under the set, which has no row. */
export type Held = { row?: BlotterRow; position?: Position; closed: boolean };

/** The trades under the set at `path`, with `positions` - the agreement's own, null on a desk that
 * keeps no record, where each row stands as the file writes it. */
export function heldUnder(path: string | null, rows: Map<string, BlotterRow>,
                          positions: Position[] | null): Held[] {
  const children = (path === null ? undefined : rows.get(path))?.children ?? [];
  if (!positions) return children.map((row) => ({ row, closed: false }));
  const at = byPath(positions);
  const filed = new Set(children.map((row) => row.path));
  return [
    ...children.flatMap((row): Held[] => (at.has(row.path) ? at.get(row.path)!.map((position) => ({
      row: { ...row, units: position.quantity }, position, closed: closed(position.quantity) }))
      : [{ row, closed: true }])),
    ...positions.filter((position) => !filed.has(position.deal_paths[0]))
      .map((position) => ({ position, closed: closed(position.quantity) })),
  ];
}

/** A mark as the desk says it: the amount in its currency, and in whose favour it stands. */
export function markText(value: number, currency: string | null): string {
  const favour = value > 0 ? "in the desk's favour" : value < 0 ? "in the client's favour" : 'flat';
  return `${formatNumber(value, 'money')}${currency ? ` ${currency}` : ''}, ${favour}`;
}

// ---- what happened ----------------------------------------------------------------------------

/** A client's history off the record's reads asked for it - by its entity, its declarations, and
 * by each of its agreements, a booking, a signature, a restatement - each row once though two reads
 * answer it, newest first. */
export const touching = (reads: ActivityRow[][]): ActivityRow[] =>
  [...new Map(reads.flat().map((row) => [row.lsn, row])).values()].sort((a, b) => b.lsn - a.lsn);

/** How many trades were booked under a set since its run: the tickets filed under its agreement
 * after the run's instant, each once though a restrike books several positions under one. */
export function bookedSince(run: XvaSet, positions: Position[]): number {
  const ran = instantMs(run.as_of);
  return ran === undefined ? 0 : new Set(positions
    .filter((position) => position.agreement === run.reference).flatMap((position) => position.tickets)
    .filter((ticket) => (instantMs(ticket.at) ?? -Infinity) > ran).map((ticket) => ticket.ticket)).size;
}

/** The count in a run's cell, in the warn tone where it is any. */
export const bookedText = (count: number): string => `${plural(count, 'trade')} booked since`;

// ---- every client as a credit officer reads it ------------------------------------------------

/** One netting set a client trades under: where the book holds it, the paper it is under - its
 * agreement's kind, null where none is in force - and whether its terms collateralise, its last
 * run, its trades and those waiting on a second seat, its value to the desk now in the risk's
 * currency, its call on the marked close, and how many trades were booked under it since its run -
 * null where the record's positions are not read. */
export type CreditLine = {
  reference: string; path: string | null; paper: string | null; collateralised: boolean;
  run: XvaSet | null; trades: number; pending: number; value: number | null;
  call: CollateralCall | null; booked: number | null;
};

/** What the lines are read off: the projection's sets, the client's agreements (`agreementsOf`),
 * the file's rows, and - each null where it is not read - the record's positions, the risk's marks
 * and the calls on the marked close. */
export type CreditReads = {
  sets: XvaSet[]; agreements: Agreement[]; rows: Map<string, BlotterRow>;
  positions: Position[] | null; marks: DealMark[] | null; calls: CollateralCall[] | null;
};

/** A client's lines: a set the book holds or held naming it, under its own agreement where one is
 * in force - its trades the file's at the units the record holds, or as the file writes them where
 * no agreement covers them - then every agreement nothing is booked under yet, worth nothing. */
export function creditLines(id: string, reads: CreditReads, schema: Schema): CreditLine[] {
  const runs = reads.sets.filter((set) => set.counterparty === id);
  const line = (reference: string, run: XvaSet | null): CreditLine => {
    const signed = reads.agreements.find((row) => row.agreement === reference);
    const held = signed && reads.positions
      && reads.positions.filter((position) => position.agreement === reference);
    const path = run?.deal_path ?? null;
    return {
      reference, path, run, paper: signed ? signed.kind || 'its own terms' : null,
      collateralised: signed ? termsReading(signed.terms, schema).collateralised : !!run?.collateralized,
      trades: heldUnder(path, reads.rows, held || null).filter((trade) => !trade.closed).length,
      pending: (held || []).filter((position) => position.pending !== 0).length,
      value: path === null ? (run ? null : 0)
        : reads.marks && (markOf(reads.marks, path).own?.value ?? null),
      call: reads.calls?.find((call) => call.agreement === reference) ?? null,
      booked: run && reads.positions ? bookedSince(run, reads.positions) : null,
    };
  };
  return [...runs.map((run) => line(run.reference, run)),
          ...reads.agreements.filter((row) => !runs.some((run) => run.reference === row.agreement))
            .map((row) => line(row.agreement, null))];
}

/** Whether the collateral read's refusal says no close stands to read a call on: no market the
 * record designates for the P&L, none declared under that name, or no marks for the book's day. */
export const unmarked = (refusal: string | null): boolean =>
  /^(this record designates no market|no market is declared|no marks stand)/.test(refusal ?? '');

/** What a set's own terms say the bank holds, on a desk keeping no record: its opening balance -
 * at its declared convention where it states none - in its balance currency, else its agreement's,
 * else the market's base, with the market's spots in that currency; no call is worked without a
 * marked close. */
export function statedHolding(terms: Record<string, unknown>, fields: Record<string, Descriptor>,
                              market: Market | null): CollateralCall {
  const factors = market?.['Price Factors'] ?? {};
  const base = label(token(market?.['System Parameters'], 'Base_Currency'));
  const currency = setCurrency(terms) || base;
  const spot = (of: string) => Number(token(factors[`FxRate.${of}`], 'Spot') ?? NaN);
  const balance = Number(terms.Opening_Balance ?? fields.Opening_Balance?.value ?? 0);
  return {
    agreement: String(terms.Reference), entity: '', currency, exposure: null, balance,
    fx: Object.fromEntries(Object.keys(factors).filter((name) => name.startsWith('FxRate.'))
      .map((name) => [name.slice(7), spot(name.slice(7)) / spot(currency)])),
    held: balance ? { [currency]: balance } : {}, margin: {}, required: null, call: null,
    direction: null, minimum_transfer: null, unknown: [],
  };
}

/** What the desk stands to lose on a line today, in the value's `currency`: its value to the desk
 * less the collateral held under it at the close's spots - what the bank posted adding to it - and
 * nothing where the client is owed; null where the value, or a collateralised set's call, is unread. */
export function exposureAfter(line: CreditLine, currency: string | null): number | null {
  if (line.value === null) return null;
  if (!line.collateralised) return Math.max(0, line.value);
  const rate = currency ? line.call?.fx[currency] : undefined;
  return line.call?.balance === null || line.call?.balance === undefined || !rate ? null
    : Math.max(0, line.value - line.call.balance / rate);
}

/** A sum over a client's lines where every one reads, and null where one does not: a client is
 * never said to owe less for a set nobody read. */
export const summed = (values: (number | null)[]): number | null =>
  (values.includes(null) ? null : values.reduce<number>((total, value) => total + value!, 0));

/** A client's adjustment over the book's own sets, in the one currency their runs reported in:
 * null where none ran, where one has none to add, or where two currencies would be added. */
export function runSum(lines: CreditLine[], key: 'cva' | 'fva'):
  { value: number | null; currency: string | null } {
  const runs = lines.flatMap((line) => (line.run && !line.run.note ? [line.run] : []));
  const currencies = new Set(runs.map((run) => run.currency));
  return { currency: currencies.size === 1 ? [...currencies][0] : null,
           value: currencies.size === 1 ? summed(runs.map((run) => run[key])) : null };
}

/** What a line's paper reads: no agreement in force, or whether its terms collateralise. */
export const paperText = (line: CreditLine): string => (line.paper === null ? 'no agreement'
  : line.collateralised ? 'collateralised' : 'uncollateralised');

/** The credit table as its export writes it, its heads first: a row per set under its client, a
 * client with none a row of its own; every amount the number itself, its currency in a column
 * beside it and a peak's day in the next; a PFE column per percentile the runs kept; and the
 * service's note on a set that has left the book. A client's sums are left to the reader. */
export function creditExport(
  clients: { name: string; spread?: [number, number]; lines: CreditLine[] }[],
  currency: string): unknown[][] {
  const kept = (line?: CreditLine) => peaks(line?.run?.peaks ?? null);
  const quantiles = [...new Set(clients.flatMap((client) => client.lines)
    .flatMap((line) => kept(line).pfe.map((peak) => peak.quantile)))].sort((a, b) => a - b);
  const peak = (found?: Peak) => [found?.value, found?.day];
  return [
    ['Client', 'Netting set', 'Spread bp', 'Spread tenor', 'Paper', 'Trades',
      'Waiting on a second seat', 'Value to the desk', 'After collateral', 'Ccy', 'Collateral held',
      'Call', 'Collateral ccy', 'Peak EE', 'Peak EE day',
      ...quantiles.flatMap((quantile) => [`Peak PFE ${quantile}`, `Peak PFE ${quantile} day`]),
      'CVA', 'FVA', 'Run ccy', 'Run', 'Note'],
    ...clients.flatMap(({ name, spread, lines }) => (lines.length ? lines : [undefined]).map((line) => {
      const held = line?.collateralised && typeof line.call?.balance === 'number' ? line.call : null;
      return [name, line?.reference, spread && spread[1] * 1e4, spread && tenorLabel(spread[0]),
        line && paperText(line), line?.trades, line?.pending, line?.value,
        line && exposureAfter(line, currency), line && currency, held?.balance, held?.call,
        held?.currency, ...peak(kept(line).ee),
        ...quantiles.flatMap((quantile) => peak(kept(line).pfe.find((at) => at.quantile === quantile))),
        line?.run?.cva, line?.run?.fva, line?.run?.currency, line?.run?.as_of, line?.run?.note];
    })),
  ];
}

/** What a row says to act on, worst first. */
export const WRONG = {
  failed: 'run failed', unpapered: 'trading with no agreement in force',
  pending: 'waiting on a second seat', unrun: 'never run',
} as const;

/** One thing to act on under a set, in business words, and who puts it right. */
export type Wrong = { kind: keyof typeof WRONG; reference: string; text: string; who: string };

const RECALC = 'a recalc asked for through the chat';

/** What every XVA row is, said once where the rows are read. */
export const RUNS = 'each row is the last run of its set, refreshed at the close, and a recalc is '
  + 'asked for through the chat';

/** Who puts a failed run right, off the factor the engine could not find: a credit curve whoever
 * marks states, any other factor the market's owner adds. */
export function ownerOf(error: string | null): string {
  const factor = /Cannot find (\S+)/.exec(error ?? '')?.[1];
  return !factor ? "the engine's sentence names no factor: read it, then " + RECALC
    : factor.startsWith(`${SURVIVAL}.`) ? `no credit curve: whoever marks states ${factor}`
      : `a market factor is missing: the market's owner adds ${factor}`;
}

/** What is wrong under a client's lines, worst first - a set that has left the book reads its last
 * run and asks for nothing. */
export function wrongWith(lines: CreditLine[]): Wrong[] {
  const order = Object.keys(WRONG);
  return lines.filter((line) => !line.run?.note).flatMap(({
    reference, paper, run, trades, pending }) => [
    run?.status === 'failed' && { kind: 'failed' as const, reference,
      text: run.error ?? 'its last run failed', who: ownerOf(run.error) },
    trades > 0 && paper === null && { kind: 'unpapered' as const, reference,
      who: 'legal declares the agreement on Paper',
      text: `${plural(trades, 'trade')} under ${reference}, which no agreement in force covers` },
    pending > 0 && { kind: 'pending' as const, reference, who: 'another seat signs on the Back Office',
      text: `${plural(pending, 'trade')} under it waiting on a second seat` },
    trades > 0 && (!run || run.status === 'never run') && {
      kind: 'unrun' as const, reference, text: 'no XVA run yet', who: RECALC },
  ].filter((wrong): wrong is Wrong => !!wrong))
    .sort((a, b) => order.indexOf(a.kind) - order.indexOf(b.kind));
}

/** Clients in the order a credit officer works them: those with something to act on, then the
 * largest exposure after collateral, an unread one last, then by name. */
export function actionFirst<T extends { name: string; wrong: Wrong[]; after: number | null }>(
  rows: T[]): T[] {
  return [...rows].sort((a, b) => Number(!a.wrong.length) - Number(!b.wrong.length)
    || (b.after ?? -1) - (a.after ?? -1) || a.name.localeCompare(b.name));
}
