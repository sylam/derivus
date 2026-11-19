// One client as an onboarding officer and a salesperson read it - pure, free of React and of the
// network, the way `blotter.ts` is for the blotter: whether it is ready to trade, its agreements'
// terms in words off the netting set's declaration, what the desk has on with it, and the record's
// rows and XVA rows that are its own - on a desk that records and on one that does not.

import type { Dispatch } from 'react';
import { rowsByPath, toRows, type BlotterRow } from './blotter';
import { SURVIVAL, entityText, instantMs, plural } from './desk';
import { byPath, type Agreement, type Entity, type Position } from './positions';
import { standing, type Action } from './state';
import {
  COLLATERAL, currencyOf, formatNumber, label, sharedSections, token, uncollateralised,
} from './tokens';
import type { ActivityRow, Descriptor, JobDoc, Reconcile, Schema, XvaSet } from './types';

/** The screen a client's page is, and the store's pick of which client it shows. */
export const CLIENT = 'client';

/** Open one client's page. */
export function openClient(dispatch: Dispatch<Action>, id: string) {
  dispatch({ type: 'PICK', screen: CLIENT, id });
  dispatch({ type: 'TAB', tab: CLIENT });
}

/** A client's agreements: on a desk that records, the ones its record declares with the entity; on
 * one that does not, every node of the book whose credit support names it - a netting set's terms
 * being its agreement. */
export function agreementsOf(id: string, declared: Agreement[] | null,
                             doc: JobDoc | null): Agreement[] {
  if (declared) return declared.filter((row) => row.entity === id);
  return [...rowsByPath(toRows(doc?.Calc.Deals.Deals.Children ?? [], [], undefined)).values()]
    .filter((row) => token(row.deal.Credit_Support_Amounts, 'Counterparty') === id)
    .map((row) => ({ agreement: row.reference, entity: id, kind: '', terms: row.deal }));
}

// ---- is it ready to trade ----------------------------------------------------------------------

/** A fact the page leads with: `ok` stands, `todo` stops a booking until it is put right, `warn` is
 * wrong and books all the same, `note` is how the desk is; `fix` names where it is put right. */
export type Line = {
  tone: 'ok' | 'todo' | 'warn' | 'note'; text: string; fix?: 'entity' | 'agreement' | 'curve';
};

/** Whether a client is ready to trade, line by line: its entity declared, an agreement in force
 * and where the book holds its netting set - each said on a desk that records, the netting sets
 * naming it on one that does not - and its credit curve, without which no CVA reads and its first
 * trade is refused; a booking is refused for a factor the book newly lacks, so a set the book
 * already prices without one still takes a trade. */
export function readiness(id: string, entities: Entity[] | null, agreements: Agreement[],
                          doc: JobDoc | null, reconcile: Reconcile | null,
                          curve: unknown): Line[] {
  const credit = `${SURVIVAL}.${id}`;
  const lines: Line[] = entities ? paperLines(id, entities, agreements, doc, reconcile) : [{
    tone: 'note', text: `This desk keeps no record: ${id} is the counterparty ${
      agreements.map((row) => row.agreement).join(', ')} name${agreements.length === 1 ? 's' : ''}.`,
  }];
  const held = agreements.some((row) => standing(doc, reconcile, row).path !== null);
  return [...lines, curve ? { tone: 'ok', text: `Its credit curve ${credit} is stated.` } : {
    tone: 'todo', fix: 'curve', text: `No credit curve ${credit}: ${held
      ? 'its CVA cannot be read, though a trade still books under a set the book already prices '
        + 'without one' : 'its first trade is refused for want of it, and its CVA cannot be read'}.`,
  }];
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

const said = (value: unknown) => value !== undefined && value !== null && value !== '';

function term(key: string, field: Descriptor, value: unknown,
              currency: (description: string) => string | undefined): Term {
  const held = said(value) ? value : field.value;
  const list = token(held, '.CreditSupportList');
  const amount = Array.isArray(list) ? (list[0] as unknown[])?.[1] : held;
  const phrase = /,\s*in ([^,:;]+)/.exec(field.description)?.[1] ?? '';
  return {
    key, words: field.description.split(/[,:]/)[0].trim() || key,
    value: typeof amount === 'number' ? formatNumber(amount, list ? 'money' : 'plain')
      : field.widget !== 'Table' ? label(amount) || 'none'
        : Array.isArray(held) ? plural(held.length, 'row') : 'none',
    unit: currency(field.description) ?? (phrase.startsWith('the ') ? '' : phrase),
    convention: !said(value), side: SIDE.exec(field.description)?.[1],
  };
}

/** An agreement's terms in words, off its type's own sections - a container's terms each a term
 * of its own: a collateralised set's every collateral term first, at its convention or not, then
 * each other term stated, the rest at their conventions apart; a set whose collateral is off only
 * what it states, collateral it does not have meaning nothing. The flag itself is the reading's. */
export function termsReading(terms: Record<string, unknown>, schema: Schema): TermsReading {
  const shared = sharedSections(schema.Instrument.types);
  const fields: Record<string, Descriptor> = Object.assign({}, ...(
    schema.Instrument.types[String(terms.Object)] ?? [])
    .filter((section) => !shared.includes(section))
    .map((section) => schema.Instrument.sections[section]));
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
    collateralised: !off, stated: said(terms[COLLATERAL]),
    sides: [...new Set(moving.map((row) => row.side!))],
    currency: moving.map((row) => row.unit).find(Boolean),
    terms: off ? all.filter((row) => !row.convention)
      : [...moving, ...rest.filter((row) => !row.convention)],
    conventions: off ? [] : rest.filter((row) => row.convention),
  };
}

// ---- what the desk has on with it -------------------------------------------------------------

/** One trade under a client's agreement: the file's row under its set, held at the position the
 * record files it as - or CLOSED, where a desk that records holds nothing of it - or a position the
 * file does not hold under the set, which has no row. */
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
      row: { ...row, units: position.quantity }, position, closed: false })) : [{ row, closed: true }])),
    ...positions.filter((position) => !filed.has(position.deal_paths[0]))
      .map((position) => ({ position, closed: false })),
  ];
}

/** A mark as the desk says it: the amount in its currency, and in whose favour it stands. */
export function markText(value: number, currency: string | null): string {
  const favour = value > 0 ? "in the desk's favour" : value < 0 ? "in the client's favour" : 'flat';
  return `${formatNumber(value, 'money')}${currency ? ` ${currency}` : ''}, ${favour}`;
}

// ---- what happened ----------------------------------------------------------------------------

/** The record's rows touching a client, newest first: its own declaration, and every row naming
 * one of its agreements - a booking, a signature, a restatement. */
export function touching(rows: ActivityRow[], id: string, agreements: string[]): ActivityRow[] {
  return rows.filter((row) => row.entity === id
    || (row.agreement !== null && agreements.includes(row.agreement))).reverse();
}

/** The record's rows under a set since its XVA row was run - the ones naming its agreement,
 * written after the run by the instant each stamp names. */
export function since(set: XvaSet, rows: ActivityRow[]): ActivityRow[] {
  const ran = instantMs(set.as_of);
  return ran === undefined ? [] : rows.filter((row) => row.agreement === set.reference
    && (instantMs(row.record_time) ?? -Infinity) > ran);
}
