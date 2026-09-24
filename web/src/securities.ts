// The Securities screen's arithmetic - pure, free of React, the way `curves.ts` is for its own. A
// ticker here is a STRING the service verified, a verdict is the service's word and a refusal is
// its sentence: nothing in this module knows what a pillar or a fixing is, only what shape the
// three `/book/securities` verbs answer in and what shape they take back.

import { isObject } from './tokens';
import type { Descriptor, Evidence, MapEntry, UsedRow } from './types';

/** The join's columns, in the order an IPV reader compares them: what the book quoted, then what
 * a terminal ever said about the security it quoted it off. */
export const USED_COLUMNS = ['curve', 'tenor', 'security', 'use', 'quote', 'print',
                             'verdict', 'name', 'verified'] as const;

/** One row of the join, flat. `print` is the BOOK's timestamp and `verified` the MAP's, which are
 * different dates and never the same column. */
export type UsedTableRow =
  Record<typeof USED_COLUMNS[number], string | number | null> & { tone: string };

/** One field of an entry at the path a panel names it by - `['conventions', 'spot_days']`. Every
 * dict on the way is copied, so the entry a table is rendering from never moves underneath it. */
export function setAt(entry: unknown, path: string[], value: unknown): unknown {
  if (path.length === 0) return value;
  const node = isObject(entry) ? entry : {};
  return { ...node, [path[0]]: setAt(node[path[0]], path.slice(1), value) };
}

/** One member of a list: past the end appends, `null` removes, and the array handed in is left
 * standing. A year, a pair, a pillar - what the member IS is the service's business. */
export function withMember<T>(list: T[], index: number, member: T | null): T[] {
  if (member === null) return list.filter((_, i) => i !== index);
  if (index >= list.length) return [...list, member];
  return list.map((item, i) => (i === index ? member : item));
}

/** The editor one seed value gets, off the SHAPE it already stands in - which is the shape an edit
 * must put back, the seed being the desk's own JSON and no declaration of the engine's. Numbers
 * stay numbers whether or not one reads whole, because a scale of 1.0 edited to 0.01 is not 0. */
export function seedDescriptor(key: string, value: unknown): Descriptor {
  const widget = typeof value === 'boolean' ? 'Checkbox'
    : typeof value === 'number' ? 'Float' : 'Text';
  return { widget, description: key, value };
}

/** The body `POST /book/securities` takes. A block is keyed UNIFORMLY - whatever key the read
 * answers an entry under is what the merge names it by - and `entry` always travels, as null where
 * the desk removed it, because a key `JSON.stringify` drops reads to the verb as a removal. */
export function mergeRequest(block: string, key: string, entry: unknown): Record<string, unknown> {
  return { block, key: key.trim(), entry: entry === undefined ? null : entry };
}

/** The scope `POST /book/securities/verify` takes, trimmed to what was STATED: a block, one key
 * inside it, or tickers by name. Nothing stated is the whole vocabulary and travels as an empty
 * body; a blank is not a scope, and would reach the verb as a name it refuses. */
export function verifyRequest(scope: {
  block?: string | null; key?: string | null; securities?: string[];
}): Record<string, unknown> {
  const request: Record<string, unknown> = {};
  if (scope.block) request.block = scope.block;
  if (scope.block && scope.key) request.key = scope.key;
  if (scope.securities?.length) request.securities = scope.securities;
  return request;
}

/** The verdict column: the word the SERVICE answered - a rejection's own, its `unmapped` where the
 * map never heard of the ticker - or `verified` where the entry carries the date a terminal
 * answered on. The tone is the only rule here, and it is a colour. */
export function verdictOf(evidence: Evidence): { text: string; tone: string } {
  if (evidence.verdict) {
    return { text: evidence.verdict, tone: evidence.verdict === 'unmapped' ? '' : 'error' };
  }
  return evidence.verified ? { text: 'verified', tone: 'done' } : { text: '', tone: '' };
}

/** The join as the table renders it, in the order the book's own ladder carries it - which is the
 * order the curve was solved in. `curve` narrows it to one. */
export function usedRows(used: UsedRow[], curve?: string): UsedTableRow[] {
  return used.filter((row) => curve === undefined || row.curve === curve).map((row) => {
    const verdict = verdictOf(row.evidence);
    return {
      curve: row.curve, tenor: row.tenor, security: row.security, use: row.use, quote: row.quote,
      print: row.timestamp, verdict: verdict.text, tone: verdict.tone,
      name: row.evidence.name ?? '', verified: row.evidence.verified ?? '',
    };
  });
}

/** Every leaf of the map's blocks as `(path, entry)` - a leaf is a dict naming a `security`, and
 * anything else is a container or the metadata a block files beside its entries. The service's own
 * walk in the client's spelling, the map being JSON no schema declares. */
export function mapRows(node: unknown, prefix: string[] = []): { path: string; entry: MapEntry }[] {
  if (!isObject(node)) return [];
  if (typeof node.security === 'string') {
    return [{ path: prefix.join('/'), entry: node as unknown as MapEntry }];
  }
  return Object.entries(node).flatMap(([key, value]) => mapRows(value, [...prefix, key]));
}

/** The map as the pages the evidence files it on, blocks in order: a block whose entries all sit in
 * containers - a curve's strip, a pair's quote grid - one page per container, and a block carrying
 * an entry directly one page whole, a page per security being a page per row. */
export function mapPages(blocks: Record<string, unknown>) {
  return Object.keys(blocks).sort().flatMap((block) => {
    const node = blocks[block];
    const children = Object.entries(isObject(node) ? node : {}).filter(
      (pair): pair is [string, Record<string, unknown>] => isObject(pair[1]));
    return children.length && children.every(([, child]) => typeof child.security !== 'string')
      ? children.map(([key, child]) => ({ block, key: key as string | null, node: child }))
      : [{ block, key: null as string | null, node }];
  });
}

/** The rejected ledger as the table renders it, in ticker order - a rejection is keyed by TICKER
 * because a candidate that never became an entry has no path. The terminal's own error travels:
 * it is why the name was refused. */
export function rejectedRows(rejected: Record<string, Evidence>) {
  return Object.keys(rejected).sort().map((security) => ({
    security, verdict: rejected[security].verdict ?? '', name: rejected[security].name ?? '',
    last_update: rejected[security].last_update ?? '', error: rejected[security].error ?? '',
  }));
}
