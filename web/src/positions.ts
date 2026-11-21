// The record's positions as the desk navigates them - pure, free of React and of the network, the
// way `tree.ts` and `blotter.ts` are. A position is keyed where it sits, instrument by agreement by
// portfolio: the PORTFOLIO tree files it under its path, whose top node is the book, and the CLIENT
// tree under its counterparty - a legal entity, grouped under the parent legal declared for it -
// and the agreement. Both are trees the navigation renders, every leaf one position.

import { formatNumber } from './tokens';
import { GROUP, type TreeNode } from './tree';
import type { DealNode, SettlementState } from './types';

/** One row of `GET /book/positions`: the fold's own row, and where the book file holds it. */
export type Position = {
  instrument: string;
  agreement: string;
  portfolio: string;
  book: string | null;
  counterparty: string | null;
  quantity: number;
  clips: number;
  first_lsn: number;
  last_lsn: number;
  deal_paths: string[];
  reference: string | null;
  object: string | null;
  /** The day the deal expired, and the day it settles - an expired position stands until then. */
  expired: string | null;
  settles: string | null;
  /** What its tickets read - `rejected`, `pending`, `approved` or `unticketed` - and the quantity
   * awaiting a second seat. Workflow, not economics: a pending trade prices all the same. */
  status: string;
  pending: number;
  tickets: Ticket[];
};

/** One ticket a position is booked by: a fill's, or a restrike's in place of the clip it restruck
 * - `fill` the LSN of the fill, `lsn` the ticket's own, `actor` and `at` the ticket's, the `tier`
 * it fell in, the sentence of each check an earlier tier failed it on (`fell_past`: none where it
 * fell in the first, null where it was filed before the record kept them) and, while pending, what
 * it `waits_on` - every one in the service's words. */
export type Ticket = {
  ticket: string; lsn: number; fill: number; actor: string; at: string;
  execution_reference: string; quantity: number; status: string; tier: string | null;
  fell_past: string[] | null; waits_on: string | null;
};

export type Entity = { entity: string; name: string; parent: string | null };
export type Agreement = {
  agreement: string; entity: string; kind: string; terms: Record<string, unknown>;
};

/** What the grouped views read: the positions standing, the paper they sit under with what that
 * paper never states, and the nodes of the book's tree the record declares. */
export type Paper = {
  positions: Position[]; entities: Entity[]; agreements: Agreement[]; unstated: SettlementState;
  nodes: string[];
};

/** The paper of a desk that has read none - or records nothing. */
export const NO_PAPER: Paper = { positions: [], entities: [], agreements: [],
                                 unstated: { fields: [], collateral_columns: [] }, nodes: [] };

/** What the Paper tab says in place of the paper where no record is kept, null on a desk that
 * records: agreements are not declared here, and a set's own terms are read on its client's page
 * on the desk's `live` book - on the set alone in a file opened in this browser, which has none. */
export function noPaper(recording: boolean, live: boolean): { title: string; hint: string } | null {
  if (recording) return null;
  const terms = "A netting set's own terms are the agreement";
  return live ? {
    title: 'This desk keeps no record, so agreements are not declared here.',
    hint: `${terms}: each client's page under Clients reads them set by set, and they are `
      + 'amended on the set in the Portfolio.',
  } : {
    title: 'A file opened here has no record, so no agreements are declared for it.',
    hint: `${terms}, read on the set in the Portfolio.`,
  };
}

/** How a screen groups the book: as the file nests it, or by one of the record's two keys. */
export type Grouping = 'book' | 'portfolio' | 'client';

export const GROUPINGS: { id: Grouping; label: string }[] = [
  { id: 'book', label: 'Book' },
  { id: 'portfolio', label: 'Portfolio' },
  { id: 'client', label: 'Client' },
];

/** A position's id: its key behind a prefix no deal path carries, since two positions can hold one
 * node of the file - the same terms in two portfolios under one agreement. */
export const POSITION = 'position:';

export const positionId = (position: Position): string =>
  POSITION + JSON.stringify([position.instrument, position.agreement, position.portfolio]);

/** A position by the reference it was booked under, else by the head of its address. */
export const positionName = (position: Position): string =>
  position.reference ?? position.instrument.slice(0, 12);

/** Whether the record holds nothing of a trade - a position netted to nothing, which stands only to
 * say it closed, or a row of the file the record holds at a net of none. Closed counts as live
 * nowhere. */
export const closed = (units: number | null | undefined): boolean => units === 0;

/** What the record holds of a node at the net of its positions, and none where it holds none. */
export const netOf = (held: Position[]): number | undefined =>
  (held.length ? held.reduce((sum, row) => sum + row.quantity, 0) : undefined);

/** The positions by their ids. */
export const byId = (positions: Position[]): Map<string, Position> =>
  new Map(positions.map((position) => [positionId(position), position]));

/** The node of the file at a deal path ('0/2/1'), or undefined where it holds none. */
export function nodeAt(nodes: DealNode[], path: string): DealNode | undefined {
  let node: DealNode | undefined;
  let level = nodes;
  for (const position of path.split('/').map(Number)) {
    node = level[position];
    if (!node) return undefined;
    level = node.Children ?? [];
  }
  return node;
}

/** The positions each node of the file CARRIES, by its deal path: the first node holding the
 * terms, which the book prices at the net, and never a later clip of the same terms, which it
 * ignores. */
export function byPath(positions: Position[]): Map<string, Position[]> {
  const held = new Map<string, Position[]>();
  for (const position of positions) {
    const [path] = position.deal_paths;
    if (path !== undefined) held.set(path, [...(held.get(path) ?? []), position]);
  }
  return held;
}

/** A counterparty as a reader knows it: the name legal declared, else the id the record holds. */
export function clientName(entities: Entity[] | null, id: string | null): string {
  return entities?.find((entity) => entity.entity === id)?.name ?? id ?? '';
}

/** What the tickets of `held` read, and the quantity awaiting a second seat where that is not the
 * whole of what is held - a trade pending whole reads `pending` once. */
export function statusText(held: Position[]): string {
  const pending = held.reduce((sum, row) => sum + row.pending, 0);
  const quantity = held.reduce((sum, row) => sum + row.quantity, 0);
  const read = [...new Set(held.map((row) => row.status))].join(', ');
  return pending && pending !== quantity ? `${read} · ${formatNumber(pending)} pending` : read;
}

/** What the record holds of a row as the blotter's four record columns: the net quantity - none
 * where nothing is held - the portfolios and the clients it sits under, and what its tickets read,
 * or that it closed. */
export function heldColumns(held: Position[], entities: Entity[]):
  [number | undefined, string, string, string] {
  const distinct = (values: string[]) => [...new Set(values)].join(', ');
  const net = netOf(held);
  return [net, distinct(held.map((row) => row.portfolio)),
    distinct(held.map((row) => clientName(entities, row.counterparty))),
    closed(net) ? 'closed' : statusText(held)];
}

/** One position as a leaf, by the reference a desk knows it by. `other` is the coordinate the tree
 * does not file it by - the agreement in the portfolio tree, the portfolio in the client tree - so
 * the same terms held twice under one folder still read as two things. A position the file does
 * not hold says so, one netted to nothing that it closed, and one whose deal has expired says when
 * it settles. */
function leaf(position: Position, other: string): TreeNode {
  return {
    id: positionId(position),
    label: position.reference ?? `${position.instrument.slice(0, 12)}…`,
    hint: `${closed(position.quantity) ? 'closed' : `×${formatNumber(position.quantity)}`} · ${
      position.deal_paths.length ? other : 'not in the file'}${
      position.expired ? ` · expired ${position.expired}, settles ${position.settles}` : ''}`,
  };
}

/** Every level ordered by `rank`, then by what a reader sees - the label, then the hint. */
function ordered(nodes: TreeNode[], rank: (node: TreeNode) => number): TreeNode[] {
  return [...nodes]
    .sort((a, b) => rank(a) - rank(b) || a.label.localeCompare(b.label)
      || (a.hint ?? '').localeCompare(b.hint ?? ''))
    .map((node) => (node.children ? { ...node, children: ordered(node.children, rank) } : node));
}

/** The portfolio tree: a folder per segment of each position's path and of every node the record
 * DECLARES - standing whether or not anything is booked under it - the book outermost, and the
 * folders under one ahead of the positions filed in it. */
export function portfolioTree(positions: Position[], nodes: string[] = []): TreeNode[] {
  const top: TreeNode[] = [];
  const folders = new Map<string, TreeNode>();
  const folder = (portfolio: string): TreeNode[] => {
    let level = top;
    let path = '';
    for (const segment of portfolio ? portfolio.split('/') : []) {
      path = path ? `${path}/${segment}` : segment;
      if (!folders.has(path)) {
        const made = { id: `${GROUP}portfolio:${path}`, label: segment, group: true, children: [] };
        folders.set(path, made);
        level.push(made);
      }
      level = folders.get(path)!.children!;
    }
    return level;
  };
  nodes.forEach(folder);
  positions.forEach((position) => folder(position.portfolio).push(
    leaf(position, position.agreement)));
  return ordered(top, (node) => (node.group ? 0 : 1));
}

/** Whether the declared parents above `id` come back round to one already walked. */
function loops(parents: Map<string, string | null>, id: string): boolean {
  const seen = new Set<string>();
  for (let at: string | null | undefined = id; at; at = parents.get(at)) {
    if (seen.has(at)) return true;
    seen.add(at);
  }
  return false;
}

/** The client tree: a folder per legal entity holding its agreements and then the entities
 * declared under it, every declared entity and agreement standing whether or not anything is
 * booked under it. A position files under its OWN counterparty - the record's word on the fill -
 * and an agreement the record never declared (a netting set booked before the paper was) sits
 * there as it was booked. A parent nobody declared is a folder under its id, and a chain of
 * parents that loops is broken where it closes. */
export function clientTree({ positions, entities, agreements }: Paper): TreeNode[] {
  const parents = new Map(entities.map((entity) => [entity.entity, entity.parent]));
  const clients = new Map<string, TreeNode>();
  const papers = new Map<string, TreeNode>();

  const client = (id: string): TreeNode => {
    if (!clients.has(id)) {
      clients.set(id, { id: `${GROUP}client:${JSON.stringify([id])}`,
                        label: clientName(entities, id) || '(no counterparty)',
                        group: true, children: [] });
    }
    return clients.get(id)!;
  };
  const agreement = (entity: string, id: string): TreeNode => {
    const key = JSON.stringify([entity, id]);
    if (!papers.has(key)) {
      papers.set(key, { id: `${GROUP}client:${key}`, label: id, group: true, children: [] });
      client(entity).children!.push(papers.get(key)!);
    }
    return papers.get(key)!;
  };

  entities.forEach((entity) => client(entity.entity));
  agreements.forEach((row) => agreement(row.entity, row.agreement));
  positions.forEach((position) => agreement(position.counterparty ?? '', position.agreement)
    .children!.push(leaf(position, position.portfolio)));

  const top: TreeNode[] = [];
  for (const [id, folder] of [...clients]) {
    const parent = parents.get(id);
    if (parent && !loops(parents, id)) client(parent).children!.push(folder);
    else top.push(folder);
  }
  // a parent nobody declared was minted by the walk above, and stands at the top
  for (const [id, folder] of clients) {
    if (!parents.has(id) && !top.includes(folder)) top.push(folder);
  }
  const entityFolders = new Set(clients.values());
  return ordered(top, (node) => (entityFolders.has(node) ? 1 : 0));
}
