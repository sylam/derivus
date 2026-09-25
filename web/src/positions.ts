// The record's positions as the desk navigates them - pure, free of React and of the network, the
// way `tree.ts` and `blotter.ts` are. A position is keyed where it sits, instrument by agreement by
// portfolio: the PORTFOLIO tree files it under its path, whose top node is the book, and the CLIENT
// tree under its counterparty - a legal entity, grouped under the parent legal declared for it -
// and the agreement. Both are trees the navigation renders, every leaf one position.

import { formatNumber } from './tokens';
import { GROUP, type TreeNode } from './tree';

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
};

export type Entity = { entity: string; name: string; parent: string | null };
export type Agreement = { agreement: string; entity: string; kind: string };

/** What the grouped views read: the positions standing, and the paper they sit under. */
export type Paper = { positions: Position[]; entities: Entity[]; agreements: Agreement[] };

/** The paper of a desk that has read none - or records nothing. */
export const NO_PAPER: Paper = { positions: [], entities: [], agreements: [] };

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

/** The positions by their ids. */
export const byId = (positions: Position[]): Map<string, Position> =>
  new Map(positions.map((position) => [positionId(position), position]));

/** The positions the file holds at each node, by its deal path. */
export function byPath(positions: Position[]): Map<string, Position[]> {
  const held = new Map<string, Position[]>();
  for (const position of positions) {
    for (const path of position.deal_paths) held.set(path, [...(held.get(path) ?? []), position]);
  }
  return held;
}

/** A counterparty as a reader knows it: the name legal declared, else the id the record holds. */
export function clientName(entities: Entity[], id: string | null): string {
  return entities.find((entity) => entity.entity === id)?.name ?? id ?? '';
}

/** One position as a leaf, by the reference a desk knows it by. `other` is the coordinate the tree
 * does not file it by - the agreement in the portfolio tree, the portfolio in the client tree - so
 * the same terms held twice under one folder still read as two things. A position the file does
 * not hold says so. */
function leaf(position: Position, other: string): TreeNode {
  return {
    id: positionId(position),
    label: position.reference ?? `${position.instrument.slice(0, 12)}…`,
    hint: `×${formatNumber(position.quantity)} · ${
      position.deal_paths.length ? other : 'not in the file'}`,
  };
}

/** Every level ordered by `rank`, then by what a reader sees - the label, then the hint. */
function ordered(nodes: TreeNode[], rank: (node: TreeNode) => number): TreeNode[] {
  return [...nodes]
    .sort((a, b) => rank(a) - rank(b) || a.label.localeCompare(b.label)
      || (a.hint ?? '').localeCompare(b.hint ?? ''))
    .map((node) => (node.children ? { ...node, children: ordered(node.children, rank) } : node));
}

/** The portfolio tree: a folder per segment of each position's path, the book outermost, and the
 * folders under one ahead of the positions filed in it. */
export function portfolioTree(positions: Position[]): TreeNode[] {
  const top: TreeNode[] = [];
  const folders = new Map<string, TreeNode>();
  for (const position of positions) {
    let level = top;
    let path = '';
    for (const segment of position.portfolio ? position.portfolio.split('/') : []) {
      path = path ? `${path}/${segment}` : segment;
      if (!folders.has(path)) {
        const folder = { id: `${GROUP}portfolio:${path}`, label: segment, group: true, children: [] };
        folders.set(path, folder);
        level.push(folder);
      }
      level = folders.get(path)!.children!;
    }
    level.push(leaf(position, position.agreement));
  }
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
