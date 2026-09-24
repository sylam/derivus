// The navigation tree's arithmetic, kept out of the component the way `blotter.ts` keeps the
// blotter's: how a screen's pages file under their groups, what a filter leaves standing, which
// folders stand open, and which page a screen shows when nobody has picked one.

export type TreeNode = {
  id: string;
  label: string;
  type?: string;
  muted?: boolean;
  /** An action rather than a thing - "a new curve", "another family" - drawn in the accent. */
  accent?: boolean;
  badge?: string;
  /** A FOLDER and not an item: nothing to select, so a click opens or closes it. */
  group?: boolean;
  children?: TreeNode[];
};

/** A group of this many leaves or fewer stands open until somebody closes it; a bigger one stands
 * closed, so a family of a few hundred blocks is one row until it is asked for. */
export const OPEN_LEAVES = 12;

/** A group's id: its name behind a prefix no item id carries, so a group and an item spelled the
 * same are never one row. */
export const GROUP = 'group:';

/** Items filed under the folder each names, in the order they came - a group sits where its first
 * member did - and an item naming none left at the top level. */
export function grouped(items: (TreeNode & { folder?: string })[]): TreeNode[] {
  const nodes: TreeNode[] = [];
  const groups = new Map<string, TreeNode[]>();
  for (const { folder, ...item } of items) {
    if (!folder) {
      nodes.push(item);
      continue;
    }
    if (!groups.has(folder)) {
      groups.set(folder, []);
      nodes.push({ id: GROUP + folder, label: folder, group: true, children: groups.get(folder) });
    }
    groups.get(folder)!.push(item);
  }
  return nodes;
}

/** The items under a node: a group counts its members, an item with children itself as well, and
 * an action - "another family" - is not a thing the folder holds. */
export const leaves = (node: TreeNode): number =>
  (node.group || node.accent ? 0 : 1)
  + (node.children ?? []).reduce((sum, child) => sum + leaves(child), 0);

/** What a filter leaves standing: a node whose name carries the text, whatever the case, with
 * everything under it, and the path down to every one - so a family's name finds the family, a
 * currency its blocks across every family, and a block's full name the block. A name is the
 * folder's, the type's and the label's joined as the store spells them; an id is never read, a
 * deal's being its position, which every digit would match. */
export function filtered(nodes: TreeNode[], text: string, folder = ''): TreeNode[] {
  const wanted = text.trim().toLowerCase();
  if (!wanted) return nodes;
  return nodes.flatMap((node) => {
    const name = [folder, node.type, node.label].filter(Boolean).join('.');
    if (name.toLowerCase().includes(wanted)) return [node];
    const children = filtered(node.children ?? [], wanted, node.group ? node.label : '');
    return children.length ? [{ ...node, children }] : [];
  });
}

/** The ids on the path down to `id`, outermost first - `[]` where it sits at the top level and
 * `null` where the tree does not carry it. */
export function ancestors(nodes: TreeNode[], id: string | null): string[] | null {
  for (const node of nodes) {
    if (node.id === id) return [];
    const below = ancestors(node.children ?? [], id);
    if (below) return [node.id, ...below];
  }
  return null;
}

/** Whether a node's children show: all of them while a filter narrows, since what matched has to
 * be seen; else the desk's own toggle where it made one; else open on the way to the page on
 * screen, which is never inside a closed folder until somebody closes it; else an item's subtree
 * open and a group open only where it is small enough to read at a glance. */
export function isOpen(node: TreeNode, toggled: Record<string, boolean>, around: string[],
                       filtering: boolean): boolean {
  if (filtering) return true;
  if (node.id in toggled) return toggled[node.id];
  return around.includes(node.id) || !node.group || leaves(node) <= OPEN_LEAVES;
}

/** The page a screen shows when nobody picked one, or picked one this document no longer carries:
 * the first there is, so a screen never opens on an empty panel. */
export function firstItem(nodes: TreeNode[]): string | null {
  for (const node of nodes) {
    const found = node.group ? firstItem(node.children ?? []) : node.id;
    if (found) return found;
  }
  return null;
}
