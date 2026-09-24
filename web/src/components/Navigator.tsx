import type { ReactNode } from 'react';
import { useApp } from '../state';
import { ancestors, firstItem, grouped, type TreeNode } from '../tree';
import { Tree } from './Tree';

/** One page of a navigated screen: the row it files as - the folder it sits in, its label - and what
 * the panel shows while it is the page picked. */
export type Page = TreeNode & { folder?: string; content: ReactNode };

/** A screen as a tree of its pages beside the page picked - the market screens' one navigation,
 * each filing its blocks by what they are. The pick is the store's, per screen, so a page stays
 * picked across tabs and a new document starts on its first. `head` stands above every page;
 * `side` above the tree. `keep` holds every page MOUNTED and shows one, for a screen whose pages
 * hold edits a click elsewhere must not throw away. */
export function Navigator({ screen, pages, head, side, keep, empty }: {
  screen: string; pages: Page[]; head?: ReactNode; side?: ReactNode; keep?: boolean;
  empty?: string;
}) {
  const { state, dispatch } = useApp();
  const nodes = grouped(pages);
  const picked = state.selection.picks[screen] ?? null;
  const selected = ancestors(nodes, picked) ? picked : firstItem(nodes);

  return (
    <div className="main">
      <div className="sidebar">
        {side}
        <Tree key={screen} id={screen} nodes={nodes} selected={selected}
              onSelect={(id) => dispatch({ type: 'PICK', screen, id })} />
      </div>
      <div className="panel">
        {head}
        {!pages.length && empty && <div className="placeholder">{empty}</div>}
        {pages.map((page) => (keep || page.id === selected) && (
          <div key={page.id} hidden={page.id !== selected}>{page.content}</div>
        ))}
      </div>
    </div>
  );
}
