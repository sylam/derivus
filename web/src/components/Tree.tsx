import { Fragment, useEffect, useState, type ReactNode } from 'react';
import { ancestors, filtered, isOpen, leaves, type TreeNode } from '../tree';

export type { TreeNode };

/** A tree of more items than this carries a filter above it. */
const FILTER_FROM = 12;

/** The folders a desk opened or closed on one tree, remembered on this browser - a convenience, so
 * a storage that refuses (a private window) costs the memory and nothing else. */
function useToggles(id?: string) {
  const key = id && `derivus.tree.${id}`;
  const [toggled, setToggled] = useState<Record<string, boolean>>(() => {
    try {
      return key ? JSON.parse(localStorage.getItem(key) ?? '{}') : {};
    } catch {
      return {};
    }
  });
  useEffect(() => {
    try {
      if (key) localStorage.setItem(key, JSON.stringify(toggled));
    } catch { /* remembered for the session alone */ }
  }, [key, toggled]);
  return [toggled, (node: string, open: boolean) =>
    setToggled((standing) => ({ ...standing, [node]: open }))] as const;
}

/** A navigation tree: an item selects on a click and a folder opens or closes on one, its caret
 * doing the same for an item that has children. `id` names the tree whose toggles are remembered;
 * the arithmetic of what shows is `src/tree.ts`'s. */
export function Tree({ id, nodes, selected, onSelect }: {
  id?: string; nodes: TreeNode[]; selected: string | null; onSelect: (id: string) => void;
}) {
  const [toggled, toggle] = useToggles(id);
  const [text, setText] = useState('');
  const count = nodes.reduce((sum, node) => sum + leaves(node), 0);
  const around = ancestors(nodes, selected) ?? [];
  const shown = filtered(nodes, text);

  function row(node: TreeNode, depth: number): ReactNode {
    const open = isOpen(node, toggled, around, text.trim() !== '');
    const folder = !!node.children?.length || node.badge === 'group';
    const flip = () => toggle(node.id, !open);
    return (
      <Fragment key={node.id}>
        <div
          className={`node${node.group ? ' group' : ''}${selected === node.id ? ' selected' : ''}`
            + `${node.muted ? ' muted' : ''}${node.accent ? ' accent' : ''}`}
          style={{ paddingLeft: 10 + depth * 16 }} title={node.label}
          onClick={() => (node.group ? flip() : onSelect(node.id))}
        >
          <span className="caret" onClick={(e) => {
            if (!folder) return;
            e.stopPropagation();
            flip();
          }}>
            {folder ? (open ? '▾' : '▸') : ''}
          </span>
          {node.type && <span className="type">{node.type}.</span>}
          <span className="label">{node.label}</span>
          {node.muted && <span className="badge">ignored</span>}
          {node.group && <span className="count">{leaves(node)}</span>}
        </div>
        {open && node.children?.map((child) => row(child, depth + 1))}
      </Fragment>
    );
  }

  return (
    <div className="tree">
      {count > FILTER_FROM && (
        <div className="filter">
          <input type="search" value={text} placeholder={`filter ${count}`}
                 onChange={(event) => setText(event.target.value)} />
        </div>
      )}
      {shown.map((node) => row(node, 0))}
      {text.trim() !== '' && !shown.length && <div className="empty">nothing matches</div>}
    </div>
  );
}
