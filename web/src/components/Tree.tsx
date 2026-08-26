import { useState } from 'react';

export type TreeNode = {
  id: string;
  label: string;
  type?: string;
  muted?: boolean;
  badge?: string;
  children?: TreeNode[];
};

function Node({ node, depth, selected, onSelect }: {
  node: TreeNode; depth: number; selected: string | null; onSelect: (id: string) => void;
}) {
  const [open, setOpen] = useState(true);
  const isFolder = !!node.children?.length || node.badge === 'group';

  return (
    <>
      <div
        className={`node${selected === node.id ? ' selected' : ''}${node.muted ? ' muted' : ''}`}
        style={{ paddingLeft: 10 + depth * 16 }}
        onClick={() => onSelect(node.id)}
      >
        <span
          className="caret"
          onClick={(e) => { e.stopPropagation(); setOpen(!open); }}
        >
          {isFolder ? (open ? '▾' : '▸') : ''}
        </span>
        {node.type && <span className="type">{node.type}.</span>}
        <span>{node.label}</span>
        {node.muted && <span className="badge">ignored</span>}
      </div>
      {open && node.children?.map((child) => (
        <Node key={child.id} node={child} depth={depth + 1} selected={selected} onSelect={onSelect} />
      ))}
    </>
  );
}

export function Tree({ nodes, selected, onSelect }: {
  nodes: TreeNode[]; selected: string | null; onSelect: (id: string) => void;
}) {
  return (
    <div className="tree">
      {nodes.map((node) => (
        <Node key={node.id} node={node} depth={0} selected={selected} onSelect={onSelect} />
      ))}
    </div>
  );
}
