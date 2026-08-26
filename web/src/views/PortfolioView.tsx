import { amendDeal, getBook } from '../api';
import { DescriptorPanel, type AmendField } from '../components/FieldView';
import { Tree, type TreeNode } from '../components/Tree';
import { useApp } from '../state';
import type { DealNode } from '../types';

/** A tree node's id is its POSITIONAL path ('0/2/1') - the same identity the service's
 * `deal_path` uses, because references are not unique in a book. */
function toTree(nodes: DealNode[], containers: string[], path = ''): TreeNode[] {
  return nodes.map((node, position) => {
    const id = path ? `${path}/${position}` : String(position);
    const deal = node.Instrument['.Deal'];
    const type = String(deal.Object ?? '?');
    return {
      id,
      type,
      label: String(deal.Reference ?? ''),
      muted: node.Ignore === 'True',
      badge: containers.includes(type) ? 'group' : undefined,
      children: node.Children ? toTree(node.Children, containers, id) : undefined,
    };
  });
}

function nodeAt(nodes: DealNode[], path: string): DealNode | undefined {
  let node: DealNode | undefined;
  let level = nodes;
  for (const position of path.split('/').map(Number)) {
    node = level[position];
    if (!node) return undefined;
    level = node.Children ?? [];
  }
  return node;
}

export function PortfolioView() {
  const { state, dispatch } = useApp();
  const { doc, schema, selection } = state;
  if (!doc || !schema) return null;

  const children = doc.Calc.Deals.Deals.Children;
  const selected = selection.deal !== null ? nodeAt(children, selection.deal) : undefined;
  const deal = selected?.Instrument['.Deal'];
  const sections = deal ? schema.Instrument.types[String(deal.Object)] : undefined;

  // editing exists only over the LIVE BOOK - a local file has nothing server-side to amend.
  // A successful amendment refreshes the book at once rather than waiting a poll tick.
  const onAmend: AmendField | undefined =
    state.source?.kind === 'book' && selection.deal !== null
      ? async (key, wireValue) => {
          const outcome = await amendDeal(selection.deal!, { [key]: wireValue });
          if (!outcome.written) return outcome.refused ?? ['refused'];
          const live = await getBook();
          dispatch({
            type: 'DOC_LOADED', doc: live.document, refresh: true,
            source: { kind: 'book', etag: live.etag, path: live.path },
          });
          return null;
        }
      : undefined;

  return (
    <div className="main">
      <div className="sidebar">
        <Tree
          nodes={toTree(children, schema.Instrument.containers)}
          selected={selection.deal}
          onSelect={(id) => dispatch({ type: 'SELECT_DEAL', path: id })}
        />
      </div>
      <div className="panel">
        {!deal && <div className="placeholder">select a deal</div>}
        {deal && sections && sections.map((section) => (
          <DescriptorPanel
            key={section}
            title={section}
            fields={schema.Instrument.sections[section]}
            values={deal}
            onAmend={onAmend}
          />
        ))}
        {deal && !sections && (
          <DescriptorPanel title={`${deal.Object} (undeclared type)`} values={deal} />
        )}
        {deal && sections && <UndeclaredPanel deal={deal} sections={sections} />}
      </div>
    </div>
  );
}

/** Every key of the deal no section covers - the viewer's honesty panel. */
function UndeclaredPanel({ deal, sections }: {
  deal: Record<string, unknown>; sections: string[];
}) {
  const { state } = useApp();
  const declared = new Set(
    sections.flatMap((s) => Object.keys(state.schema?.Instrument.sections[s] ?? {})));
  const extra = Object.fromEntries(
    Object.entries(deal).filter(([key]) => !declared.has(key)));
  if (!Object.keys(extra).length) return null;
  return <DescriptorPanel title="Undeclared fields" values={extra} />;
}
