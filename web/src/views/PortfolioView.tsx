import { amendDeal, getBook } from '../api';
import { DescriptorPanel, type AmendField } from '../components/FieldView';
import { Tree, type TreeNode } from '../components/Tree';
import {
  GROUPINGS, NO_PAPER, POSITION, byId, byPath, clientName, clientTree, portfolioTree, type Paper,
  type Position,
} from '../positions';
import { useApp } from '../state';
import type { DealNode, Descriptor } from '../types';

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

/** The book as a tree beside the deal picked. Where the desk keeps a record the tree groups by
 * the book file's own nesting, by portfolio or by client, and a position picked in either of the
 * last two shows the deal the file holds for it. */
export function PortfolioView() {
  const { state, dispatch } = useApp();
  const { doc, schema, selection } = state;
  if (!doc || !schema) return null;

  const children = doc.Calc.Deals.Deals.Children;
  // the grouped trees are the record's, and a copy opened from a file has none
  const recording = state.source?.kind === 'book' && state.record.spine !== null;
  const grouping = recording ? state.grouping : 'book';
  const paper = state.paper.data ?? NO_PAPER;
  const nodes = grouping === 'book' ? toTree(children, schema.Instrument.containers)
    : grouping === 'portfolio' ? portfolioTree(paper.positions) : clientTree(paper);

  // a picked position stands for the first node the file holds it at; a picked node for itself
  const picked = selection.deal;
  const position = picked?.startsWith(POSITION) ? byId(paper.positions).get(picked) : undefined;
  const dealPath = position ? position.deal_paths[0] : picked ?? undefined;
  const held = position ? [position] : (dealPath && byPath(paper.positions).get(dealPath)) || [];
  const selected = dealPath !== undefined ? nodeAt(children, dealPath) : undefined;
  const deal = selected?.Instrument['.Deal'];
  const sections = deal ? schema.Instrument.types[String(deal.Object)] : undefined;

  // editing exists only over the LIVE BOOK - a local file has nothing server-side to amend.
  // A successful amendment refreshes the book at once rather than waiting a poll tick.
  const onAmend: AmendField | undefined =
    state.source?.kind === 'book' && dealPath !== undefined
      ? async (key, wireValue) => {
          const outcome = await amendDeal(dealPath, { [key]: wireValue });
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
        {recording && (
          <div className="sidehead">
            {GROUPINGS.map((choice) => (
              <button key={choice.id} className={grouping === choice.id ? 'on' : ''}
                      onClick={() => dispatch({ type: 'GROUP', grouping: choice.id })}>
                {choice.label}
              </button>
            ))}
          </div>
        )}
        {grouping !== 'book' && state.paper.error && (
          <div className="empty">the record did not answer: {state.paper.error}</div>
        )}
        {grouping !== 'book' && !state.paper.data && state.paper.loading && (
          <div className="empty">reading the record…</div>
        )}
        <Tree
          id={`portfolio.${grouping}`}
          nodes={nodes}
          selected={selection.deal}
          onSelect={(id) => dispatch({ type: 'SELECT_DEAL', path: id })}
        />
      </div>
      <div className="panel">
        {!deal && !held.length && <div className="placeholder">select a deal</div>}
        {held.map((row) => <Held key={`${row.agreement}/${row.portfolio}`} position={row}
                                 paper={paper} />)}
        {position && !deal && (
          <div className="banner">
            The record holds this position and the book file does not hold its instrument under
            this agreement - the banner above names where the two disagree.
          </div>
        )}
        {deal && sections && sections.map((section) => (
          <DescriptorPanel
            key={section}
            title={section}
            fields={schema.Instrument.sections[section]}
            values={Object.fromEntries(Object.entries(deal).filter(
              ([key]) => key in schema.Instrument.sections[section]))}
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

const said = (widget: string, description: string): Descriptor =>
  ({ widget, description, value: null });

/** What a position's card states, each field saying what it is where the pointer rests. */
const HELD: Record<string, Descriptor> = {
  Quantity: said('Float', 'The net of every fill, in units of the deal the book file holds - 1 is '
    + 'the deal as written'),
  Agreement: said('Text', 'The agreement the position sits under, whose netting set its credit '
    + 'exposure is measured over'),
  Counterparty: said('Text', 'The legal entity on the other side of the agreement, by the name the '
    + 'seat that keeps the legal documents declared'),
  Clips: said('Float', 'How many fills the position is the net of - every booking, close-out and '
    + 'unwind of it'),
  First_Fill: said('Text', 'Where on the record the first fill of the position sits, as the '
    + 'strip below numbers it'),
  Last_Moved: said('Text', 'Where on the record the position last moved - a fill, or an amendment '
    + 'carrying it onto new terms'),
};

/** Where one position sits and how big it is, as the record says. */
function Held({ position, paper }: { position: Position; paper: Paper }) {
  return (
    <DescriptorPanel title={`Position in ${position.portfolio || 'no portfolio'}`} fields={HELD}
                     values={{
                       Quantity: position.quantity,
                       Agreement: position.agreement,
                       Counterparty: clientName(paper.entities, position.counterparty),
                       Clips: position.clips,
                       First_Fill: `LSN ${position.first_lsn}`,
                       Last_Moved: `LSN ${position.last_lsn}`,
                     }} />
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
