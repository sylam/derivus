import { DescriptorPanel } from '../components/FieldView';
import { Tree, type TreeNode } from '../components/Tree';
import {
  GROUPINGS, NO_PAPER, POSITION, byId, byPath, clientName, clientTree, closed, nodeAt,
  portfolioTree, type Paper, type Position,
} from '../positions';
import { openTrade, spineOf, unread, useApp } from '../state';
import { tradeAt } from '../trade';
import type { DealNode, Descriptor } from '../types';
import { DealEditor } from '../components/Desk';

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

/** The book as a tree beside the deal picked. Where the desk keeps a record the tree groups by
 * the book file's own nesting, by portfolio or by client, and a position picked in either of the
 * last two shows the deal the file holds for it. */
export function PortfolioView() {
  const { state, dispatch } = useApp();
  const { doc, schema, selection } = state;
  if (!doc || !schema) return null;

  const children = doc.Calc.Deals.Deals.Children;
  // the grouped trees are the record's, and a copy opened from a file has none
  const recording = spineOf(state) !== null;
  const grouping = recording ? state.grouping : 'book';
  const paper = state.paper.data ?? NO_PAPER;
  const nodes = grouping === 'book' ? toTree(children, schema.Instrument.containers)
    : grouping === 'portfolio' ? portfolioTree(paper.positions, paper.nodes) : clientTree(paper);

  // a picked position stands for the first node the file holds it at; a picked node for itself
  const picked = selection.deal;
  const position = picked?.startsWith(POSITION) ? byId(paper.positions).get(picked) : undefined;
  const dealPath = position ? position.deal_paths[0] : picked ?? undefined;
  const held = position ? [position] : (dealPath && byPath(paper.positions).get(dealPath)) || [];
  const selected = dealPath !== undefined ? nodeAt(children, dealPath) : undefined;
  const deal = selected?.Instrument['.Deal'];
  const trade = dealPath !== undefined ? tradeAt(children, dealPath) : undefined;

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
        {grouping !== 'book' && unread(state.paper) && (
          <div className="empty">{unread(state.paper)}</div>
        )}
        <Tree
          id={`portfolio.${grouping}`}
          nodes={nodes}
          selected={selection.deal}
          onSelect={(id) => dispatch({ type: 'SELECT_DEAL', path: id })}
        />
      </div>
      <div className="panel">
        {trade && (
          <div className="statusrow">
            <button className="ghost" onClick={() => openTrade(dispatch, trade.path)}>
              open the trade page ›
            </button>
          </div>
        )}
        {!deal && !held.length && <div className="placeholder">select a deal</div>}
        {held.map((row) => <Held key={`${row.agreement}/${row.portfolio}`} position={row}
                                 paper={paper} />)}
        {position && !deal && (
          <div className="banner">
            The record holds this position and the book file does not hold its instrument under
            this agreement - the banner above names where the two disagree.
          </div>
        )}
        {selected && <DealEditor path={dealPath!} node={selected} />}
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

/** What an expired position still held says: it stands until it settles, and rolls off then. */
const SETTLING: Record<string, Descriptor> = {
  Expired: said('Text', 'The day the deal expired - the position is held until it settles'),
  Settles: said('Text', 'The day it settles, on which it rolls off the book'),
};

/** Where one position sits and how big it is, as the record says - or that it closed. */
function Held({ position, paper }: { position: Position; paper: Paper }) {
  return (
    <DescriptorPanel title={`Position in ${position.portfolio || 'no portfolio'}${
                       closed(position.quantity) ? ', closed: the record holds none of it' : ''}`}
                     fields={position.expired ? { ...HELD, ...SETTLING } : HELD}
                     values={{
                       Quantity: position.quantity,
                       Agreement: position.agreement,
                       Counterparty: clientName(paper.entities, position.counterparty),
                       Clips: position.clips,
                       First_Fill: `LSN ${position.booked.lsn}`,
                       Last_Moved: `LSN ${position.last_lsn}`,
                       ...(position.expired
                         ? { Expired: position.expired, Settles: position.settles } : {}),
                     }} />
  );
}
