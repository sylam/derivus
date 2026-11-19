import { useMemo, useState } from 'react';
import {
  HEADS, RECORD_HEADS, WINDOWS, baseDateOf, cellText, csvText, describe, exported, filterRows,
  flatten, groupedRows, heldRows, holdsTrades, inWindow, nameOf, quoted, rollingCount, rowsByPath,
  sizeOf, sortRows, toRows, type BlotterRow, type Picked,
} from '../blotter';
import { DealPanels } from '../components/FieldView';
import {
  GROUPINGS, NO_PAPER, POSITION, byId, byPath, clientTree, heldColumns, portfolioTree, type Paper,
  type Position,
} from '../positions';
import { marketOf, openTrade, useApp } from '../state';
import { formatNumber, type Kind } from '../tokens';
import type { Market, Schema } from '../types';

/** The desk blotter: one row per booked deal over the live book, the tree flattened with
 * containers still holding their legs, sorted by days-to-roll so "who rolls off this week" is one
 * glance. Read-only by construction - the amendment surface is the portfolio view, and a blotter
 * that edited would be a blotter nobody could scan. Where the desk keeps a record it reads the
 * Portfolio screen's grouping - the file's own nesting, by portfolio or by client - and every row
 * says what the record holds of it.
 *
 * It rides the SAME etag poll every other view does: the document in the store is the truth, so a
 * booking, an amendment or a market tick from any client repaints this table on the next tick with
 * no machinery of its own. */
export function BlotterView() {
  const { state, dispatch } = useApp();
  const { doc, schema, selection } = state;
  const [horizonId, setHorizonId] = useState('all');
  const [sort, setSort] = useState<'days' | 'reference'>('days');
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const [opened, setOpened] = useState<string | null>(null);

  const market = marketOf(state);
  const base = useMemo(() => (doc ? baseDateOf(doc, market) : undefined), [doc, market]);
  // an empty book is a book: a document carrying no Deals block still renders the empty blotter
  const book = useMemo(() => (doc
    ? toRows(doc.Calc.Deals?.Deals?.Children ?? [], schema?.Instrument.containers ?? [], base)
    : []), [doc, schema, base]);
  // the record's columns and groupings are the live book's: a copy opened from a file has none
  const recording = state.source?.kind === 'book' && state.record.spine !== null;
  const grouping = recording ? state.grouping : 'book';
  const paper = state.paper.data ?? NO_PAPER;
  const heldAt = useMemo(() => byPath(paper.positions), [paper]);
  const positions = useMemo(() => byId(paper.positions), [paper]);
  const tree = useMemo(() => (grouping !== 'book' ? groupedRows(
    grouping === 'portfolio' ? portfolioTree(paper.positions, paper.nodes) : clientTree(paper),
    rowsByPath(book), byId(paper.positions), base) : recording ? heldRows(book, heldAt) : book),
  [grouping, book, paper, base, recording, heldAt]);

  if (!doc) return null;

  const horizon = WINDOWS.find((w) => w.id === horizonId)?.days ?? null;
  const rows = flatten(sortRows(filterRows(tree, horizon), sort), collapsed);
  // the deal COUNT is /describe's answer and the header already states it - this bar states only
  // what is the blotter's own reading, the roll-off
  const rolling = rollingCount(tree, 30);
  const heads = recording ? [...HEADS, ...RECORD_HEADS] : HEADS;
  const heldOf = (row: BlotterRow): Position[] | null => (!recording ? null
    : row.position ? [row.position] : grouping === 'book' ? heldAt.get(row.path) ?? [] : []);
  // the deal path a trade's row opens its page at: its own, or the node its position is held at
  const tradeOf = (row: BlotterRow): string | undefined => (holdsTrades(row) ? undefined
    : row.target!.startsWith(POSITION) ? positions.get(row.target!)?.deal_paths[0] : row.target!);

  function toggle(path: string) {
    setCollapsed((current) => {
      const next = new Set(current);
      if (next.has(path)) next.delete(path); else next.add(path);
      return next;
    });
  }

  /** The rows as shown, in the columns shown, saved as CSV - with a byte-order mark, so a
   * spreadsheet reads the dots and dashes of a row's terms as written. */
  function download() {
    const table = rows.map((row) => [...exported(row, market, schema),
      ...(recording ? heldColumns(heldOf(row)!, paper.entities) : [])]);
    const link = document.createElement('a');
    link.href = URL.createObjectURL(new Blob(['﻿', csvText([heads, ...table])],
                                             { type: 'text/csv' }));
    link.download = `blotter ${base ?? 'undated'}.csv`;
    link.click();
    URL.revokeObjectURL(link.href);
  }

  return (
    <div className="main">
      <div className="panel">
        <div className="blotterbar">
          {recording && GROUPINGS.map((choice) => (
            <button key={choice.id} className={`ghost${grouping === choice.id ? ' on' : ''}`}
                    onClick={() => dispatch({ type: 'GROUP', grouping: choice.id })}>
              {choice.label}
            </button>
          ))}
          <span className="hint">roll-off window</span>
          {WINDOWS.map((window) => (
            <button
              key={window.id}
              className={`ghost${horizonId === window.id ? ' on' : ''}`}
              onClick={() => setHorizonId(window.id)}
            >
              {window.label}
            </button>
          ))}
          {!recording && (
            <span className="hint">
              {state.source?.kind === 'book' ? 'this desk keeps no record'
                : 'a file opened here has no record'}, so no grouping, quantity, portfolio, client
              or status
            </span>
          )}
          <span className="spacer" />
          <span className="hint" title={'a trade rolls off on its own expiry; a fixing or a payment '
                                           + 'before it is its next date, and does not roll it'}>
            <b>{rolling}</b> {rolling === 1 ? 'trade rolls' : 'trades roll'} off within 30 days
          </span>
          <span className="hint">
            base date <span className="mono">{base ?? 'not stated'}</span>
          </span>
          <button className="ghost" onClick={() => setSort(sort === 'days' ? 'reference' : 'days')}>
            sort: {sort === 'days' ? 'days to roll' : 'reference'}
          </button>
          <button className="ghost" onClick={download}
                  title="the rows as shown, in the columns shown, every number as itself">
            export CSV
          </button>
        </div>

        {!base && (
          <div className="banner">
            This book states no <b>Base_Date</b> — the date columns still read, but days-to-roll
            cannot be computed against anything, so the window filter shows everything.
          </div>
        )}

        <div className="tablewrap">
          <table className="data blotter">
            <thead>
              <tr>
                {heads.map((head) => (
                  <th key={head} className={NUMERIC.includes(head) ? 'n' : undefined}
                      title={head === 'Amount' && !recording ? undefined : TITLES[head]}>
                    {head}
                  </th>
                ))}
                <th />
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <Row
                  key={row.path}
                  row={row}
                  schema={schema}
                  market={market}
                  held={heldOf(row)}
                  entities={paper.entities}
                  selected={row.target !== null && selection.deal === row.target}
                  emphasised={horizon !== null ? inWindow(row, horizon) : inWindow(row, 7)}
                  collapsed={collapsed.has(row.path)}
                  opened={opened === row.path}
                  onToggle={() => toggle(row.path)}
                  onOpen={() => setOpened(opened === row.path ? null : row.path)}
                  onSelect={() => row.target !== null
                    && dispatch({ type: 'SELECT_DEAL', path: row.target })}
                  trade={tradeOf(row)}
                />
              ))}
              {rows.length === 0 && (
                <tr><td colSpan={heads.length + 1} className="hint">
                  nothing rolls off in this window.
                </td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

/** Whose side the Side column states. */
const SIDE = 'the desk\'s side of the trade, as the desk booked it: Sell is the desk selling, '
  + 'whatever the client calls the same trade';

const HELD = 'under a position, the written amount times the units the record holds of it';

const LEVELS = 'an FX level as the market quotes its pair, base then quote';

/** The heads that say more on hover, and the ones whose column is a number. */
const TITLES: Record<string, string> = {
  'Desk side': SIDE,
  On: 'what the trade is on - the pair as the market writes it, else the commodity, equity or '
    + 'index it names - and an option\'s sense on it',
  Amount: HELD,
  Strike: `${LEVELS}; a forward's outright, a swap's far one`,
  Barrier: LEVELS,
  Days: 'days to the roll: the expiry, else the next date',
};
const NUMERIC = ['Amount', 'Strike', 'Barrier', 'Days', 'Qty'];

function Row(props: {
  row: BlotterRow; schema: Schema | null; market: Market | null; held: Position[] | null;
  entities: Paper['entities']; selected: boolean; emphasised: boolean; collapsed: boolean;
  opened: boolean; onToggle: () => void; onOpen: () => void; onSelect: () => void;
  trade?: string;
}) {
  const { dispatch } = useApp();
  const { row, schema, selected, emphasised, collapsed, opened } = props;
  const size = sizeOf(row.deal, row.object, schema, row.units);
  const read = quoted(row, props.market, schema);

  return (
    <>
      <tr
        className={`${emphasised ? 'rolling ' : ''}${selected ? 'selected ' : ''}${
          row.ignored ? 'muted ' : ''}${row.container ? 'group' : ''}`}
        onClick={props.onSelect}
      >
        <td className="ref" style={{ paddingLeft: 10 + row.depth * 18 }} title={row.reference}>
          <span
            className="caret"
            onClick={(e) => { e.stopPropagation(); props.onToggle(); }}
          >
            {row.children.length ? (collapsed ? '▸' : '▾') : ''}
          </span>
          {props.trade !== undefined ? (
            <button className="tradelink" title="open the trade's page"
                    onClick={() => openTrade(dispatch, props.trade!)}>
              {nameOf(row) || '(no reference)'}
            </button>
          ) : nameOf(row) || <span className="hint">(no reference)</span>}
          {row.ignored && <span className="badge">ignored</span>}
        </td>
        <td className="type" title={row.object}>{row.object}</td>
        <td><Side side={size.side} /></td>
        <td title={read.on?.title ?? read.on?.key}>
          {read.on ? cellText(read.on.value) : ''}{read.sense && <span> {read.sense}</span>}
        </td>
        <td title={size.of?.key}>{size.of ? cellText(size.of.value) ?? '' : ''}</td>
        <Value picked={size.amount} object={row.object} schema={schema} kind="money" />
        <Value picked={read.strike} object={row.object} schema={schema} />
        <Value picked={read.barrier} object={row.object} schema={schema} />
        {/* a wrapped line never opens on the separator */}
        <td className="terms" title={read.terms?.title}>{read.terms?.parts.join('\u00a0· ')}</td>
        {/* a container carrying no dates of its own shows the roll it INHERITS, marked, so the
            days beside it are never a number with nothing behind it */}
        <td className="date">
          {row.expiry ?? (!row.next && row.roll ? <span className="hint">↳ {row.roll}</span> : '')}
        </td>
        <td className="date">{row.next && row.next !== row.expiry ? row.next : ''}</td>
        <Days row={row} />
        {props.held && <Held held={props.held} entities={props.entities} />}
        <td className="opener">
          <span onClick={(e) => { e.stopPropagation(); props.onOpen(); }}>
            {opened ? '×' : '⋯'}
          </span>
        </td>
      </tr>
      {opened && (
        <tr className="detail">
          <td colSpan={HEADS.length + (props.held ? RECORD_HEADS.length : 0) + 1}>
            <DealPanels deal={row.deal} schema={schema} />
          </td>
        </tr>
      )}
    </>
  );
}

/** What the record holds of a row: the net quantity, the portfolios and clients it sits under,
 * and what its tickets read with the quantity awaiting a second seat - one position where the book
 * is grouped by the record, every one the file's node holds where it is not. */
function Held({ held, entities }: { held: Position[]; entities: Paper['entities'] }) {
  const [quantity, portfolios, clients, status] = heldColumns(held, entities);
  return (
    <>
      <td className="n">{quantity === undefined ? '' : formatNumber(quantity)}</td>
      <td className="clip" title={portfolios}>{portfolios}</td>
      <td className="clip" title={clients}>{clients}</td>
      <td className={held.some((row) => row.status === 'rejected') ? 'rejected' : ''}>
        {status}
      </td>
    </>
  );
}

/** Buy or sell as a chip, whatever the family calls it - Buy_Sell, Payer_Receiver, a lender - and
 * always the desk's. */
export function Side({ side }: { side?: Picked }) {
  const text = side ? cellText(side.value) : undefined;
  if (!text) return null;
  const short = ['Sell', 'Pay', 'Payer', 'Lender'].includes(text) ? 'sell' : 'buy';
  return (
    <span className={`side ${short}`} title={`${SIDE} (${side!.title ?? side!.key})`}>{text}</span>
  );
}

/** One picked column. A shape a cell cannot state says so and points at the expander, which
 * renders it whole - the viewer never hides a value it cannot format. */
export function Value({ picked, object, schema, kind }: {
  picked?: Picked; object: string; schema: Schema | null; kind?: Kind;
}) {
  if (!picked) return <td className="n" />;
  const text = cellText(picked.value, kind);
  const exact = typeof picked.value === 'number' ? `: ${picked.value}` : '';
  const title = picked.title ?? describe(schema, object, picked.key) + exact;
  if (text === undefined) {
    return <td className="n"><span className="hint" title={title}>{'{…}'}</span></td>;
  }
  return <td className="n" title={title}>{text}</td>;
}

/** Days to roll: past is spent, this week is hot, the month is warm, undated is silent. */
function Days({ row }: { row: BlotterRow }) {
  if (row.days === undefined) return <td className="n hint">—</td>;
  const tone = row.days < 0 ? 'past' : row.days <= 7 ? 'hot' : row.days <= 30 ? 'warm' : '';
  return <td className={`n days ${tone}`} title={row.roll}>{row.days}</td>;
}
