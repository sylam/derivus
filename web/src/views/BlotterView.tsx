import { useMemo, useState } from 'react';
import {
  WINDOWS, baseDateOf, cellText, describe, filterRows, flatten, inWindow, sortRows, tagText,
  toRows, type BlotterRow,
} from '../blotter';
import { DescriptorPanel } from '../components/FieldView';
import { useApp } from '../state';
import type { Schema } from '../types';

/** The desk blotter: one row per booked deal over the live book, the tree flattened with
 * containers still holding their legs, sorted by days-to-roll so "who rolls off this week" is one
 * glance. Read-only by construction - the amendment surface is the portfolio view, and a blotter
 * that edited would be a blotter nobody could scan.
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

  const base = useMemo(() => (doc ? baseDateOf(doc) : undefined), [doc]);
  // an empty book is a book: a document carrying no Deals block still renders the empty blotter
  const tree = useMemo(() => (doc
    ? toRows(doc.Calc.Deals?.Deals?.Children ?? [], schema?.Instrument.containers ?? [], base)
    : []), [doc, schema, base]);

  if (!doc) return null;

  const horizon = WINDOWS.find((w) => w.id === horizonId)?.days ?? null;
  const rows = flatten(sortRows(filterRows(tree, horizon), sort), collapsed);
  // the deal COUNT is /describe's answer and the header already states it - this bar states only
  // what is the blotter's own reading, the roll-off
  const rolling = countRows(filterRows(tree, 30));

  function toggle(path: string) {
    setCollapsed((current) => {
      const next = new Set(current);
      if (next.has(path)) next.delete(path); else next.add(path);
      return next;
    });
  }

  return (
    <div className="main">
      <div className="panel">
        <div className="blotterbar">
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
          <span className="spacer" />
          <span className="hint">
            <b>{rolling}</b> within 30 days
          </span>
          <span className="hint">
            base date <span className="mono">{base ?? 'not stated'}</span>
          </span>
          <button className="ghost" onClick={() => setSort(sort === 'days' ? 'reference' : 'days')}>
            sort: {sort === 'days' ? 'days to roll' : 'reference'}
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
                <th>Reference</th>
                <th>Type</th>
                <th>Side</th>
                <th>Ccy</th>
                <th className="n">Amount</th>
                <th className="n">Strike</th>
                <th className="n">Barrier</th>
                <th>Expiry</th>
                <th>Next date</th>
                <th className="n">Days</th>
                <th>Tag</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <Row
                  key={row.path}
                  row={row}
                  schema={schema}
                  selected={selection.deal === row.path}
                  emphasised={horizon !== null ? inWindow(row, horizon) : inWindow(row, 7)}
                  collapsed={collapsed.has(row.path)}
                  opened={opened === row.path}
                  onToggle={() => toggle(row.path)}
                  onOpen={() => setOpened(opened === row.path ? null : row.path)}
                  tagTitles={tagText(doc.Calc.Deals?.Tag_Titles)}
                  onSelect={() => dispatch({ type: 'SELECT_DEAL', path: row.path })}
                />
              ))}
              {rows.length === 0 && (
                <tr><td colSpan={12} className="hint">nothing rolls off in this window.</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function Row(props: {
  row: BlotterRow; schema: Schema | null; selected: boolean; emphasised: boolean;
  collapsed: boolean; opened: boolean; tagTitles: string;
  onToggle: () => void; onOpen: () => void; onSelect: () => void;
}) {
  const { row, schema, selected, emphasised, collapsed, opened } = props;
  const sections = schema?.Instrument.types[row.object];

  return (
    <>
      <tr
        className={`${emphasised ? 'rolling ' : ''}${selected ? 'selected ' : ''}${
          row.ignored ? 'muted ' : ''}${row.container ? 'group' : ''}`}
        onClick={props.onSelect}
      >
        <td style={{ paddingLeft: 10 + row.depth * 18 }}>
          <span
            className="caret"
            onClick={(e) => { e.stopPropagation(); props.onToggle(); }}
          >
            {row.children.length ? (collapsed ? '▸' : '▾') : ''}
          </span>
          {row.reference || <span className="hint">(no reference)</span>}
          {row.ignored && <span className="badge">ignored</span>}
        </td>
        <td className="type">{row.object}</td>
        <td><Side row={row} /></td>
        <td>{row.currency ? cellText(row.currency.value) ?? '' : ''}</td>
        <Value row={row} field="amount" schema={schema} />
        <Value row={row} field="strike" schema={schema} />
        <Value row={row} field="barrier" schema={schema} />
        {/* a container carrying no dates of its own shows the roll it INHERITS, marked, so the
            days beside it are never a number with nothing behind it */}
        <td className="date">
          {row.expiry ?? (!row.next && row.roll ? <span className="hint">↳ {row.roll}</span> : '')}
        </td>
        <td className="date">{row.next && row.next !== row.expiry ? row.next : ''}</td>
        <Days row={row} />
        <td title={props.tagTitles || undefined}>{row.tag}</td>
        <td className="opener">
          <span onClick={(e) => { e.stopPropagation(); props.onOpen(); }}>
            {opened ? '×' : '⋯'}
          </span>
        </td>
      </tr>
      {opened && (
        <tr className="detail">
          <td colSpan={12}>
            {sections
              ? sections.map((section) => (
                  <DescriptorPanel
                    key={section}
                    title={section}
                    fields={schema?.Instrument.sections[section]}
                    values={row.deal}
                  />
                ))
              : <DescriptorPanel title={`${row.object} (undeclared type)`} values={row.deal} />}
          </td>
        </tr>
      )}
    </>
  );
}

/** Buy or sell as a chip, whatever the family calls it - Buy_Sell, Payer_Receiver, a lender. */
function Side({ row }: { row: BlotterRow }) {
  const text = row.side ? cellText(row.side.value) : undefined;
  if (!text) return null;
  const short = ['Sell', 'Pay', 'Payer', 'Lender'].includes(text) ? 'sell' : 'buy';
  return <span className={`side ${short}`} title={row.side!.key}>{text}</span>;
}

/** One picked column. A shape a cell cannot state says so and points at the expander, which
 * renders it whole - the viewer never hides a value it cannot format. */
function Value({ row, field, schema }: {
  row: BlotterRow; field: 'amount' | 'strike' | 'barrier'; schema: Schema | null;
}) {
  const picked = row[field];
  if (!picked) return <td className="n" />;
  const text = cellText(picked.value);
  const title = describe(schema, row.object, picked.key);
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

function countRows(rows: BlotterRow[]): number {
  return rows.reduce((total, row) => total + 1 + countRows(row.children), 0);
}
