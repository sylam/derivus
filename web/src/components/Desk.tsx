// Shared desk primitives: back navigation, side/value/money cells, valuation strip, deal editor.

import { useState, type ReactNode } from 'react';
import { amendDeal } from '../api';
import { cellText, describe, quoted, toRows, type Picked } from '../blotter';
import { LIMIT_USAGE, type LimitRead } from '../client';
import { marketSource, stampText, valuationDay } from '../desk';
import { byPath } from '../positions';
import { verdictOf } from '../quote';
import { heldBack, marketOf, spineOf, useApp, written } from '../state';
import { formatNumber, type Kind } from '../tokens';
import { judgedAt, tradeAt } from '../trade';
import type { DealNode, JobDoc, Schema } from '../types';
import { DealPanels, type AmendField } from './FieldView';

/** Back button to the tab the current page was opened from, if any. */
export function Back() {
  const { state, dispatch, labels } = useApp();
  const back = state.back === null ? undefined : labels[state.back];
  return back ? (
    <button className="ghost" onClick={() => dispatch({ type: 'TAB', tab: state.back! })}>
      ‹ back to {back}
    </button>
  ) : null;
}

/** Tooltip explaining whose side the Side column states. */
export const SIDE = 'the desk\'s side of the trade, as the desk booked it: '
  + 'Sell is the desk selling, whatever the client calls the same trade';

/** Buy or sell chip in the desk's frame, whatever field name the family uses. */
export function Side({ side }: { side?: Picked }) {
  const text = side ? cellText(side.value) : undefined;
  if (!text) return null;
  const short = ['Sell', 'Pay', 'Payer', 'Lender'].includes(text) ? 'sell' : 'buy';
  return (
    <span className={`side ${short}`} title={`${SIDE} (${side!.title ?? side!.key})`}>{text}</span>
  );
}

/** One picked numeric or scalar column cell; complex shapes show an expander hint. */
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

/** Formatted money figure: always amount followed by currency. */
export function Money({ value, currency, kind = 'money' }: {
  value: number; currency?: string | null; kind?: 'whole' | 'money';
}) {
  return (
    <>{formatNumber(value, kind)}{currency && <> <span className="unit">{currency}</span></>}</>
  );
}

/** One credit limit in a line (`limitRead`): whose it is where the line stands off its entity's
 * own row, its amounts, the flag where it is breached, and what is said after. */
export function LimitLine({ read, whose }: { read: LimitRead; whose?: string }) {
  return (
    <div className="ticketline limitline" title={LIMIT_USAGE}>
      {whose && <b>{whose}</b>}
      {read.figures.map((figure) => (
        <span key={figure.label}><span className="hint">{figure.label}</span>{' '}
          <b className={figure.value < 0 ? 'neg' : undefined}>
            <Money value={figure.value} currency={read.currency} kind="whole" /></b></span>
      ))}
      {read.flag && <span className="chip error">{read.flag}</span>}
      {read.notes.map((note) => <span key={note} className="hint">{note}</span>)}
    </div>
  );
}

/** Table cell for an optional amount (whole units by default), a negative one in the loss
 * colour; null shows a dash under the hover `absent`, and nothing where none is given. */
export function Amount({ value, currency, absent, kind = 'whole' }: {
  value: number | null; currency?: string | null; absent?: string; kind?: 'whole' | 'money';
}) {
  return (
    <td className={`n${value !== null && value < 0 ? ' neg' : ''}`}
        title={value === null ? undefined : String(value)}>
      {value !== null ? <Money value={value} currency={currency} kind={kind} />
        : absent && <span className="hint" title={absent}>—</span>}
    </td>
  );
}

/** A queued job's wait, and what ended it where that was a refusal (`useQueued`). */
export function Waiting({ run, error }: {
  run: { status: string; since: number } | null; error: string | null;
}) {
  return (
    <>
      {run && (
        <span className="chip running">
          pricing - {run.status}, {Math.floor((Date.now() - run.since) / 1000)} s
        </span>
      )}
      {error && <div className="error-box">{error}</div>}
    </>
  );
}

/** Header strip stating valuation day, timestamp, currency and market for a screen's numbers. */
export function OnStrip({ computed, title, currency, lagging, loading, doc, again }: {
  computed: ReactNode; title?: string; currency: ReactNode; lagging: boolean; loading: boolean;
  doc?: JobDoc; again?: string;
}) {
  const { state } = useApp();
  const on = doc ?? state.doc;
  const market = marketSource(on, marketOf(state));
  return (
    <div className={`onstrip${lagging ? ' behind' : ''}`}>
      <span><span className="k">valued on</span><b>{valuationDay(on) || '—'}</b></span>
      <span title={title}><span className="k">computed</span>{computed}</span>
      <span><span className="k">reported in</span><b>{currency || '—'}</b></span>
      <span>
        <span className="k">market</span>
        {market.file ? <><b>{market.file}</b> with the book's own block over it</>
          : <>the book's own block</>}
        {market.seen && <>, quotes last seen {stampText(market.seen)}</>}
      </span>
      {lagging && (
        <span className="lag">
          These numbers are of an earlier book: it or its record has moved since they were read.
          {again ?? (loading ? ' A fresh read is on its way.'
            : ' They are read again on the next poll.')}
        </span>
      )}
    </div>
  );
}

const RESTRIKE = 'Filing this restrikes the trade: its new terms are filed under a new instrument '
  + 'and go through approval again.';

/** Deal panels at `path` with market-quoted hints and live scalar editing (restrikes on record). */
export function DealEditor({ path, node }: { path: string; node: DealNode }) {
  const { state, dispatch } = useApp();
  const [said, setSaid] = useState<string[]>([]);
  const recording = spineOf(state) !== null;
  const read = quoted(toRows([node], state.schema?.Instrument.containers ?? [], undefined)[0],
                      marketOf(state));
  // the positions of the trade the path names, a leg's being its structure's
  const trade = tradeAt(state.doc?.Calc.Deals.Deals.Children ?? [], path)?.path ?? path;
  const held = byPath(state.paper.data?.positions ?? []).get(trade) ?? [];
  const beside = (key: string) => {
    const level = [read.strike, read.barrier].find((picked) => picked?.key === key);
    if (level?.title) return <span className="readas">the market quotes {level.title}</span>;
    if (key === 'Barrier_Type' && read.terms) {
      return <span className="readas">the market reads {read.terms.parts.join(' · ')}</span>;
    }
    return key === 'Option_Type' && read.on && read.sense && (
      <span className="readas">
        a {read.sense} on {String(read.on.value)} as the market quotes it</span>);
  };
  const onAmend: AmendField | undefined = state.source?.kind === 'book'
    ? async (key, wireValue) => {
      const answer = await amendDeal(path, { [key]: wireValue });
      setSaid(answer.written ? verdictOf(answer).words : []);
      return written(dispatch, answer);
    } : undefined;
  return (
    <>
      {said.map((words) => <div key={words} className="note">{words}</div>)}
      <DealPanels deal={node.Instrument['.Deal']} schema={state.schema} onAmend={onAmend}
                  hold={heldBack(state, 'amendment', judgedAt(held))}
                  beside={beside} ask={recording ? RESTRIKE : undefined} />
    </>
  );
}
