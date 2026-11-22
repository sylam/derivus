// Quote ticket card and its expandable leg table.

import { useRef, useState } from 'react';
import { askCharge } from '../api';
import { dateOf } from '../blotter';
import { fundedAt, limitRead } from '../client';
import { lagsBook, stampText } from '../desk';
import { NO_PAPER } from '../positions';
import {
  STANDS, chargeAsk, chargeLines, clientText, firmText, onMarketAxis, salesName, solvedNumbers,
  spreadText, ticketLegs, type TicketLeg,
} from '../quote';
import { clockOf, runHeld, useApp, useQueued, useStanding } from '../state';
import { formatNumber } from '../tokens';
import { tradeAt } from '../trade';
import type { ChargeAsk, Increment, PendingQuote, Schema } from '../types';
import { Gated, LimitLine, Money, TradeCells, Waiting } from './Desk';
import { DealPanels, FieldView } from './FieldView';

/** Quote ticket card: client, solved terms, net, market axis levels, spread, margin and window. */
export function TicketCard({ pending }: { pending: PendingQuote }) {
  const { state } = useApp();
  const { schema } = state;
  const paper = state.paper.data ?? NO_PAPER;
  const { quote } = pending;
  const legs = ticketLegs(quote, pending.deal, schema);
  const first = legs[0]?.row.deal ?? {};
  const surface = first.FX_Volatility;
  const spot = onMarketAxis(quote.spot.value_market, quote.params.pair, surface);
  const forward = onMarketAxis(quote.forward, quote.params.pair, surface);
  const standing = { ...pending, quote_id: quote.quote_id };
  const word = useStanding([standing], pending.as_of)(standing);
  const delivery = dateOf(first.Delivery_Date);
  const nodes = state.doc?.Calc.Deals?.Deals?.Children ?? [];
  const charge = chargeAsk(pending, nodes, state.tab === 'trade'
    ? tradeAt(nodes, state.selection.picks.trade ?? '')?.path : undefined);
  const level = (on: { pair: string; level: number } | undefined, raw: number) =>
    (on ? `${on.pair} ${formatNumber(on.level)}` : formatNumber(raw));

  return (
    <section className="card ticket">
      <h3 title={quote.structure}>
        {salesName(schema, quote.structure)}{quote.variation && ` · ${quote.variation}`}
      </h3>
      <div className="nested">
        <div className="ticketline">
          <b>{clientText(quote.netting_set, paper.entities, paper.agreements)}</b>
          {quote.client && (
            <span>the client buys {quote.client.buys}, sells {quote.client.sells}</span>
          )}
          <span>{String(quote.params.pair)} · {String(quote.params.expiry)}
            {legs[0]?.row.expiry && ` · expires ${legs[0].row.expiry}`}
            {delivery && ` · delivers ${delivery}`}</span>
        </div>
        <div className="stats">
          {/* unsigned on purpose: the net's sign is the client's direction, not a loss */}
          <div className="stat">
            <div className="label">net, in the client's sign</div>
            <div className="value">
              <Money value={quote.net} currency={quote.currency} />
            </div>
          </div>
          {solvedNumbers(legs).map((solved) => (
            <div key={solved.what} className="stat solved">
              <div className="label">solved · {solved.what}</div>
              <div className="value">
                {solved.on ? `${solved.on} ` : ''}{formatNumber(solved.value)}
              </div>
            </div>
          ))}
          <div className="stat">
            <div className="label">
              spot · {quote.spot.source === 'terminal' ? 'live' : "book's"}
            </div>
            <div className="value">{level(spot, quote.spot.value_market)}</div>
          </div>
          <div className="stat">
            <div className="label">outright forward{delivery ? ` to ${delivery}` : ''}</div>
            <div className="value">{level(forward, quote.forward)}</div>
          </div>
          {pending.firm_until && (
            <div className={`stat${word === 'lapsed' ? ' warn' : ''}`}>
              <div className="label">{word === 'lapsed' ? STANDS.lapsed : 'firm until'}</div>
              <div className="value">{stampText(pending.firm_until)}</div>
            </div>
          )}
        </div>
        <TicketLegs legs={legs} currency={quote.currency} />
        <div className="note">{spreadText(quote)}</div>
        {quote.margin && (
          <div className="note">
            Sales margin {formatNumber(quote.margin.amount, 'money')} {quote.margin.currency},{' '}
            {formatNumber(quote.margin.value, 'money')} {quote.margin.pricing_currency} at the
            book's spot, charged where the spread is.
          </div>
        )}
        {[quote.spot.note, ...(quote.notes ?? []),
          ...quote.legs.map((leg) => leg.note && `${leg.role}: ${leg.note}`)]
          .filter((note): note is string => !!note)
          .map((note) => <div key={note} className="hint ticketnote">{note}</div>)}
        <div className="ticketline">
          <span>quoted by <b>{pending.quoted_by ?? 'no seat named'}</b> at{' '}
            {stampText(pending.quoted_at)}</span>
          <span className="hint">{firmText(pending, state.record.spine !== null)}</span>
        </div>
        {charge && <CreditCharge key={JSON.stringify(charge)} ask={charge} />}
      </div>
    </section>
  );
}

/** What a trade does to its client's credit numbers, asked on demand and never on a poll - of a
 * quote's trade before it books, of a trade the book holds from then: its netting set's CVA, FVA
 * and peak exposures with the trade and without it, both runs on the same paths, or the service's
 * own sentence. A what-if, held from a seat that may run none. */
export function CreditCharge({ ask }: { ask: ChargeAsk }) {
  const { state } = useApp();
  const [read, setRead] = useState<{ answer: Increment; clock: string | null } | null>(null);
  const asked = useRef<string | null>(null);
  const asking = useQueued(askCharge, 'Increment',
                           (answer: Increment) => setRead({ answer, clock: asked.current }));
  const start = () => {
    asked.current = clockOf(state);
    setRead(null);
    asking.start(ask);
  };
  return (
    <>
      <div className="statusrow">
        <Gated label="credit charge" hold={runHeld(state)}>
          <button className="ghost" disabled={!!asking.run} onClick={start}>
            credit charge
          </button>
          <Waiting {...asking} />
          {!asking.run && !asking.error && (
            <span className="hint">what {'quote_id' in ask ? 'booking it does' : 'the trade adds'}
              {' '}to the client's CVA, FVA and peak exposure: a what-if, which writes
              nothing</span>
          )}
        </Gated>
      </div>
      {read && (
        <>
          <div className="tablewrap">
            <table className="data desk">
              <thead>
                <tr>
                  <th>{read.answer.netting_set}</th><th className="n">With the trade</th>
                  <th className="n">Without it</th><th className="n">The trade</th>
                </tr>
              </thead>
              <tbody>
                {chargeLines(read.answer).map((line) => (
                  <tr key={line.what}>
                    <td>{line.what}
                      {line.plus && <span className="hint"> · {line.plus}</span>}</td>
                    {(['with', 'without'] as const).map((run) => (
                      <td key={run} className="n">
                        <Money value={line[run]} currency={read.answer.currency}
                               kind={line.kind} />
                        {line.days && <div className="hint">{line.days[run]}</div>}
                      </td>
                    ))}
                    <td className="n">{line.moves}{line.by !== 0 && (
                      <> <Money value={line.by} currency={read.answer.currency}
                                kind={line.kind} /></>
                    )}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {read.answer.limits?.map((entry) => (
            <LimitLine key={entry.entity} read={limitRead(entry)}
                       whose={entry.name ?? entry.entity} />
          ))}
          <div className="hint">
            Both runs drew the same {formatNumber(read.answer.paths.count, 'whole')} paths
            {lagsBook(read.clock, clockOf(state))
              && ': the book or its record has moved since, so ask again'}.
            {state.xva.data && ` ${fundedAt(state.xva.data.funding)}`}
          </div>
        </>
      )}
    </>
  );
}

/** Whose side a quoted leg states. */
const CLIENT_SIDE = "the client's side of the leg, as quoted: the desk books its mirror";

/** Expandable table of quoted legs in the client's frame with solved levels highlighted. */
function TicketLegs({ legs, currency }: { legs: TicketLeg[]; currency: string }) {
  const { state } = useApp();
  const [opened, setOpened] = useState<string | null>(null);
  const midTip = 'what the leg is worth to the client at mid; '
    + 'the net adds the spread and the margin';
  return (
    <div className="tablewrap">
      <table className="data blotter ticketlegs">
        <thead>
          <tr>
            <th>Leg</th><th>Client</th><th>On</th><th className="n">Amount</th><th>Ccy</th>
            <th className="n">Strike</th><th className="n">Barrier</th><th>Terms</th>
            <th className="n" title={midTip}>At mid {currency}</th><th />
          </tr>
        </thead>
        <tbody>
          {legs.map((leg) => (
            <TicketLegRow key={leg.leg.reference} leg={leg}
                          opened={opened === leg.leg.reference} schema={state.schema}
                          onOpen={() => setOpened(opened === leg.leg.reference ? null
                            : leg.leg.reference)} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function TicketLegRow({ leg, opened, schema, onOpen }: {
  leg: TicketLeg; opened: boolean; schema: Schema | null; onOpen: () => void;
}) {
  return (
    <>
      <tr onClick={onOpen}>
        <td className="ref">{leg.leg.role}</td>
        <TradeCells size={leg.size} read={leg.read} object={leg.row.object} schema={schema}
                    solved={leg.solved} whose={CLIENT_SIDE} />
        <td className="n">{formatNumber(leg.leg.premium, 'money')}</td>
        <td className="opener"><span>{opened ? '×' : '⋯'}</span></td>
      </tr>
      {leg.schedule && (
        <tr className="schedule">
          <td />
          <td colSpan={9}>
            <FieldView value={leg.schedule.value} descriptor={leg.schedule.declared} />
          </td>
        </tr>
      )}
      {opened && (
        <tr className="detail">
          <td colSpan={10}><DealPanels deal={leg.row.deal} schema={schema} /></td>
        </tr>
      )}
    </>
  );
}
