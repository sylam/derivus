import { Fragment, useMemo, useState, type ReactNode } from 'react';
import { NO_SEAT } from '../api';
import {
  factorAxes, factorFamily, lagsBook, markHolds, marketSource, marksTotal, onPair, perOf,
  perUnitText, plural, positionsUnder, quoteText, quoteView, reconciles, riskOf, riskReading,
  sharedPer, sortGreeks, stampText, tenorText, valuationDay, type GreekKey, type Ladder,
} from '../desk';
import { EmptyState } from '../components/EmptyState';
import type { Position } from '../positions';
import { clockOf, marketOf, openTrade, useApp } from '../state';
import { axisText, formatNumber } from '../tokens';
import { tradeAt } from '../trade';
import type { BookRisk, GreekRow, JobDoc } from '../types';

/** The consolidated risk: the book's own mark and its whole-book gradient, counterparty-blind.
 *
 * COUNTERPARTIES DO NOT ENTER HERE. This is the desk's risk across everything it holds, aggregated
 * over the entire book, which is why there is nothing on this screen to slice by - the per-set
 * reading is the XVA view, and it is a different kind of number for a different reason.
 *
 * READ IN A DESK'S UNITS: every row per the move its factor is quoted by - a spot per 1% of
 * itself, a curve per basis point, a vol per vol point - spots together, each curve by tenor, each
 * surface by expiry; the per-unit derivative a row was read from is its hover.
 */
export function RiskView() {
  const { state, dispatch } = useApp();
  const { risk, source } = state;
  const [key, setKey] = useState<GreekKey>('magnitude');
  const [descending, setDescending] = useState(true);
  // the desk's grouped reading by default; the quote ladder, where the book's factors are built
  // from quotes, one click away
  const quotes = (risk.data?.quotes ?? []).length > 0 && state.selection.picks.risk === 'quotes';

  const market = marketOf(state);
  const rows = useMemo(
    () => (risk.data && quotes ? sortGreeks(quoteView(risk.data), key, descending)
      .map((row) => onPair(row, market)) : []),
    [risk.data, quotes, key, descending, market]);

  // the risk verb answers over the LIVE book; a document opened from a file has no server-side
  // book to differentiate, and saying so is better than showing another book's numbers under it
  if (source?.kind !== 'book') {
    return (
      <EmptyState
        title="Consolidated risk is a reading of the live book."
        hint="This document was opened from a file. Start the service with --book <job file> and the desk's own risk appears here."
      />
    );
  }
  if (risk.status === 404) {
    return (
      <EmptyState
        title="The service is serving no book."
        hint="It was started without --book, or the book has been closed since this page loaded."
      />
    );
  }
  // a refusal is the service's own wording, verbatim, and it stands in place of the numbers -
  // nothing partial was served and nothing was cached, so there is nothing else to show
  if (risk.error) {
    return (
      <div className="main">
        <div className="panel">
          <div className={risk.error === NO_SEAT ? 'banner' : 'error-box'}>{risk.error}</div>
          <div className="hint">
            {risk.error === NO_SEAT
              ? 'The risk is read again as soon as a seat is set.'
              : risk.status === 422
              ? 'Nothing was cached and nothing partial was served. Fix the book — the etag moves, and this screen asks again on its own.'
              : 'The consolidated risk could not be read.'}
          </div>
        </div>
      </div>
    );
  }
  if (!risk.data) {
    return (
      <EmptyState
        title={risk.loading ? 'Pricing the book…' : 'No risk has been read yet.'}
        hint={risk.loading
          ? 'One base valuation with first-order greeks over the whole book. A standing book answers from the cache; a moved one is run.'
          : 'The risk is fetched off the book poll, when this view is the one being read.'}
      />
    );
  }

  const data = risk.data;
  const lagging = lagsBook(risk.clock, clockOf(state));
  const left = data.quotes.filter((row) => row.value === 0).length;
  const currency = data.currency ?? '';

  function sortBy(next: GreekKey) {
    if (next === key) setDescending(!descending);
    else { setKey(next); setDescending(true); }
  }

  return (
    <div className="main">
      <div className="panel">
        <OnStrip
          computed={stampText(data.as_of)} title={`risk etag ${data.etag}`} currency={currency}
          lagging={lagging} loading={risk.loading}
        />
        {data.quote_note && <div className="banner">{data.quote_note}</div>}

        {data.per_deal.length === 0 && data.greeks.length === 0 ? (
          <div className="note">
            Nothing was marked and nothing was differentiated — an empty book answers zeros
            without running, because there is no value to take a gradient of. Book a trade and the
            numbers appear here on the next tick.
          </div>
        ) : (
          <>
            <Marks risk={data} currency={currency} />
            <div className="blotterbar" style={{ marginTop: 18 }}>
              {data.quotes.length > 0 && [['factors', 'by factor'], ['quotes', 'by quote']].map(
                ([view, words]) => (
                  <button key={view} className={`ghost${(view === 'quotes') === quotes ? ' on' : ''}`}
                          onClick={() => dispatch({ type: 'PICK', screen: 'risk', id: view })}>
                    {words}
                  </button>
                ))}
              <span className="hint">
                {quotes ? (
                  <><b>Risk in quotes</b> — the report-currency change per each quote's move,
                    through the calibration that builds the factor: a vol quote per vol point, any
                    other quote per unit as quoted; a factor no quote builds keeps its own row
                    {left > 0 && `, and the ${plural(left, 'quote')} the book does not move with `
                      + `${left === 1 ? 'is' : 'are'} left off`}</>
                ) : (
                  <><b>Risk as a desk reads it</b> — the report-currency change per the move each
                    factor is quoted by, over the whole book; hover a number for the derivative per
                    unit it was read from</>
                )}
              </span>
              <span className="spacer" />
              {risk.loading && <span className="hint">refreshing…</span>}
            </div>
            {quotes ? (
              <div className="tablewrap">
                <table className="data desk">
                  <thead>
                    <tr>
                      <SortTh
                        label="Block · Factor" mine="factor" key_={key} descending={descending}
                        onSort={sortBy}
                        title="sorted by name, each factor's coordinates in curve order beneath it"
                      />
                      <th>Quote · Tenor</th>
                      <SortTh
                        label={`Risk${currency ? ` (${currency})` : ''}`} mine="magnitude"
                        key_={key} descending={descending} onSort={sortBy} numeric
                        title="sorted by size of exposure per each row's move, sign carried but not sorted on"
                      />
                      <th>per</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((row, position) => (
                      <tr key={`${row.factor}:${tenorText(row.tenor)}:${position}`}>
                        <FactorCell factor={row.factor} pair={row.pair} />
                        <td className="tenor" title={row.quote ?? row.tenor?.join(' / ')}>
                          {row.quote === undefined ? tenorText(row.tenor, factorAxes(
                            state.schema, row.factor, row.tenor?.length ?? 0)) : quoteText(row.quote)}
                        </td>
                        <Risk row={row} />
                        <td className="hint">{perOf(row)}</td>
                      </tr>
                    ))}
                    {rows.length === 0 && (
                      <tr><td colSpan={4} className="hint">this run moved with no quote.</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
            ) : (
              <DeskReading rows={data.greeks} currency={currency} />
            )}
          </>
        )}
      </div>
    </div>
  );
}

/** What the numbers on a desk screen are ON: the day the book values on, when they were computed,
 * the currency they report in and where the market came from as far as the book says - and, where
 * the answer was read at an older clock than the book on screen, that in words. `doc` is the
 * document they were computed over where that is not the one on screen, and `again` what brings
 * fresh ones where nothing reads them again on its own. */
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

/** The gradient as a desk reads it, every number per the move its factor is quoted by: the spot
 * deltas together, each curve by tenor, each vol surface by expiry with its nodes a click away, and
 * whatever has no declared move per unit, apart. */
function DeskReading({ rows, currency }: { rows: GreekRow[]; currency: string }) {
  const { state } = useApp();
  const reading = useMemo(() => riskReading(rows, state.schema), [rows, state.schema]);
  const unit = currency ? ` (${currency})` : '';
  if (rows.length === 0) {
    return (
      <div className="hint">
        this run reported no gradient — the book marks, but nothing in it moves with a factor.
      </div>
    );
  }
  const per = sharedPer(reading.spots);
  const spots = reading.spots.map((row) => onPair(row, marketOf(state)));
  return (
    <>
      {spots.length > 0 && (
        <section className="card ladder">
          <h3>Spot delta{spots.some((row) => row.pair) && (
            <span className="hint">an FX spot reads on its pair as the market quotes it, and its
              delta per 1% of that pair</span>)}</h3>
          <table className="data desk">
            <thead>
              <tr>
                <th>Factor</th>
                <th className="n">Level</th>
                <th className="n">{`Delta ${per ?? ''}${unit}`}</th>
              </tr>
            </thead>
            <tbody>
              {spots.map((row) => (
                <tr key={row.factor}>
                  <FactorCell factor={row.factor} pair={row.pair}
                              per={per ? undefined : perOf(row)} />
                  <td className="n">{row.level === null || row.level === undefined
                    ? '' : formatNumber(row.level)}</td>
                  <Risk row={row} />
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
      {([['Curves, by tenor', reading.curves], ['Vol surfaces, vega by expiry', reading.surfaces]] as
        [string, Ladder[]][]).map(([title, ladders]) => ladders.length > 0 && (
          <Fragment key={title}>
            <h4 className="ladders-title">{title}</h4>
            <div className="ladders">
              {ladders.map((ladder) => (
                <LadderCard key={ladder.factor} ladder={ladder} unit={unit} />
              ))}
            </div>
          </Fragment>
        ))}
      {reading.units.length > 0 && (
        <section className="card ladder">
          <h3>Read per unit — no move is declared for these</h3>
          <table className="data desk">
            <thead>
              <tr><th>Factor</th><th>Coordinates</th><th className="n">{`per unit${unit}`}</th></tr>
            </thead>
            <tbody>
              {reading.units.map((row, position) => (
                <tr key={`${row.factor}:${position}`}>
                  <FactorCell factor={row.factor} />
                  <td className="tenor">{tenorText(row.tenor, factorAxes(
                    state.schema, row.factor, row.tenor?.length ?? 0))}</td>
                  <Risk row={row} />
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </>
  );
}

/** One curve's ladder or one surface's buckets: a row per tenor or expiry, the total its parallel
 * move, and a surface bucket opening onto the nodes it sums. */
function LadderCard({ ladder, unit }: { ladder: Ladder; unit: string }) {
  const [opened, setOpened] = useState<string | null>(null);
  const summed = ladder.buckets.some((bucket) => bucket.nodes.length > 1);
  const axes = ladder.axes.filter((_, at) => at !== ladder.across);
  return (
    <section className="card ladder">
      <h3 title={ladder.factor}>
        <span className="family">{factorFamily(ladder.factor)}</span>
        {ladder.factor.slice(factorFamily(ladder.factor).length)}
      </h3>
      <table className="data desk">
        <thead>
          <tr>
            <th>{axes.join(' / ') || 'coordinates'}</th>
            <th className="n">{`${ladder.per ?? 'risk'}${unit}`}</th>
            {summed && <th className="n">nodes</th>}
          </tr>
        </thead>
        <tbody>
          {ladder.buckets.map((bucket) => {
            const id = JSON.stringify(bucket.key);
            return (
              <Fragment key={id}>
                <tr className={summed ? 'pick' : undefined}
                    onClick={() => summed && setOpened(opened === id ? null : id)}>
                  <td className="tenor">{tenorText(bucket.key, axes)}</td>
                  {bucket.nodes.length === 1 ? <Risk row={bucket.nodes[0]} /> : (
                    <td className={`n${bucket.risk < 0 ? ' neg' : ''}`}
                        title={`the ${bucket.nodes.length} nodes at this expiry summed`}>
                      {formatNumber(bucket.risk, 'money')}
                    </td>
                  )}
                  {summed && (
                    <td className="n hint">{opened === id ? '▾' : '▸'} {bucket.nodes.length}</td>
                  )}
                </tr>
                {opened === id && bucket.nodes.map((node) => (
                  <tr key={JSON.stringify(node.tenor)} className="node">
                    <td className="tenor">{ladder.axes[ladder.across]}{' '}
                      {axisText(ladder.axes[ladder.across], node.tenor![ladder.across])}</td>
                    <Risk row={node} />
                    <td />
                  </tr>
                ))}
              </Fragment>
            );
          })}
          <tr className="total">
            <td>{summed ? 'parallel, every node' : 'parallel, every tenor'}</td>
            <td className={`n${ladder.total < 0 ? ' neg' : ''}`}>
              {formatNumber(ladder.total, 'money')}
            </td>
            {summed && <td />}
          </tr>
        </tbody>
      </table>
    </section>
  );
}

/** A factor by its name, or an FX spot by the pair it is read on with its name beside. */
function FactorCell({ factor, per, pair }: { factor: string; per?: string; pair?: string }) {
  return (
    <td title={factor}>
      {pair ? <><b>{pair}</b> <span className="hint">{factor}</span></> : <>
        <span className="family">{factorFamily(factor)}</span>
        {factor.slice(factorFamily(factor).length)}</>}
      {per && <span className="hint"> {per}</span>}
    </td>
  );
}

/** A row's risk per its move, the per-unit derivative it was read from on the hover. */
function Risk({ row }: { row: GreekRow }) {
  const value = riskOf(row);
  return (
    <td className={`n${value < 0 ? ' neg' : ''}`} title={perUnitText(row)}>
      {formatNumber(value, 'money')}
    </td>
  );
}

/** The per-deal marks, the desk's side of each: one row per TOP-LEVEL trade, so a structure or a
 * netting set appears once with its net and `mtm` is exactly this column's total. The legs are not
 * here on purpose - they are inside the row their container reports, and adding them again would
 * double the book. On a desk that records, a row names the clients it is against and the trades
 * in it that wait on a second seat. A trade's row opens its page; a set's picks it here. */
function Marks({ risk, currency }: { risk: BookRisk; currency: string }) {
  const { state, dispatch } = useApp();
  const total = marksTotal(risk.per_deal);
  const agrees = reconciles(risk);
  const recording = state.record.spine !== null;
  const positions = state.paper.data?.positions;
  const under = positionsUnder(risk.per_deal, positions ?? []);
  const holds = (path: string | null) => markHolds(
    (path === null ? undefined : under.get(path)) ?? [], state.paper.data?.entities ?? []);
  const waiting = risk.per_deal.flatMap((row) => holds(row.deal_path).pending);
  const nodes = state.doc?.Calc.Deals.Deals.Children ?? [];

  return (
    <>
      <div className="stats">
        <div className="stat">
          <div className="label">Book MTM</div>
          <div className={`value${risk.mtm < 0 ? ' neg' : ''}`} title={String(risk.mtm)}>
            {formatNumber(risk.mtm, 'money')}
            {currency && <span className="unit">{currency}</span>}
          </div>
        </div>
        <div className="stat">
          <div className="label">Marked</div>
          <div className="value">
            {risk.per_deal.length}<span className="unit">
              top-level {risk.per_deal.length === 1 ? 'row' : 'rows'}</span>
          </div>
        </div>
        {recording && (
          <div className={`stat${waiting.length ? ' warn' : ''}`}>
            <div className="label">Pending approval</div>
            <div className="value">
              {positions ? waiting.length : '…'}<span className="unit">
                {positions ? `${waiting.length === 1 ? 'trade' : 'trades'} inside these marks, `
                  + 'priced all the same' : 'reading the record'}</span>
            </div>
          </div>
        )}
      </div>
      <div className="blotterbar">
        <span className="hint">
          <b>Per-deal marks</b> — the desk's side of each: positive is worth that much to the
          desk; one row per top-level node of the book, a structure and a netting set each
          reporting their net
        </span>
      </div>
      <div className="tablewrap">
        <table className="data desk marks">
          <thead>
            <tr>
              <th>Reference</th>
              {recording && <th>Client</th>}
              <th>Path</th>
              {recording && <th>Waiting on a second seat</th>}
              <th className="n" title="the mark from the desk's side: positive is worth that much to the desk">
                {`Desk's value${currency ? ` (${currency})` : ''}`}</th>
            </tr>
          </thead>
          <tbody>
            {risk.per_deal.map((row, position) => {
              const { clients, pending: held } = holds(row.deal_path);
              return (
                <tr
                  key={`${row.deal_path ?? '?'}:${position}`}
                  className={`${row.deal_path ? 'pick ' : ''}${held.length ? 'pending ' : ''}${
                    row.deal_path !== null && state.selection.deal === row.deal_path ? 'selected' : ''}`}
                  onClick={() => row.deal_path && (tradeAt(nodes, row.deal_path)
                    ? openTrade(dispatch, row.deal_path)
                    : dispatch({ type: 'SELECT_DEAL', path: row.deal_path }))}
                >
                  <td>{row.reference || <span className="hint">(no reference)</span>}</td>
                  {recording && (
                    <td>{positions ? clients.join(', ')
                      || <span className="hint">—</span> : <span className="hint">…</span>}</td>
                  )}
                  <td className="path">
                    {row.deal_path ?? (
                      <span className="hint" title="this reference is not unique in the book, so no positional identity is offered for it">
                        not unique
                      </span>
                    )}
                  </td>
                  {recording && <td><PendingCell held={held} /></td>}
                  <td className={`n${row.value < 0 ? ' neg' : ''}`} title={String(row.value)}>
                    {formatNumber(row.value, 'money')}
                  </td>
                </tr>
              );
            })}
            {risk.per_deal.length === 0 && (
              <tr><td colSpan={recording ? 5 : 3} className="hint">this book marks nothing.</td></tr>
            )}
            <tr className="total">
              <td>Book</td>
              <td />
              {recording && <><td /><td /></>}
              <td className={`n${total < 0 ? ' neg' : ''}`} title={String(total)}>
                {formatNumber(total, 'money')}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      {/* the service promises `mtm` IS this total; a disagreement is the one thing on this screen
          worth interrupting a reader for */}
      {!agrees && (
        <div className="error-box">
          The headline mark ({formatNumber(risk.mtm, 'money')}) is not the sum of the rows above
          ({formatNumber(total, 'money')}). Read neither number until that is explained.
        </div>
      )}
    </>
  );
}

function PendingCell({ held }: { held: Position[] }) {
  if (!held.length) return <span className="hint">—</span>;
  return (
    <span className="badge pending" title="booked, and waiting on a second seat to approve it: the mark prices it all the same">
      {plural(held.length, 'trade')} pending: {held.map((position) => position.reference
        ?? position.instrument.slice(0, 12)).join(', ')}
    </span>
  );
}

function SortTh({ label, mine, key_, descending, onSort, numeric, title }: {
  label: string; mine: GreekKey; key_: GreekKey; descending: boolean;
  onSort: (key: GreekKey) => void; numeric?: boolean; title: string;
}) {
  const on = key_ === mine;
  return (
    <th
      className={`sort${numeric ? ' n' : ''}`}
      onClick={() => onSort(mine)}
      title={title}
    >
      {label}
      <span className="arrow">{on ? (descending ? '▼' : '▲') : ''}</span>
    </th>
  );
}
