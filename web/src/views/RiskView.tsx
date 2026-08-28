import { useMemo, useState } from 'react';
import {
  factorFamily, lagsBook, marksTotal, reconciles, sortGreeks, stampText, tenorText,
  type GreekKey,
} from '../desk';
import { EmptyState } from '../components/EmptyState';
import { useApp } from '../state';
import { formatNumber } from '../tokens';
import type { BookRisk } from '../types';

/** The consolidated risk: the book's own mark and its whole-book gradient, counterparty-blind.
 *
 * COUNTERPARTIES DO NOT ENTER HERE. This is the desk's risk across everything it holds, aggregated
 * over the entire book, which is why there is nothing on this screen to slice by - the per-set
 * reading is the XVA view, and it is a different kind of number for a different reason.
 *
 * Read-only by construction, like the blotter: the amendment surface is the portfolio view. It
 * rides the same etag every other view does, so a booking from any client moves the book, the
 * risk is refetched behind it, and the strip says plainly while the two are apart.
 */
export function RiskView() {
  const { state } = useApp();
  const { risk, source } = state;
  const [key, setKey] = useState<GreekKey>('magnitude');
  const [descending, setDescending] = useState(true);

  const rows = useMemo(
    () => (risk.data ? sortGreeks(risk.data.greeks, key, descending) : []),
    [risk.data, key, descending]);

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
          <div className="error-box">{risk.error}</div>
          <div className="hint">
            {risk.status === 422
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
  const lagging = lagsBook(risk.bookEtag, source.etag);
  const currency = data.currency ?? '';

  function sortBy(next: GreekKey) {
    if (next === key) setDescending(!descending);
    else { setKey(next); setDescending(true); }
  }

  return (
    <div className="main">
      <div className="panel">
        <div className="stats">
          <div className="stat">
            <div className="label">Book MTM</div>
            <div className={`value${data.mtm < 0 ? ' neg' : ''}`}>
              {formatNumber(data.mtm)}
              {currency && <span className="unit">{currency}</span>}
            </div>
          </div>
          <div className="stat">
            <div className="label">Marked</div>
            <div className="value">
              {data.per_deal.length}<span className="unit">
                top-level {data.per_deal.length === 1 ? 'trade' : 'trades'}</span>
            </div>
          </div>
          <div className="stat">
            <div className="label">Gradient</div>
            <div className="value">
              {data.greeks.length}<span className="unit">
                {data.greeks.length === 1 ? 'row' : 'rows'}</span>
            </div>
          </div>
          <div className="stat">
            <div className="label">Computed</div>
            <div className="value" title={data.as_of}>{stampText(data.as_of)}</div>
          </div>
          <div className="stat">
            <div className="label">Risk etag</div>
            <div className="value" title={data.etag}>{data.etag.slice(0, 12)}</div>
          </div>
        </div>

        {/* the two etags are hashes of DIFFERENT things and are never compared to each other:
            what is asked is whether the book has moved since this answer was fetched */}
        {lagging && (
          <div className="banner">
            The book has moved since this risk was computed — it is at etag{' '}
            <span className="mono">{source.etag.slice(0, 12)}</span> and these numbers were fetched
            at <span className="mono">{(risk.bookEtag ?? '').slice(0, 12)}</span>.{' '}
            {risk.loading ? 'A fresh run is on its way.' : 'They will refresh on the next poll.'}
          </div>
        )}

        {data.per_deal.length === 0 && data.greeks.length === 0 ? (
          <div className="note">
            Nothing was marked and nothing was differentiated — an empty book answers zeros
            without running, because there is no value to take a gradient of. Book a trade and the
            numbers appear here on the next tick.
          </div>
        ) : (
          <>
            <div className="blotterbar">
              <span className="hint">
                <b>Aggregate greeks</b> — the report-currency derivative per unit of each factor,
                over the whole book
              </span>
              <span className="spacer" />
              {risk.loading && <span className="hint">refreshing…</span>}
            </div>
            <div className="tablewrap">
              <table className="data desk">
                <thead>
                  <tr>
                    <SortTh
                      label="Factor" mine="factor" key_={key} descending={descending}
                      onSort={sortBy}
                      title="sorted by name, each factor's coordinates in curve order beneath it"
                    />
                    <th>Tenor</th>
                    <SortTh
                      label={currency ? `Value (${currency})` : 'Value'} mine="magnitude"
                      key_={key} descending={descending} onSort={sortBy} numeric
                      title="sorted by size of exposure, sign carried but not sorted on"
                    />
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row, position) => (
                    <tr key={`${row.factor}:${tenorText(row.tenor)}:${position}`}>
                      <td title={row.factor}>
                        <span className="family">{factorFamily(row.factor)}</span>
                        {row.factor.slice(factorFamily(row.factor).length)}
                      </td>
                      {/* a factor with no coordinates carries no tenor key at all - the cell is
                          empty because there is nothing there, not because it was dropped */}
                      <td className="n tenor">{tenorText(row.tenor)}</td>
                      <td className={`n${row.value < 0 ? ' neg' : ''}`}>
                        {formatNumber(row.value)}
                      </td>
                    </tr>
                  ))}
                  {rows.length === 0 && (
                    <tr>
                      <td colSpan={3} className="hint">
                        this run reported no gradient — the book marks, but nothing in it moves
                        with a factor.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>

            <Marks risk={data} currency={currency} />
          </>
        )}
      </div>
    </div>
  );
}

/** The per-deal marks: one row per TOP-LEVEL trade, so a structure or a netting set appears once
 * with its net and `mtm` is exactly this column's total. The legs are not here on purpose - they
 * are inside the row their container reports, and adding them again would double the book. */
function Marks({ risk, currency }: { risk: BookRisk; currency: string }) {
  const { state, dispatch } = useApp();
  const total = marksTotal(risk.per_deal);
  const agrees = reconciles(risk);

  return (
    <>
      <div className="blotterbar" style={{ marginTop: 18 }}>
        <span className="hint">
          <b>Per-deal marks</b> — one row per top-level trade; a structure and a netting set each
          report their net
        </span>
      </div>
      <div className="tablewrap">
        <table className="data desk marks">
          <thead>
            <tr>
              <th>Reference</th>
              <th>Path</th>
              <th className="n">{currency ? `Value (${currency})` : 'Value'}</th>
            </tr>
          </thead>
          <tbody>
            {risk.per_deal.map((row, position) => (
              <tr
                key={`${row.deal_path ?? '?'}:${position}`}
                className={`${row.deal_path ? 'pick ' : ''}${
                  row.deal_path !== null && state.selection.deal === row.deal_path ? 'selected' : ''}`}
                onClick={() => row.deal_path
                  && dispatch({ type: 'SELECT_DEAL', path: row.deal_path })}
              >
                <td>{row.reference || <span className="hint">(no reference)</span>}</td>
                <td className="path">
                  {row.deal_path ?? (
                    <span className="hint" title="this reference is not unique in the book, so no positional identity is offered for it">
                      not unique
                    </span>
                  )}
                </td>
                <td className={`n${row.value < 0 ? ' neg' : ''}`}>{formatNumber(row.value)}</td>
              </tr>
            ))}
            {risk.per_deal.length === 0 && (
              <tr><td colSpan={3} className="hint">this book marks nothing.</td></tr>
            )}
            <tr className="total">
              <td>Book</td>
              <td />
              <td className={`n${total < 0 ? ' neg' : ''}`}>{formatNumber(total)}</td>
            </tr>
          </tbody>
        </table>
      </div>
      {/* the service promises `mtm` IS this total; a disagreement is the one thing on this screen
          worth interrupting a reader for */}
      {!agrees && (
        <div className="error-box">
          The headline mark ({formatNumber(risk.mtm)}) is not the sum of the rows above
          ({formatNumber(total)}). Read neither number until that is explained.
        </div>
      )}
    </>
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
