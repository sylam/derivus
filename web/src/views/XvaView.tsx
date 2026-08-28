import { useState } from 'react';
import {
  ageOf, currencyOf, isOrphan, lastRunChip, replayRows, stampText, statusChip, xvaTotals,
} from '../desk';
import { EmptyState } from '../components/EmptyState';
import { useApp } from '../state';
import { formatNumber } from '../tokens';
import type { XvaSet } from '../types';

/** The XVA projection: one row per netting set, the last run over what the book says the set is
 * now. Netting sets are the instruments here, and a CVA is minutes of device time, so this is a
 * CACHED projection read off the desk's own file - never a run, and never a live number.
 *
 * STALENESS IS DATA. Every row states its own age against the read that carried it, a set the
 * book carries with no row yet says `never run` rather than showing a zero, and a row whose set
 * has left the book stays, marked, with the last numbers it had. A number that was on the blotter
 * yesterday disappearing without a word is how a desk loses track of a position.
 *
 * READ-ONLY BY DESIGN: recalcs are asked for through the MCP verbs. There is no button here, and
 * the bar says so rather than leaving a reader to wonder where it went.
 */
export function XvaView() {
  const { state } = useApp();
  const { xva, source, doc } = state;
  const [opened, setOpened] = useState<string | null>(null);

  if (source?.kind !== 'book') {
    return (
      <EmptyState
        title="The XVA projection belongs to the live book."
        hint="This document was opened from a file. Start the service with --book <job file> and the desk's projection appears here."
      />
    );
  }
  if (xva.status === 404) {
    return (
      <EmptyState
        title="The service is serving no book."
        hint="It was started without --book, or the book has been closed since this page loaded."
      />
    );
  }
  // a corrupt projection file names itself in the refusal; the rows are still on disk, so this
  // is a screen that will come back rather than numbers that are gone
  if (xva.error) {
    return (
      <div className="main">
        <div className="panel">
          <div className="error-box">{xva.error}</div>
          <div className="hint">
            The projection file could not be read. Nothing was lost — the rows are on disk, and
            this screen shows them again as soon as the file parses.
          </div>
        </div>
      </div>
    );
  }
  if (!xva.data) {
    return (
      <EmptyState
        title={xva.loading ? 'Reading the projection…' : 'No projection has been read yet.'}
        hint="The last run per netting set, joined with the book's own set list."
      />
    );
  }

  const view = xva.data;
  const totals = xvaTotals(view);
  const currency = currencyOf(doc);

  return (
    <div className="main">
      <div className="panel">
        <div className="blotterbar">
          <span className="hint">
            <b>{totals.sets}</b> netting set{totals.sets === 1 ? '' : 's'}
          </span>
          <span className="hint">
            <b>{totals.done}</b> run · <b>{totals.failed}</b> failed ·{' '}
            <b>{totals.neverRun}</b> never run
            {totals.orphans > 0 && <> · <b>{totals.orphans}</b> off-book</>}
            {totals.running > 0 && <> · <b>{totals.running}</b> recalculating</>}
          </span>
          {totals.counted > 0 && (
            <span className="hint" title={`the sum over the ${totals.counted} of the book's own sets that have a number; a set that never ran contributes nothing, and an off-book row is not the book's exposure`}>
              CVA over the {totals.counted} that ran{' '}
              <b>{formatNumber(totals.cva)}</b> {currency}
            </span>
          )}
          <span className="spacer" />
          <span className="hint" title={view.as_of}>read {stampText(view.as_of)}</span>
        </div>

        {/* the standing note, not a disabled button: a surface that looked like it could start a
            credit Monte Carlo would start one by accident */}
        <div className="note">
          Recalcs are asked for through the MCP verbs — <code>recalc_xva</code> — and never from
          here. A credit Monte Carlo is minutes of device time, so this screen is a <b>read</b> of{' '}
          <span className="mono" title={view.path}>{view.path}</span>: the last run per set, as old
          as each row says it is.
          {totals.running > 0 && ' A recalc is in flight — this view re-reads when you return to it.'}
        </div>

        <div className="tablewrap">
          <table className="data desk xva">
            <thead>
              <tr>
                <th>Netting set</th>
                <th>Path</th>
                <th>Counterparty</th>
                <th>CSA</th>
                <th className="n">{currency ? `CVA (${currency})` : 'CVA'}</th>
                <th>Last run</th>
                <th>Status</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {view.sets.map((set, position) => {
                // a reference is not unique in a book, and an orphan shares its name with nothing
                // that is left - the row's own position is the only identity here
                const id = `${set.reference}:${position}`;
                return (
                  <Row
                    key={id}
                    set={set}
                    read={view.as_of}
                    opened={opened === id}
                    onOpen={() => setOpened(opened === id ? null : id)}
                  />
                );
              })}
              {view.sets.length === 0 && (
                <tr>
                  <td colSpan={8} className="hint">
                    this book carries no netting sets, and the projection file holds no rows for
                    any that have left it — there is no XVA to read.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function Row({ set, read, opened, onOpen }: {
  set: XvaSet; read: string; opened: boolean; onOpen: () => void;
}) {
  const orphan = isOrphan(set);
  const age = ageOf(set.as_of, read);
  const chip = statusChip(set);
  const last = lastRunChip(set);

  return (
    <>
      <tr className={orphan ? 'muted' : undefined}>
        <td>
          <b>{set.reference}</b>
          {orphan && (
            <span className="badge" title={set.note ?? undefined}>left the book</span>
          )}
        </td>
        <td className="path">
          {set.deal_path ?? <span className="hint">—</span>}
        </td>
        <td>{set.counterparty ?? <span className="hint">—</span>}</td>
        <td>
          {/* a real boolean off the wire - the book stores this as the text "True"/"False" and the
              service reads it, so nothing here has to guess what "False" means */}
          <span
            className={`chip${set.collateralized ? ' on' : ''}`}
            title={set.collateralized ? 'collateralised' : 'uncollateralised'}
          >
            {set.collateralized ? 'CSA' : 'no CSA'}
          </span>
        </td>
        <td className={`n${set.cva !== null && set.cva < 0 ? ' neg' : ''}`}>
          {set.cva === null
            ? <span className="hint" title="a CVA is reported only for a run that completed">—</span>
            : formatNumber(set.cva)}
        </td>
        <td className={`age ${age.tone}`} title={age.title}>{age.text}</td>
        <td>
          <span className={`chip ${chip.tone}`} title={chip.title}>{chip.text}</span>
          {/* a recalc in flight leads, but the number on the row is still the LAST run's */}
          {set.recalc && (
            <span className={`chip ${last.tone}`} title={last.title} style={{ marginLeft: 6 }}>
              was {last.text}
            </span>
          )}
        </td>
        <td className="opener">
          <span onClick={onOpen}>{opened ? '×' : '⋯'}</span>
        </td>
      </tr>
      {opened && (
        <tr className="detail">
          <td colSpan={8}>
            {orphan && <div className="banner">{set.note}</div>}
            {/* the engine's own wording, verbatim: reformatting it would put a second author
                between the engine and the desk */}
            {set.status === 'failed' && set.error && (
              <div className="error-box">{set.error}</div>
            )}
            <section className="card">
              <h3>The set, and the run behind its number</h3>
              <div className="fields">
                {replayRows(set).map(([name, value]) => (
                  <Field key={name} name={name} value={value} />
                ))}
              </div>
            </section>
            {set.recalc && (
              <div className="hint">
                a recalc is {set.recalc.status} as result{' '}
                <span className="mono">{set.recalc.result_id.slice(0, 12)}</span> — the row above
                is still the last one that finished.
              </div>
            )}
          </td>
        </tr>
      )}
    </>
  );
}

function Field({ name, value }: { name: string; value: string | null }) {
  return (
    <>
      <div className="k">{name}</div>
      <div className="v">
        {value === null ? <span className="hint">—</span> : <span className="mono">{value}</span>}
      </div>
    </>
  );
}
