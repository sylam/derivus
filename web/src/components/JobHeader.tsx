import { postValidate } from '../api';
import { useApp } from '../state';

/** The one-line read on the loaded document: the book by type, both halves of the factor
 * universe, the calculation, the queue's cost class - `/describe`'s answer, which App refreshes
 * whenever the document moves. Validate on demand, messages verbatim: they are the engine's own
 * words and reformatting them would put a second author between the engine and the reader. */
export function JobHeader() {
  const { state, dispatch } = useApp();
  const { doc, describe, validate } = state;
  if (!doc) return null;

  const dealCount = describe
    ? Object.values(describe.deals).reduce((a, b) => a + b, 0) : null;

  return (
    <div className="jobbar">
      {describe && (
        <>
          <span><b>{dealCount}</b> deal{dealCount === 1 ? '' : 's'}{' '}
            <span className="hint">
              ({Object.entries(describe.deals).map(([t, n]) => `${n} ${t}`).join(', ')})
            </span>
          </span>
          <span><b>{describe.factors.resolved.length}</b> factors resolved</span>
          {describe.factors.missing.length > 0 && (
            <span className="bad"><b>{describe.factors.missing.length}</b> missing:{' '}
              {describe.factors.missing.join(', ')}</span>
          )}
          <span>calc <b>{String(describe.calculation.Object ?? '')}</b></span>
          {describe.cost && (
            <span title={describe.cost.basis}>
              cost class <b>{describe.cost.class}</b> · est <b>{describe.cost.estimate}</b>
            </span>
          )}
        </>
      )}
      <button
        className="ghost"
        onClick={async () => dispatch({ type: 'VALIDATED', validate: await postValidate(doc) })}
      >
        validate
      </button>
      {validate && (
        Object.keys(validate.deals).length === 0 && validate.factors.length === 0
          ? <span style={{ color: 'var(--good)' }}>✓ validates clean</span>
          : <span className="bad">
              {Object.entries(validate.deals).map(([ref, messages]) =>
                `${ref}: ${messages.join('; ')}`).join(' · ')}
              {validate.factors.length > 0 && ` · no market data: ${validate.factors.join(', ')}`}
            </span>
      )}
    </div>
  );
}
