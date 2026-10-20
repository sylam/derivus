import { useState } from 'react';
import { failure, retireMarket } from '../api';
import { holds, marketOf, useApp, written } from '../state';

/** Whose an entry of the market is, `path` naming it from its section down: the block's where the
 * document's own explicit block states it, else the file's where the book names one and the market
 * holds it, else nothing - the engine's own value. On the live book the block's entry over a file
 * retires: deleted from the block, the file's standing again and the market re-bootstrapped, or
 * the service's refusal in its own words. */
export function Whose({ path }: { path: string[] }) {
  const { state, dispatch } = useApp();
  const [refused, setRefused] = useState<string[] | null>(null);
  const merge = state.doc?.Calc.MergeMarketData;
  const file = merge?.MarketDataFile;
  const block = holds(merge?.ExplicitMarketData, path);
  if (!block && !(file && holds(marketOf(state), path))) return null;

  async function retire() {
    try {
      setRefused(await written(dispatch, await retireMarket([path])));
    } catch (error) {
      setRefused([failure(error).error]);
    }
  }

  const title = block ? "the book's own explicit block states this"
    : `${file} states this, or the engine's own value stands`;
  return (
    <span className="editcell">
      <span className={block ? 'chip on' : 'chip'} title={title}>{block ? 'block' : 'file'}</span>
      {block && file && state.source?.kind === 'book' && (
        <button className="ghost" title={`delete the block's entry - ${file}'s stands again`}
                onClick={() => void retire()}>retire</button>
      )}
      {refused?.map((message, i) => <span key={i} className="refusal">{message}</span>)}
    </span>
  );
}
