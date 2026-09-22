import { useEffect, useState } from 'react';
import { getBookMarkets, patchMarket } from '../api';
import { DescriptorPanel, type AmendField } from '../components/FieldView';
import { Tree } from '../components/Tree';
import { stampText } from '../desk';
import { marketsView } from '../spine';
import { useApp, written } from '../state';
import { isObject } from '../tokens';
import type { BookMarkets, Schema } from '../types';

/** `ModelParams.search` ported (config.py): a `Price Models` block keyed exactly wins; else the
 * first matching Model Configuration filter row; else the type's default; else static. */
function resolveProcess(
  schema: Schema, factorName: string, block: Record<string, unknown>,
  priceModels: Record<string, unknown>, modelConfig: unknown,
): { title: string; process?: string; values?: Record<string, unknown> } {
  const [type, ...rest] = factorName.split('.');
  const name = rest.join('.');
  for (const process of schema.Process_factor_map[type] ?? []) {
    const keyed = priceModels[`${process}.${name}`];
    if (isObject(keyed)) return { title: `${process} (Price Models)`, process, values: keyed };
  }
  const params = isObject(modelConfig) ? modelConfig['.ModelParams'] : undefined;
  if (isObject(params)) {
    const pfType = type + (block.Sub_Type === 'BasisSpread' ? 'BasisSpread' : '');
    const filters = (params.modelfilters as Record<string, [[string, string], string][]>)?.[pfType];
    const candidate: Record<string, string> = { ...Object.fromEntries(
      Object.entries(block).map(([k, v]) => [k.toLowerCase(), String(v)])), id: name };
    for (const [[attribute, value], model] of filters ?? []) {
      if (candidate[attribute.toLowerCase()] === value) {
        return { title: `${model} (Model Configuration filter - no Price Models block)`, process: model };
      }
    }
    const fallback = (params.modeldefaults as Record<string, string>)?.[pfType];
    if (fallback) {
      return { title: `${fallback} (Model Configuration default - no Price Models block)`, process: fallback };
    }
  }
  return { title: 'static - no process' };
}

export function MarketDataView() {
  const { state, dispatch } = useApp();
  const { doc, schema, selection, describe } = state;
  if (!doc || !schema) return null;

  const merge = doc.Calc.MergeMarketData ?? {};
  const marketFile = merge.MarketDataFile ?? '';
  const explicit = merge.ExplicitMarketData ?? {};
  const factors = (explicit['Price Factors'] ?? {}) as Record<string, unknown>;
  const priceModels = (explicit['Price Models'] ?? {}) as Record<string, unknown>;
  const names = Object.keys(factors).sort();

  const selected = selection.factor;
  const block = selected ? factors[selected] : undefined;
  const type = selected?.split('.')[0] ?? '';

  // market VALUES edit over the live book - the bind='value' declaration is the whole predicate,
  // so structure stays read-only exactly where the engine refuses it anyway
  const onPatch: AmendField | undefined =
    state.source?.kind === 'book' && selected
      ? async (key, wireValue) =>
          written(dispatch, await patchMarket(selected, { [key]: wireValue }))
      : undefined;

  return (
    <div className="main">
      <div className="sidebar">
        <Tree
          nodes={names.map((name) => {
            const [factorType, ...rest] = name.split('.');
            return { id: name, type: factorType, label: rest.join('.') };
          })}
          selected={selected}
          onSelect={(id) => dispatch({ type: 'SELECT_FACTOR', name: id })}
        />
      </div>
      <div className="panel">
        <RecordMarkets />
        {marketFile ? (
          <div className="banner">
            This document overlays a server-side market data file (<b>{String(marketFile)}</b>) —
            only the overlay is shown here.
            {describe && ` The engine resolves ${describe.factors.resolved.length} factors from it` +
              (describe.factors.missing.length
                ? ` and is missing ${describe.factors.missing.join(', ')}.` : '.')}
          </div>
        ) : null}
        {!selected && <div className="placeholder">select a price factor</div>}
        {selected && isObject(block) && (
          <>
            <DescriptorPanel
              title={selected}
              fields={schema.Factor.types[type]}
              values={block}
              onAmend={onPatch}
              editable={(_, descriptor) => descriptor.bind === 'value'}
            />
            <ProcessPanel
              schema={schema} factorName={selected} block={block}
              priceModels={priceModels} modelConfig={explicit['Model Configuration']}
            />
            {(schema.Interpolation_factor_map[type] ?? []).length > 0 && (
              <div className="pager">
                interpolation methods: {schema.Interpolation_factor_map[type].join(', ')}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

/** What the RECORD says about this market, above what the book carries of it: the official close
 * standing per market with the close it restated, the names a values vector was declared under,
 * and the snapshots registered.
 *
 * Read-only, and invisible where this desk records nothing - which is why it heads the screen
 * rather than hiding under it: a close is what an IPV reader comes here for, and a desk with no
 * record sees exactly the screen it saw before. A `values_hash` IS the address of the vector, so
 * two rows carrying one hash are one market marked twice.
 */
function RecordMarkets() {
  const { state } = useApp();
  const [markets, setMarkets] = useState<BookMarkets | null>(null);
  const source = state.source;
  const records = state.record.spine !== null;

  // the record's own read, on the etag the whole screen rides: a close declared by any client
  // lands here on the next beat, and a desk that keeps no record is not asked at all
  useEffect(() => {
    if (source?.kind !== 'book' || !records) return;
    let live = true;
    getBookMarkets()
      .then((answer) => { if (live) setMarkets(answer); })
      .catch(() => { if (live) setMarkets(null); });
    return () => { live = false; };
  }, [source, records]);

  if (markets === null) return null;
  const { closes, names, snapshots } = marketsView(markets);

  return (
    <section className="card">
      <h3>the record · LSN {markets.lsn}</h3>
      <table className="data">
        <thead>
          <tr>
            <th>Market</th><th>Official close</th><th>Values</th><th>Stands over</th>
            <th className="n">LSN</th>
          </tr>
        </thead>
        <tbody>
          {closes.map((close) => (
            <tr key={close.lsn}>
              <td>{close.market}</td>
              <td>{stampText(close.effective_time)}</td>
              <td className="mono" title={close.values_hash}>{close.values_hash.slice(0, 12)}</td>
              <td>{close.supersedes}</td>
              <td className="n">{close.lsn}</td>
            </tr>
          ))}
          {names.map((name) => (
            <tr key={`name-${name.lsn}`}>
              <td>{name.name}</td>
              <td>declared by {name.actor}</td>
              <td className="mono" title={name.values_hash}>{name.values_hash.slice(0, 12)}</td>
              <td>a market name pointed at a values vector</td>
              <td className="n">{name.lsn}</td>
            </tr>
          ))}
          {snapshots.map((snapshot) => (
            <tr key={`snapshot-${snapshot.lsn}`}>
              <td>{snapshot.book ?? ''}</td>
              <td>a registered snapshot</td>
              <td className="mono" title={snapshot.blob}>{snapshot.blob.slice(0, 12)}</td>
              <td />
              <td className="n">{snapshot.lsn}</td>
            </tr>
          ))}
          {!closes.length && !names.length && !snapshots.length && (
            <tr><td colSpan={5}>this record holds no market declaration yet.</td></tr>
          )}
        </tbody>
      </table>
    </section>
  );
}

function ProcessPanel(props: {
  schema: Schema; factorName: string; block: Record<string, unknown>;
  priceModels: Record<string, unknown>; modelConfig: unknown;
}) {
  const { schema, factorName, block, priceModels, modelConfig } = props;
  const resolved = resolveProcess(schema, factorName, block, priceModels, modelConfig);
  if (!resolved.process) return <div className="pager">{resolved.title}</div>;
  return (
    <DescriptorPanel
      title={resolved.title}
      fields={schema.Process.types[resolved.process]}
      values={resolved.values ?? {}}
    />
  );
}
