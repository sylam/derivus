import { DescriptorPanel } from '../components/FieldView';
import { Tree } from '../components/Tree';
import { useApp } from '../state';
import { isObject } from '../tokens';
import type { Schema } from '../types';

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
