import { DescriptorPanel } from '../components/FieldView';
import { useApp } from '../state';
import { isObject } from '../tokens';
import type { Schema } from '../types';

/** One `Market Prices` block against its own family's declarations. A block is filed under
 * `<market_factor_type>.<underlying>` and that first segment IS the schema key, so the fields
 * render as declared and the quote ladders take their column names. A family the store does not
 * carry falls to the JSON view rather than to nothing. */
function MarketPriceBlock({ schema, name, block }: {
  schema: Schema; name: string; block: unknown;
}) {
  const fields = schema.MarketPrices.types[name.split('.')[0]];
  const instrument = isObject(block) ? block['instrument'] : undefined;
  return fields && isObject(instrument)
    ? <DescriptorPanel title={name} fields={fields} values={instrument} />
    : <DescriptorPanel title={name} values={isObject(block) ? block : { value: block }} />;
}

/** The configuration half of the document: System Parameters against the schema's System store,
 * the quote blocks against the MarketPrices store, and every other market-data section as itself -
 * `.ModelParams` gets its two tables through FieldView, everything else falls to the JSON view
 * rather than to nothing. */
export function SettingsView() {
  const { state } = useApp();
  const { doc, schema } = state;
  if (!doc || !schema) return null;

  const merge = doc.Calc.MergeMarketData ?? {};
  const explicit = merge.ExplicitMarketData ?? {};
  const system = explicit['System Parameters'];
  const prices = explicit['Market Prices'];
  const others = Object.entries(explicit).filter(([section]) =>
    !['System Parameters', 'Price Factors', 'Price Models', 'Market Prices'].includes(section));

  return (
    <div className="main">
      <div className="panel">
        {isObject(system) && (
          <DescriptorPanel title="System Parameters" fields={schema.System.fields} values={system} />
        )}
        {isObject(prices) && Object.entries(prices).map(([name, block]) => (
          <MarketPriceBlock key={name} schema={schema} name={name} block={block} />
        ))}
        {others.map(([section, values]) => (
          <DescriptorPanel key={section} title={section}
                           values={isObject(values) ? values : { value: values }} />
        ))}
        {merge.MarketDataFile ? (
          <div className="pager">market data file: {String(merge.MarketDataFile)}</div>
        ) : null}
      </div>
    </div>
  );
}
