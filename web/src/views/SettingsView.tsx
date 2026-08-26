import { DescriptorPanel } from '../components/FieldView';
import { useApp } from '../state';
import { isObject } from '../tokens';

/** The configuration half of the document: System Parameters against the schema's System store,
 * and every other market-data section as itself - `.ModelParams` gets its two tables through
 * FieldView, everything else falls to the JSON view rather than to nothing. */
export function SettingsView() {
  const { state } = useApp();
  const { doc, schema } = state;
  if (!doc || !schema) return null;

  const merge = doc.Calc.MergeMarketData ?? {};
  const explicit = merge.ExplicitMarketData ?? {};
  const system = explicit['System Parameters'];
  const others = Object.entries(explicit).filter(
    ([section]) => !['System Parameters', 'Price Factors', 'Price Models'].includes(section));

  return (
    <div className="main">
      <div className="panel">
        {isObject(system) && (
          <DescriptorPanel title="System Parameters" fields={schema.System.fields} values={system} />
        )}
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
