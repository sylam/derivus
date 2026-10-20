import { DescriptorPanel } from '../components/FieldView';
import { Whose } from '../components/Whose';
import { marketOf, useApp } from '../state';
import { isObject } from '../tokens';

/** What is left of the configuration half once the quotes and the bootstrap have screens of their
 * own, as the market the book prices on states it, each entry marked whose it is: System
 * Parameters against the schema's System store, and every other market-data section as itself,
 * `.ModelParams` getting its two tables through FieldView and everything else falling to the JSON
 * view rather than to nothing. */
export function SettingsView() {
  const { state } = useApp();
  const { doc, schema } = state;
  const market = marketOf(state);
  if (!doc || !schema || !market) return null;

  const merge = doc.Calc.MergeMarketData ?? {};
  const system = market['System Parameters'];
  const elsewhere = ['System Parameters', 'Price Factors', 'Price Models', 'Market Prices']
    .concat(Object.keys(schema.Configuration));
  const others = Object.entries(market).filter(([section]) => !elsewhere.includes(section));

  return (
    <div className="main">
      <div className="panel">
        {isObject(system) && (
          <DescriptorPanel title="System Parameters" fields={schema.System.fields} values={system}
                           aside={(key) => <Whose path={['System Parameters', key]} />} />
        )}
        {others.map(([section, values]) => (
          <DescriptorPanel key={section} title={section}
                           values={isObject(values) ? values : { value: values }}
                           aside={(key) => <Whose path={[section, key]} />} />
        ))}
        {merge.MarketDataFile ? (
          <div className="pager">market data file: {String(merge.MarketDataFile)}</div>
        ) : null}
      </div>
    </div>
  );
}
