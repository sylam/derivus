import { configureBook } from '../api';
import { DescriptorPanel } from '../components/FieldView';
import { useApp, written } from '../state';
import { isObject, token } from '../tokens';
import type { ConfigSection, Schema, Section } from '../types';

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

/** The sibling store a declaration NAMES rather than restating - a menu of values per key. */
function menuOf(schema: Schema, name?: string): Record<string, string[]> {
  const stores = schema as unknown as Record<string, Record<string, string[]> | undefined>;
  return (name ? stores[name] : undefined) ?? {};
}

type Configure = (entry: string, fields: Record<string, unknown>) => Promise<string[] | null>;

/** One configuration section against its declaration, rendered by SHAPE. A declaration carrying
 * `types` is an entry per key the book states - the entry's own values where it has them and the
 * declared default otherwise, so an empty `{}` shows every dial at its default, and an older
 * book's class-name key finds its entry through the `aliases` the store publishes. One carrying
 * `menu` is a single panel of that menu's choices, one row per key it names, at the engine's own
 * value where the book states none. Editable where the document is the live book. */
function ConfigurationSection({ schema, name, declared, value, configure }: {
  schema: Schema; name: string; declared: ConfigSection; value: unknown; configure?: Configure;
}) {
  if (declared.types) {
    const types = declared.types;
    return (
      <>
        {Object.entries(isObject(value) ? value : {}).map(([key, entry]) => {
          const declaration = types[key]
            ?? Object.values(types).find((type) => type.aliases.includes(key));
          return (
            <DescriptorPanel
              key={key} title={`${name} — ${key}`} fields={declaration?.fields}
              values={isObject(entry) ? entry : {}}
              onAmend={configure && ((field, wire) => configure(key, { [field]: wire }))} />
          );
        })}
      </>
    );
  }
  const half = declared.entry ?? '';
  const params = token(value, '.ModelParams');
  const stated = isObject(params) && isObject(params[half]) ? params[half] : {};
  const fields: Section = Object.fromEntries(
    Object.entries(menuOf(schema, declared.menu)).map(([key, values]) => [
      key, { widget: 'Dropdown', description: key, value: declared.value ?? '', values }]));
  return (
    <DescriptorPanel
      title={name} fields={fields} values={stated}
      onAmend={configure && ((field, wire) => configure(half, { [field]: wire }))} />
  );
}

/** The configuration half of the document: System Parameters against the schema's System store,
 * the quote blocks against the MarketPrices store, the bootstrap's own sections against the
 * Configuration store - editable, each save re-bootstrapping the market it configures - and every
 * other market-data section as itself, `.ModelParams` getting its two tables through FieldView and
 * everything else falling to the JSON view rather than to nothing. */
export function SettingsView() {
  const { state, dispatch } = useApp();
  const { doc, schema } = state;
  if (!doc || !schema) return null;

  const merge = doc.Calc.MergeMarketData ?? {};
  const explicit = merge.ExplicitMarketData ?? {};
  const system = explicit['System Parameters'];
  const prices = explicit['Market Prices'];
  const configured = Object.keys(schema.Configuration);
  const others = Object.entries(explicit).filter(([section]) =>
    !['System Parameters', 'Price Factors', 'Price Models', 'Market Prices']
      .concat(configured).includes(section));

  const configure = (section: string): Configure | undefined =>
    state.source?.kind === 'book'
      ? async (entry, fields) => written(dispatch, await configureBook(section, entry, fields))
      : undefined;

  return (
    <div className="main">
      <div className="panel">
        {isObject(system) && (
          <DescriptorPanel title="System Parameters" fields={schema.System.fields} values={system} />
        )}
        {Object.entries(schema.Configuration).map(([section, declared]) => (
          <ConfigurationSection key={section} schema={schema} name={section} declared={declared}
                                value={explicit[section]} configure={configure(section)} />
        ))}
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
