import { useState } from 'react';
import { configureBook } from '../api';
import { DescriptorPanel } from '../components/FieldView';
import { useApp, written } from '../state';
import { isObject, token } from '../tokens';
import type { ConfigSection, Schema, Section } from '../types';

type Configure = (entry: string, fields: Record<string, unknown>) => Promise<string[] | null>;

/** The sibling store a declaration NAMES rather than restating - a menu of values per key. */
function menuOf(schema: Schema, name?: string): Record<string, string[]> {
  const stores = schema as unknown as Record<string, Record<string, string[]> | undefined>;
  return (name ? stores[name] : undefined) ?? {};
}

/** Which declared entry a book's key files under - its own spelling, or the one whose aliases
 * carry it, so an older book's class-name key finds its family. */
const declarationFor = (types: NonNullable<ConfigSection['types']>, key: string) =>
  types[key] ?? Object.values(types).find((type) => type.aliases.includes(key));

/** A family the book does not configure yet, added from the store's own list: the verb completes
 * the empty entry with the family's routing stem and re-bootstraps, so the entry appears at every
 * declared default. */
function AddEntry({ missing, configure }: { missing: string[]; configure: Configure }) {
  const [family, setFamily] = useState(missing[0]);
  const [refused, setRefused] = useState<string[] | null>(null);
  const chosen = missing.includes(family) ? family : missing[0];
  return (
    <>
      <div className="pager">
        <span>configure another family</span>
        <select value={chosen} onChange={(event) => setFamily(event.target.value)}>
          {missing.map((name) => <option key={name}>{name}</option>)}
        </select>
        <button className="ghost" onClick={async () => setRefused(await configure(chosen, {}))}>
          add
        </button>
      </div>
      {refused?.map((message, i) => <div key={i} className="error-box">{message}</div>)}
    </>
  );
}

/** One configuration section against its declaration, rendered by SHAPE. A declaration carrying
 * `types` is an entry per key the book states - the entry's own values where it has them and the
 * declared default otherwise, so an empty `{}` shows every dial at its default - plus the families
 * it does not state yet. One carrying `menu` is a single panel of that menu's choices, one row per
 * key it names, at the engine's own value where the book states none - and the key itself is the
 * entry the verb takes. Editable where the document is the live book. */
function ConfigurationSection({ schema, name, declared, value, configure }: {
  schema: Schema; name: string; declared: ConfigSection; value: unknown; configure?: Configure;
}) {
  if (declared.types) {
    const types = declared.types;
    const stated = Object.keys(isObject(value) ? value : {});
    const missing = Object.keys(types).filter((family) =>
      !stated.some((key) => declarationFor(types, key) === types[family]));
    return (
      <>
        {stated.map((key) => (
          <DescriptorPanel
            key={key} title={`${name} — ${key}`} fields={declarationFor(types, key)?.fields}
            values={isObject(value) && isObject(value[key]) ? value[key] : {}}
            onAmend={configure && ((field, wire) => configure(key, { [field]: wire }))} />
        ))}
        {configure && missing.length > 0 && (
          <AddEntry missing={missing} configure={configure} />
        )}
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
      onAmend={configure && ((field, wire) => configure(field, { method: wire }))} />
  );
}

/** The bootstrap's own dials: one entry per price family under the factor it writes, the
 * interpolation the routed curve types are built with, and the one addition a book can make -
 * a family it does not configure yet. Every save re-bootstraps the market it configures, so a
 * bootstrap that complains writes nothing and hands its messages back. */
export function BootstrapperView() {
  const { state, dispatch } = useApp();
  const { doc, schema } = state;
  if (!doc || !schema) return null;

  const explicit = doc.Calc.MergeMarketData?.ExplicitMarketData ?? {};
  const configure = (section: string): Configure | undefined =>
    state.source?.kind === 'book'
      ? async (entry, fields) => written(dispatch, await configureBook(section, entry, fields))
      : undefined;

  return (
    <div className="main">
      <div className="panel">
        {Object.entries(schema.Configuration).map(([section, declared]) => (
          <ConfigurationSection key={section} schema={schema} name={section} declared={declared}
                                value={explicit[section]} configure={configure(section)} />
        ))}
      </div>
    </div>
  );
}
