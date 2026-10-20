import { useState } from 'react';
import { configureBook } from '../api';
import { DataTable } from '../components/DataTable';
import { DescriptorPanel, EditableScalar } from '../components/FieldView';
import { Navigator, type Page } from '../components/Navigator';
import { Whose } from '../components/Whose';
import { blockHint, marketOf, useApp, written } from '../state';
import { isObject, token } from '../tokens';
import type { ConfigSection, JobDoc, Schema } from '../types';

type Configure = (entry: string, fields: Record<string, unknown>) => Promise<string[] | null>;

/** The sibling store a declaration NAMES rather than restating - a menu of values per key. */
function menuOf(schema: Schema, name?: string): Record<string, string[]> {
  const stores = schema as unknown as Record<string, Record<string, string[]> | undefined>;
  return (name ? stores[name] : undefined) ?? {};
}

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

/** One rule added: the routed type it is for, the factor the attribute names, and the method that
 * one is built with. The verb takes the three together, so this is the one place on the screen a
 * button does anything - an existing row edits in place. */
function AddRule({ menu, attribute, configure }: {
  menu: Record<string, string[]>; attribute: string;
  configure: (entry: string, fields: Record<string, unknown>) => void;
}) {
  const types = Object.keys(menu);
  const [factor, setFactor] = useState(types[0] ?? '');
  const [named, setNamed] = useState('');
  const [method, setMethod] = useState('');
  const offered = menu[factor] ?? [];
  return (
    <div className="pager">
      <span>a rule for one factor</span>
      <select value={types.includes(factor) ? factor : types[0]}
              onChange={(event) => setFactor(event.target.value)}>
        {types.map((type) => <option key={type}>{type}</option>)}
      </select>
      <span className="mono">{attribute} =</span>
      <input type="text" value={named} placeholder="the factor's name"
             onChange={(event) => setNamed(event.target.value)} />
      <select value={offered.includes(method) ? method : ''}
              onChange={(event) => setMethod(event.target.value)}>
        <option value="">a method</option>
        {offered.map((one) => <option key={one}>{one}</option>)}
      </select>
      <button className="ghost" disabled={!named.trim() || !offered.includes(method)}
              onClick={() => void configure(factor, { [attribute]: named.trim(), method })}>
        add
      </button>
    </div>
  );
}

/** The section a `menu` declares, as the three-column table it has always been edited in -
 * `Risk_Factor.Method | Where | Equals`: one row per routed factor type, the method every factor
 * of it is built with and the two columns blank, then one row per rule, the attribute a single
 * factor is named by and the name it must equal. A per-curve rule set from a Curves card and one
 * set here are THE SAME ROW. Every method edits in place; a rule is cleared by the ✕ that posts a
 * blank method, which is what the verb reads as a removal. Each row is marked whose it is, a type's
 * rules being one list that a rule row marks and retires whole. */
function RuleTable({ schema, name, declared, value, configure }: {
  schema: Schema; name: string; declared: ConfigSection; value: unknown; configure?: Configure;
}) {
  const [refused, setRefused] = useState<string[] | null>(null);
  const menu = menuOf(schema, declared.menu);
  const params = token(value, '.ModelParams');
  const half = (key?: string): Record<string, unknown> => {
    const stated = isObject(params) ? params[key ?? ''] : undefined;
    return isObject(stated) ? stated : {};
  };
  const defaults = half(declared.entry);
  const attribute = declared.key ?? '';
  const save = configure && (async (factor: string, fields: Record<string, unknown>) =>
    setRefused(await configure(factor, fields)));
  const at = (key: string | undefined, factor: string) => [name, '.ModelParams', key ?? '', factor];
  const rows = [
    ...Object.keys(menu).map((factor) => ({
      factor, method: String(defaults[factor] ?? declared.value ?? ''), where: '', equals: '',
      path: at(declared.entry, factor) })),
    ...Object.entries(half(declared.rules)).flatMap(([factor, filed]) =>
      (Array.isArray(filed) ? filed as [[string, string], string][] : [])
        .map(([[where, equals], method]) => ({
          factor, method, where, equals, path: at(declared.rules, factor) }))),
  ];

  return (
    <section className="card">
      <h3>{name}</h3>
      <DataTable
        columns={['Risk_Factor.Interpolation', 'Where', 'Equals', '', '']} index={[]}
        data={rows.map((rule) => [
          `${rule.factor}.${rule.method}`, rule.where, rule.equals, null, null])}
        cell={(r, c) => {
          const rule = rows[r];
          if (c === 3) return <Whose path={rule.path} />;
          if (c === 0) {
            return save && (
              <span className="editcell">
                <span className="mono">{rule.factor}.</span>
                <EditableScalar
                  name={rule.factor} value={rule.method}
                  descriptor={{ widget: 'Dropdown', description: rule.factor, value: '',
                                values: menu[rule.factor] ?? [] }}
                  onAmend={async (_, wire) => configure!(rule.factor, rule.equals
                    ? { [attribute]: rule.equals, method: wire } : { method: wire })} />
              </span>
            );
          }
          return c === 4 && rule.equals && save ? (
            <button className="ghost"
                    onClick={() => void save(rule.factor, { [attribute]: rule.equals })}>✕</button>
          ) : undefined;
        }} />
      {save && <AddRule menu={menu} attribute={attribute} configure={save} />}
      {refused?.map((message, i) => <div key={i} className="error-box">{message}</div>)}
    </section>
  );
}

/** One configuration section of the market the book prices on against its declaration, as PAGES
 * and by shape. A declaration carrying `types` is an entry per key the market states, marked whose
 * it is - the entry's own values where it has them and the declared default otherwise, so an empty
 * `{}` shows every dial at its default - filed under the section, plus a page adding a family it
 * does not state yet. One carrying `menu` is the rule table above, one page. Editable where the
 * document is the live book, an edit to a file's entry landing on the block. */
function sectionPages(schema: Schema, doc: JobDoc, name: string, declared: ConfigSection,
                      value: unknown, configure?: Configure): Page[] {
  if (!declared.types) {
    return [{ id: name, label: name, content: (
      <RuleTable schema={schema} name={name} declared={declared} value={value}
                 configure={configure} />
    ) }];
  }
  const types = declared.types;
  const stated = Object.keys(isObject(value) ? value : {});
  const missing = Object.keys(types).filter((family) => !stated.includes(family));
  return [
    ...stated.map((key) => ({ id: `${name}/${key}`, folder: name, label: key,
                              hint: blockHint(doc, [name, key]), content: (
      <>
        <div className="pager"><Whose path={[name, key]} /></div>
        <DescriptorPanel
          title={`${name} — ${key}`} fields={types[key]?.fields}
          values={isObject(value) && isObject(value[key]) ? value[key] : {}}
          onAmend={configure && ((field, wire) => configure(key, { [field]: wire }))} />
      </>
    ) })),
    ...(configure && missing.length ? [{
      id: `${name}/`, folder: name, label: 'another family', accent: true,
      content: <AddEntry missing={missing} configure={configure} />,
    }] : []),
  ];
}

/** The bootstrap's own dials, as the market the book prices on states them: one entry per price
 * family under the factor it writes, the interpolation the routed curve types are built with as the
 * rule table it is edited in, and the one addition a book can make - a family it does not configure
 * yet. Every save re-bootstraps the market it configures, so a bootstrap that complains writes
 * nothing and hands its messages back. */
export function BootstrapperView() {
  const { state, dispatch } = useApp();
  const { doc, schema } = state;
  const market = marketOf(state);
  if (!doc || !schema || !market) return null;

  const configure = (section: string): Configure | undefined =>
    state.source?.kind === 'book'
      ? async (entry, fields) => written(dispatch, await configureBook(section, entry, fields))
      : undefined;

  return (
    <Navigator screen="bootstrap" keep pages={Object.entries(schema.Configuration).flatMap(
      ([section, declared]) => sectionPages(schema, doc, section, declared, market[section],
                                            configure(section)))} />
  );
}
