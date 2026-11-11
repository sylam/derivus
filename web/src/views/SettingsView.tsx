import { savePaths } from '../api';
import { DataTable } from '../components/DataTable';
import { DescriptorPanel, EditableScalar } from '../components/FieldView';
import { Whose } from '../components/Whose';
import { edited, mapRows, type MapName } from '../files';
import { marketOf, useApp, written } from '../state';
import { isObject } from '../tokens';
import type { Descriptor, NamedFile } from '../types';

/** Which file a named row is, by the key the job names it under. */
const WHICH: Record<NamedFile['key'], string> = {
  MarketDataFile: 'market data', CalendDataFile: 'calendar',
};

/** A map's right side, the folder or file name this box reads as the box spells it. */
const HERE: Descriptor = { widget: 'Text', description: 'on this box', value: null };

/** This workstation's first, where it reads the files a job names, which is no part of the
 * document and stands whether or not its market read; then the document's settings. */
export function SettingsView() {
  return (
    <div className="main">
      <div className="panel">
        <Files />
        <Stated />
      </div>
    </div>
  );
}

/** What is left of the configuration half once the quotes and the bootstrap have screens of their
 * own, as the market the book prices on states it, each entry marked whose it is: System
 * Parameters against the schema's System store, and every other market-data section as itself,
 * `.ModelParams` getting its two tables through FieldView and everything else falling to the JSON
 * view rather than to nothing. */
function Stated() {
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
    <>
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
    </>
  );
}

/** The service's own fact, kept on this box and never written into a job: the files the served
 * book names, where this box reads each and what it read, then the two maps a name is read
 * through, each entry edited in place and an emptied one removed. */
function Files() {
  const { state, dispatch } = useApp();
  const paths = state.paths.data;
  if (!paths) {
    return state.paths.error ? (
      <div className="error-box">this workstation's files did not answer: {state.paths.error}</div>
    ) : null;
  }

  // both maps go whole and the answer is the paths on screen; the book is re-read so the market,
  // the description and every screen follow the new reading
  const save = (map: MapName) => async (key: string, value: unknown) => {
    const answer = await savePaths(edited(paths, map, key, String(value)));
    dispatch({ type: 'PATHS_LOADED', paths: answer });
    return written(dispatch, { written: true });
  };
  const table = (map: MapName) => {
    const rows = mapRows(paths, map);
    return (
      <DataTable columns={['as written', 'on this box']} index={[]}
                 data={rows.map((row) => [row.written, row.here])}
                 cell={(r, c) => (c === 1 ? (
                   <EditableScalar name={rows[r].written} descriptor={HERE} value={rows[r].here}
                                   onAmend={save(map)} />
                 ) : undefined)} />
    );
  };

  return (
    <section className="card files">
      <h3>This workstation: where a job's files are read</h3>
      <div className="nested">
        <div className="pager">the files the served book names</div>
        <DataTable columns={['file', 'as written', 'on this box', 'content']} index={[]}
                   data={paths.named.map((named) =>
                     [WHICH[named.key], named.name, named.path, null])}
                   cell={(r, c) => {
                     if (c !== 3) return undefined;
                     const { digest, unread } = paths.named[r];
                     return unread ? <span className="bad">{unread}</span>
                       : <span className="mono">{digest?.slice(0, 12)}</span>;
                   }} />
        <div className="pager">folders</div>
        {table('path_transform')}
        <div className="pager">file names</div>
        {table('file_transform')}
      </div>
    </section>
  );
}
