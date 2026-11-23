import { getBookMarkets, patchMarket } from '../api';
import { HeldCurve } from '../components/CreditCard';
import { DescriptorPanel, type AmendField } from '../components/FieldView';
import { Navigator } from '../components/Navigator';
import { Whose } from '../components/Whose';
import { factorFamily, fileName, spotQuoted, stampText } from '../desk';
import { marketsView } from '../spine';
import {
  blockHint, heldBack, heldSpreads, marketOf, spineOf, useApp, useRead, written,
} from '../state';
import { curveOf, formatNumber, isObject } from '../tokens';
import type { Market, Schema } from '../types';

/** One price factor of the market the book prices on, marked whose it is: its declared fields,
 * VALUES editable over the live book - the bind='value' declaration is the whole predicate, so
 * structure stays read-only exactly where the engine refuses it anyway, and an edit to a file's
 * factor lands on the block - and the process that simulates it. A value a desk quotes otherwise
 * says beside it how the desk reads it. */
function FactorBlock({ schema, name, block, priceModels, process, live, market }: {
  schema: Schema; name: string; block: Record<string, unknown>;
  priceModels: Record<string, unknown>; process?: string | null; live: boolean; market: Market;
}) {
  const { dispatch } = useApp();
  const type = factorFamily(name);
  const fields = schema.Factor.types[type];
  const onPatch: AmendField | undefined = live
    ? async (key, wireValue) => written(dispatch, await patchMarket(name, { [key]: wireValue }))
    : undefined;
  return (
    <>
      <div className="pager"><Whose path={['Price Factors', name]} /></div>
      <DescriptorPanel
        title={name} fields={fields} values={block} onAmend={onPatch}
        editable={(_, descriptor) => descriptor.bind === 'value'}
        beside={(key) => fields?.[key]?.bind === 'value'
          && <ReadAs name={name} value={block[key]} market={market} />} />
      <ProcessPanel schema={schema} factorName={name} process={process}
                    priceModels={priceModels} />
      {(schema.Interpolation_factor_map[type] ?? []).length > 0 && (
        <div className="pager">
          interpolation methods: {schema.Interpolation_factor_map[type].join(', ')}
        </div>
      )}
    </>
  );
}

export function MarketDataView() {
  const { state } = useApp();
  const { doc, schema, describe } = state;
  const market = marketOf(state);
  if (!doc || !schema || !market) return null;

  const marketFile = doc.Calc.MergeMarketData?.MarketDataFile ?? '';
  const factors = market['Price Factors'] ?? {};
  const priceModels = market['Price Models'] ?? {};
  const hold = heldBack(state, 'market_tick');
  const live = state.source?.kind === 'book';

  return (
    <Navigator
      screen="market" empty="this document carries no price factor"
      head={<>
        <RecordMarkets />
        {hold && <div className="note">{hold}</div>}
        {marketFile ? (
          <div className="banner">
            {live
              ? <>This book prices on the market data file{' '}
                <b title={marketFile}>{fileName(marketFile)}</b> with its own block
                merged over it: an entry the block states is marked <b>block</b>.</>
              : <>This document overlays a server-side market data file (
                <b title={marketFile}>{fileName(marketFile)}</b>) —
                only the overlay is shown here.</>}
          </div>
        ) : null}
      </>}
      pages={Object.keys(factors).sort().map((name) => {
        const [type, ...rest] = name.split('.');
        const block = factors[name];
        return {
          id: name, folder: type, label: rest.join('.') || type,
          hint: blockHint(doc, ['Price Factors', name]),
          content: isObject(block) ? (
            <FactorBlock schema={schema} name={name} block={block} priceModels={priceModels}
                         process={describe && (describe.processes[name] ?? null)}
                         live={live && !hold} market={market} />
          ) : <DescriptorPanel title={name} values={{ value: block }} />,
        };
      })} />
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
  const source = state.source;
  const records = spineOf(state) !== null;

  // the record's own read, on the etag the whole screen rides: a close declared by any client
  // lands here on the next beat, and a desk that keeps no record is not asked at all
  const markets = useRead(records ? getBookMarkets : null, [source, records]).data;

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

/** A factor's value as a desk reads it, beside the box holding it as the engine does: an FX spot
 * on its pair as the market quotes it, a credit curve as the spreads a credit officer typed. */
function ReadAs({ name, value, market }: { name: string; value: unknown; market: Market }) {
  const held = heldSpreads(useApp().state, name);
  const spot = typeof value === 'number' ? spotQuoted(name, value, market) : undefined;
  if (spot) {
    return (
      <span className="readas">
        the market quotes <b>{spot.pair} {formatNumber(spot.level)}</b>
        {spot.turned ? <> · this box holds {spot.held}, as the engine does</>
          : <>, as this box holds it</>}
      </span>
    );
  }
  if (!held || !curveOf(value)) return null;
  return (
    <div className="readas">
      <div>the curve is the cumulative hazard the engine solved from the spreads a credit officer typed:</div>
      <HeldCurve id={name.slice(name.indexOf('.') + 1)} />
    </div>
  );
}

/** The process the engine simulates a factor with, as `/describe` answers it, beside its `Price
 * Models` block; a factor it answers none for is static, and nothing shows before it answers. */
function ProcessPanel({ schema, factorName, process, priceModels }: {
  schema: Schema; factorName: string; process?: string | null; priceModels: Record<string, unknown>;
}) {
  if (process === undefined) return null;
  if (!process) return <div className="pager">static - no process</div>;
  const values = priceModels[`${process}.${factorName.split('.').slice(1).join('.')}`];
  return <DescriptorPanel title={process} fields={schema.Process.types[process]}
                          values={isObject(values) ? values : {}} />;
}
