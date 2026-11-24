// The desk's two data views, as arithmetic - kept out of the components the way `blotter.ts` is.
// `/book/risk` is the consolidated mark and the whole-book gradient; `/book/xva` is the last run
// per netting set. Nothing here touches React or the network, so everything the two screens SAY
// about a book - what lags, what is stale, what the rows add up to - is one pure function of the
// two answers, and can be run in node against the live service.
//
// BOTH VIEWS ARE READS. The risk verb computes on a cache miss and answers from the cache after;
// the XVA verb never runs anything at all. A recalc is asked for through the MCP verbs, and this
// module has no vocabulary for one on purpose.

import { MARKED, levelOnAxis, rowsByPath, toRows } from './blotter';
import { clientName, positionName, under, type Entity, type Position } from './positions';
import {
  axisText, formatNumber, isObject, label, tenorLabel, token, ungrouped,
} from './tokens';
import type {
  BookRisk, BookXva, DealMark, Descriptor, GreekRow, HeldSpreads, JobDoc, Market, RefusalCode,
  Schema, XvaSet,
} from './types';
import { axisNames } from './vols';

/** A file as a reader names it: its name, never the folders it sits in on the server. */
export const fileName = (path: string): string => path.split(/[\\/]/).pop()!;

/** An entity as a reader tells it from another: its name with its id beside it, the id alone where
 * legal declared no name. */
export function entityText(entities: Entity[] | null, id: string): string {
  const name = clientName(entities, id);
  return name === id ? id : `${name} (${id})`;
}

// ---- stamps -------------------------------------------------------------------------------------
// The service stamps every answer on its own clock to milliseconds, with the offset it runs at. Two
// stamps from the same service are the only clocks compared here: an age is measured from a row's
// run to the READ that carried it, never to the browser's clock, which belongs to another machine.

const STAMP = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?(Z|[+-][\d:]{4,5})?/;

/** A service stamp's wall-clock components in UTC ms, before applying its zone offset. */
function stampMs(stamp: string | null | undefined): number | undefined {
  const parts = stamp === null || stamp === undefined ? null : STAMP.exec(stamp);
  if (!parts) return undefined;
  const fraction = parts[7] === undefined ? 0 : Number(parts[7].slice(0, 3).padEnd(3, '0'));
  return Date.UTC(+parts[1], +parts[2] - 1, +parts[3], +parts[4], +parts[5], +parts[6], fraction);
}

const pad = (value: number) => String(value).padStart(2, '0');

/** A stamp as the instant it names, its offset applied, so the record's UTC and a run's local
 * time compare; one naming no zone is read as UTC, and anything that is not a stamp is undefined. */
export function instantMs(stamp: string | null | undefined): number | undefined {
  const zone = (stamp && STAMP.exec(stamp)?.[8]) || 'Z';
  const [hours, minutes] = zone === 'Z' ? [0, 0] : zone.slice(1).split(/:?(?=\d{2}$)/);
  const east = (zone[0] === '-' ? -1 : 1) * (+hours * 60 + +minutes);
  const ms = stampMs(stamp);
  return ms === undefined ? undefined : ms - east * 60_000;
}

/** A stamp as a desk reads it: on this browser's clock with its offset named, the seconds kept -
 * `2026-08-28 22:51:34 UTC+2` - so every time on a screen compares with every other. A stamp naming
 * no zone reads as written; anything that is not a stamp reads as itself. */
export function stampText(stamp: string | null | undefined): string {
  const parts = stamp === null || stamp === undefined ? null : STAMP.exec(stamp);
  if (!parts) return stamp ?? '';
  if (!parts[8]) return `${parts[1]}-${parts[2]}-${parts[3]} ${parts[4]}:${parts[5]}:${parts[6]}`;
  const at = new Date(instantMs(stamp)!);
  const west = at.getTimezoneOffset();
  const size = Math.abs(west);
  const zone = size === 0 ? 'UTC'
    : `UTC${west > 0 ? '-' : '+'}${Math.floor(size / 60)}${size % 60 ? `:${pad(size % 60)}` : ''}`;
  return `${at.getFullYear()}-${pad(at.getMonth() + 1)}-${pad(at.getDate())} ${
    pad(at.getHours())}:${pad(at.getMinutes())}:${pad(at.getSeconds())} ${zone}`;
}

export function plural(count: number, unit: string): string {
  return `${count} ${unit}${count === 1 ? '' : 's'}`;
}

/** A duration as an age, coarsening as it grows: nobody reads a CVA to the minute once it is a
 * day old, and a stamp that precise invites a precision the number does not have. FLOORED at every
 * unit, so an age understates - `3 hours ago` for a row of three hours and fifty minutes is at
 * least true, where rounding it up to four would age a number that had not aged. */
function ageText(ms: number): string {
  const minutes = Math.floor(ms / 60_000);
  if (minutes < 1) return 'moments ago';
  if (minutes < 60) return `${plural(minutes, 'minute')} ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${plural(hours, 'hour')} ago`;
  return `${plural(Math.floor(hours / 24), 'day')} ago`;
}

/** How old a projection row may be before this view calls it stale. A CVA recalc is a deliberate
 * act minutes of device time long, so a desk runs one and reads it for the rest of the session;
 * a row carried over from yesterday is a number about a different market. Half a day is that line,
 * and it is a READING of the number, not a claim the service makes. */
const FRESH_MS = 12 * 3_600_000;

type Age = { text: string; tone: 'fresh' | 'stale' | 'none'; title: string };

/** A row's age, measured against the read that carried it. `ran` null is the 'never run' case -
 * the book carries the set and no recalc has ever produced a row for it - which is not old, it is
 * absent, and says so. */
export function ageOf(ran: string | null, read: string): Age {
  if (ran === null) {
    return {
      text: 'never run', tone: 'none',
      title: 'the book carries this set, but no recalc has ever produced a row for it',
    };
  }
  const from = instantMs(ran);
  const to = instantMs(read);
  if (from === undefined || to === undefined) return { text: ran, tone: 'none', title: ran };
  // a row stamped after the read is a clock that moved, not a number from the future
  const ms = Math.max(0, to - from);
  return { text: ageText(ms), tone: ms < FRESH_MS ? 'fresh' : 'stale', title: stampText(ran) };
}

// ---- the consolidated risk ----------------------------------------------------------------------

export type GreekKey = 'factor' | 'magnitude';

/** A factor's coordinates as one cell: empty where the factor has none (the key is absent on a
 * scalar like an FxRate spot), one for a curve point, two for a vol-surface node - each as its
 * axis is named, so a tenor reads as one and an excel day as its date. */
export function tenorText(tenor?: number[], axes: string[] = []): string {
  return tenor === undefined ? ''
    : tenor.map((value, at) => axisText(axes[at] ?? '', value)).join(' / ');
}

/** The names of a factor's coordinates, off the curve or the surface its type declares. */
export function factorAxes(schema: Schema | null, factor: string, count: number): string[] {
  const shape = Object.values(schema?.Factor.types[factorFamily(factor)] ?? {})
    .find((field) => field.widget === (count === 1 ? 'Curve' : 'Surface'));
  return axisNames(shape?.description, count + 1);
}

/** A vol quote as a dealer names it - `2M ATM`, `1M 25D RR` - off the engine's own name for it, its
 * type then its pillar delta then its expiry in years; any other name reads as itself. */
export function quoteText(quote: string): string {
  const [type, ...rest] = quote.split(' ');
  const numbers = rest.map(Number);
  if (!['ATM', 'RR', 'BF'].includes(type) || numbers.length !== (type === 'ATM' ? 1 : 2)
      || numbers.some((value) => !isFinite(value))) return quote;
  const expiry = tenorLabel(numbers[numbers.length - 1]);
  return type === 'ATM' ? `${expiry} ATM` : `${expiry} ${formatNumber(100 * numbers[0])}D ${type}`;
}

/** Coordinates in reading order, a factor with none first - a spot sits above its own curve. */
function compareTenor(a: number[] | undefined, b: number[] | undefined): number {
  if (a === undefined || b === undefined) return a === b ? 0 : a === undefined ? -1 : 1;
  for (let index = 0; index < Math.max(a.length, b.length); index += 1) {
    const difference = (a[index] ?? -Infinity) - (b[index] ?? -Infinity);
    if (difference !== 0) return difference < 0 ? -1 : 1;
  }
  return 0;
}

/** The gradient sorted for reading. By MAGNITUDE - the default - it is a ranking: what the book
 * is most exposed to per the move each row is read per, sign carried but not sorted on, so the
 * biggest short shows beside the biggest long. By FACTOR it is a curve read in order: the factors
 * alphabetical, and each one's coordinates always ascending inside it, because a curve reversed is
 * not a ranking of anything. Never mutates the answer it is given. */
export function sortGreeks(rows: GreekRow[], key: GreekKey, descending: boolean): GreekRow[] {
  const direction = descending ? -1 : 1;
  return [...rows].sort((a, b) => {
    if (key === 'magnitude') {
      const difference = Math.abs(riskOf(a)) - Math.abs(riskOf(b));
      if (difference !== 0) return direction * (difference < 0 ? -1 : 1);
      return a.factor.localeCompare(b.factor) || compareTenor(a.tenor, b.tenor);
    }
    const named = a.factor.localeCompare(b.factor);
    return named !== 0 ? direction * named : compareTenor(a.tenor, b.tenor);
  });
}

/** The gradient read in QUOTE space: every quote the book moves with, under the block it came off
 * and in the order the block quotes them, then every factor row no quote block stands for - a
 * spot, a curve set up by hand - in its own units. A factor a block builds is read on its quotes
 * and never twice; a quote past everything the book holds reads exactly zero and is left off the
 * screen, the service's answer keeping the whole ladder. */
export function quoteView(risk: BookRisk): GreekRow[] {
  const quoted = new Set(risk.quoted);
  return [...risk.quotes.filter((row) => row.value !== 0)
            .map(({ block, ...row }) => ({ ...row, factor: block })),
          ...risk.greeks.filter((row) => !quoted.has(row.factor))];
}

/** A row's report-currency change per the move it is read per, and the words for that move - the
 * per-unit derivative, and `per unit`, where the answer read none or could not, its level absent. */
export const riskOf = (row: GreekRow): number => row.risk ?? row.value;
export const perOf = (row: GreekRow): string =>
  (row.risk === null || row.risk === undefined ? undefined : row.per) ?? 'per unit';

/** The words of the move every row shares, or null where they differ: a table names one move in
 * its header, or each row beside itself. */
export function sharedPer(rows: GreekRow[]): string | null {
  const words = new Set(rows.map(perOf));
  return words.size === 1 ? [...words][0] : null;
}

/** What the raw derivative a row was read from says, for a reader who wants it: per unit of the
 * factor as stored, at the level it was taken. */
export function perUnitText(row: GreekRow): string {
  const at = row.level === null || row.level === undefined ? '' : ` at ${formatNumber(row.level)}`;
  return `${formatNumber(row.value)} per unit${at}`;
}

/** One bucket of a ladder: its coordinates less the one summed across, the risk of its nodes. */
type Bucket = { key: number[]; risk: number; nodes: GreekRow[] };

/** One factor's buckets in coordinate order, the names of its axes - `across` the position of the
 * one its buckets sum over, -1 where none is - and its parallel move, every node at once. */
export type Ladder = {
  factor: string; per: string | null; axes: string[]; across: number; buckets: Bucket[];
  total: number;
};

/** The gradient as a desk reads it: every spot's delta together, largest first; each curve's
 * ladder by tenor; each surface's buckets by expiry - its nodes summed across the axis its type
 * declares as moneyness, the nodes kept on the bucket - and every row read per unit apart. */
export function riskReading(rows: GreekRow[], schema: Schema | null) {
  const spots: GreekRow[] = [];
  const units: GreekRow[] = [];
  const held = new Map<string, GreekRow[]>();
  for (const row of rows) {
    if (perOf(row) === 'per unit') units.push(row);
    else if (row.tenor === undefined) spots.push(row);
    else held.set(row.factor, [...(held.get(row.factor) ?? []), row]);
  }
  const curves: Ladder[] = [];
  const surfaces: Ladder[] = [];
  for (const factor of [...held.keys()].sort()) {
    const nodes = held.get(factor)!;
    const count = nodes[0].tenor!.length;
    const axes = factorAxes(schema, factor, count).slice(0, count);
    const across = axes.findIndex((axis) => /moneyness/i.test(axis));
    const buckets = new Map<string, Bucket>();
    for (const node of nodes) {
      const key = node.tenor!.filter((_, at) => at !== across);
      const bucket = buckets.get(JSON.stringify(key)) ?? { key, risk: 0, nodes: [] };
      bucket.risk += riskOf(node);
      bucket.nodes.push(node);
      buckets.set(JSON.stringify(key), bucket);
    }
    (count === 1 ? curves : surfaces).push({
      factor, per: sharedPer(nodes), axes, across,
      buckets: [...buckets.values()].sort((a, b) => compareTenor(a.key, b.key)),
      total: nodes.reduce((sum, node) => sum + riskOf(node), 0),
    });
  }
  spots.sort((a, b) => Math.abs(riskOf(b)) - Math.abs(riskOf(a))
    || a.factor.localeCompare(b.factor));
  return { spots, curves, surfaces, units };
}

/** What a marked row holds, off the positions inside it: the clients it is against, each by name
 * and id, and the trades in it that wait on a second seat. */
export function markHolds(held: Position[], entities: Entity[]):
  { clients: string[]; pending: Position[] } {
  return {
    clients: [...new Set(held.flatMap((position) => position.counterparty ?? []))]
      .map((id) => entityText(entities, id)),
    pending: held.filter((position) => position.pending !== 0),
  };
}

/** The positions inside each marked row, by its deal path: a position the row's node holds, or one
 * any node beneath it does. */
export function positionsUnder(marks: DealMark[], positions: Position[]): Map<string, Position[]> {
  const inside = new Map<string, Position[]>();
  for (const mark of marks) {
    if (mark.deal_path === null) continue;
    const path = mark.deal_path;
    const held = positions.filter((position) => position.deal_paths.some((at) => under(at, path)));
    if (held.length) inside.set(path, held);
  }
  return inside;
}

/** An FX spot on its pair as the market quotes it. `FxRate.ZAR` is the base currency's price of
 * one rand - a level of a deal on ZAR priced in USD - so it reads by the blotter's own rule: USDZAR
 * 17.35 for 0.0576, `turned` where that is its reciprocal. Undefined where no pair is known. */
export function spotQuoted(factor: string, level: number, market: Market | null):
  { pair: string; level: number; turned: boolean; held: string } | undefined {
  const [family, currency] = factor.split('.');
  const base = String(token(market?.['System Parameters'], 'Base_Currency') || '');
  if (family !== 'FxRate' || !base) return undefined;
  const read = levelOnAxis(currency, base, level, market);
  return read && { ...read, held: `${base} per 1 ${currency}` };
}

/** A gradient row of an FX spot re-read on its pair: the level the market quotes, the derivative
 * per unit of it, and the risk per 1% of it - the engine's 1% of a reciprocal is minus 1% of the
 * pair, to first order exactly. Any other row as it came. */
export function onPair(row: GreekRow, market: Market | null): GreekRow & { pair?: string } {
  const spot = typeof row.level === 'number'
    ? spotQuoted(row.factor, row.level, market) : undefined;
  if (!spot || !spot.turned) return spot ? { ...row, pair: spot.pair } : row;
  return { ...row, pair: spot.pair, level: spot.level, value: -row.value * row.level! ** 2,
           risk: row.risk === null || row.risk === undefined ? row.risk : -row.risk };
}

// ---- the market as a desk reads it --------------------------------------------------------------

/** The price-factor family a counterparty's credit curve is filed under, `SurvivalProb.<id>`. */
export const SURVIVAL = 'SurvivalProb';

/** The quotes family a credit curve is solved from, its block `SurvivalProbPrices.<id>`. */
export const SURVIVAL_QUOTES = `${SURVIVAL}Prices`;

/** The recovery a credit curve's quotes are priced at where none is stated, its declared default. */
export const recoveryConvention = (schema: Schema | null): number =>
  Number(schema?.MarketPrices.types[SURVIVAL_QUOTES]?.Recovery_Rate?.value);

/** A credit curve as the officer types it: the recovery, the currency its default swaps are struck
 * in and the curve they discount on, and one flat spread - the first row's - or a spread in basis
 * points per tenor. */
export type SpreadForm = {
  recovery: string; currency: string; discount_rate: string; flat: boolean;
  rows: { tenor: string; spread: string }[];
};

/** What `POST /book/survival` takes: a typed number as the number, anything else as typed. */
export type SpreadRequest = {
  counterparty: string; recovery: number | string; currency: string; discount_rate: string;
  spread: number | string | { tenor: string; spread: number | string }[];
};

const typed = (text: string): number | string => {
  const plain = ungrouped(text.trim());
  return plain !== '' && isFinite(Number(plain)) ? Number(plain) : plain;
};

/** The request a form states: a flat spread as one number, else every row anything is typed on. */
export function spreadRequest(counterparty: string, form: SpreadForm): SpreadRequest {
  return {
    counterparty: counterparty.trim(), recovery: typed(form.recovery),
    currency: form.currency.trim(), discount_rate: form.discount_rate.trim(),
    spread: form.flat ? typed(form.rows[0]?.spread ?? '')
      : form.rows.filter((row) => `${row.tenor}${row.spread}`.trim())
        .map((row) => ({ tenor: row.tenor.trim(), spread: typed(row.spread) })),
  };
}

/** The form a counterparty's curve opens in: the spreads its quotes block states (`HeldSpreads`),
 * a spread per tenor as they stand, at their recovery, currency and discount curve - else a flat
 * spread left blank at the convention's recovery, the currency and the curve chosen. */
export function spreadForm(held: HeldSpreads | undefined, schema: Schema | null): SpreadForm {
  const rows = (held?.rows ?? []).filter((row) => row.type === 'Par_Spread')
    .map((row) => ({ tenor: row.tenor, spread: String(row.quote ?? '') }));
  return { recovery: String(held?.recovery ?? recoveryConvention(schema)),
           currency: held?.currency ?? '', discount_rate: held?.discount_rate ?? '',
           flat: !rows.length, rows: rows.length ? rows : [{ tenor: '', spread: '' }] };
}

/** Where the survival route's refusal reads: beside the field of `request` its `code` names, any
 * other - the tenors, a seat without the grant - by the act. */
export const refusedAt = (code: RefusalCode | null, request: SpreadRequest):
  keyof SpreadRequest | 'act' => (code && code in request ? code as keyof SpreadRequest : 'act');

/** Who a credit curve is stated for: on a desk that records, the entities its record declares; on
 * one that does not, the counterparties the book's netting sets name and those a curve is held for. */
export function creditCounterparties(entities: Entity[] | null, doc: JobDoc | null,
                                     factors: Record<string, unknown>): string[] {
  if (entities) return entities.map((entity) => entity.entity).sort();
  const named = [...rowsByPath(toRows(doc?.Calc.Deals.Deals.Children ?? [], [], undefined)).values()]
    .map((row) => token(row.deal.Credit_Support_Amounts, 'Counterparty'));
  const held = Object.keys(factors).filter((name) => factorFamily(name) === SURVIVAL)
    .map((name) => name.slice(SURVIVAL.length + 1));
  return [...new Set([...named, ...held])]
    .filter((id): id is string => typeof id === 'string' && id !== '').sort();
}

// ---- what a table of a run's results is in ------------------------------------------------------

/** A potential future exposure column, the percentile of the mark it is at in its name. */
const PFE = /^PFE_(\d+(?:\.\d+)?)$/;

/** The columns the engine's exposure summary names, in words. */
const EXPOSURE: [RegExp, (found: RegExpExecArray) => string][] = [
  [/^EE$/, () => 'expected exposure, the mean of the mark where it is positive'],
  [/^ENE$/, () => 'expected negative exposure, the mean of the mark where it is negative'],
  [PFE, (found) => `the ${found[1]}th percentile of the mark`],
];

/** The largest number of a profile's column and the day it falls on. */
export type Peak = { value: number; day: string };

/** How large a run's exposure gets, off the peaks its row keeps: its expected exposure's and each
 * PFE column's, with its day, and the percentile that column names. */
export function peaks(held: XvaSet['peaks']): { ee?: Peak; pfe: (Peak & { quantile: number })[] } {
  return { ee: held?.EE, pfe: Object.entries(held ?? {}).flatMap(([name, peak]) => {
    const found = PFE.exec(name);
    return found ? [{ ...peak, quantile: Number(found[1]) }] : [];
  }) };
}

type TableReading = { currency: string; collateral: string[]; columns: [string, string][] };

/** What a run's table is in: one filed under a currency the market holds a spot for -
 * `cashflows/ZAR` - in that currency, any other in the run's report currency; one over dates of a
 * run holding sets it collateralises (`collateralised`) is net of what they hold, which a set
 * over-collateralised reads below nought; its exposure columns in words, and numbered columns over
 * dates a path each. */
export function tableReading(name: string, columns: unknown[], dated: boolean, doc: JobDoc,
                             collateralised: string[], market: Market | null): TableReading {
  const filed = name.includes('/') ? fileName(name) : '';
  const own = !!filed && `FxRate.${filed}` in (market?.['Price Factors'] ?? {});
  const paths = dated && columns.length > 0 && columns.every((column, at) => column === at);
  return {
    currency: own ? filed : reportCurrency(doc),
    collateral: dated && !own ? collateralised : [],
    columns: paths ? [[`0–${columns.length - 1}`, 'one simulated path each']]
      : columns.map(String).flatMap((column) => EXPOSURE.flatMap(([rule, words]) => {
        const found = rule.exec(column);
        return found ? [[column, words(found)] as [string, string]] : [];
      })),
  };
}

/** The family a factor name opens with - `FxRate`, `InterestRate`, `FXVol`. What a desk groups by
 * when it reads a gradient, and it is the name's own first segment, not a list kept here. */
export function factorFamily(factor: string): string {
  const stop = factor.indexOf('.');
  return stop === -1 ? factor : factor.slice(0, stop);
}

/** What a field naming price factors offers: every factor of the market of a type its declaration
 * names, by the name after the type - `InterestRate.ZAR` as `ZAR`. A table's cell is declared per
 * column, so it is asked with its column's own. */
export function factorNames(descriptor: Descriptor | undefined, market: Market | null): string[] {
  const types = descriptor?.factors;
  // asked once per input on a screen, and most inputs name no factor
  if (!Array.isArray(types) || !types.length) return [];
  return [...new Set(Object.keys(market?.['Price Factors'] ?? {})
    .filter((name) => types.includes(factorFamily(name)))
    .map((name) => name.slice(name.indexOf('.') + 1)))].sort();
}

// ---- the exports: a table's rows under its own heads, every number as itself --------------------

/** The marks table's heads: on a desk that records, each row's clients and what waits in it too. */
export const marksHeads = (recording: boolean): string[] => ['Reference',
  ...(recording ? ['Client'] : []), 'Path', ...(recording ? ['Waiting on a second seat'] : []), MARKED];

/** The marks as their export writes them, their heads first: each mark the number itself, its
 * currency in a column of its own, and - with `holds`, on a desk that records - each row's clients
 * and the trades waiting in it by name. */
export function marksExport(risk: BookRisk,
                            holds?: (path: string | null) => ReturnType<typeof markHolds>):
  unknown[][] {
  return [[...marksHeads(!!holds), 'Ccy'], ...risk.per_deal.map((row) => {
    const held = holds?.(row.deal_path);
    return [row.reference, ...(held ? [held.clients.join(', ')] : []), row.deal_path,
      ...(held ? [held.pending.map(positionName).join(', ')] : []), row.value, risk.currency];
  })];
}

/** The by-quote table's heads. */
export const QUOTE_HEADS = ['Block · Factor', 'Quote · Tenor', 'Risk', 'per'];

/** Where a gradient row sits, as its cell reads: its quote as a dealer names it, else its
 * coordinates as its factor's axes name them. */
export const placeText = (row: GreekRow, schema: Schema | null): string => (row.quote === undefined
  ? tenorText(row.tenor, factorAxes(schema, row.factor, row.tenor?.length ?? 0))
  : quoteText(row.quote));

/** The axes a ladder's buckets are keyed by: its factor's, less the one they sum across. */
export const bucketAxes = (ladder: Ladder): string[] =>
  ladder.axes.filter((_, at) => at !== ladder.across);

/** A gradient row's factor as its cell reads: an FX spot the pair it is read on, its name beside. */
const factorText = (row: GreekRow & { pair?: string }): string =>
  (row.pair ? `${row.pair} ${row.factor}` : row.factor);

/** The quote reading's `rows` as their export writes them, under the table's own heads: each
 * risk per its move the number itself, the report currency in a column of its own. */
export const quotesExport = (rows: GreekRow[], schema: Schema | null, currency: string):
  unknown[][] => [[...QUOTE_HEADS, 'Ccy'], ...rows.map((row) =>
  [factorText(row), placeText(row, schema), riskOf(row), perOf(row), currency])];

/** The desk reading as the one table its export writes: every spot on its pair at its level, each
 * curve's and surface's buckets under its factor - a surface's the sum its row shows - then each
 * row read per unit; every risk the number itself, the report currency in a column of its own. */
export function readingExport(rows: GreekRow[], schema: Schema | null, market: Market | null,
                              currency: string): unknown[][] {
  const reading = riskReading(rows, schema);
  return [['Factor', 'Coordinates', 'Level', 'Risk', 'per', 'Ccy'],
    ...reading.spots.map((spot) => onPair(spot, market)).map((row) =>
      [factorText(row), undefined, row.level, riskOf(row), perOf(row), currency]),
    ...[...reading.curves, ...reading.surfaces].flatMap((ladder) => ladder.buckets.map((bucket) =>
      [ladder.factor, tenorText(bucket.key, bucketAxes(ladder)), undefined, bucket.risk,
        sharedPer(bucket.nodes), currency])),
    ...reading.units.map((row) =>
      [row.factor, placeText(row, schema), undefined, riskOf(row), perOf(row), currency])];
}

/** What stands under the service's refusal to price the book's risk (its `code` `unpriced`): that
 * nothing was served, and who puts it right - the desk on its own book, its sender on a document
 * `opened` here from another system. Undefined for any other refusal - a seat's - which says for
 * itself. */
export function unpriced(code: RefusalCode | null, opened?: string): string | undefined {
  if (code !== 'unpriced') return undefined;
  return `Nothing was cached and nothing partial was served. ${opened
    ? `${opened} came from another system and will not price as it was sent: its sender puts it `
      + 'right, the desk cannot, and the corrected file is opened here again.'
    : 'The book will not price as it stands, and the desk puts it right in its own book: this '
      + 'screen asks again as soon as the book moves.'}`;
}

/** What leaving the served book asks first, where it is a document `opened` here whose copy may
 * hold edits made on this desk: that they go with it unless it is downloaded. Null on the desk's
 * own book, and on a copy standing at the etag its open answered this client (`seen`). */
export function leavingAsks(served: { etag: string; path: string; opened?: string },
                            seen: { etag: string; path: string } | null): string | null {
  if (!served.opened || (seen?.path === served.path && seen.etag === served.etag)) return null;
  return `Edits made to ${served.opened} on this desk are lost unless it is downloaded first.`;
}

/** The label of the header's control that leaves an opened file or a read-only copy for the book. */
export const BACK_TO_BOOK = 'back to book';
const LEAVE_COPY = `Press "${BACK_TO_BOOK}" at the top to leave it`;

/** What a screen reading the desk's own book says over a file the desk shows as a read-only copy:
 * the control that leaves the copy, and `shown`, what the screen then shows of the book. */
export const offBook = (shown: string): { title: string; hint: string } => ({
  title: "This screen reads the desk's own book.",
  hint: `The document on screen is a read-only copy of a file. ${LEAVE_COPY}: this screen then `
    + `shows ${shown}.`,
});

/** What a file the service would not open is said as over its read-only copy where the desk
 * records (`code` `recorded`): the service's own sentence, a state of the desk and no error, with
 * the control that leaves the copy. Null for any other refusal, which says for itself. */
export const recordedCopy = (refused: { error: string; code: RefusalCode | null }): string | null =>
  (refused.code === 'recorded' ? `${refused.error} ${LEAVE_COPY}.` : null);

/** What a screen with no book under it says the reader can do, beneath the service's sentence. */
export const OPEN_FILE = 'Press "open a job file" at the top to work on one here.';

/** The one clock an answer read off the live book is held against: the book's etag, and on a
 * desk that records how far the record stands (`pinnedAt`) - a booking moves the first, and an
 * approval, a declaration or a fill the record alone holds the second. */
export const bookClock = (etag: string, recordAt: string | null): string =>
  `${etag}@${recordAt ?? ''}`;

/** Whether an answer on screen was read at an OLDER clock than the one the app now holds. Asked of
 * the clock the answer was FETCHED at and never of the risk's own `etag`: the two are hashes of
 * different things, and comparing them would read stale forever. */
export function lagsBook(fetchedAt: string | null, now: string | null | undefined): boolean {
  return fetchedAt !== null && !!now && fetchedAt !== now;
}

/** The report currency, off the document the app already holds - the one the risk answer states
 * and a recalc would report in now; an XVA row states its own. */
export function reportCurrency(doc: JobDoc | null): string {
  const currency = doc?.Calc?.Calculation?.['Currency'];
  return typeof currency === 'string' ? currency : '';
}

/** The day the book values on, its calculation's `Base_Date` as written. */
export const valuationDay = (doc: JobDoc | null): string =>
  label(doc?.Calc?.Calculation?.Base_Date);

/** Where the book says its market comes from: the file it names, null where its own block is all
 * of it, and the latest moment any quote of the market it prices on says it was seen. */
export function marketSource(doc: JobDoc | null, market: Market | null) {
  const named = doc?.Calc?.MergeMarketData?.MarketDataFile;
  const seen = [
    ...Object.values(market?.['Market Prices'] ?? {}).flatMap((block) => {
      const points = token(token(block, 'instrument'), 'Points');
      return Array.isArray(points) ? points.map((point) => token(point, 'Timestamp')) : [];
    }),
    ...Object.values(market?.['Price Factors'] ?? {})
      .map((block) => token(block, 'Quote_Timestamp')),
  ].map((stamp) => (isObject(stamp) ? String(stamp['.Timestamp']) : ''))
    .filter((stamp) => instantMs(stamp) !== undefined)
    .sort((a, b) => instantMs(a)! - instantMs(b)!);
  return { file: named ? fileName(named) : null, seen: seen.pop() ?? null };
}

// ---- a named calculation ------------------------------------------------------------------------

/** The dials a named calculation's type declares and its block does not state, at the book's
 * own: `POST /calculations/run` merges the block over the book's, so that is what they run at. */
export function ownDials(own: Record<string, unknown>, block: Record<string, unknown>,
                         declared: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(own).filter(([key]) => key in declared && !(key in block)));
}

// ---- the XVA projection -------------------------------------------------------------------------

/** A row whose set has left the book. It keeps its last numbers - a number that was on the
 * blotter yesterday vanishing without a word is how a desk loses track of a position - and the
 * service sorts these after the book's own sets, an order this module never re-imposes. */
function isOrphan(set: XvaSet): boolean {
  return set.note !== null;
}

/** One currency's sums: the CVA over the book's own sets that have RUN in it - never an orphan,
 * whose exposure has left the book, nor a set with no number - and the FVA over the rows that CARRY
 * one, counted apart since a row filed before the column existed reads null there. `currency` is
 * null for rows filed before the run kept one. */
type XvaSum = {
  currency: string | null; cva: number; counted: number; fva: number; countedFva: number;
};

type XvaTotals = {
  /** Sets the book holds now. Orphans are counted apart, because they are not the book's. */
  sets: number;
  done: number;
  failed: number;
  neverRun: number;
  running: number;
  orphans: number;
  /** One sum per currency the rows were run in: amounts in two currencies never add. */
  sums: XvaSum[];
};

/** What a positive credit adjustment is, as every screen showing one says it beside the number. */
export const PLUS = { CVA: '+ is an expected loss', FVA: '+ is a funding cost' } as const;

export function xvaTotals(view: BookXva): XvaTotals {
  const totals: XvaTotals = {
    sets: 0, done: 0, failed: 0, neverRun: 0, running: 0, orphans: 0, sums: [],
  };
  for (const set of view.sets) {
    if (set.recalc) totals.running += 1;
    if (isOrphan(set)) {
      totals.orphans += 1;
      continue;
    }
    totals.sets += 1;
    if (set.status === 'done') totals.done += 1;
    else if (set.status === 'failed') totals.failed += 1;
    else if (set.status === 'never run') totals.neverRun += 1;
    if (set.status !== 'done') continue;
    let sum = totals.sums.find((held) => held.currency === set.currency);
    if (!sum) {
      sum = { currency: set.currency, cva: 0, counted: 0, fva: 0, countedFva: 0 };
      totals.sums.push(sum);
    }
    if (typeof set.cva === 'number') {
      sum.cva += set.cva;
      sum.counted += 1;
    }
    if (typeof set.fva === 'number') {
      sum.fva += set.fva;
      sum.countedFva += 1;
    }
  }
  return totals;
}

/** The replay tuple, labelled: what the row IS to the book over what the RUN was, in that order.
 * A null reads as absent rather than as a blank - the expander is where a desk goes to find out
 * whether a number can be reproduced, and half a tuple must look like half a tuple. `FVA` is here
 * as well as in the table because its absence is a fact about the RUN: a null beside a `done` row
 * says the row predates the column, which is what an old `as_of` and a recalc would fix. */
export function replayRows(set: XvaSet): [string, string | null][] {
  return [
    ['Deal path', set.deal_path],
    ['Counterparty', set.counterparty],
    ['Collateralised', set.collateralized ? 'yes' : 'no'],
    ['Reported in', set.currency],
    ['FVA', set.fva === null ? null : formatNumber(set.fva, 'money')],
    ['Last run', set.as_of === null ? null : stampText(set.as_of)],
    ['Result id', set.result_id],
    ['Plan hash', set.plan_hash],
    ['Values hash', set.values_hash],
    ['Seed', set.seed === null ? null : String(set.seed)],
    ['Quasi seed', set.quasi_seed === null ? null : String(set.quasi_seed)],
  ];
}
