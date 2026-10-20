"""SIMM off a source CRIF: the trades it names, each loaded from its own job document, valued as
one book of a netting set per counterparty by the framework's SIMM calculation and merged back into
the source CRIF.

    python experiments/simm_from_crif.py CRIF FOLDER --parameters PARAMETERS [--mapping MAPPING]
        [--calendar CALENDAR] [--currency CCY] [--workers N] [--output DIR]

Run from the repo root. FOLDER holds `<TradeID>.json` for every trade the CRIF names, a job document
whose one netting set holds the trade. The book is one job document: the SIMM block, the union of
the documents' market data, the calendar, and per counterparty a NettingCollateralSet holding its
trades and posting and collecting margin under the regimes the CRIF's rows state - the job a desk
hands the engine, written as it runs to `SIMM_JOB_<date>.json`. A trade whose document the run
refuses by name is set aside and the book run again without it.

The run reads the mapping a bank states in MAPPING, none unless named, completed off the CRIF's own
rows (`derived_mapping`) for every factor a trade reads that it states nothing for and every
product class it lacks: what it states is kept, a derivation that disagrees is listed, and a factor
nothing pins is left unstated, so the run refuses its trades by name.

Written under the output folder (artifacts/simm_from_crif unless stated): SIMM_MAPPING_<date>.json,
the completed mapping; MAPPING_<date>.txt, tab separated, every derivation, pin not made and
disagreement with the mapping stated; CRIF_REVAL_<date>.tsv, the source CRIF with every row of each
trade the run valued replaced by the run's in the source's layout; CRIF_<date>.tsv, the run's own;
TRADES_<date>.csv, its report per trade; and COMPARE_<date>.csv, per trade the rows matched, missing
and extra against the source and the largest relative difference of a matched amount, with every
trade the run could not value and the engine's sentence for it.
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import pandas as pd

import derivus
import derivus_batch
from derivus import utils
from derivus.config import decode_wire

#: What a row of a CRIF is keyed by, its vertex read whatever its case.
KEY = ['TradeID', 'RiskType', 'Qualifier', 'Bucket', 'Label1', 'Label2']
#: The factor types whose coordinates a trade's own rows pin, each the risk type it is reported as.
PINNED = {'EquityPrice': 'Risk_Equity', 'EquityPriceVol': 'Risk_EquityVol', 'FXVol': 'Risk_FXVol',
          'CommodityPrice': 'Risk_Commodity', 'ForwardPrice': 'Risk_Commodity',
          'FuturesPrice': 'Risk_Commodity', 'CommodityPriceVol': 'Risk_CommodityVol'}
#: The risk types a survival curve is reported as, which of the two the source's own rows say.
CREDIT = ('Risk_CreditQ', 'Risk_CreditNonQ')
#: The factor types ISDA names no risk factor for, mapped null: a dividend curve, a correlation, a
#: commodity's carry curve, a basis on no mapped commodity spot, a sampling rule, the delivery a
#: fixing reads and an index's prints.
UNNAMED = ('DividendRate', 'Correlation', 'ForwardRate', 'ObservedBasis', 'ForwardPriceSample',
           'ReferencePrice', 'PriceIndex')
#: The columns of MAPPING_<date>.txt.
NOTE = ['Outcome', 'Kind', 'Name', 'Stated', 'Derived', 'Detail']
#: What a mapping states or a derivation finds for nothing, where None is the null.
NOTHING = object()


def read_crif(path):
    """A CRIF as text, every column as written."""
    return pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)


def terms(source):
    """Each trade's terms off its first row of the `source` CRIF, by TradeID."""
    return source.drop_duplicates('TradeID').set_index('TradeID')


def agreed(stated, what):
    """The one value `{trade: value}` states for `what`, refused naming the trades that differ."""
    first, value = next(iter(stated.items()))
    differ = [trade for trade, other in stated.items() if other != value]
    if differ:
        raise ValueError('{} differs between the documents of {} and {}'.format(
            what, first, ', '.join(differ)))
    return value


def translated(node):
    """`node` and what is under it as the engine reads them: the extract's one translation, an
    EquityBinaryOption stating its cash amount as `Cash_Payoff`, which states it as `Payoff`."""
    deal = node['Instrument']['.Deal']
    if deal['Object'] == 'EquityBinaryOption' and 'Cash_Payoff' in deal and 'Payoff' not in deal:
        deal = dict({key: value for key, value in deal.items() if key != 'Cash_Payoff'},
                    Payoff=deal['Cash_Payoff'])
    return dict(node, Instrument=dict(node['Instrument'], **{'.Deal': deal}), **(
        {'Children': [translated(child) for child in node['Children']]} if 'Children' in node else {}))


def references(node):
    """Every deal reference in `node`'s tree."""
    return {node['Instrument']['.Deal'].get('Reference')}.union(
        *map(references, node.get('Children', [])))


def book(source, documents, calculation, calendar):
    """The job document of `documents`, `{trade: document}`: `calculation` its block, the union of
    their market data, a factor stated two ways refused by name, and per counterparty of the
    `source` CRIF a netting set of its trades on the terms their own sets state."""
    stated_terms = terms(source)
    markets = {trade: document['Calc']['MergeMarketData']['ExplicitMarketData']
               for trade, document in documents.items()}
    explicit, stated = {'System Parameters': agreed(
        {trade: market['System Parameters'] for trade, market in markets.items()},
        'System Parameters')}, {}
    for trade, market in markets.items():
        for section, blocks in market.items():
            for name, block in blocks.items() if section != 'System Parameters' else ():
                merged = explicit.setdefault(section, {})
                if merged.setdefault(name, block) != block:
                    raise ValueError('{} {} is stated one way by {} and another by {}'.format(
                        section, name, stated[section, name], trade))
                stated.setdefault((section, name), trade)
    sets = {}
    for trade, document in documents.items():
        sets.setdefault(stated_terms.loc[trade, 'Counterparty'], {})[trade] = document['Calc'][
            'Deals']['Deals']['Children'][0]
    children = []
    for counterparty, held in sets.items():
        agreement = agreed({trade: {key: value for key, value in node['Instrument']['.Deal'].items()
                                    if key != 'Reference'} for trade, node in held.items()},
                           'the netting set of ' + counterparty)
        post, collect = agreed({trade: tuple(stated_terms.loc[trade, ['PostRegulations',
                                                                       'CollectRegulations']])
                                for trade in held}, 'the regulations of ' + counterparty)
        children.append({'Instrument': {'.Deal': dict(
            agreement, Reference=counterparty, Post_Regulations=post, Collect_Regulations=collect)},
            'Children': [translated(child) for node in held.values() for child in node['Children']]})
    return {'Calc': {'Calculation': calculation,
                     'Deals': {'Reference': 'simm_from_crif', 'Deals': {'Children': children}},
                     'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': explicit},
                     'CalendDataFile': calendar}}


def loaded(job):
    """A context holding `job`."""
    context = derivus.Context()
    context.load_json((json.dumps(job), 'simm_from_crif'), compress=False)
    return context


def pins(source, reads, selected):
    """`(found, failed)` over the trades' `reads`: per factor `selected` names, the coordinates each
    reader's own rows of `source` pin it to, `{factor: {trade: coordinates}}`, and the sentences of
    the readers whose rows pin it to nothing. `selected` maps a factor to the rows it is reported
    in, `(risk types, Qualifier or None, columns)`: a trade pins the one factor of those rows it
    reads where they carry one value of `columns`."""
    found, failed = {}, {}
    for trade, factors in reads.items():
        rows, groups = source[source['TradeID'] == trade], {}
        for factor in factors:
            if factor in selected:
                groups.setdefault(selected[factor], []).append(factor)
        for (risk_types, qualifier, columns), names in groups.items():
            carried = sorted(set(rows[rows['RiskType'].isin(risk_types) & (
                (rows['Qualifier'] == qualifier) | (qualifier is None))][list(columns)].itertuples(
                index=False, name=None)))
            if len(names) == 1 and len(carried) == 1:
                found.setdefault(names[0], {})[trade] = dict(zip(columns, carried[0]))
                continue
            said = ' '.join(filter(None, ('/'.join(risk_types), qualifier)))
            for name in names:
                failed.setdefault(name, []).append('{} reads {} and its {} rows carry {}'.format(
                    trade, ', '.join(names), said, ', '.join(map('/'.join, carried)) or 'none'))
    return found, failed


def baskets(source, reads, found):
    """`{spot: sentence}` of each spot a basket's rows pin, the pin put in `found`: a trade reading
    one spot and its own vol, the vol pinned and its delta rows looked through to constituents -
    several qualifiers, none the vol's - has the one spot here carry the basket's delta, so it takes
    the vol's coordinates."""
    said = {}
    for trade, factors in reads.items():
        spots, vols = ([factor for factor in factors if utils.check_rate_name(factor)[0] == kind]
                       for kind in ('EquityPrice', 'EquityPriceVol'))
        if not (len(spots) == len(vols) == 1 and trade in found.get(vols[0], {})
                and utils.check_rate_name(spots[0])[1] == utils.check_rate_name(vols[0])[1]):
            continue
        rows = source[(source['TradeID'] == trade) & (source['RiskType'] == PINNED['EquityPrice'])]
        vol = found[vols[0]][trade]
        if rows['Qualifier'].nunique() > 1 and vol['Qualifier'] not in set(rows['Qualifier']):
            found.setdefault(spots[0], {})[trade] = vol
            said[spots[0]] = 'a basket: {} looks its delta through to {} constituents and books ' \
                             'its vega on {}'.format(trade, rows['Qualifier'].nunique(), vols[0])
    return said


def disagrees(stated, derived):
    """Whether `derived` states a class, a null or a coordinate other than `stated` does."""
    if isinstance(stated, dict) and isinstance(derived, dict):
        return any(stated.get(key, '') != value for key, value in derived.items())
    return stated != derived


def derived_mapping(source, documents, reads, factors, mapping, parameters):
    """`(completed, notes)`: `mapping` completed off the `source` CRIF's own rows for every factor
    the trades' `reads`, `{trade: [factor]}`, name that it states nothing for and every product
    class of the trades of `documents` it lacks, `factors` their `Price Factors` and `parameters`
    the ISDA parameters file; `notes` the `NOTE` lines of each derivation, pin not made and
    disagreement with `mapping`, whose own statements are kept.

    A spot, a commodity's price factor, or an equity, FX or commodity vol takes the qualifier and
    bucket the rows of its risk type carry for a trade reading no other factor reported as that
    risk type (`pins`), or a basket's spot its vol's (`baskets`); one pinned two ways or by no
    trade is left unstated, and a survival curve so takes the risk type, qualifier, bucket and
    Label2 of its credit rows. An inflation curve is Risk_Inflation in its currency. A
    basis whose primary, its name's first part, is a mapped commodity spot takes the
    spot's coordinates, Risk_Commodity under its qualifier and bucket, being part of the
    commodity's price; one on any other primary maps null, a spot's or a currency's move
    scaling the one factor it names so that an entry there would be read by nothing, as a
    commodity's carry curve, a sampling rule, a fixing's delivery and an index's prints
    map null (`UNNAMED`). A curve is Risk_IRCurve in its currency, in the bucket of the volatility
    group the parameters file puts the currency in, Label2 as the rows of a trade reading it alone
    of its currency pin it, and left unstated where none does, a rate row carrying no sub-curve
    being no row; a child of a curve that is a basis spread is Risk_XCcyBasis in its currency and
    any other rides its parent, null; a rate vol surface is Risk_IRVol in the currency of the curve
    it is named for; an FX rate is Risk_FX. A structure's product class is its trade's rows', by
    reference, and another trade's by instrument type where every trade of the type carries one."""
    ir = next(iter(parameters['horizons'].values()))['ir']
    groups, buckets = ir['ccy']['Weights'], {group: bucket for bucket, group in ir[
        'bucket_vol_map'].items()}
    kinds = {factor: utils.check_rate_name(factor)
             for factor in sorted({factor for names in reads.values() for factor in names})}
    found, failed = pins(source, reads, dict(
        {factor: ((PINNED[kind[0]],), None, ('Qualifier', 'Bucket'))
         for factor, kind in kinds.items() if kind[0] in PINNED},
        **{factor: (CREDIT, None, ('RiskType', 'Qualifier', 'Bucket', 'Label2'))
           for factor, kind in kinds.items() if kind[0] == 'SurvivalProb'},
        **{factor: (('Risk_IRCurve',), factors[factor]['Currency'], ('Label2',))
           for factor, kind in kinds.items() if kind[0] == 'InterestRate' and len(kind) == 2}))
    basket = baskets(source, reads, found)
    completed, notes = copy.deepcopy(mapping), []

    def agreement(factor):
        """The coordinates `factor`'s readers pin it to and who pins it, or None and why not."""
        held = {}
        for trade, value in found.get(factor, {}).items():
            held.setdefault(tuple(value.items()), []).append(trade)
        if len(held) == 1:
            (value, trades), = held.items()
            return {key: part for key, part in value if part}, 'pinned by ' + ', '.join(trades)
        return None, '; '.join('pinned to {} by {}'.format('/'.join(part for _, part in value),
                                                            ', '.join(trades))
                               for value, trades in held.items()) or '; '.join(failed[factor])

    def derivation(factor):
        """`(kind, coordinates, why)` of `factor`, the coordinates NOTHING where nothing derives
        them."""
        kind = kinds[factor]
        if kind[0] in PINNED:
            value, why = agreement(factor)
            return 'basket' if factor in basket else 'pin', NOTHING if value is None else dict(
                RiskType=PINNED[kind[0]], **value), '; '.join(filter(None, (
                    why, basket.get(factor))))
        if kind[0] == 'SurvivalProb':
            value, why = agreement(factor)
            return 'credit', NOTHING if value is None else value, why
        if kind[0] == 'InterestRate' and len(kind) == 2:
            currency = factors[factor]['Currency']
            group = groups['Currency'].get(currency, groups['default'])
            value, why = agreement(factor)
            if value is None:
                return 'curve', NOTHING, 'Label2 ' + why
            return 'curve', dict({'RiskType': 'Risk_IRCurve', 'Qualifier': currency,
                                  'Bucket': buckets[group]}, **value), \
                '{} volatility; Label2 {}'.format(group, why)
        if kind[0] == 'InterestRate':
            # the engine's own sub-type of a curve that is a basis over its parent
            if factors[factor].get('Sub_Type') == 'BasisSpread':
                return 'basis', {'RiskType': 'Risk_XCcyBasis',
                                 'Qualifier': factors[factor]['Currency']}, 'a basis spread'
            return 'spread', None, 'a spread riding its parent, which its quotes move'
        if kind[0] in utils.ThreeDimensionalFactors and (
                utils.FactorRiskClass[kind[0]] == 'InterestRate'):
            curve = utils.check_tuple_name(utils.Factor('InterestRate', kind[1:]))
            return 'vol', {'RiskType': 'Risk_IRVol', 'Qualifier': factors[curve]['Currency']}, \
                'the currency of ' + curve
        if kind[0] == 'InflationRate':
            return 'inflation', {'RiskType': 'Risk_Inflation',
                                 'Qualifier': factors[factor]['Currency']}, 'its currency'
        if kind[0] == 'FxRate':
            return 'fx', {'RiskType': 'Risk_FX', 'Qualifier': kind[1]}, ''
        if kind[0] == 'ObservedBasis':
            # part of its commodity's price, whose spot `kinds`, being sorted, settles first
            spot = utils.check_tuple_name(utils.Factor('CommodityPrice', kind[1:2]))
            held = completed['Factors'][spot]['SIMM'] if spot in completed['Factors'] else None
            if held is not None:
                return 'commodity', held, 'the price of ' + spot
        if kind[0] in UNNAMED:
            return 'unnamed', None, 'ISDA names no risk factor for a ' + kind[0]
        return '', NOTHING, 'nothing derives a {} factor'.format(kind[0])

    def settled(name, stated, kind, derived, why):
        """Whether `name`'s derivation is written - where the mapping states nothing for it - each
        derivation, pin not made and disagreement noted beside what the mapping states."""
        said = '' if stated is NOTHING else json.dumps(stated)
        if derived is NOTHING:
            notes.append(('unpinned', kind, name, said, '', why))
        elif stated is NOTHING:
            notes.append(('derived', kind, name, said, json.dumps(derived), why))
        elif disagrees(stated, derived):
            notes.append(('disagrees', kind, name, said, json.dumps(derived), why))
        return stated is NOTHING and derived is not NOTHING

    for factor in kinds:
        stated = mapping['Factors'][factor]['SIMM'] if factor in mapping['Factors'] else NOTHING
        kind, derived, why = derivation(factor)
        if settled(factor, stated, kind, derived, why):
            completed['Factors'][factor] = {'SIMM': derived}
    stated_terms, classes = terms(source), {}
    for trade, document in documents.items():
        for node in document['Calc']['Deals']['Deals']['Children'][0]['Children']:
            deal = node['Instrument']['.Deal']
            classes.setdefault(('Structures', deal['Reference'], 'structure') if node.get(
                'Children') else ('Instruments', deal['Object'], 'instrument'), {}).setdefault(
                stated_terms.loc[trade, 'ProductClass'], []).append(trade)
    for (section, key, kind), held in classes.items():
        product = next(iter(held)) if len(held) == 1 else NOTHING
        if settled('{}.{}'.format(section, key), mapping['Product_Class'][section].get(
                key, NOTHING), kind, product, '; '.join('{} carried by {}'.format(
                    carried, ', '.join(carriers)) for carried, carriers in held.items())):
            completed['Product_Class'][section][key] = product
    return completed, notes


def run(job, output):
    """`(calculation, results, refused)` - `job` run as SIMM, its curves quoted as the batch runner
    quotes them, each trade whose document the run refuses by name set aside and the job run again,
    `refused` the engine's sentence per trade set aside."""
    refused = {}
    while True:
        context = loaded(job)
        config = context.current_cfg
        date = config.deals['Calculation']['Base_Date']
        derivus_batch.quote_curves(config, date)
        context.save_json(os.path.join(output, 'SIMM_JOB_{:%Y-%m-%d}.json'.format(date)))
        try:
            calc, results = context.run_job()
            return calc, results['Results'], refused
        except Exception as error:
            said = '{}: {}'.format(type(error).__name__, error)
            sets = job['Calc']['Deals']['Deals']['Children']
            named = [trade['Instrument']['.Deal']['Reference'] for agreement in sets
                     for trade in agreement['Children']
                     if any(reference and reference in str(error) for reference in references(trade))]
            if len(named) != 1:
                raise
            refused[named[0]] = said
            print('set aside {} - {}'.format(named[0], said[:300]), flush=True)
            for agreement in sets:
                agreement['Children'] = [trade for trade in agreement['Children']
                                         if trade['Instrument']['.Deal']['Reference'] != named[0]]
            job['Calc']['Deals']['Deals']['Children'] = [
                agreement for agreement in sets if agreement['Children']]


def compared(source, crif, unvalued):
    """Per trade of `source`, its rows the run matched, missed and added and the largest relative
    difference of a matched amount, keyed by `KEY` with the vertex in any case; `unvalued` the
    engine's sentence per trade the run could not value."""
    def amounts(frame):
        return frame.assign(Label1=frame['Label1'].str.lower(),
                            Amount=frame['Amount'].astype(float)).groupby(KEY)['Amount'].sum()

    theirs, mine = amounts(source), amounts(crif)
    rows = []
    for trade in source['TradeID'].drop_duplicates():
        a, b = theirs[theirs.index.get_level_values(0) == trade], mine[
            mine.index.get_level_values(0) == trade]
        both = a.index.intersection(b.index)
        relative = (np.abs(b[both] - a[both]) / np.abs(a[both])).max() if len(both) else np.nan
        rows.append({'TradeID': trade, 'Valued': trade not in unvalued, 'Matched': len(both),
                     'Missing': len(a) - len(both), 'Extra': len(b) - len(both),
                     'Largest_Relative': relative, 'Engine': unvalued.get(trade, '')})
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('crif', help='the source CRIF, tab separated')
    parser.add_argument('folder', help='the folder holding <TradeID>.json per trade')
    parser.add_argument('--parameters', required=True, help='the ISDA SIMM parameters file')
    parser.add_argument('--mapping', help='the regime mapping the bank states, completed off the '
                                          'CRIF; none unless stated')
    parser.add_argument('--calendar', default='', help='the calendar file, none unless stated')
    parser.add_argument('--currency', help="the calculation currency, the CRIF's AmountCurrency "
                                           'unless stated')
    parser.add_argument('--workers', type=int, default=1, help='processes the moves are dealt over')
    parser.add_argument('--output', default=os.path.join('artifacts', 'simm_from_crif'),
                        help='the folder the outputs are written to')
    args = parser.parse_args(argv)
    started = time.time()

    source = read_crif(args.crif)
    trades = list(source['TradeID'].drop_duplicates())
    paths = {trade: os.path.join(args.folder, trade + '.json') for trade in trades}
    missing = [trade for trade, path in paths.items() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError('{} holds no job document for {}'.format(
            args.folder, ', '.join(missing)))
    documents = {}
    for trade, path in paths.items():
        with open(path, 'rt', encoding='utf-8') as f:
            documents[trade] = json.load(f)
    stated = {'Product_Class': {'Instruments': {}, 'Structures': {}}, 'Factors': {}}
    if args.mapping:
        with open(args.mapping, 'rt', encoding='utf-8') as f:
            stated = json.load(f)
    with open(args.parameters, 'rt', encoding='utf-8') as f:
        parameters = json.load(f)
    currency = args.currency or agreed(dict(zip(source['TradeID'], source['AmountCurrency'])),
                                       'AmountCurrency')
    base_date = agreed({trade: document['Calc']['Calculation']['Base_Date']
                        for trade, document in documents.items()}, 'Base_Date')
    date = '{:%Y-%m-%d}'.format(decode_wire(base_date, None))
    calculation = {
        'Object': 'SIMM', 'Currency': currency, 'Parameters': args.parameters,
        'Mapping': os.path.join(args.output, 'SIMM_MAPPING_{}.json'.format(date)),
        'Workers': args.workers, 'Base_Date': base_date}
    reads = {trade: loaded(book(source, {trade: document}, calculation, args.calendar)).current_cfg
             .factor_universe()['resolved'] for trade, document in documents.items()}
    job = book(source, documents, calculation, args.calendar)
    mapping, notes = derived_mapping(
        source, documents, reads, job['Calc']['MergeMarketData']['ExplicitMarketData'][
            'Price Factors'], stated, parameters)
    os.makedirs(args.output, exist_ok=True)
    with open(calculation['Mapping'], 'wt', encoding='utf-8') as f:
        json.dump(mapping, f, indent=1)
    notes = pd.DataFrame(notes, columns=NOTE)
    notes.to_csv(os.path.join(args.output, 'MAPPING_{}.txt'.format(date)), sep='\t', index=False)
    calc, results, refused = run(job, args.output)

    crif, report = results['CRIF'], results['Trades']
    calc.write(crif, os.path.join(args.output, 'CRIF_{}.tsv'.format(date)))
    report.to_csv(os.path.join(args.output, 'TRADES_{}.csv'.format(date)), index=False)
    valued = report[np.isfinite(report['Value'].astype(float))]['TradeID']
    derivus_batch.merge_crif(source, crif.astype(str), currency, valued).to_csv(
        os.path.join(args.output, 'CRIF_REVAL_{}.tsv'.format(date)), sep='\t', index=False)
    unvalued = dict(refused, **dict(zip(report[~report['TradeID'].isin(valued)]['TradeID'],
                                        report[~report['TradeID'].isin(valued)]['Skipped'])))
    comparison = compared(source, crif, unvalued)
    comparison.to_csv(os.path.join(args.output, 'COMPARE_{}.csv'.format(date)), index=False)
    print(comparison.to_string(index=False))
    print('mapping: {}'.format(', '.join('{} {}'.format(outcome, count) for outcome, count in
                                         notes['Outcome'].value_counts().items())))
    print('trades {}, valued {}, not valued {}; rows matched {}, missing {}, extra {}; largest '
          'relative difference {:.3g}; {:.1f} seconds'.format(
              len(trades), len(valued), len(unvalued), comparison['Matched'].sum(),
              comparison['Missing'].sum(), comparison['Extra'].sum(),
              comparison['Largest_Relative'].max(), time.time() - started))


if __name__ == '__main__':
    main()
