"""SIMM off a source CRIF: the trades it names, each loaded from its own job document, valued as
one book of a netting set per counterparty by the framework's SIMM calculation and merged back into
the source CRIF.

    python experiments/simm_from_crif.py CRIF FOLDER --mapping MAPPING --parameters PARAMETERS
        [--calendar CALENDAR] [--currency CCY] [--workers N] [--output DIR]

Run from the repo root. FOLDER holds `<TradeID>.json` for every trade the CRIF names, a job document
whose one netting set holds the trade. The book is one job document: the SIMM block, the union of
the documents' market data, the calendar, and per counterparty a NettingCollateralSet holding its
trades and posting and collecting margin under the regimes the CRIF's rows state - the job a desk
hands the engine, written as it runs to `SIMM_JOB_<date>.json`. A trade whose document the run
refuses by name is set aside and the book run again without it.

Written under the output folder (artifacts/simm_from_crif unless stated): CRIF_REVAL_<date>.tsv, the
source CRIF with every row of each trade the run valued replaced by the run's in the source's
layout; CRIF_<date>.tsv, the run's own; TRADES_<date>.csv, its report per trade; and
COMPARE_<date>.csv, per trade the rows matched, missing and extra against the source and the largest
relative difference of a matched amount, with every trade the run could not value and the engine's
sentence for it.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import derivus
import derivus_batch

#: What a row of a CRIF is keyed by, its vertex read whatever its case.
KEY = ['TradeID', 'RiskType', 'Qualifier', 'Bucket', 'Label1', 'Label2']


def read_crif(path):
    """A CRIF as text, every column as written."""
    return pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)


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
    terms = source.drop_duplicates('TradeID').set_index('TradeID')
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
        sets.setdefault(terms.loc[trade, 'Counterparty'], {})[trade] = document['Calc']['Deals'][
            'Deals']['Children'][0]
    children = []
    for counterparty, held in sets.items():
        agreement = agreed({trade: {key: value for key, value in node['Instrument']['.Deal'].items()
                                    if key != 'Reference'} for trade, node in held.items()},
                           'the netting set of ' + counterparty)
        post, collect = agreed({trade: tuple(terms.loc[trade, ['PostRegulations',
                                                                'CollectRegulations']])
                                for trade in held}, 'the regulations of ' + counterparty)
        children.append({'Instrument': {'.Deal': dict(
            agreement, Reference=counterparty, Post_Regulations=post, Collect_Regulations=collect)},
            'Children': [translated(child) for node in held.values() for child in node['Children']]})
    return {'Calc': {'Calculation': calculation,
                     'Deals': {'Reference': 'simm_from_crif', 'Deals': {'Children': children}},
                     'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': explicit},
                     'CalendDataFile': calendar}}


def run(job, output):
    """`(calculation, results, refused)` - `job` run as SIMM, its curves quoted as the batch runner
    quotes them, each trade whose document the run refuses by name set aside and the job run again,
    `refused` the engine's sentence per trade set aside."""
    refused = {}
    while True:
        context = derivus.Context()
        context.load_json((json.dumps(job), 'simm_from_crif'), compress=False)
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
    parser.add_argument('--mapping', required=True, help='the regime mapping SIMM reads')
    parser.add_argument('--parameters', required=True, help='the ISDA SIMM parameters file')
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
    currency = args.currency or agreed(dict(zip(source['TradeID'], source['AmountCurrency'])),
                                       'AmountCurrency')
    calculation = {
        'Object': 'SIMM', 'Currency': currency, 'Mapping': args.mapping,
        'Parameters': args.parameters, 'Workers': args.workers,
        'Base_Date': agreed({trade: document['Calc']['Calculation']['Base_Date']
                             for trade, document in documents.items()}, 'Base_Date')}
    os.makedirs(args.output, exist_ok=True)
    calc, results, refused = run(book(source, documents, calculation, args.calendar), args.output)

    date = '{:%Y-%m-%d}'.format(calc.params['Base_Date'])
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
    print('trades {}, valued {}, not valued {}; rows matched {}, missing {}, extra {}; largest '
          'relative difference {:.3g}; {:.1f} seconds'.format(
              len(trades), len(valued), len(unvalued), comparison['Matched'].sum(),
              comparison['Missing'].sum(), comparison['Extra'].sum(),
              comparison['Largest_Relative'].max(), time.time() - started))


if __name__ == '__main__':
    main()
