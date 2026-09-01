########################################################################
# Copyright (C)  Shuaib Osman (vretiel@gmail.com)
# This file is part of Derivus.
#
# Derivus is free for noncommercial use under the terms of the PolyForm
# Noncommercial License 1.0.0. You should have received a copy of the license
# along with Derivus. If not, see
# <https://polyformproject.org/licenses/noncommercial/1.0.0>.
#
# Derivus is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
########################################################################

__author__ = "Shuaib Osman"
__license__ = "Free for non-commercial use"
__all__ = ['version_info', '__version__', '__author__', '__license__', 'Context', 'makeflatcurve', 'getpath',
           'set_collateral', 'load_market_data', 'run_baseval', 'run_cmc', 'update_dict']

import os
import json
import torch
import hashlib
import pathlib
import logging
import numpy as np
import pandas as pd
import collections.abc
import torch.multiprocessing as mp

from ._version import version_info, __version__
# schema FIRST: it assembles `mapping` from the declaring modules, and importing one of those
# first would have it read a half-initialised module
from . import schema
from . import fields
from . import utils
from .instruments import construct_instrument
from .config import CustomJsonEncoder, Config, deal_at


def update_dict(d, u):
    for k, v in u.items():
        d[k] = update_dict(d.get(k, {}), v) if isinstance(v, collections.abc.Mapping) else v
    return d


def makeflatcurve(curr, bps, daycount='ACT_365', tenor=30):
    """
    generates a constant (flat) curve in basis points with the given daycount and tenor
    :return: a dictionary containing the curve definition
    """
    return {'Currency': curr, 'Curve': utils.Curve([], [[0, bps * 0.01 * 0.01], [tenor, bps * 0.01 * 0.01]]),
            'Day_Count': daycount, 'Property_Aliases': None, 'Sub_Type': 'None'}


def getpath(pathlist):
    """
    returns the first valid path in pathlist
    """
    for path in pathlist:
        if os.path.isdir(path):
            return path


def content_hash(obj):
    """sha256 over a CANONICAL dump - sorted keys, and `CustomJsonEncoder` for everything a config
    tree holds that JSON has no form for: curves, deals, timestamps, offsets, model parameters."""
    return hashlib.sha256(json.dumps(
        obj, sort_keys=True, separators=(',', ':'), cls=CustomJsonEncoder).encode()).hexdigest()


def summarize_data(data, percentiles):
    pos_exposure = data.clip(0.0, np.inf)
    neg_exposure = data.clip(-np.inf, 0.0)

    exposure = {
        'EE': np.mean(pos_exposure, axis=1),
        'ENE': np.mean(neg_exposure, axis=1)}

    if percentiles:
        extra = {'PFE_{}'.format(x): np.percentile(data, float(x), axis=1) for x in percentiles.split(',')}
        exposure.update(extra)

    return pd.DataFrame(exposure)

def set_collateral(cfg, Agreement_Currency, Balance_Currency, Opening_Balance, Received_Threshold=0.0,
                   Posted_Threshold=0.0, Minimum_Received=100000.0, Minimum_Posted=100000.0, Liquidation_Period=10.0):
    """
    Loads CSA details on the root netting set in the given context
    """
    cfg.deals['Deals']['Children'][0]['Instrument'].field.update(
        {'Agreement_Currency': Agreement_Currency, 'Opening_Balance': Opening_Balance,
         'Apply_Closeout_When_Uncollateralized': 'No', 'Collateralized': 'True', 'Settlement_Period': 0.0,
         'Balance_Currency': Balance_Currency, 'Liquidation_Period': Liquidation_Period,
         'Credit_Support_Amounts':
             {'Received_Threshold': utils.CreditSupportList({1: Received_Threshold}),
              'Minimum_Received': utils.CreditSupportList({1: Minimum_Received}),
              'Posted_Threshold': utils.CreditSupportList({1: Posted_Threshold}),
              'Minimum_Posted': utils.CreditSupportList({1: Minimum_Posted})
              }
         }
    )


def load_market_data(rundate, path, json_name='MarketData.json', calendar_name='calendars.cal'):
    """
    Loads a json marketdata file and corresponding calendar (assumed to be named 'calendars.cal')
    :param rundate: folder inside path where the marketdata file resides
    :param path: root folder for the marketdata, calendar and trades
    :param json_name: name of the marketdata json file (default MarketData.json)
    :return: a context object with the data and calendars loaded
    """

    config = Config()
    config.parse_json(os.path.join(path, rundate, json_name))
    if os.path.isfile(os.path.join(path, 'calendars.cal')):
        config.parse_calendar_file(os.path.join(path, calendar_name))
    else:
        logging.warning('Calendar file {} not loaded'.format(calendar_name))

    config.params['System Parameters']['Base_Date'] = pd.Timestamp(rundate)

    return config


def solve_deal_field(document, deal_path, field, target=0.0, bounds=None, tolerance=0.01,
                     max_iterations=30):
    """Solve ONE field of a deal so the deal's own value lands on `target` - par forwards
    (target 0), sales margins, a collar's second strike, a strike to a premium.

    Each iterate rewrites the field in a copy of the wire document, loads and runs it, and reads
    the deal's own row off the `mtm` frame. The document's seed is fixed, so a Monte-Carlo-priced
    objective is DETERMINISTIC and the solved field is conditional on that draw. A deal field is
    structural, so every iterate recompiles.

    With `bounds`, bracketed brentq; without, a secant from the field's current value - exact in
    two evaluations for anything affine in the field. Raises when the residual cannot reach
    `tolerance`, naming it.

    Returns `(solved_value, evaluations, residual, out)`, `out` being the full output of the run
    AT the solved value.
    """
    from scipy import optimize

    reference = deal_at(document, deal_path)['Instrument']['.Deal'].get('Reference')
    priced = {}

    def value(x):
        x = float(x)
        if x not in priced:
            iterate = json.loads(json.dumps(document))
            deal_at(iterate, deal_path)['Instrument']['.Deal'][field] = x
            _, out = Context().load_json((json.dumps(iterate), 'solve')).run_job()
            frame = out['Results']['mtm']
            priced[x] = (float(frame[frame['Reference'] == reference]['Value'].iloc[0]), out)
        return priced[x][0]

    if bounds is not None:
        solved = float(optimize.brentq(lambda x: value(x) - target, bounds[0], bounds[1],
                                       maxiter=max_iterations))
    else:
        start = deal_at(document, deal_path)['Instrument']['.Deal'].get(field)
        x0 = float(start) if isinstance(start, (int, float)) and start else 1.0
        solved = float(optimize.newton(lambda x: value(x) - target, x0, x1=x0 * 1.0001 + 1.0,
                                       maxiter=max_iterations))

    residual = value(solved) - target
    if abs(residual) > tolerance:
        raise ValueError('solve for {} stopped {:.6g} from the target after {} pricings'.format(
            field, residual, len(priced)))
    return solved, len(priced), residual, priced[solved][1]


def run_baseval(context, prec=torch.float64, overrides=None):
    """
    Runs a base valuation calculation on the provided context
    :param context: a Context object
    :param prec: the numerical precision to use (default float64)
    :param overrides: a dictionary of overrides merged over the context's calculation parameters
    :return: a tuple containing the calculation object and the output dictionary
    """
    from .calculation import construct_calculation
    calc_params = context.deals.get(
        'Calculation',
        {'Base_Date': context.params['System Parameters']['Base_Date'],
         'Currency': context.params['System Parameters']['Base_Currency']})

    if torch.cuda.is_available():
        # CUBLAS_WORKSPACE_CONFIG is what makes cuBLAS reductions deterministic run to run
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")

    rundate = calc_params['Base_Date'].strftime('%Y-%m-%d')
    # only the runtime-derived key is injected; every declared default comes from the schema
    params_bv = {'Run_Date': rundate}
    params_bv.update(calc_params)

    if overrides is not None:
        update_dict(params_bv, overrides)

    calc = construct_calculation('Base_Revaluation', context, device=device, prec=prec)
    out = calc.execute(params_bv)
    return calc, out


def run_hedgemontecarlo(context, prec=torch.float32, overrides=None):
    from .calculation import construct_calculation, HedgeMonteCarlo
    from .schema import declared_defaults

    calc_params = context.deals.get(
        'Calculation',
        {'Base_Date': context.params['System Parameters']['Base_Date'],
         'Currency': context.params['System Parameters']['Base_Currency']})

    if torch.cuda.is_available():
        # CUBLAS_WORKSPACE_CONFIG is what makes cuBLAS reductions deterministic run to run
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")

    rundate = calc_params['Base_Date'].strftime('%Y-%m-%d')
    time_grid = str(declared_defaults(HedgeMonteCarlo, calc_params)['Time_Grid'])

    # only the runtime-derived keys are injected; every declared default comes from the schema
    params_mc = {'Time_grid': time_grid, 'Run_Date': rundate}

    params_mc.update(calc_params)

    if overrides is not None:
        update_dict(params_mc, overrides)

    calc = construct_calculation('HedgeMonteCarlo', context, device=device, prec=prec)
    out = calc.execute(params_mc)

    return calc, out


def run_cmc(context, prec=torch.float32, overrides=None, job_id=0, num_jobs=1, res_queue=None):
    """
    Runs a credit monte carlo calculation on the provided context
    :param res_queue: If not None, returns the results in this queue
    :param num_jobs: number of jobs spawned - usually just 1 (i.e. the parent)
    :param job_id: used if multiprocessing is set (index of this process in a group of workers)
    :param context: a Context object
    :param overrides: a dictionary of overrides merged over the context's calculation parameters
    :param prec: the numerical precision to use (default float32)
    :return: a tuple containing the calculation object, output dictionary and exposure profile
    """
    from .calculation import construct_calculation, Credit_Monte_Carlo
    from .schema import declared_defaults
    calc_params = context.deals.get(
        'Calculation',
        {'Base_Date': context.params['System Parameters']['Base_Date'],
         'Currency': context.params['System Parameters']['Base_Currency']})

    if torch.cuda.is_available():
        # CUBLAS_WORKSPACE_CONFIG is what makes cuBLAS reductions deterministic run to run
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        device = torch.device("cuda", job_id)
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")

    rundate = calc_params['Base_Date'].strftime('%Y-%m-%d')
    time_grid = str(declared_defaults(Credit_Monte_Carlo, calc_params)['Time_Grid'])

    # only the runtime-derived keys are injected; every declared default comes from the schema
    params_mc = {'Time_grid': time_grid, 'Run_Date': rundate}

    params_mc.update(calc_params)

    if overrides is not None:
        update_dict(params_mc, overrides)

    if params_mc.get('Credit_Valuation_Adjustment', {}).get('Calculate', 'No') == 'Yes':
        cva_sect = params_mc['Credit_Valuation_Adjustment']
        if cva_sect.get('CDS_Tenors'):
            # add extra tenors to the survival probability curve and interpolate it to calculate CDS rates
            survivalprob = context.params['Price Factors']['SurvivalProb.{}'.format(cva_sect['Counterparty'])]
            daycount = lambda time_in_days: utils.get_day_count_accrual(
                params_mc['Base_Date'], time_in_days, utils.DAYCOUNT_ACT365)
            to_add = [daycount((x - params_mc['Base_Date']).days) for x in
                      utils.cds_dates(params_mc['Base_Date'], max(cva_sect.get('CDS_Tenors')) * 12)]
            new_terms = np.union1d(to_add, survivalprob['Curve'].array[:, 0])
            survivalprob['Curve'].array = np.array(
                list(zip(new_terms, np.interp(new_terms, *survivalprob['Curve'].array.T))))

    if params_mc.get('Collateral_Valuation_Adjustment', {}).get('Calculate', 'No') == 'Yes':
        ns = context.deals['Deals']['Children'][0]['Instrument'].field
        collva_sect = context.deals['Calculation']['Collateral_Valuation_Adjustment']
        calc_currency = ns.get('Balance_Currency', 'ZAR')

        if 'Cash_Collateral' not in ns.get('Collateral_Assets', {}):
            collateral_curve = collva_sect['Collateral_Curve']
            funding_curve = collva_sect['Funding_Curve']

            if collva_sect['Collateral_Spread']:
                collateral_curve = '{}.COLLATERAL'.format(collateral_curve)
                context.params['Price Factors']['InterestRate.{}'.format(collateral_curve)] = makeflatcurve(
                    calc_currency, collva_sect['Collateral_Spread'])

            if collva_sect['Funding_Spread']:
                funding_curve = '{}.FUNDING'.format(funding_curve)
                context.params['Price Factors']['InterestRate.{}'.format(funding_curve)] = makeflatcurve(
                    calc_currency, collva_sect['Funding_Spread'])

            ns['Collateral_Assets'] = {
                'Cash_Collateral': [{
                    'Currency': calc_currency,
                    'Collateral_Rate': collateral_curve,
                    'Funding_Rate': funding_curve,
                    'Haircut_Posted': 0.0,
                    'Amount': 1.0}]}

        elif len(ns['Collateral_Assets']['Cash_Collateral']) > 1:
            # make sure we just take the first definition
            ns['Collateral_Assets']['Cash_Collateral'] = [ns['Collateral_Assets']['Cash_Collateral'][0]]

    calc = construct_calculation('Credit_Monte_Carlo', context, device=device, prec=prec)
    out = calc.execute(params_mc, job_id, num_jobs)
    out['Results']['exposure_profile'] = summarize_data(
        out['Results']['mtm'], params_mc.get('Percentile', '95').replace(' ', ''))

    if res_queue is not None:
        # parent process must summarize data
        res_queue.put({'Results': out['Results'], 'Stats': out['Stats'], 'Params':params_mc, 'Reference':context.deals['Attributes']['Reference']})

    return calc, out


def quote_delta(name, points, values):
    """One `Market Prices` block's patched value rows: the delta merged onto what each row carries.

    `Points` is the one key a quote patch may name - everything else on a block is plan - and the
    row list is exactly as long as the block's, because dropping or adding a quote re-authors the
    instrument set rather than moving a number. Per row: a named field replaces, an omitted one
    keeps, and `null` clears every value key outside `schema.MARKET_QUOTE_REQUIRED`; a null
    `Quoted_Market_Value` refuses, because a mid is moved. WHICH keys may be cleared is a
    declaration, not a name spelled here.
    """
    for field in values:
        if field != 'Points':
            raise ValueError('{}: {} is structural, not a value'.format(name, field))
    patched = values.get('Points', [])
    if len(patched) != len(points):
        raise ValueError(
            '{}: the patch carries {} Points row(s) against the block\'s {} - a changed row count '
            'is a new plan; re-author the block deliberately'.format(
                name, len(patched), len(points)))
    rows = []
    for point, row in zip(points, patched):
        # key-presence, not content: a null the patch does not name is the row's own current
        # content and keeps, which is what "an omitted field keeps" says about it
        current = {key: point[key] for key in schema.MARKET_QUOTE_VALUES if key in point}
        for field, content in row.items():
            if field not in schema.MARKET_QUOTE_VALUES:
                raise ValueError('{}: {} is structural, not a value'.format(name, field))
            if content is None:
                if field in schema.MARKET_QUOTE_REQUIRED:
                    raise ValueError(
                        '{}: {} cannot be cleared - a mid is moved, never removed; a two-way '
                        'side or a Timestamp is what null clears'.format(name, field))
                current.pop(field, None)
            else:
                current[field] = content
        rows.append(current)
    return rows


class Context:
    def __init__(self, path_transform={}, file_transform={}):
        # needed if the json file contains paths to window's files but linux is needed
        self.path_map = path_transform
        # needed if the name of the file referenced needs to be changed (e.g. from .dat to .json)
        self.file_map = file_transform
        self.config_cache = {}
        self.holiday_cfg_cache = {}
        self.current_cfg = Config()
        # there may be a stressed config file attached to the context
        self.stressed_config_file = None

    def load_config(self, path_name):
        new_cfg = Config()
        new_cfg.parse_json(path_name)

        # check we need to set the base_date
        if new_cfg.params['System Parameters'].get('Base_Date') is None:
            # today as a DATE: a wall-clock default is a nondeterministic plan input - two loads
            # of one job must hash the same, and Base_Date is a date everywhere it is read
            new_cfg.params['System Parameters']['Base_Date'] = pd.Timestamp.now().normalize()

        new_cfg.version = ['JSONVersion', '22.05.30']
        return new_cfg

    def parse_path(self, file_path):
        # file_path is assumed to be a window's path - so we need to check if we're in a posix world
        if os.name == 'posix':
            file_path = pathlib.PureWindowsPath(file_path).as_posix()

        path, filename = os.path.split(file_path)
        return os.path.join(self.path_map.get(path, path), self.file_map.get(filename, filename))

    def load_json(self, jobfilename, compress=True):

        def resolve_deferred_deals(obj, val_config):
            if isinstance(obj, utils.DeferredDeal):
                return construct_instrument(obj.payload, val_config)

            if isinstance(obj, dict):
                return {k: resolve_deferred_deals(v, val_config) for k, v in obj.items()}

            if isinstance(obj, list):
                return [resolve_deferred_deals(v, val_config) for v in obj]

            return obj

        cfg = self.current_cfg
        # read the raw json data (skips deals) - needs to load later
        data = self.current_cfg.read_json(jobfilename)

        if 'MergeMarketData' in data['Calc']:
            market_data = data['Calc']['MergeMarketData']
            # check if there's a stressed marketdata defined (record but don't load it)
            self.stressed_config_file = market_data.get('StressedMarketDataFile')

            if market_data.get('MarketDataFile'):
                if market_data['MarketDataFile'] not in self.config_cache:
                    new_cfg = self.load_config(self.parse_path(market_data['MarketDataFile']))
                    self.config_cache[market_data['MarketDataFile']] = new_cfg

                cfg = self.config_cache[market_data['MarketDataFile']]
                for section, section_data in market_data['ExplicitMarketData'].items():
                    cfg.params[section].update(section_data)
            else:
                cfg = Config()
                for section, section_data in market_data.get('ExplicitMarketData', {}).items():
                    cfg.params[section].update(section_data)

        if data['Calc'].get('CalendDataFile'):
            if data['Calc']['CalendDataFile'] not in self.holiday_cfg_cache:
                cfg.parse_calendar_file(self.parse_path(data['Calc']['CalendDataFile']))
                self.holiday_cfg_cache[data['Calc']['CalendDataFile']] = cfg.holidays
            cfg.holidays = self.holiday_cfg_cache[data['Calc']['CalendDataFile']]

        if 'Deals' in data['Calc']:
            cfg.deals = {'Attributes': {
                'Tag_Titles': data['Calc']['Deals'].get('Tag_Titles', ''),
                'Reference': data['Calc']['Deals'].get('Reference')}}
            valuation_config = cfg.params.get('Valuation Configuration', {})
            deals = resolve_deferred_deals ( data['Calc']['Deals']['Deals'], valuation_config )
            # TODO: the compression only reaches one level of nesting
            if compress:
                for i in deals['Children']:
                    if 'Children' in i:
                        i['Children'] = utils.compress_deal_data(i['Children'])
        else :
            # a job that carries no deal tree (a hedging problem builds its own at execute time)
            # still loads an empty BOOK, which is the shape every walk of it reads
            deals = {'Children': []}

        cfg.deals.update({'Deals': deals})
        cfg.deals.update({'Calculation': data['Calc']['Calculation']})

        self.current_cfg = cfg
        return self

    def save_json(self, json_filename):
        '''
        Writes the loaded job back out as a job JSON - experimental, not fully implemented
        :param json_filename: destination path, or None to return the JSON string instead
        :return: None when a filename is given, otherwise the JSON string
        '''

        def write_final_json(out_json, cfg, section):
            if cfg.params[section]:
                out_json[section] = cfg.params[section]

        cfg = self.current_cfg
        try:
            md, cal = list(self.config_cache.keys())[0], list(self.holiday_cfg_cache.keys())[0]
        except:
            md, cal = '', ''

        final_json = {
            "Calc":
                {
                    "Calculation": cfg.deals['Calculation'],
                    "Deals": {
                        "Tag_Titles": cfg.deals['Attributes'].get('Tag_Titles', ''),
                        "Reference": cfg.deals['Attributes'].get('Reference', ''),
                        "Deals": cfg.deals['Deals']
                        },
                    "MergeMarketData": {
                        "MarketDataFile": md,
                        "ExplicitMarketData": {
                        }
                    },
                    "CalendDataFile": cal
                }
            }

        out_json = final_json['Calc']['MergeMarketData']['ExplicitMarketData']
        if md:
            # only write out the price factors if the market data is defined
            write_final_json(out_json, cfg, 'Price Factors')
        else:
            # write out everything
            write_final_json(out_json, cfg, 'System Parameters')
            write_final_json(out_json, cfg, 'Model Configuration')
            write_final_json(out_json, cfg, 'Price Factors')
            write_final_json(out_json, cfg, 'Price Models')
            write_final_json(out_json, cfg, 'Correlations')

        data = json.dumps(final_json, separators=(',', ':'), cls=CustomJsonEncoder)
        if json_filename is not None:
            with open(json_filename, 'wt') as f:
                f.write(data)
        else:
            return data

    def validate(self):
        return self.current_cfg.validate()

    def describe(self):
        return self.current_cfg.describe()

    def bootstrap(self):
        """Run every configured bootstrapper - quotes in `Market Prices` become the price factors
        and model parameters the pricers read. Delegates to the loaded config, so the harvested
        calibration tensors (`Quote_Sensitivity`) land where `_build_factor_state` reads them."""
        return self.current_cfg.bootstrap()

    def market_patch(self):
        """The VALUES half of the market data: `{name: {field: content}}` over both market sections.

        Everything a job may change without recompiling - spots, the rate column of every curve,
        the vol column of every surface, the calibrated model parameters, and every quote a
        `Market Prices` block carries on a `Points` row. What sizes a tenor grid, wires a process or
        picks a code path is plan and does not appear, nor does a quote key holding `null`, which is
        a source with no print rather than a number.

        The two sections cannot collide in one dict: every family type string ends in `Prices` and
        no factor type does. A market-price entry is keyed `{'Points': [row, ...]}`, the shape
        `patch_market` takes it back in.
        """
        patch = {}
        for name, block in self.current_cfg.params['Price Factors'].items():
            values = schema.partition_factor(utils.check_rate_name(name)[0], block)[1]
            if values:
                patch[name] = values
        for name, block in self.current_cfg.params.get('Market Prices', {}).items():
            values = schema.partition_market_price(block)[1]
            if values:
                patch[name] = {'Points': values}
        return patch

    def patch_market(self, patch):
        """Apply a values patch in place, and fail loud on anything else.

        A patch is a DELTA: a field it names is replaced, a value field it omits keeps its current
        content, so `{"FxRate.EUR": {"Spot": 1.24}}` is a complete patch. A key naming no
        value-bound field of that factor raises - including a field the block does not carry, since
        a key set that grows or shrinks is a different plan, not a different value.

        A name resolves against `Price Factors` first and `Market Prices` second, and one in neither
        section says so naming both. A market-price name takes exactly one key, `Points`, carrying a
        list as long as the block's own and only `schema.MARKET_QUOTE_VALUES` per row; `null` clears
        a two-way side or a `Timestamp`, and a null mid refuses.

        A quote patch does NOT re-bootstrap: the price factors the last bootstrap wrote stand as
        they are, and `values_hash` records the board that is actually standing.
        """
        factors = self.current_cfg.params['Price Factors']
        prices = self.current_cfg.params['Market Prices']
        for name, values in patch.items():
            if name in factors:
                type_name = utils.check_rate_name(name)[0]
                structural, value_fields = schema.partition_factor(type_name, factors[name])
                for field in values:
                    if field not in value_fields:
                        raise ValueError(
                            '{}: {} is structural, not a value'.format(name, field))
                factors[name] = schema.apply_values(
                    type_name, structural, {**value_fields, **values})
            elif name in prices:
                prices[name] = schema.apply_market_values(
                    schema.partition_market_price(prices[name])[0],
                    quote_delta(name, prices[name]['instrument'].get('Points') or [], values))
            else:
                raise KeyError(
                    '{} is neither a price factor nor a market price'.format(name))

    def plan_hash(self):
        """The content hash of the PROGRAM: everything a run compiles, market values excluded.

        A plan is `params` and `deals` less the THREE things that are replay coordinates of their
        own - every `bind='value'` field of a price factor, `schema.MARKET_QUOTE_VALUES` on a
        `Market Prices` `Points` row, both of which `values_hash` carries, and `Random_Seed`.
        Everything else is in, `Batch_Size` and `Simulation_Batches` included: they change the
        realized numbers, so a replay has to pin them. A vol tick therefore moves `values_hash` and
        leaves this bit-identical; a moved pillar, a flipped `Use` or a re-authored benchmark lands
        here, because that is a different program rather than a different number.

        `params['Correlations']` is the SIMULATION matrix feeding the cholesky - a compile input, and
        a different thing entirely from the `Correlation` price factor a quanto reads. It is re-keyed
        off its name PAIR here only because JSON has no tuple key.
        """
        cfg = self.current_cfg
        params = dict(cfg.params)
        params['Price Factors'] = {
            name: schema.partition_factor(utils.check_rate_name(name)[0], block)[0]
            for name, block in cfg.params['Price Factors'].items()}
        params['Market Prices'] = {
            name: schema.partition_market_price(block)[0]
            for name, block in cfg.params.get('Market Prices', {}).items()}
        params['Correlations'] = {'{}/{}'.format(*pair): value
                                  for pair, value in cfg.params['Correlations'].items()}
        calculation = {k: v for k, v in cfg.deals['Calculation'].items() if k != 'Random_Seed'}
        return content_hash({'params': params, 'deals': dict(cfg.deals, Calculation=calculation)})

    def values_hash(self):
        """The content hash of `market_patch()` - the market VALUES, and nothing else.

        With `plan_hash`, `__version__` and the seed this is the replay identity: two runs agreeing
        on all four report the same numbers.
        """
        return content_hash(self.market_patch())

    # The book-of-record verbs. Each hands PLAIN DATA to `derivus_spine`, reached lazily inside
    # `derivus.spine` so nothing under `derivus/` depends on the extra; a missing extra or an unset
    # `DV_SPINE_HOME` refuses by name.

    def book(self, deal, quantity, counterparty, netting_set, execution_reference,
             actor=None, book=None, effective_time=None):
        """Book a fill: the canonical instrument registered, the SIGNED quantity recorded.

        The instrument id is the content hash of the deal's canonical JSON, so booking the same
        strike twice files two events against one instrument. `execution_reference` has no default
        anywhere on this path: it is what makes a retry the same fact and two identical clips two.
        """
        from .spine import book as record

        return record(deal, quantity, counterparty, netting_set, execution_reference,
                      actor_name=actor, book_name=book, effective_time=effective_time)

    def amend(self, deal, amended_to, actor=None, book=None, effective_time=None):
        """Record that these terms became those terms - a NEW instrument hash linked to the old one.
        Economics are never edited; an amendment is a second row, never a changed one."""
        from .spine import amend as record

        return record(deal, amended_to, actor_name=actor, book_name=book,
                      effective_time=effective_time)

    def apply_lifecycle(self, event_type, body, actor=None, book=None, effective_time=None):
        """File an election, a fixing observation or a determination - and nothing else.

        A knock, an expiry or an accrual is a CONSEQUENCE of terms plus one of those three, so it is
        a projection and this verb refuses it rather than storing a second source of truth.
        """
        from .spine import apply_lifecycle as record

        return record(event_type, body, actor_name=actor, book_name=book,
                      effective_time=effective_time)

    def declare_market(self, name, actor=None, effective_time=None):
        """Point a market NAME at the values vector THIS context is carrying.

        Officialness is a property of the name rather than of the data: every values vector lives
        identically in the store, and `official` moves onto one only by declaration from a
        `mark`-scoped actor, which the writer enforces.
        """
        from .spine import declare_market as record, values_of

        return record(name, values_of(self), actor_name=actor, effective_time=effective_time)

    def pin_result(self, job, values, result, claim, actor=None, book=None, effective_time=None):
        """Promote a replay claim this box did not witness: re-execute it, or find it already
        attested, and only then record it.

        The executor is built from THIS context's own class, so a subclass that prices differently
        pins through the pricing it actually does. A version mismatch refuses by name, and bytes
        that will not reproduce within the declared tolerance policy are refused, not attested.
        """
        from .spine import executor, pin_result as record

        return record(claim, job, values, result, actor_name=actor, book_name=book,
                      effective_time=effective_time, execute=executor(type(self)))

    def run_job(self, overrides=None, runparallel=False):
        # check what calc we should run
        if self.current_cfg.deals['Calculation']['Object'] == 'CreditMonteCarlo':
            return self.Credit_Monte_Carlo(overrides, runparallel)
        elif self.current_cfg.deals['Calculation']['Object'] == 'BaseValuation':
            return self.Base_Valuation(overrides)
        elif self.current_cfg.deals['Calculation']['Object'] == 'HedgeMonteCarlo':
            return self.Hedge_Monte_Carlo(overrides)
        else:
            raise Exception('Unknown Calculation {}'.format(self.current_cfg.deals['Calculation']['Object']))

    def Credit_Monte_Carlo(self, overrides=None, runparallel=False):
        if runparallel:
            num_workers = torch.cuda.device_count()
            results = mp.Queue()
            workers = [mp.Process(target=run_cmc, args=(
                self.current_cfg, torch.float32, overrides, False, i, num_workers, results))
                            for i in range(num_workers)]

            for w in workers:
                w.start()

            post_processing = []
            for i in range(num_workers):
                post_processing.append(results.get())

            results.close()

            for w in workers:
                w.join()
                if w.is_alive():
                    w.close()

            post_results = {}
            for output in post_processing:
                data = dict(output)
                for k, v in data.items():
                    post_results.setdefault(k, []).append(v)
            return post_results
        else:
            return run_cmc(self.current_cfg, overrides=overrides)

    def Base_Valuation(self, overrides=None):
        return run_baseval(self.current_cfg, overrides=overrides)

    def Hedge_Monte_Carlo(self, overrides=None):
        return run_hedgemontecarlo(self.current_cfg, overrides=overrides)


class StressedContext(Context):

    def __init__(self, path_transform={}, file_transform={}):
        super(StressedContext, self).__init__(path_transform, file_transform)
        self.stressed_cfg = None
        self.current_models = None
        self.current_factors = None

    def restore_config(self):
        if self.current_models is not None:
            self.current_cfg.params['Price Models'].update(self.current_models)
        if self.current_factors is not None:
            self.current_cfg.params['Price Factors'].update(self.current_factors)

    def stress_config(self, rate_group):
        factors_to_override, models_to_override = self.calc_stress_config(rate_group)
        self.current_factors = {k: self.current_cfg.params['Price Factors'][k] for k in factors_to_override.keys()}
        # .get: an implied model may be overridden without carrying an entry of its own
        self.current_models = {k: self.current_cfg.params['Price Models'].get(k) for k in models_to_override.keys()}
        self.current_cfg.params['Price Factors'].update(factors_to_override)
        self.current_cfg.params['Price Models'].update(models_to_override)

    def calc_stress_config(self, rate_group):
        '''
        Loads the stressed market data file and returns what it has to say about the rate group
        :param rate_group: Rate group (InterestRate, FxRate etc.)
        :return: a tuple of the price factors and the price models to override
        '''
        if self.stressed_config_file not in self.config_cache:
            self.stressed_cfg = self.load_config(self.parse_path(self.stressed_config_file))
            self.config_cache[self.stressed_config_file] = self.stressed_cfg

        self.stressed_cfg = self.config_cache[self.stressed_config_file]

        factors_to_override = {}
        models_to_override = {}

        for factor_type in rate_group:
            factors = [utils.Factor(factor_type, utils.check_rate_name(i)[1:])
                       for i in self.current_cfg.params['Price Factors'].keys() if i.startswith(factor_type)]
            factor_models, additional_factors = self.current_cfg.find_models(factors)
            for factor in [utils.check_tuple_name(x) for x in additional_factors.values()]:
                try:
                    factors_to_override[factor] = self.stressed_cfg.params['Price Factors'][factor]
                except KeyError as k:
                    logging.warning(
                        "Skipping Stressed Price Factor {} - not present in stressed file".format(k))
            for factor_model in [utils.check_tuple_name(x) for x in factor_models.keys()]:
                try:
                    models_to_override[factor_model] = self.stressed_cfg.params['Price Models'][factor_model]
                except KeyError as k:
                    logging.warning(
                        "Skipping Stressed Price Model {} - not present in stressed file".format(k))

        return factors_to_override, models_to_override
