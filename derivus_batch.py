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

import os
import sys
import time
import json
import glob
import traceback
import pandas as pd
import numpy as np

from collections import defaultdict
from multiprocessing import Process, Queue, Lock
from xml.etree.ElementTree import SubElement, Element, tostring


def parse_output_results(data):
    """A Credit Monte Carlo exposure profile as the legacy `<Results><Series>` XML an .aaj file
    carries: the peak per series as its value, Excel-offset day numbers on X."""
    from derivus.utils import excel_offset

    results = Element('Results')
    header = SubElement(results, 'Header')
    calc_type = SubElement(header, 'CalcType')
    calc_type.text = 'Credit Monte Carlo'
    series = SubElement(results, 'Series')

    for index, (column, items) in enumerate(data.T.iterrows()):
        if 'PFE' in column:
            name = 'Bank Exposure ({}%)'.format(column[4:])
        elif column == 'EE':
            name = 'Bank Expected Exposure'
        else:
            continue
        value = items.max()
        series_item = SubElement(
            series, 'SeriesItem', Name=name, Index=str(index), InitialState='On',
            SeriesType='XYPlot', LineStyle='Solid', Value='{:.0f}'.format(value),
            Description='Maximum value {:,.3f}'.format(value))
        x_values = [str((x - excel_offset).days) for x in items.index]
        y_values = ['{:.2f}'.format(x) for x in items.values]
        x = SubElement(series_item, 'X')
        x.text = ','.join(x_values)
        y = SubElement(series_item, 'Y')
        y.text = ','.join(y_values)

    return tostring(results, encoding='utf-8').decode('utf-8')


def rename_factor(cx, old_name, new_name):
    for old_local_df in [x for x in cx.current_cfg.params['Price Models'] if x.endswith(old_name)]:
        new_local_df = old_local_df.replace(old_name, new_name)
        cx.current_cfg.params['Price Models'][new_local_df] = cx.current_cfg.params['Price Models'][old_local_df]

    for old_local_param in [x for x in cx.current_cfg.params['Price Factors'] if x.endswith(
            'Parameters.{}'.format(old_name))]:
        new_local_param = old_local_param.replace(old_name, new_name)
        cx.current_cfg.params['Price Factors'][new_local_param] = cx.current_cfg.params['Price Factors'][
            old_local_param]


def merge_crif(original, crif, currency, valued=None):
    """`original` with every row of each `valued` trade - each trade `crif` carries unless named -
    replaced by `crif`'s rows, both read as text, in the original's column layout: the columns the
    run writes are its own, its vertex spelled as the original spells it, and the rest the trade's
    own original terms. Amount is in the trade's AmountCurrency, the run's own where that is the
    calculation `currency` and its AmountUSD at the rate the original's two amounts state where it
    is not. A trade the original does not carry is left out."""
    terms = original.drop(columns=[column for column in crif.columns if column != 'TradeID']
                          ).groupby('TradeID', sort=False).first()
    amounts = original[['Amount', 'AmountUSD']].astype(float)
    rates = (amounts['Amount'] / amounts['AmountUSD'])[amounts['AmountUSD'] != 0.0].groupby(
        original['AmountCurrency']).median()
    spelled = {label.lower(): label for label in original['Label1']}
    valued = terms.index.intersection(crif['TradeID'] if valued is None else valued)
    mine = crif[crif['TradeID'].isin(valued)]
    usd = mine['AmountUSD'].astype(float)
    held = terms['AmountCurrency'].reindex(mine['TradeID']).to_numpy()
    amount = np.where(held == currency, mine['Amount'].astype(float), usd * pd.Series(held).map(
        rates).to_numpy())
    rows = mine.assign(
        Label1=mine['Label1'].map(lambda label: spelled.get(label.lower(), label)),
        Amount=[repr(float(value)) for value in amount],
        AmountUSD=[repr(float(value)) for value in usd]).join(terms, on='TradeID')
    order = {trade: index for index, trade in enumerate(terms.index)}
    merged = pd.concat([original[~original['TradeID'].isin(valued)], rows])
    return merged.sort_values('TradeID', key=lambda trades: trades.map(order), kind='stable')[
        list(original.columns)]


def quote_curves(config, base_date):
    """Par quotes on each curve's own knots for every curve `config`'s book reads that no block
    quotes, under the curve family's declared conventions and configured, so the bootstrap returns
    the curve as it stands and a basis point moves a market rate."""
    from derivus import bootstrappers, utils

    prices = config.params['Market Prices']
    family = bootstrappers.InterestRate.market_factor_type
    for curve in sorted({utils.check_rate_name(name)[1]
                         for name in config.factor_universe()['resolved']
                         if utils.check_rate_name(name)[0] == 'InterestRate'}):
        if '{}.{}'.format(family, curve) not in prices:
            name, block = bootstrappers.knot_prices(
                curve, config.params['Price Factors'],
                config.params['Price Factor Interpolation'], base_date, config.holidays)
            prices[name] = block
            config.params['Bootstrapper Configuration'].setdefault(
                'InterestRate', {'Prices': 'InterestRate'})


class JOB(object):
    #: whether a netting set's deals are compressed on load
    compress = True
    #: the processes a run's paths are sharded over, set by the Parent
    shards = 1

    def __init__(self, cx, rundate, input_path, outputdir, netting_set, stats, log):
        self.cx = cx
        self.rundate = rundate
        self.input_path = input_path
        self.outputdir = outputdir
        self.netting_set = netting_set
        self.stats = stats
        self.logger = log
        self.splits = None
        self.params = {'Calc_Scenarios': 'No',
                       'Generate_Cashflows': 'No'}

        # load trades (and marketdata)
        self.cx.load_json(os.path.join(self.input_path, self.rundate, self.netting_set), self.compress)

        # get the netting set
        self.ns = self.cx.current_cfg.deals['Deals']['Children'][0]['Instrument']
        # get the agreement currency
        self.agreement_currency = self.ns.field.get('Agreement_Currency', 'ZAR')
        # get the balance currency
        self.balance_currency = self.ns.field.get('Balance_Currency', self.agreement_currency)

    def perform_calc(self):
        # create the calculation    
        if self.valid():
            try:
                self.run_calc()
            except Exception as e:
                self.logger(self.netting_set, "!! CRITICAL ERROR In Calc !! - {} - Skipping".format(e.args))

    def valid(self):
        return False

    def run_calc(self):
        calc, out = self.calculate(self.params)
        self.stats.setdefault('Stats', {})[self.netting_set] = out['Stats']

    def calculate(self, params):
        """`cx.run_job`, its answer pooled to the one `out` of an unsharded run under `shards`."""
        if self.shards == 1:
            return self.cx.run_job(params)
        return None, pooled(self.cx.run_job(params, runparallel=self.shards))

    @classmethod
    def collate(cls, outputdir, rundate, stats):
        """What the parent writes off every netting set's stats beyond their total: nothing."""


class FVA_GRAD(JOB):
    '''Bare CRB job'''
    def valid(self):
        return True

    def run_calc(self):
        self.params.update({'Funding_Valuation_Adjustment': {'Gradient': 'Yes'}})
        calc, out = self.calculate(self.params)
        self.stats.setdefault('Stats', {})[self.netting_set] = out['Stats']
    

class PFE(JOB):
    def __init__(self, cx, rundate, input_path, outputdir, netting_set, stats, log):
        super(PFE, self).__init__(cx, rundate, input_path, outputdir, netting_set, stats, log)
        # load up a calendar (for theta)
        self.business_day = self.cx.current_cfg.holidays['Johannesburg']['businessday']
        # set the OIS cashflow flag to speed up prime linked swaps
        self.cx.current_cfg.params[
            'Valuation Configuration']['CFFloatingInterestListDeal']['OIS_Cashflow_Group_Size'] = 1

    def valid(self):
        if not self.cx.current_cfg.deals['Deals']['Children'][0]['Children']:
            return False
        else:
            return True

    def run_calc(self):
        self.params['Run_Date'] = self.rundate
        filename = 'PFE_{}_{}.csv'.format(
            self.params['Run_Date'], self.cx.current_cfg.deals['Attributes']['Reference'])
        legacy_filename = 'OutputAAJ_{}.aaj'.format(self.cx.current_cfg.deals['Attributes']['Reference'])

        # load up CVA calc params
        if self.cx.current_cfg.deals['Deals']['Children'][0]['Instrument'].field.get('Collateralized') == 'True':
            self.logger(self.netting_set, 'is collateralized')
            # turn on dynamic scenarios (more accurate)
            self.params['Dynamic_Scenario_Dates'] = 'Yes'
            # do 20000 sims in batches of 1024
            self.params['Simulation_Batches'] = 10
            self.params['Batch_Size'] = 2048
        else:
            self.logger(self.netting_set, 'is uncollateralized')
            # do 20000 sims in batches of 1024
            self.params['Simulation_Batches'] = 5
            self.params['Batch_Size'] = 4096
            self.params['Dynamic_Scenario_Dates'] = 'No'

        # set the percentiles to 95 and 99
        self.params['Percentile'] = '95, 99'
        # do more MCMC simulations
        self.params['MCMC_Simulations'] = 4096

        calc, out = self.cx.run_job(overrides=self.params)
        profile = out['Results']['exposure_profile']
        legacy_exposure_profile = parse_output_results(profile)
        PFE_key = [x for x in profile.keys() if x.startswith('PFE')][0]
        out['Stats'].update({PFE_key: profile[PFE_key].max(), 'Currency': calc.params['Currency']})
        # set the currency in the profile
        profile.index.name = calc.params['Currency']
        # write the results
        profile.to_csv(os.path.join(self.outputdir, filename))
        if not os.path.isdir(os.path.join(self.outputdir, self.rundate)):
            os.mkdir(os.path.join(self.outputdir, self.rundate))
        with open(os.path.join(self.outputdir, self.rundate, legacy_filename), 'w') as f:
            f.write(legacy_exposure_profile)
        self.stats.setdefault('Stats', {})[self.netting_set] = out['Stats']
        
        
class PFEGross(PFE):

    def valid(self):
        if not self.cx.current_cfg.deals['Deals']['Children'][0]['Children']:
            return False
        else:
            if self.cx.current_cfg.deals['Deals']['Children'][0]['Instrument'].field.get('Collateralized') == 'True':
                self.cx.current_cfg.deals['Deals']['Children'][0]['Instrument'].field['Collateralized'] = 'False'
                return True
            else:
                return False


class CollateralBaseVal(JOB):
    def __init__(self, cx, rundate, input_path, outputdir, netting_set, stats, log):
        super(CollateralBaseVal, self).__init__(cx, rundate, input_path, outputdir, netting_set, stats, log)

        # get the collateral assets
        try:
            assets = self.ns.field['Collateral_Assets'].keys()[0]
            if assets is None and self.ns.field['Collateralized'] == 'True':
                assets = 'Cash'
        except:
            assets = 'Cash' if self.ns.field['Collateralized'] == 'True' else 'None'

        self.collateral = assets
        self.params = {'Run_Date': rundate}

    def valid(self):
        if not self.cx.current_cfg.deals['Deals']['Children'][0]['Children'] or self.collateral != 'Cash':
            return False
        else:
            return True

    def run_calc(self):
        import derivus as rf

        if 'Reference' not in self.cx.current_cfg.deals['Attributes']:
            self.cx.current_cfg.deals['Attributes']['Reference'] = os.path.splitext(self.netting_set)[0]
        ending = self.params['Run_Date'] + '_' + self.cx.current_cfg.deals['Attributes']['Reference'] + '.csv'
        filename = 'BaseVal_' + ending
        filename_greeks = 'BaseVal_Delta_' + ending
        filename_greeks_second = 'BaseVal_Gamma_' + ending

        self.params.update({'Currency': self.agreement_currency, 'Greeks': 'All'})

        try:
            try:
                _, out = self.cx.Base_Valuation(overrides=self.params)
            except rf.utils.SecondOrderRefused as refused:
                # only the SECOND order block is refused - the mark and the first order one are
                # unaffected - so this set gets them rather than nothing. The message names the
                # deals the refusal is over, which is what a book owner needs to act on.
                self.logger(self.netting_set, 'Warning: {} Falling back to Greeks: First for '
                                              'this netting set.'.format(refused))
                self.params['Greeks'] = 'First'
                _, out = self.cx.Base_Valuation(overrides=self.params)
            out['Results']['mtm'].to_csv(os.path.join(self.outputdir, 'Greeks', filename))
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)
            self.logger(self.netting_set, 'Failed to baseval')
        else:
            self.stats.setdefault('Stats', {})[self.netting_set] = out['Stats']
            out['Stats'].update({'Currency': self.params['Currency']})
            out['Stats'].update({'MTM': out['Results']['mtm']['Value'].head(1).values[0]})
            out['Results']['Greeks_First'].to_csv(os.path.join(self.outputdir, 'Greeks', filename_greeks))
            if 'Greeks_Second' in out['Results']:
                out['Results']['Greeks_Second'].to_csv(os.path.join(self.outputdir, 'Greeks', filename_greeks_second))


class CVA_GRAD(JOB):
    def __init__(self, cx, rundate, input_path, outputdir, netting_set, stats, log):
        super(CVA_GRAD, self).__init__(cx, rundate, input_path, outputdir, netting_set, stats, log)
        # load up a calendar (for theta)
        self.business_day = cx.current_cfg.holidays['Johannesburg']['businessday']
        # set the OIS cashflow flag to speed up prime linked swaps
        self.cx.current_cfg.params['Valuation Configuration']['CFFloatingInterestListDeal']['OIS_Cashflow_Group_Size'] = 1
        # change the currency
        self.params['Currency'] = self.cx.current_cfg.deals['Calculation']['Currency']
        # does this netting set have any combination deals?
        self.cominations = 0
        # rename some curves to better handle CSA discounting
        rename_factor(self.cx, 'ZAR-SWAP', 'ZAR-OIS')
        rename_factor(self.cx, 'ZAR-JIBAR-3M', 'ZAR-JIBAR-3M-OIS')

    def valid(self):
        if not self.cx.current_cfg.deals['Deals']['Children'][0]['Children']:
            return False
        else:  
            combo_deals = [x for x in self.cx.current_cfg.deals['Deals']['Children'][0]['Children'] if 'Combination' in x['Instrument'].field.get('Tags',[''])[0]]
            self.combinations = len(combo_deals)
            
            cpy = self.cx.current_cfg.deals['Calculation']['Credit_Valuation_Adjustment'].get('Counterparty')        
            if self.cx.current_cfg.params['Price Factors'].get('SurvivalProb.{}'.format(cpy) ,{}).get('Recovery_Rate')==1.0:
                #100% recovery - skip
                self.logger(self.netting_set, 'Netting set has 100% Recovery - skipping')
                return False
            else:
                return True
                

    def run_calc(self):
        import derivus as rf

        filename = 'CVA_' + self.rundate + '_' + self.params['Currency']+ '_' + self.cx.current_cfg.deals['Attributes']['Reference'] + '.csv'
         # make sure other xva is switched off
        if 'Funding_Valuation_Adjustment' in self.cx.current_cfg.deals['Calculation']:
            del self.cx.current_cfg.deals['Calculation']['Funding_Valuation_Adjustment']
        if 'Collateral_Valuation_Adjustment' in self.cx.current_cfg.deals['Calculation']:
            del self.cx.current_cfg.deals['Calculation']['Collateral_Valuation_Adjustment']
            
        # how many simulations do we want (in thousands)
        num_1ksims = 16
        if self.combinations:
            # use smaller batches if there are comination deals present
            self.logger(self.netting_set, 'Combination deals present - dropping batch size to 256')
            self.params['Simulation_Batches'] = 2 * num_1ksims
            self.params['Batch_Size'] = 512
        else:
            self.params['Simulation_Batches'] = num_1ksims//2
            self.params['Batch_Size'] = 1024*2
            
        # load up CVA calc params
        if self.cx.current_cfg.deals['Deals']['Children'][0]['Instrument'].field.get('Collateralized') == 'True':
            self.logger(self.netting_set, 'is collateralized')
            # turn on dynamic scenarios (more accurate)
            self.params['Dynamic_Scenario_Dates'] = 'Yes'
        else:
            self.logger(self.netting_set, 'is uncollateralized')
            self.params['Dynamic_Scenario_Dates'] = 'No'

        if os.path.isfile(os.path.join(self.outputdir, 'Greeks', filename)):
            self.logger(self.netting_set, 'Warning: skipping gradient CVA calc as file already exists')
            self.params['Credit_Valuation_Adjustment'] = {'Gradient': 'No'}
        else:
            self.params['Credit_Valuation_Adjustment'] = {'Gradient': 'Yes'}

        calc_complete = False
        num_tries = 0

        while not calc_complete:
            try:
                calc, out = self.calculate(self.params)
            except RuntimeError as e:  # Out of memory
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          limit=2, file=sys.stdout)
                self.params['Simulation_Batches'] *= 2
                self.params['Batch_Size'] //= 2
                self.logger(self.netting_set,
                            'Exception: Runtime Error - Halving to {} Batchsize'.format(self.params['Batch_Size']))
                num_tries += 1
                if num_tries>2:
                    self.logger(self.netting_set,
                            'Tried twice with last batchsize of {} Skipping'.format(self.params['Batch_Size']))                        
                    out = {'Stats': {}}
                    out['Stats'].update({'CVA': np.nan, 'Currency': self.params['Currency']})
                    calc_complete = True
            except KeyError as key:
                self.logger(self.netting_set, 'Exception: Key Error {} - skipping'.format(key.args))
                calc_complete = True
                out = {'Stats': {}}
                out['Stats'].update({'CVA': np.nan, 'Currency': self.params['Currency']})
            else:
                if 'grad_cva' in out['Results']:
                    grad_cva = out['Results']['grad_cva'].rename(
                        columns={'Gradient': self.cx.current_cfg.deals['Attributes']['Reference']})
                    grad_cva.to_csv(os.path.join(self.outputdir, 'Greeks', filename))
                    # store the CVA as part of the stats
                    out['Stats'].update({
                        'CVA': out['Results']['cva'],
                        'Currency': self.params['Currency']})                        
                    # log the netting set
                    self.logger(self.netting_set, 'CVA calc complete')
                else:
                    out['Stats'].update({
                        'CVA': out['Results']['cva'] if 'cva' in out['Results'] else np.nan,
                        'Currency': self.params['Currency']})
                    self.logger(self.netting_set, 'CVA calc complete - (gradients already present)')
                                    
                calc_complete = True

        self.stats.setdefault('Stats', {})[self.netting_set] = out['Stats']


class COLLVA(JOB):
    def __init__(self, cx, rundate, input_path, outputdir, netting_set, stats, log):
        super(COLLVA, self).__init__(cx, rundate, input_path, outputdir, netting_set, stats, log)
        # load up a calendar (for theta)
        self.business_day = cx.current_cfg.holidays['Johannesburg']['businessday']
        # set the OIS cashflow flag to speed up prime linked swaps
        self.cx.current_cfg.params['Valuation Configuration']['CFFloatingInterestListDeal']['OIS_Cashflow_Group_Size'] = 1
        from derivus.utils import Curve
        # change the currency
        self.params['Currency'] = self.balance_currency
        # does this netting set have any combination deals?
        self.cominations = 0

        if self.ns.field.get('Collateral_Assets') is None:
            self.logger(self.netting_set, 'Check balance currency {}'.format(self.balance_currency))

    def valid(self):
        if not self.cx.current_cfg.deals['Deals']['Children'][0]['Children'] or self.ns.field.get(
            'Collateralized', 'False') == 'False':
            return False
        else:
            combo_deals = [x for x in self.cx.current_cfg.deals['Deals']['Children'][0]['Children'] if 'Combination' in x['Instrument'].field.get('Tags',[''])[0]]
            self.combinations = len(combo_deals)
            return True

    def run_calc(self):
        filename = 'COLLVA_' + self.rundate + '_' + self.cx.current_cfg.deals['Attributes']['Reference'] + '.csv'
        num_deals = len(self.cx.current_cfg.deals['Deals']['Children'][0]['Children'])
        self.logger(self.netting_set, 'Netting set has {} deals'.format(num_deals))

        spreads = {
            'USD': {'collateral': 0, 'funding': 65},
            'EUR': {'collateral': 0, 'funding': 65},
            'GBP': {'collateral': 0, 'funding': 65},
            'ZAR': {'collateral': -10, 'funding': 15}
            }
    
        curves = {'USD': {'collateral': 'USD-OIS', 'funding': 'USD-SOFR.USD-SOFR3M_CAS'},
                  'EUR': {'collateral': 'EUR-EONIA', 'funding': 'EUR-EURIBOR-3M'},
                  'GBP': {'collateral': 'GBP-SONIA', 'funding': 'GBP-SONIA'},
                  'ZAR': {'collateral': 'ZAR-SWAP', 'funding': 'ZAR-SWAP'}}

        collva_sect = self.cx.current_cfg.deals['Calculation'].get(
            'Collateral_Valuation_Adjustment', {'Calculate': 'Yes'})
        # sensible defaults
        collva_sect['Collateral_Curve'] = collva_sect.get(
            'Collateral_Curve', curves[self.balance_currency]['collateral'])
        collva_sect['Funding_Curve'] = collva_sect.get(
            'Funding_Curve', curves[self.balance_currency]['funding'])
        collva_sect['Collateral_Spread'] = collva_sect.get(
            'Collateral_Spread', spreads[self.balance_currency]['collateral'])
        collva_sect['Funding_Spread'] = collva_sect.get(
            'Funding_Spread', spreads[self.balance_currency]['funding'])
        self.cx.current_cfg.deals['Calculation']['Collateral_Valuation_Adjustment'] = collva_sect
        # make sure other xva is switched off
        if 'Credit_Valuation_Adjustment' in self.cx.current_cfg.deals['Calculation']:
            del self.cx.current_cfg.deals['Calculation']['Credit_Valuation_Adjustment']
        if 'Funding_Valuation_Adjustment' in self.cx.current_cfg.deals['Calculation']:
            del self.cx.current_cfg.deals['Calculation']['Funding_Valuation_Adjustment']
        # how many simulations do we want (in thousands)
        num_1ksims = 16
        if self.combinations:
            # use smaller batches if there are comination deals present
            self.logger(self.netting_set, 'Combination deals present - dropping batch size to 256')
            self.params['Simulation_Batches'] = 4 * num_1ksims
            self.params['Batch_Size'] = 256
        else:
            self.params['Simulation_Batches'] = 2 * num_1ksims
            self.params['Batch_Size'] = 512

        self.params['Dynamic_Scenario_Dates'] = 'Yes'
        if os.path.isfile(os.path.join(self.outputdir, 'Greeks', filename)):
            self.logger(self.netting_set, 'Warning: skipping gradient COLLVA calc as file already exists')
            self.params['COLLVA'] = {'Gradient': 'No'}
            self.params['Simulation_Batches'] = num_1ksims
            self.params['Batch_Size'] = 1024
        else:
            self.params['COLLVA'] = {'Gradient': 'Yes'}

        calc_complete = False
        num_tries = 0
        CSA = {k: v.value() if hasattr(v, 'value') else v for k, v in self.ns.field['Credit_Support_Amounts'].items()}
        while not calc_complete:
            try:
                calc, out = self.calculate(self.params)
            except RuntimeError as e:  # Out of memory
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          limit=2, file=sys.stdout)
                self.params['Simulation_Batches'] *= 2
                self.params['Batch_Size'] //= 2
                self.logger(self.netting_set,
                            'Exception: Runtime Error - Halving to {} Batchsize'.format(self.params['Batch_Size']))
                num_tries += 1
                if num_tries > 2:
                    self.logger(self.netting_set,
                            'Tried twice with last batchsize of {} Skipping'.format(self.params['Batch_Size']))                        
                    out = {'Stats': CSA}
                    out['Stats'].update({'CollVA': np.nan, 'Currency': self.params['Currency']})
                    calc_complete = True
            except KeyError as key:
                self.logger(self.netting_set, 'Exception: Key Error {} - skipping'.format(key.args))
                calc_complete = True
                out = {'Stats': CSA}
                out['Stats'].update({'CollVA': np.nan, 'Currency': self.params['Currency']})
            else:
                if 'grad_collva' in out['Results']:
                    grad_collva = out['Results']['grad_collva'].rename(
                        columns={'Gradient': self.cx.current_cfg.deals['Attributes']['Reference']})
                    grad_collva.to_csv(os.path.join(self.outputdir, 'Greeks', filename))
                    out['Stats'].update(CSA)
                    # store the CollVA as part of the stats
                    out['Stats'].update({
                        'CollVA': out['Results']['collva'],
                        'Opening_CSA_Balance': self.ns.field['Opening_Balance'],
                        'MtM_t0': out['Results']['mtm'][0].mean(),
                        'Calc_Collateral_t2': out['Results']['collateral'][1].mean(),
                        'Agreement_Currency': self.agreement_currency,
                        'Currency': self.params['Currency']})
                    # log the netting set
                    self.logger(self.netting_set, 'CollVA calc complete')
                else:
                    out['Stats'].update({
                        'CollVA': out['Results']['collva'] if 'collva' in out['Results'] else np.nan,
                        'Opening_CSA_Balance': self.ns.field['Opening_Balance'],
                        'MtM_t0': out['Results']['mtm'][0].mean() if 'mtm' in out['Results'] else np.nan,
                        'Calc_Collateral_t2': out['Results']['collateral'][1].mean() if 'collateral' in out['Results'] else np.nan,
                        'Agreement_Currency': self.agreement_currency,
                        'Currency': self.params['Currency']})
                    self.logger(self.netting_set, 'CollVA calc complete - (gradients already present)')

                calc_complete = True

        self.stats.setdefault('Stats', {})[self.netting_set] = out['Stats']


class FVA(JOB):
    def __init__(self, cx, rundate, input_path, outputdir, netting_set, stats, log):
        super(FVA, self).__init__(cx, rundate, input_path, outputdir, netting_set, stats, log)
        # load up a calendar (for theta)
        self.business_day = cx.current_cfg.holidays['Johannesburg']['businessday']
        # set the OIS cashflow flag to speed up prime linked swaps
        self.cx.current_cfg.params['Valuation Configuration']['CFFloatingInterestListDeal']['OIS_Cashflow_Group_Size'] = 1
        from derivus.utils import Curve
        # change the currency
        self.params['Currency'] = self.cx.current_cfg.deals['Calculation']['Currency']
        # does this netting set have any combination deals?
        self.cominations = 0
        # rename some curves to better handle CSA discounting
        rename_factor(self.cx, 'ZAR-SWAP', 'ZAR-OIS')
        rename_factor(self.cx, 'ZAR-JIBAR-3M', 'ZAR-JIBAR-3M-OIS')

    def valid(self):
        if not self.cx.current_cfg.deals['Deals']['Children'][0]['Children']:
            return False
        else:
            combo_deals = [x for x in self.cx.current_cfg.deals['Deals']['Children'][0]['Children'] if 'Combination' in x['Instrument'].field.get('Tags',[''])[0]]
            self.combinations = len(combo_deals)
            
            # if assets.replace('_Collateral','')=='Equity':
            if self.ns.field.get('Credit_Support_Amounts', {}).get('Independent_Amount') is not None:
                self.logger(self.netting_set, 'Netting set has an Independent Amount - assuming Equity collar deals and Skipping')
                return False
            else:
                return True

    def run_calc(self):
        
        filename = 'FVA_' + self.rundate + '_' + self.params['Currency']+ '_' + self.cx.current_cfg.deals['Attributes']['Reference'] + '.csv'
        num_deals = len(self.cx.current_cfg.deals['Deals']['Children'][0]['Children'])
        self.logger(self.netting_set, 'Netting set has {} deals'.format(num_deals))

        # make sure other xva is switched off
        if 'Credit_Valuation_Adjustment' in self.cx.current_cfg.deals['Calculation']:
            del self.cx.current_cfg.deals['Calculation']['Credit_Valuation_Adjustment']
        if 'Collateral_Valuation_Adjustment' in self.cx.current_cfg.deals['Calculation']:
            del self.cx.current_cfg.deals['Calculation']['Collateral_Valuation_Adjustment']
        # how many simulations do we want (in thousands)
        num_1ksims = 16
        if self.combinations:
            # use smaller batches if there are comination deals present
            self.logger(self.netting_set, 'Combination deals present - dropping batch size to 256')
            self.params['Simulation_Batches'] = 4 * num_1ksims
            self.params['Batch_Size'] = 256
        else:
            self.params['Simulation_Batches'] = num_1ksims
            self.params['Batch_Size'] = 1024

        self.params['Dynamic_Scenario_Dates'] = 'Yes'
        if os.path.isfile(os.path.join(self.outputdir, 'Greeks', filename)):
            self.logger(self.netting_set, 'Warning: skipping gradient FVA calc as file already exists')
            self.params['Funding_Valuation_Adjustment'] = {'Gradient': 'No'}
        else:
            self.params['Funding_Valuation_Adjustment'] = {'Gradient': 'Yes'}

        calc_complete = False
        num_tries = 0

        while not calc_complete:
            try:
                calc, out = self.calculate(self.params)
            except RuntimeError as e:  # Out of memory
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          limit=2, file=sys.stdout)
                self.params['Simulation_Batches'] *= 2
                self.params['Batch_Size'] //= 2
                self.logger(self.netting_set,
                            'Exception: Runtime Error - Halving to {} Batchsize'.format(self.params['Batch_Size']))
                num_tries += 1
                if num_tries > 2:
                    self.logger(self.netting_set,
                            'Tried twice with last batchsize of {} Skipping'.format(self.params['Batch_Size']))                        
                    out = {'Stats': {}}
                    out['Stats'].update({'FVA': np.nan, 'Currency': self.params['Currency']})
                    calc_complete = True
            except KeyError as key:
                self.logger(self.netting_set, 'Exception: Key Error {} - skipping'.format(key.args))
                calc_complete = True
                out = {'Stats': {}}
                out['Stats'].update({'FVA': np.nan, 'Currency': self.params['Currency']})
            else:
                if 'grad_fva' in out['Results']:
                    grad_fva = out['Results']['grad_fva'].rename(
                        columns={'Gradient': self.cx.current_cfg.deals['Attributes']['Reference']})
                    grad_fva.to_csv(os.path.join(self.outputdir, 'Greeks', filename))
                    # store the FVA as part of the stats
                    out['Stats'].update({
                        'FVA': out['Results']['fva'],
                        'Currency': self.params['Currency']})
                    # log the netting set
                    self.logger(self.netting_set, 'FVA calc complete')
                else:
                    out['Stats'].update({
                        'FVA': out['Results']['fva'] if 'fva' in out['Results'] else np.nan,
                        'Currency': self.params['Currency']})
                    self.logger(self.netting_set, 'FVA calc complete - (gradients already present)')

                calc_complete = True

        self.stats.setdefault('Stats', {})[self.netting_set] = out['Stats']


class SA_CVA(JOB):
    def __init__(self, cx, rundate, input_path, outputdir, netting_set, stats, log):
        super(SA_CVA, self).__init__(cx, rundate, input_path, outputdir, netting_set, stats, log)
        # set the OIS cashflow flag to speed up prime linked swaps
        self.cx.current_cfg.params['Valuation Configuration']['CFFloatingInterestListDeal']['OIS_Cashflow_Group_Size'] = 1
        # does this netting set have any combination deals?
        self.cominations = 0
        if 'HullWhite2FactorModelParameters.USD-OIS' not in self.cx.current_cfg.params['Price Factors']:
            self.cx.current_cfg.params['Price Factors']['HullWhite2FactorModelParameters.USD-OIS'] = self.cx.current_cfg.params['Price Factors']['HullWhite2FactorModelParameters.USD-SOFR']
        if self.cx.stressed_config_file is None:
            # DERIVUS_STRESSED_MARKETDATA is the directory holding the vega-stressed calibrations
            root = os.environ.get('DERIVUS_STRESSED_MARKETDATA')
            if root:
                short_date = ''.join(rundate[2:].split('-')[::-1])
                self.cx.stressed_config_file = os.path.join(
                    root, 'CVAMarketData_Calibrated_Vega_{}.json'.format(short_date))
                self.logger(self.netting_set, 'stressed market file resolved to {}'.format(
                    self.cx.stressed_config_file))

    def valid(self):
        if not self.cx.current_cfg.deals['Deals']['Children'][0]['Children']:
            return False
        else:
            combo_deals = [x for x in self.cx.current_cfg.deals['Deals']['Children'][0]['Children'] if 'Combination' in x['Instrument'].field.get('Tags',[''])[0]]
            self.combinations = len(combo_deals) 
            return True

    def calc_vega(self, num_2ksims, calc, output):
    
        def run_vega(vega_factors):
            if [x for x in calc.all_factors.keys() if x.type in vega_factors]:
                # stress the cfg
                self.cx.stress_config(vega_factors)
                try:
                    _, vega = self.cx.run_job(self.params)
                except:
                    self.logger(self.netting_set, 'vega failed')
                    vega = output
                # restore the config
                self.cx.restore_config()
                vega_result = vega['Results']['cva'] - output['Results']['cva']
                self.logger(self.netting_set, 'vega calc for {} is {}'.format(vega_factors, vega_result))
                return vega_result
            else:
                self.logger(self.netting_set, 'skipping vega calc for {}'.format(vega_factors))
                return 0.0
                
        # turn off gradients
        self.params['Credit_Valuation_Adjustment']['Gradient'] = 'No'
        
        results = {}
        results['CVA_IR_Vega'] = run_vega(['InterestRate', 'InflationRate'])
        results['CVA_CM_Vega'] = run_vega(['ForwardPrice'])  
          
        return results


    def run_calc(self):
        filename = self.rundate + '_' + self.cx.current_cfg.deals['Attributes']['Reference'] + '.csv'
        num_deals = len(self.cx.current_cfg.deals['Deals']['Children'][0]['Children'])
        self.logger(self.netting_set, 'Netting set has {} deals'.format(num_deals))
        # how many simulations do we want (in thousands) - must be divisible by 4
        num_2ksims = 8
        # load up CVA calc params
        if self.cx.current_cfg.deals['Deals']['Children'][0]['Instrument'].field.get('Collateralized') == 'True':
            self.logger(self.netting_set, 'is collateralized')
            # turn on dynamic scenarios (more accurate)
            self.params['Dynamic_Scenario_Dates'] = 'Yes'
            # change the liquidation period to 10 days (as per regs)
            self.ns.field['Liquidation_Period'] = 10.0
            if self.combinations:
                # use smaller batches if there are comination deals present
                self.logger(self.netting_set, 'Combination deals present - dropping batch size to 256')
                self.params['Simulation_Batches'] = 4 * num_2ksims
                self.params['Batch_Size'] = 512
            else:
                self.params['Simulation_Batches'] = 2 * num_2ksims
                self.params['Batch_Size'] = 1024
        else:
            self.params['Dynamic_Scenario_Dates'] = 'No'
            self.params['Simulation_Batches'] = num_2ksims
            self.params['Batch_Size'] = 2048
            self.logger(self.netting_set, 'is uncollateralized')

        # get the calculation parameters for CVA
        cva_sect = self.cx.current_cfg.deals['Calculation']['Credit_Valuation_Adjustment']

        # update the params
        self.params['Currency'] = 'ZAR'
        self.params['Deflation_Interest_Rate'] = 'ZAR-SWAP'
        self.params['Credit_Valuation_Adjustment'] = cva_sect
        self.params['Credit_Valuation_Adjustment']['CDS_Tenors'] = [0.5, 1, 3, 5, 10]
        self.params['Credit_Valuation_Adjustment']['Gradient'] = 'Yes'

        # this produces (or not) the gamma matrix
        cva_sect['Hessian'] = 'No'
        calc_complete = False
        num_tries = 0
        while not calc_complete:
            try:
                calc, out = self.cx.run_job(self.params)
            except RuntimeError as e:  # Out of memory
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          limit=2, file=sys.stdout)
                self.params['Simulation_Batches'] *= 2
                self.params['Batch_Size'] //= 2
                self.logger(self.netting_set,
                            'Exception: Runtime Error - Halving to {} Batchsize'.format(self.params['Batch_Size']))
                num_tries += 1
                if num_tries>2:
                    self.logger(self.netting_set,
                            'Tried twice with last batchsize of {} Skipping'.format(self.params['Batch_Size']))
                    out = {'Stats': {}}
                    out['Stats'].update({'CVA_IR_Vega': np.nan, 'CVA_CM_Vega': np.nan, 'CVA': np.nan, 'Currency': self.params['Currency']})
                    calc_complete = True
            except KeyError as key:
                self.logger(self.netting_set, 'Exception: Key Error {} - skipping'.format(key.args))
                calc_complete = True
                out = {'Stats': {}}
                out['Stats'].update({'CVA_IR_Vega': np.nan, 'CVA_CM_Vega': np.nan, 'CVA': np.nan, 'Currency': self.params['Currency']})
            else:
                if 'grad_cva' in out['Results']:
                    grad_cva = out['Results']['grad_cva'].rename(
                        columns={'Gradient': self.cx.current_cfg.deals['Attributes']['Reference']})
                    out['Stats'].update({'CVA': out['Results']['cva'], 'Currency': self.params['Currency']})
                    grad_cva.to_csv(os.path.join(self.outputdir, 'Greeks', 'SACVA_' + filename))
                    # write out the CS01
                    if 'CS01' in out['Results']: 
                        out['Results']['CS01'].to_csv(os.path.join(self.outputdir, 'Greeks', 'CS01_'+filename))
                    # now calculate the vega sensitivities
                    out['Stats'].update(self.calc_vega(num_2ksims, calc, out))
                    
                    # log the netting set
                    self.logger(self.netting_set, 'CVA calc complete')
                else:
                    self.logger(self.netting_set, '!!! Critical !!! - CVA NOT calc complete - Check gradients and Vega calc')

                calc_complete = True

        self.stats.setdefault('Stats', {})[self.netting_set] = out['Stats']


class SIMM(JOB):
    """ISDA SIMM sensitivities of each booked trade of the netting set by bump and revaluation, its
    CRIF written to `CRIF/CRIF_<rundate>_<file>.tsv`. DERIVUS_SIMM_MAPPING and
    DERIVUS_SIMM_PARAMETERS name the mapping and the ISDA parameters file, DERIVUS_SIMM_CURRENCY the
    calculation currency (USD unless stated); DERIVUS_SIMM_CRIF may name an original CRIF, whose
    regulations a netting set declaring none takes for its trades and onto which `collate` merges
    the run."""
    # a sensitivity is a booked trade's, and compression folds trades into one
    compress = False

    def __init__(self, cx, rundate, input_path, outputdir, netting_set, stats, log):
        super(SIMM, self).__init__(cx, rundate, input_path, outputdir, netting_set, stats, log)
        config = self.cx.current_cfg
        stated = config.deals['Calculation']
        # the document's own quasi seed, so trades' Monte Carlo errors stay independent across a
        # book
        config.deals['Calculation'] = dict(
            {key: stated[key] for key in ('Quasi_Seed',) if key in stated}, Object='SIMM',
            Base_Date=stated['Base_Date'], Currency=os.environ.get('DERIVUS_SIMM_CURRENCY', 'USD'),
            Mapping=os.environ['DERIVUS_SIMM_MAPPING'], Parameters=os.environ['DERIVUS_SIMM_PARAMETERS'])
        quote_curves(config, stated['Base_Date'])
        if os.environ.get('DERIVUS_SIMM_CRIF'):
            self.regulations(pd.read_csv(
                os.environ['DERIVUS_SIMM_CRIF'], sep='\t', dtype=str, keep_default_na=False,
                usecols=['TradeID', 'PostRegulations', 'CollectRegulations']))

    def regulations(self, original):
        """The regimes a netting set declaring none posts and collects margin under, where the
        original CRIF's rows for its trades state one pair."""
        for agreement in self.cx.current_cfg.deals['Deals']['Children']:
            terms = agreement['Instrument'].field
            trades = {node['Instrument'].field.get('Reference') for node in agreement.get('Children', [])}
            stated = original[original['TradeID'].isin(trades)].drop_duplicates(
                ['PostRegulations', 'CollectRegulations'])
            if len(stated) == 1:
                for field, column in (('Post_Regulations', 'PostRegulations'),
                                      ('Collect_Regulations', 'CollectRegulations')):
                    terms[field] = terms.get(field) or stated[column].iloc[0]

    def valid(self):
        return bool(self.cx.current_cfg.deals['Deals']['Children'][0].get('Children'))

    def run_calc(self):
        filename = os.path.join(self.outputdir, 'CRIF', 'CRIF_{}_{}.tsv'.format(
            self.rundate, os.path.splitext(self.netting_set)[0]))
        try:
            calc, out = self.cx.run_job()
        except Exception as e:
            # a set the engine refuses keeps the original's rows; its stats say why
            self.logger(self.netting_set, 'SIMM refused - {}'.format(e))
            self.stats.setdefault('Stats', {})[self.netting_set] = {
                'Error': '{}: {}'.format(type(e).__name__, e)}
            return
        crif, trades = out['Results']['CRIF'], out['Results']['Trades']
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        calc.write(crif, filename)
        skipped = '; '.join(said for said in trades['Skipped'] if said)
        if skipped:
            self.logger(self.netting_set, 'skipped inside a trade - {}'.format(skipped))
        self.stats.setdefault('Stats', {})[self.netting_set] = {
            'CRIF': filename, 'Trades': len(trades), 'Rows': len(crif), 'Moves': trades['Moves'].sum(),
            'Value': trades['Value'].sum(), 'Currency': calc.params['Currency'], 'Skipped': skipped}

    @classmethod
    def collate(cls, outputdir, rundate, stats):
        """The run's CRIF over every netting set, and its merge onto the original CRIF that
        DERIVUS_SIMM_CRIF names."""
        written = stats['CRIF'].dropna() if 'CRIF' in stats else []
        if not len(written):
            return
        crif = pd.concat([pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
                          for path in written])
        stem = os.path.join(outputdir, 'CRIF', 'SIMM_CRIF_{}'.format(rundate))
        crif.to_csv(stem + '_Total.tsv', sep='\t', index=False)
        if os.environ.get('DERIVUS_SIMM_CRIF'):
            merge_crif(pd.read_csv(os.environ['DERIVUS_SIMM_CRIF'], sep='\t', dtype=str,
                                   keep_default_na=False), crif,
                       os.environ.get('DERIVUS_SIMM_CURRENCY', 'USD')).to_csv(
                stem + '_Merged.tsv', sep='\t', index=False)


def pooled(answer):
    """One run's `out` from a sharded run's per-worker lists: the xva and its gradient averaged, as
    every shard ran as many batches, and the shards' paths side by side. Whatever else a shard
    reports does not pool and is left out."""
    results, out = answer['Results'], {}
    for key, part in results[0].items():
        parts = [shard[key] for shard in results]
        if key.startswith('grad_'):
            out[key] = pd.concat(parts).groupby(
                level=list(range(part.index.nlevels)), sort=False).mean()
        elif key in ('mtm', 'collateral'):
            out[key] = pd.concat(parts, axis=1, ignore_index=True)
        elif isinstance(part, (float, np.floating)):
            out[key] = np.mean(parts)
    stats = dict(answer['Stats'][0], Simulation_Batches=sum(
        stats['Simulation_Batches'] for stats in answer['Stats']))
    return {'Results': out, 'Stats': stats}


def logger(id, lock):
    def log(netting_set, msg):
        with lock:
            print('JOB %s:' % id, '{0}: {1}'.format(netting_set, msg))
    return log


def context(job):
    """The context a job's netting sets load on."""
    import derivus as rf
    # A deal file can carry the absolute paths of whatever machine wrote it; path_transform
    # rewrites those roots onto local mounts. Set DERIVUS_PATH_TRANSFORM to a JSON object of
    # {written_root: local_root} when that applies - it is empty when the paths already resolve.
    dpt = os.environ.get('DERIVUS_PATH_TRANSFORM')
    path_transform = json.load(open(dpt,'rt')) if dpt is not None else {}
    # a context object loads (and caches) one or more configs
    # remap any paths as necessary
    if job == 'PFE':
        cx = rf.Context(path_transform=path_transform, file_transform={
            'CVAMarketData_Calibrated_New.json': 'MarketData.json',
            'MarketData.dat': 'MarketData.json'
        })
    elif job == 'SA_CVA':
        # this uses 2 marketdata files - a "normal" file and a "stressed" file which contains the vega recalibrations
        cx = rf.StressedContext(path_transform=path_transform)
    else:
        cx = rf.Context(path_transform=path_transform)
    return cx


def perform(job, tasks, rundate, input_path, outputdir, id, log):
    """`job` over each netting set of `tasks` on one context: `[('Stats', file)]` or `[]`."""
    cx, logs = context(job), {}
    for task in tasks:
        try:
            obj = globals().get(job)(cx, rundate, input_path, outputdir, task, logs, log)
        except Exception as e:
            log(task, "!! CRITICAL ERROR In JSON !! - {} - Skipped".format(e.args))
        else:
            obj.perform_calc()

    # write out the logs
    if 'Stats' not in logs:
        return []
    stats_file = os.path.join(outputdir, 'Stats', '{0}_Stats_{1}_JOB_{2}.csv'.format(job, rundate, id))
    pd.DataFrame(data=logs['Stats']).T.to_csv(stats_file)
    return [('Stats', stats_file)]


def work(id, lock, queue, results, job, rundate, input_path, outputdir):
    # set the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(id)
    result = perform(job, iter(queue.get, None), rundate, input_path, outputdir, id, logger(id, lock))
    # hand the sentinel on so the next worker ends too
    queue.put(None)
    results.put(result)


class Parent:
    def __init__(self, num_jobs):
        self.queue = Queue()
        self.results = Queue()
        self.lock = Lock()
        self.NUMBER_OF_PROCESSES = num_jobs

    def start(self, job, rundate, input_path, outputdir, wildcard='CrB*.json', shard=False):
        crbs = [os.path.split(x)[-1] for x in sorted(
            glob.glob(os.path.join(input_path, rundate, wildcard)), key=os.path.getsize)[::-1]]

        if shard:
            # one process runs the netting sets in turn, the engine sharding each over NUMBER_OF_PROCESSES
            print("sharding each of {0} netting sets in {1} over {2} workers".format(
                len(crbs), input_path, self.NUMBER_OF_PROCESSES))
            globals()[job].shards = self.NUMBER_OF_PROCESSES
            post_processing = [perform(
                job, crbs, rundate, input_path, outputdir, 0, logger(0, self.lock))]
        else:
            print("starting {0} workers in {1}".format(self.NUMBER_OF_PROCESSES, input_path))

            self.workers = [Process(target=work, args=(
                i, self.lock, self.queue, self.results, job, rundate, input_path, outputdir))
                            for i in range(self.NUMBER_OF_PROCESSES)]

            # start all children
            for w in self.workers:
                w.start()

            # load the crbs on the queue
            for netting_set in crbs:
                self.queue.put(netting_set)

            self.queue.put(None)

            # now collate the results
            post_processing = []
            for i in range(self.NUMBER_OF_PROCESSES):
                post_processing.append(self.results.get())

            # close the queue
            self.queue.close()
            self.results.close()

            # terminate all worker processes
            for w in self.workers:
                w.join(timeout=300)
                if w.is_alive():
                    w.close()

        post_results = {}
        for output in post_processing:
            data = dict(output)
            for k, v in data.items():
                post_results.setdefault(k, []).append(pd.read_csv(v, index_col=0))

        # write out the combined data
        for k, v in post_results.items():
            if v:
                out_path = os.path.join(outputdir, k, '{0}_{1}_{2}_Total.csv'.format(job, k, rundate))
                pd.concat(v).to_csv(out_path)
        globals()[job].collate(outputdir, rundate, pd.concat(post_results.get('Stats', [pd.DataFrame()])))


def main():
    import argparse

    jobs = [cls.__name__ for cls in globals().values() if
            isinstance(cls, type) and hasattr(cls, 'valid') and cls.__name__ != 'JOB']
    # setup the arguments
    parser = argparse.ArgumentParser(description='Run a derivus batch on a directory of .json netting sets.')
    parser.add_argument('num_jobs', type=int, help='the number of gpu\'s to use (if available) else cpu\'s')
    parser.add_argument('job', type=str, help='the job name', choices=jobs)
    parser.add_argument('rundate', type=str, help='batch rundate')
    parser.add_argument('input_path', type=str, help='directory containing the input files (note that the rundate is '
                                                     'assumed to be a directory within)')
    parser.add_argument('output_path', type=str, help='output directory')
    parser.add_argument('filename', type=str, help='filename(s) in input_path to run - wildcards allowed')
    parser.add_argument('--shard', action='store_true', help=(
        'read num_jobs as the shards each netting set\'s paths are split over, the netting sets run in '
        'turn, rather than the workers that take netting sets between them; for a very large netting '
        'set\'s xva sensitivities (CVA_GRAD, FVA, COLLVA)'))

    # get the arguments
    args = parser.parse_args()

    Parent(args.num_jobs).start(
        args.job, args.rundate, args.input_path, args.output_path, args.filename, args.shard)
    return 0


if __name__ == '__main__':
    sys.exit(main())
