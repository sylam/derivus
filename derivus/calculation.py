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


import contextlib
import copy
import math
import time
import json
import pickle
import logging
import itertools
import traceback
import pandas as pd
import numpy as np
import scipy.stats
import torch
import torch.multiprocessing as mp
from queue import Empty
from functools import partial, reduce

from collections import namedtuple, defaultdict
from .riskfactors import construct_factor
from .stochasticprocess import REVEAL_CONTINUOUS, SuppliedValuesModel, construct_process
from .instruments import (Deal, barrier_monitoring_rows, get_fxrate_factor, get_survival_component,
                          get_interest_factor, get_survival_factor)
from .pricing import SensitivitiesEstimator, fixed_payments
from . import utils, instruments, pricing, riskfactors
from . import schema
from .schema import (F, REQUIRED, Row, declared_defaults, declared_settlements, index_named,
                     settlement_days)
from .utils import (CASHFLOW_INDEX_FixedAmt, CASHFLOW_INDEX_FixedRate, CASHFLOW_INDEX_Nominal,
                    CASHFLOW_INDEX_Pay_Day, CASHFLOW_INDEX_Year_Frac, RESET_INDEX_Reset_Day,
                    RESET_INDEX_Value, TensorCashFlows, TensorResets, walk_schedules)
from .hedge_runtime import construct_hedge_runtime
from .hedge_bundle import Bundle, run_hedge_execution, HedgeRuntimeExecutionResult
from .hedge_solver import StreamingSolve

class Aggregation(object):
    '''Container class that represents the base Instrument for aggregation'''
    def __init__(self, name):
        self.reval_dates = None
        self.field = {'Reference': name}
        self.accum_dependencies = True

    def calc_dependencies(self, base_date, static_offsets, stochastic_offsets,
        all_factors, all_tenors, time_grid, calendars):
        pass

    def get_report_dates(self):
        return self.reval_dates

    def set_report_dates(self, reval_dates):
        self.reval_dates = reval_dates

    def post_process(self, accum, shared, time_grid, deal_data, child_dependencies):
        # every netting set hands back the global grid and the root reports its own dates off it; a
        # book that valued nothing sums to the unit, which has no rows to report
        if time_grid.report_index is not None and accum.shape[:1] == time_grid.mtm_time_grid.shape:
            accum = accum[time_grid.report_index]
        # Honour store_results=False (Calc_res is None) exactly as pricing.interpolate does.
        if deal_data.Calc_res is not None:
            shared.save_results(deal_data.Calc_res, {'Value': accum})
        return accum

class DealStructure(object):
    def __init__(self, obj, store_results=False):
        self.obj = utils.DealDataType(
            Instrument=obj, Factor_dep={}, Time_dep=None, Calc_res={} if store_results else None)
        self.dependencies = []
        self.sub_structures = []
        self.store_results = store_results

    @staticmethod
    def calc_time_dependency(base_date, deal, time_grid):
        """The deal's time dependency on `time_grid`, or None where its last date is behind the
        base date - expired under a scenario grid exactly as under a single date."""
        reval_dates = deal.get_reval_dates(clip_expiry=True)
        if reval_dates and max(reval_dates) < base_date:
            logging.warning('skipping expired deal {0}'.format(
                deal.field.get('Reference', 'Unknown Instrument Reference')))
            return None
        return time_grid.calc_deal_grid(
            {base_date} if len(time_grid.scenario_dates) == 1 else reval_dates)

    def add_deal_to_structure(self, base_date, deal, static_offsets, stochastic_offsets,
                              all_factors, all_tenors, time_grid, calendars, stats, unit,
                              valuation_options=None, exclude=True):
        """Compile `deal` into this structure, the history fold first. A deal the compile cannot
        read is skipped and counted, or kept and marked at zero on every date where `exclude` is
        False, the exception stamped on it either way (`unpriced_because`, which the diary reads);
        a framework fault or a schedule refused by name (`utils.is_fatal_pricing_error`) is counted
        and stamped too, and raised, never swallowed."""
        try:
            deal = deal.resolve_history(base_date, calendars, valuation_options or {})
            deal_time_dep = self.calc_time_dependency(base_date, deal, time_grid)
            if deal_time_dep is None:
                return
            factor_dep = utils.bind_schedules(deal.calc_dependencies(
                base_date, static_offsets, stochastic_offsets, all_factors, all_tenors, time_grid,
                calendars), unit)
            stats['Deals loaded'] = stats.setdefault('Deals loaded', 0) + 1
        except Exception as e:
            fatal = utils.is_fatal_pricing_error(e)
            logging.error('{0} {1} {2} - {3}'.format(
                deal.field['Object'], deal.field.get('Reference'), e.args,
                'Skipped' if exclude or fatal else 'kept at zero'))
            stats['Deals Skipped'] = stats.setdefault('Deals Skipped', 0) + 1
            deal.unpriced_because = e
            if fatal:
                raise
            if exclude:
                return
            deal_time_dep, factor_dep = utils.DealTimeDependencies(
                time_grid.mtm_time_grid, np.arange(time_grid.mtm_time_grid.size)), None
        self.dependencies.append(utils.DealDataType(
            Instrument=deal, Factor_dep=factor_dep, Time_dep=deal_time_dep,
            Calc_res={} if self.store_results else None))

    @staticmethod
    def unpriced(shared, structure, error):
        """A structure the run could not value, counted once under `Structs Skipped` however many
        batches meet it."""
        instrument = structure.obj.Instrument
        unpriced = getattr(shared, 'unpriced', None)
        if unpriced is not None and instrument not in unpriced:
            unpriced.add(instrument)
            instrument.unpriced_because = error
            shared.calc_stats['Structs Skipped'] = shared.calc_stats.get('Structs Skipped', 0) + 1

    def structures(self):
        """Every structure beneath this one, depth first."""
        for structure in self.sub_structures:
            yield structure
            yield from structure.structures()

    def finalize_struct(self, base_date, time_grid):
        windows = {}
        for structure in self.structures():
            window = structure.obj.Instrument.closeout_window()
            if window is not None:
                windows.setdefault(window, []).append(structure.obj.Instrument.field.get('Reference'))
        if len(windows) > 1:
            raise utils.UnpriceableSchedule(
                'Collateralised netting sets in one job close out over one Liquidation_Period and '
                'Settlement_Period, and these differ: {}'.format('; '.join(
                    '{} over (Liquidation_Period, Settlement_Period) = {}'.format(', '.join(refs), window)
                    for window, refs in sorted(windows.items()))))
        all_report_dates = [set(instrument.get_report_dates(time_grid, base_date)) for instrument in
                            [x.obj.Instrument for x in self.sub_structures] +
                            [x.Instrument for x in self.dependencies]]
        self.obj.Instrument.set_report_dates(
            reduce(set.union, all_report_dates) if all_report_dates else time_grid.mtm_dates)
        time_grid.set_report_dates(base_date, self.obj.Instrument.get_report_dates())

    def add_structure_to_structure(self, struct, base_date, static_offsets, stochastic_offsets,
                                   all_factors, all_tenors, time_grid, calendars, stats, unit):
        struct_time_dep = self.calc_time_dependency(base_date, struct.obj.Instrument, time_grid)
        if struct_time_dep is not None:
            try:
                struct.obj = utils.DealDataType(
                    Instrument=struct.obj.Instrument,
                    Factor_dep=utils.bind_schedules(struct.obj.Instrument.calc_dependencies(
                        base_date, static_offsets, stochastic_offsets,
                        all_factors, all_tenors, time_grid, calendars), unit),
                    Time_dep=struct_time_dep,
                    Calc_res={} if self.store_results or struct.obj.Instrument.accum_dependencies else None)
                self.sub_structures.append(struct)
                stats['Structs loaded'] = stats.setdefault('Structs loaded', 0) + 1
            except Exception as e:
                logging.error('{0} {1} - Skipped'.format(struct.obj.Instrument.field['Object'], e.args))
                # same rule as `add_deal_to_structure`: a skipped STRUCTURE takes its whole netting
                # set's deals out of the report with it
                if utils.is_fatal_pricing_error(e):
                    raise
                stats['Structs Skipped'] = stats.setdefault('Structs Skipped', 0) + 1

    def deals(self):
        """Every deal beneath this structure, sub-structures first - the order `report` lists
        them in."""
        for structure in self.sub_structures:
            for deal in structure.deals():
                yield deal
        for deal in self.dependencies:
            yield deal

    def resolve_structure(self, shared, time_grid):
        """Price every deal and sub-structure and return the accumulated MTM."""

        accum = 0.0 * shared.one
        # a boundary set registered past this mark is a decision taken BENEATH this structure
        # - post_process runs only once the children are priced, so the mark cannot wait
        mark = len(shared.boundary_sets) if getattr(shared, 'boundary_aad', False) else None

        if self.sub_structures:
            for structure in self.sub_structures:
                logging.root.name = structure.obj.Instrument.field.get('Reference', 'root')
                if structure.obj.Instrument.accum_dependencies and hasattr(shared, 'reset_cashflows'):
                    shared.reset_cashflows(time_grid)
                struct = structure.resolve_structure(shared, time_grid)
                if (struct != struct).any():
                    logging.critical('Netting set contains NANS! Please Investigate! - skipping for now')
                    self.unpriced(shared, structure, 'its value holds NaN')
                    continue
                if structure.obj.Instrument.accum_dependencies and hasattr(shared, 'save_cashflows'):
                    shared.save_cashflows(structure.obj.Calc_res, time_grid)
                if structure.obj.Instrument.field.get('Reference', 'root').startswith('FLIP'):
                    logging.warning('Netting set starts with FLIP - inverting MTM')
                    struct = -struct
                accum = accum + struct

        if self.dependencies and self.obj.Instrument.accum_dependencies:
            deal_tensors = 0.0

            for deal_data in self.dependencies:
                logging.root.name = deal_data.Instrument.field.get('Reference', 'root')
                deal_mark = len(shared.boundary_sets) if mark is not None else None
                mtm = deal_data.Instrument.calculate(shared, time_grid, deal_data)
                if deal_mark is not None:
                    utils.BoundarySet.stamp(shared, deal_mark, logging.root.name)
                deal_tensors = deal_tensors + mtm

            accum = accum + deal_tensors

        if hasattr(self.obj.Instrument, 'post_process'):
            logging.root.name = self.obj.Instrument.field.get('Reference', 'root')
            try:
                accum = self.obj.Instrument.post_process(accum, shared, time_grid, self.obj, self.dependencies)
            except RuntimeError as e:
                logging.error('Runtime error Deal skipped - {}'.format(e.args))
                raise
            except Exception as e:
                logging.critical('Deal skipped - {}'.format(e.args))
                # the same rule as `Deal.calculate`: a post_process that prices a leg can meet a
                # named refusal, and swallowing one leaves the structure marking at its accumulation
                if utils.is_fatal_pricing_error(e):
                    raise
                self.unpriced(shared, self, e.args)
            finally:
                if mark is not None:
                    utils.BoundarySet.claim(shared, mark)
                    # whatever post_process itself registered (a margin call) has no deal to
                    # name it, so the structure does
                    utils.BoundarySet.stamp(shared, mark, logging.root.name)

        return accum

    def resolve_hedge_structure(self, shared, time_grid):
        """Accumulate the liability MTM across the structure and return `{'mtm': tensor}`.

        Skips `post_process`/`save_results` - and so the per-batch GPU->CPU copy of the mark -
        which is why the hedge sim loop and the inner MC use it for the liability."""
        def merge_features(cumulative, new_features):
            new_mtm = new_features.get('mtm')
            if new_mtm is not None:
                cumulative['mtm'] = new_mtm if cumulative.get('mtm') is None else cumulative['mtm'] + new_mtm

        accum = {}

        if self.sub_structures:
            for structure in self.sub_structures:
                logging.root.name = structure.obj.Instrument.field.get('Reference', 'root')
                features = structure.resolve_hedge_structure(shared, time_grid)
                merge_features(accum, features)


        if self.dependencies and self.obj.Instrument.accum_dependencies:
            deal_features = {}

            for deal_data in self.dependencies:
                logging.root.name = deal_data.Instrument.field.get('Reference', 'root')
                features = deal_data.Instrument.build_features(shared, time_grid, deal_data)
                merge_features(deal_features, features)

            merge_features(accum, deal_features)

        return accum

    def aggregate_leg_descriptors(self):
        """Reduce the per-deal cashflow descriptors over this structure and its
        sub-structures: summed |notional| across all legs, and the latest pay-day. A leg
        with no schedule contributes (0.0, None)."""
        total_volume, last_payment_day = 0.0, None
        for vol, pay in ([dd.Instrument.leg_descriptors(dd) for dd in self.dependencies] +
                         [sub.aggregate_leg_descriptors() for sub in self.sub_structures]):
            total_volume += vol
            if pay is not None:
                last_payment_day = pay if last_payment_day is None else max(last_payment_day, pay)
        return total_volume, last_payment_day

    def tensor_marks(self):
        """Stored per-deal price series keyed by deal Reference, recursing sub-structures.
        Only deals whose `Calc_res` holds a kept 'tensor' (set by `pricing.interpolate`
        under `shared.keep_tensor`) appear."""
        marks = {}
        for sub in self.sub_structures:
            marks.update(sub.tensor_marks())
        for deal_data in self.dependencies:
            tensor = (deal_data.Calc_res or {}).get('tensor')
            if tensor is not None:
                marks[deal_data.Instrument.field['Reference']] = tensor
        return marks

    @staticmethod
    def max_settlement_date(deals, calendars):
        """Latest clipped reval/settlement date across a set of (un-built) deal nodes - the
        liability-terminal horizon that caps the sim time grid. Each instrument is reset from its
        `field` first (idempotent), since the structure is not built when the horizon is needed."""
        dates = set()
        for node in deals:
            node['Instrument'].reset(calendars)
            dates |= node['Instrument'].get_reval_dates(clip_expiry=True)
        return max(dates)


class ScenarioTimeGrid(object):
    def __init__(self, cutoff_date, global_time_grid, base_date):
        scen_grid = global_time_grid.scen_time_grid
        offset = scen_grid.searchsorted((cutoff_date - base_date).days) + 1
        self.scen_time_grid = scen_grid[:offset]
        self.time_grid_years = self.scen_time_grid / utils.DayCount.DAYS_IN_YEAR
        self.scenario_grid = global_time_grid.scenario_grid[:offset]


class Calculation(object):

    #: Whether this calculation VALUES its book: a compile that only reads it drops an unreadable
    #: deal whatever `Exclude_Deals_With_Missing_Market_Data` says.
    valuation = True

    #: `Deterministic_Kernels`, declared once for the calculations that take a seed.
    DETERMINISTIC_KERNELS = F(
        'Deterministic_Kernels', 'Text', default='No', values=['Yes', 'No'],
        description="Pin the GPU backward to torch's deterministic kernels (warn_only) so a "
                    "gradient's last bits repeat between runs on one machine and one build, at "
                    'about 1.03x the wall clock')

    #: `Random_Seed`, declared once for the calculations that simulate scenarios.
    RANDOM_SEED = F('Random_Seed', 'Integer', default=5120,
                    description='Seed of the scenario stream the simulated processes draw from')

    #: `Quasi_Seed`, declared once for the calculations whose pricers draw a quasi-random stream.
    QUASI_SEED = F(
        'Quasi_Seed', 'Integer', default=schema.QUASI_SEED, convention=True,
        description='Scramble seed of every quasi-random draw: the inner Monte Carlo paths, the '
                    'Sobol-driven processes and the walks keyed off it')

    def __init__(self, config, prec=torch.float32, device=torch.device('cpu')):
        """Construct a new calculation - all calculations set up their own tensors."""

        self.config = config
        self.dtype = prec
        self.time_grid = None
        self.device = device

        self.static_factors = {}
        self.static_var = {}
        self.stoch_factors = {}
        self.stoch_var = {}
        self.all_factors = {}
        self.all_tenors = {}

        self.base_date = None
        self.tenor_size = None
        self.tenor_offset = None

        self.netting_sets = None

        # what the run reports, from the digests of the files its job named, and the deals its
        # pricers skipped, counted there once each
        self.calc_stats, self.unpriced = {'Files': config.files}, set()
        self.params = {}
        self.gradient_index = None
        self.output = {}

    def execute(self, params):
        pass

    @classmethod
    def runner_keys(cls, block):
        """The keys a run derives off its job's `block` and sets beneath it: none."""
        return {}

    @classmethod
    def parameters(cls, context, overrides=None):
        """The block a run of `context` reads, each step over the last: its `runner_keys`, the
        job's `Calculation` block (the system parameters' date and currency where it has none) and
        `overrides` merged deeply."""
        system = context.params['System Parameters']
        block = context.deals.get('Calculation', {
            'Base_Date': system['Base_Date'], 'Currency': system['Base_Currency']})
        params = dict(cls.runner_keys(block), **block)
        if overrides is not None:
            utils.update_dict(params, overrides)
        return params

    @classmethod
    def run(cls, context, overrides=None):
        """Runs this calculation on `context`, its block completed by `parameters`:
        `(calculation, output)`."""
        calc = cls(context)
        return calc, calc.execute(cls.parameters(context, overrides))

    @classmethod
    def refusals(cls, params):
        """What this calculation's declarations say against `params` completed as a run completes
        them (`declared_defaults`): a REQUIRED field left unstated, or a value that is not what its
        field declares."""
        return schema.validate_block(cls, declared_defaults(cls, params))

    @classmethod
    def judged(cls, params):
        """`params`, or a `ValueError` naming every one of its `refusals` at once."""
        refused = cls.refusals(params)
        if refused:
            raise ValueError('{}: {}'.format(cls.__name__, '; '.join(refused)))
        return params

    def factor_leaf(self, factor, current_val, requires_grad, offset=0.0):
        """The AAD leaf for `factor`, connected to a calibration where one exists.

        Every leaf is minted from a numpy array, which severs whatever produced those numbers: it
        does not raise, it reports a zero gradient. A curve - or one named parameter of a calibrated
        model - the library bootstrapped and kept the graph of is offered as
        `leaf + (theta - theta.detach())` instead: worth zero forward, derivative one, so the factor
        greek is the number it always was and dV/dq arrives in the same reverse sweep. A non-zero
        `offset` (Tenor_Offset) declines the attachment - a shifted curve is a different curve.

        Under `Quote_Propagation` the same seam changes the VALUE: the leaf is minted from the last
        artifact ridden to the quotes standing now, `theta* + dtheta/dq (q_now - q0)`, so a tick
        reaches a valuation without a re-solve. The ride is derived per call, never stored, and
        leaves the artifact id in `calc_stats['Calibrations']` - nothing else in the replay tuple
        distinguishes a ride from a refit. A `Tenor_Offset` REFUSES that ride rather than declining
        it: the shifted curve interpolates off coefficients fitted before the ride.
        """
        ridden = self.config.propagated_factor(factor)
        if ridden is not None:
            if offset:
                raise Exception(
                    'Quote_Propagation: {} carries a Tenor_Offset, so the curve the calculation '
                    'consumes is interpolated off coefficients fitted before the ride - riding it '
                    'would price a curve nobody solved for. Run the offset off a re-bootstrap, or '
                    'set Quote_Propagation to No.'.format(utils.check_tuple_name(factor)))
            current_val, artifact_id = ridden
            self.calc_stats.setdefault('Calibrations', {})[
                utils.check_tuple_name(factor)] = artifact_id
        leaf = torch.tensor(current_val, device=self.device, dtype=self.dtype,
                            requires_grad=requires_grad)
        theta = self.config.calibrated_factors.get(factor)
        if theta is None or not requires_grad or offset:
            return leaf
        theta = theta.to(device=self.device, dtype=self.dtype)
        connected = leaf + (theta - theta.detach())
        connected.retain_grad()
        return connected

    def leaves(self, moved, factors=None):
        """`[(key, value)]` - each leaf of `factors`, the compiled price factors where None, that
        the price factors `moved` write, keyed as the compile keys it, and the value written."""
        found, interp = [], self.config.params['Price Factor Interpolation']
        for factor in [f for f in (self.all_factors if factors is None else factors)
                       if utils.check_tuple_name(f) in moved]:
            found.extend(self.split(factor, construct_factor(
                factor, moved, interp, base_date=self.base_date).current_value()))
        return found

    @staticmethod
    def split(factor, value):
        """`[(key, value)]` - the leaves price factor `factor` standing at `value` compiles to: a
        surface stating parameters a leaf a parameter, any other factor its own."""
        return [(utils.Factor(factor.type, factor.name + (k,)), v) for k, v in value.items()] if (
            isinstance(value, dict)) else [(factor, value)]

    def make_factor_index(self, tensors):
        tenors = utils.get_tenors(self.all_factors)
        self.gradient_index = {}
        for name, var in tensors:
            factor_name = utils.check_scope_name(name)
            name_size, pad_size = tenors[factor_name].shape
            padding = 3 - pad_size
            indices = np.pad(tenors[factor_name], [[0, 0], [0, padding]], 'constant')
            self.gradient_index[factor_name] = (indices, pad_size)

    def gradients_as_df(self, grad, header='Gradient', display_val=False):
        if isinstance(grad, dict):
            factor_values = {utils.check_scope_name(k): v.factor if hasattr(v, 'factor') else v
                             for k, v in self.all_factors.items()} if display_val else {}
            hessian_index = ([], [])
            factor, rate, tenor, values = [], [], [], []
            for name, v in grad.items():
                # first derivative
                non_zero = np.where(v)[0]
                grad_index, index_len = self.gradient_index[name]
                hessian_index[0].append([name] * v.shape[0])
                hessian_index[1].append(grad_index)
                values.append(v[non_zero])
                rate.append([name] * non_zero.size)
                tenor.append(grad_index[non_zero])
                if display_val:
                    if name in factor_values:
                        rate_non_zero = grad_index[non_zero][:, :index_len].astype(np.float64)
                        factor_val = factor_values[name].current_value(rate_non_zero).flatten()
                    else:
                        sub_rate, param = name.rsplit('.', 1)
                        sub_rate_non_zero = grad_index[non_zero].shape[0]
                        factor_val = factor_values[sub_rate].current_value()[param].flatten()[:sub_rate_non_zero]

                    if factor_val.size == non_zero.size:
                        factor.append(factor_val)

            tenors = np.vstack(tenor)
            self.hessian_index = (np.hstack(hessian_index[0]), np.vstack(hessian_index[1]))

            data = {'Rate': np.hstack(rate), 'Tenor': tenors[:, 0], 'Tenor2': tenors[:, 1],
                    'Tenor3': tenors[:, 2], header: np.hstack(values)}
            index = ['Rate', 'Tenor', 'Tenor2', 'Tenor3']

            if display_val:
                data['Value'] = np.hstack(factor)

            df = pd.DataFrame(data).set_index(index).sort_index(level=[0, 1, 2, 3])
        else:
            # second derivative
            multi_index = []
            non_zero = np.where(grad)
            headers = [None, header]
            for extra_index, full_index in zip(headers, non_zero):
                index = np.unique(full_index)
                rate = self.hessian_index[0][index]
                tenor = self.hessian_index[1][index]
                m_index = [rate, tenor[:, 0], tenor[:, 1], tenor[:, 2]]
                if extra_index is not None:
                    m_index = [[extra_index] * rate.size] + m_index
                multi_index.append(pd.MultiIndex.from_arrays(m_index))

            values = grad[grad.any(axis=1)][:, grad.any(axis=0)]
            df = pd.DataFrame(values, index=multi_index[0], columns=multi_index[1]).sort_index(
                level=[0, 1, 2, 3], axis=0).sort_index(level=[0, 1, 2, 3], axis=1)

        return df

    def set_deal_structures(self, deals, output, unit, deal_level_mtm=False, exclude=None):
        """Compile the deal tree, binding each deal's schedules to `unit`, the calculation's dtype
        and device. The base currency is stamped here, a deal being constructed before the book it
        prices against is known, and so is whether it has children, which its node carries and its
        block does not; `Exclude_Deals_With_Missing_Market_Data` is read once, at the top: `Yes`,
        the default, drops a deal that cannot be read, `No` keeps it marked at zero,
        and a compile that only reads the book drops one whatever the document says. The job's
        `Valuation Configuration` is handed down for the history fold's substitute type. A node
        that never became a `Deal` refuses here, naming every such node at once."""
        system = self.config.params['System Parameters']
        base_currency = utils.check_rate_name(system['Base_Currency'])
        if exclude is None:
            exclude = not self.valuation or system.get(
                'Exclude_Deals_With_Missing_Market_Data', 'Yes') != 'No'
        valuation_options = self.config.params.get('Valuation Configuration', {})
        for node in deals:
            instrument = node['Instrument']
            if node.get('Ignore') == 'True':
                self.calc_stats['Ignored'] = self.calc_stats.setdefault('Ignored', 0) + 1
                continue
            if not isinstance(instrument, Deal):
                raise ValueError('; '.join(self.config.nameless_deals().values()))

            instrument.base_currency = base_currency
            instrument.has_children = bool(node.get('Children'))
            logging.root.name = instrument.field.get('Reference', '<undefined>')
            if node.get('Children'):
                struct = DealStructure(instrument, store_results=deal_level_mtm)
                self.set_deal_structures(node['Children'], struct, unit, deal_level_mtm, exclude)
                output.add_structure_to_structure(
                    struct, self.base_date, self.static_factors, self.stoch_factors, self.all_factors,
                    self.all_tenors, self.time_grid, self.config.holidays, self.calc_stats, unit)
                continue

            output.add_deal_to_structure(
                self.base_date, instrument, self.static_factors, self.stoch_factors, self.all_factors,
                self.all_tenors, self.time_grid, self.config.holidays, self.calc_stats, unit,
                valuation_options, exclude)


def batch_seed(random_seed, batch_index):
    """The seed a batch runs under: a SplitMix64 mix of (seed, global batch), not `seed + batch`.

    Deterministic sharding reseeds ONCE PER BATCH, so a 1024-batch job asks for 1024 seeds and
    consecutive integers are the weakest input to a generator's initialization. CUDA's Philox is
    counter-based and does not care; CPU MT19937 expands its 624-word state from the seed by a
    linear recurrence, where near seeds giving near initial states is not ruled out. One SplitMix64
    round costs three multiplies and closes it for both backends.

    ONE spelling, here, or a sharded run and an unsharded-but-batch-seeded run would derive the same
    batch's seed two ways. Masked to 63 bits: `torch.manual_seed` takes a signed 64-bit value.
    """
    z = (random_seed + (batch_index + 1) * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (z ^ (z >> 31)) & 0x7FFFFFFFFFFFFFFF


class CMC_State(utils.Calculation_State):
    def __init__(self, cholesky, static_buffer, batch_size, one, mcmc_sims, report_currency,
                 seed, quasi_seed, job_id, num_jobs, scale_survival=False, nomodel='Constant',
                 keep_tensor=False):
        """Per-calculation Monte Carlo state: correlated random numbers, scenario buffers and the
        caches a batched exposure run needs on top of `Calculation_State`.

        `t_Bridge_Variance_Rate` holds each factor's annualized log-variance RATE: a continuously
        monitored barrier crossing between two deal-grid dates needs the SIMULATION variance of that
        interval, not an implied vol.
        """
        super(CMC_State, self).__init__(
            static_buffer, one, mcmc_sims, report_currency, nomodel, batch_size, keep_tensor, quasi_seed)
        # decisions taken on simulated state, recorded forward so their derivative can be restored
        # before the reverse sweep. Per BATCH like t_Buffer - backward() runs once per batch
        self.boundary_aad = False
        self.boundary_sets = []
        self.t_Bridge_Variance_Rate = {}
        self.t_cholesky = cholesky
        self.t_random_numbers = None
        self.t_Scenario_Buffer = {}
        self.t_quasi_rng_batch = {}
        # Where each REQUESTED width's stream stands, and separately the engine cached per ENGINE
        # width with its own standing position - a chunked draw's two keys are not the same number.
        # Tracking position is what lets one `quasi_rng` serve both arms - see there.
        self.sobol_position = {}
        self.sobol_engine = {}
        # The GLOBAL index of the batch being run, set by `set_quasi_batch` under deterministic
        # sharding only. None leaves `quasi_rng` indexing by this process's own draw count.
        self.quasi_batch = None
        # seed each job by its offset
        torch.manual_seed(seed + job_id)
        self.job_id = job_id
        self.num_jobs = num_jobs
        self.scale_survival = scale_survival

    def quasi_rng(self, dimension, sample_size):
        """One quasi-random draw of `sample_size` points in `dimension` dimensions.

        SHARDING CHANGES ONLY WHERE THE DRAW IS TAKEN FROM. The stream is a scrambled Sobol sequence
        seeded by `Quasi_Seed` at a fixed offset, so it is reproducible; what is not is POSITION.
        The historical arm indexes by THIS PROCESS's own draw count and reads wherever its engine
        has reached, so a worker holding part of a job reads the points an earlier batch should
        have had. The anchored arm (`set_quasi_batch`, deterministic sharding only) indexes by the
        GLOBAL batch and reads `QUASI_ANCHOR + index * sample_size`.

        The two agree on an unsharded run of one draw per batch per shape - the engine advances by
        `sample_size` a batch, landing exactly on the anchored position - so anchoring is a
        repositioning rather than a second model. The historical arm still reads its engine's
        STANDING position rather than recomputing it: one dimension drawn at two sample sizes shares
        one engine and interleaves, and that interleaving must survive byte for byte.

        ABOVE the engine's `MAXDIM` the draw is taken in `span` dimension chunks at successive
        positions and concatenated, each chunk advancing the stream by `sample_size` exactly as one
        draw of that width does. A wide draw therefore CONSUMES `span * sample_size` of the stream,
        which is the stride both arms step by, or consecutive batches would read the same points.
        One chunk - every dimension at or below the cap - is the single draw unchanged, engine,
        position and bytes.

        The clamp and the icdf happen once, below, for both arms. THE DRAW IS TAKEN IN
        DOUBLE and returned three ways - `(icdf, u, u in double)` - the first two in the job's own
        dtype and bit-identical to a float32 draw, a Sobol point being a dyadic rational both
        conversions round the same way. A float32 uniform holds 24 bits, so `1 - u` at the clamp's
        own margin carries 3% error: the third is for a consumer that reads the TAIL, the
        inverse-Gaussian mixer being the one that does.
        """
        batch_key = (dimension, sample_size)
        call = self.t_quasi_rng_batch.setdefault(batch_key, 0)
        cap = torch.quasirandom.SobolEngine.MAXDIM
        span = -(-dimension // cap)

        if self.quasi_batch is None:
            position = self.sobol_position.get(dimension, self.QUASI_ANCHOR)
        else:
            if call:
                # a genuine second draw of one shape inside one batch, which has no distinct batch
                # position
                raise RuntimeError(
                    'deterministic sharding cannot cover a second draw of one quasi-random stream '
                    'inside a batch: (dimension={}, sample_size={}) was asked for a {} time while '
                    'running global batch {}, and anchoring gives a draw the position of its '
                    'BATCH, so two of them in one batch have no distinct position to take. Two '
                    'factors sharing a shape is the usual source - run such a job unsharded, or '
                    'give the second consumer its own dimension.'.format(
                        dimension, sample_size, call + 1, self.quasi_batch))
            position = self.QUASI_ANCHOR + self.quasi_batch * span * sample_size

        chunks, at = [], position
        for start in range(0, dimension, cap):
            engine = self._sobol_at(min(cap, dimension - start), at)
            chunks.append(engine.draw(sample_size, dtype=torch.float64))
            at += sample_size
            self.sobol_engine[engine.dimension] = (engine, at)
        self.sobol_position[dimension] = at
        sample_sobol = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=1)
        margin = 1.0e-6
        u = sample_sobol.clamp(min=margin, max=1.0 - margin).to(self.one.device)
        self.t_quasi_rng_batch[batch_key] += 1
        return utils.norm_icdf(u.to(self.one.dtype)), u.to(self.one.dtype), u

    def _sobol_at(self, dimension, position):
        """This dimension's engine, standing at `position`.

        Seeks FORWARD and rebuilds to go backwards. The historical arm only ever asks for the
        position it already stands at, so it never seeks; a sharded worker pays the jump to the
        start of its own slice once and then advances the same way.
        """
        engine, standing = self.sobol_engine.get(dimension, (None, self.QUASI_ANCHOR))
        if engine is None or position < standing:
            engine = torch.quasirandom.SobolEngine(
                dimension=dimension, scramble=True, seed=self.quasi_seed)
            engine.fast_forward(position)
        elif position > standing:
            engine.fast_forward(position - standing)
        return engine

    def set_quasi_batch(self, batch_num):
        """Anchor the quasi stream to a GLOBAL batch index (deterministic sharding only).

        Resets the per-`(dimension, sample_size)` counter along with it, because under anchoring
        that counter means "draws so far WITHIN this batch" - which is what makes a repeated draw
        detectable instead of silently aliasing onto the next batch's position.
        """
        self.quasi_batch = batch_num
        self.t_quasi_rng_batch = {}

    def reset_cashflows(self, time_grid):
        self.t_Cashflows = {k: {t_i: self.one.new_zeros(self.simulation_batch)
                                for t_i in np.where(v >= 0)[0]} for k, v in time_grid.CurrencyMap.items()}

    def save_cashflows(self, output, time_grid):
        dates = np.array(sorted(time_grid.mtm_dates))
        for currency, values in self.t_Cashflows.items():
            rows = [v for _, v in sorted(values.items())]
            # one stacked copy off the device and one block for pandas, not a column at a time
            output.setdefault('cashflows', {}).setdefault(currency, []).append(pd.DataFrame(
                torch.stack(rows).detach().cpu().numpy() if rows else None,
                index=dates[sorted(values.keys())]))

    @staticmethod
    def save_results(output, tensors):
        for k, v in tensors.items():
            output.setdefault(k, []).append(v.detach().cpu().numpy().astype(np.float64))

    def reset(self, num_factors, time_grid: utils.TimeGrid, use_antithetic=False):
        sample_size = self.simulation_batch // 2 if use_antithetic else self.simulation_batch
        correlated_sample = torch.matmul(
            self.t_cholesky, torch.randn(
                num_factors, sample_size * time_grid.scen_time_grid.size,
                dtype=self.one.dtype, device=self.one.device)
        ).reshape(num_factors, time_grid.scen_time_grid.size, -1)

        if use_antithetic:
            self.t_random_numbers = torch.concat([correlated_sample, -correlated_sample], dim=-1)
        else:
            self.t_random_numbers = correlated_sample

        self.reset_cashflows(time_grid)

        self.t_Buffer.clear()
        self.boundary_sets.clear()


class CMC_State_Inner(CMC_State):
    """Inner-MC variant of `CMC_State` for nested simulation: each of `simulation_batch` outer-path
    states fans out into `simulation_sub_batch` (B2) independent forward paths.

    `reset()` is inherited unchanged, so outer-mode use is transparent; `reset_inner()` swaps in
    random numbers shaped `(num_factors, T, simulation_batch, simulation_sub_batch)`. Stochastic
    processes dispatch on `Z.ndim` to pick the outer or the inner path."""

    def __init__(self, cholesky, static_buffer, batch_size, one, mcmc_sims, report_currency,
                 seed, quasi_seed, job_id, num_jobs, simulation_sub_batch=0,
                 scale_survival=False, nomodel='Constant', keep_tensor=False):
        super().__init__(cholesky, static_buffer, batch_size, one, mcmc_sims, report_currency,
                         seed, quasi_seed, job_id, num_jobs, scale_survival=scale_survival,
                         nomodel=nomodel, keep_tensor=keep_tensor)
        # 0 (default) = inner mode unused; base `reset()` works, `reset_inner()` raises.
        self.simulation_sub_batch = simulation_sub_batch

    def reset_inner(self, num_factors, time_grid: utils.TimeGrid, use_antithetic=False,
                    use_random=False):
        """Draw the inner-mode correlated Gaussians, shaped
        `(num_factors, T, simulation_batch, simulation_sub_batch)`, and clear the per-batch caches.

        `use_random` (`Inner_Draws='random'`) swaps the shared Sobol tensor for iid Gaussians: one
        low-discrepancy stream strided across (T,B,B2) loses its uniformity on the per-(t,b)
        B2-slices as B grows (measured at B=512), while iid draws keep per-fork label noise
        B-independent.

        `use_antithetic` (`Inner_Antithetic='Yes'`) draws B2/2 quasi-normals per (t, outer path) and
        mirrors them (z, -z), halving the label/argmax variance of the E[C] estimate. Unbiased
        because the folded emissions are symmetric in z; auxiliary streams stay iid.
        """
        if self.simulation_sub_batch <= 1:
            raise ValueError(
                f'reset_inner requires simulation_sub_batch > 1; got {self.simulation_sub_batch}. '
                f'Pass a positive Inner_Sub_Batch in params to enable nested simulation.')
        T = time_grid.scen_time_grid.size
        B = self.simulation_batch
        B2 = self.simulation_sub_batch
        if use_antithetic and B2 % 2:
            raise ValueError(f'Inner_Antithetic requires an even Inner_Sub_Batch; got {B2}.')
        if use_random:
            # iid inner draws: no cross-(B,B2) coupling, unlike one Sobol stream strided over them
            half = B2 // 2 if use_antithetic else B2
            z = torch.randn(num_factors, T, B, half, dtype=self.one.dtype, device=self.one.device)
            z = torch.einsum('fg,gtbi->ftbi', self.t_cholesky, z)
            self.t_random_numbers = torch.cat([z, -z], dim=-1) if use_antithetic else z
            self.reset_cashflows(time_grid)
            self.t_Buffer.clear()
            return
        # Sobol-based correlated Gaussian: draw T*B*B2 quasi-normal vectors of dim num_factors,
        # transpose to (num_factors, T*B*B2), correlate via cholesky, reshape.
        if use_antithetic:
            Z_normal = self.quasi_rng(num_factors, T * B * (B2 // 2))[0]
            half = torch.matmul(
                self.t_cholesky, Z_normal.transpose(0, 1)
            ).reshape(num_factors, T, B, B2 // 2)
            self.t_random_numbers = torch.cat([half, -half], dim=-1)
        else:
            Z_normal = self.quasi_rng(num_factors, T * B * B2)[0]                    # (T*B*B2, num_factors)
            self.t_random_numbers = torch.matmul(
                self.t_cholesky, Z_normal.transpose(0, 1)
            ).reshape(num_factors, T, B, B2)

        self.reset_cashflows(time_grid)
        self.t_Buffer.clear()


class CreditMonteCarlo(Calculation):
    documentation = ('Calculations', [
        'A credit Monte Carlo simulates every price factor that has a model along `Batch_Size` x',
        '`Simulation_Batches` paths, values the book on each path at each date of its grid, and reduces the',
        'paths to an exposure profile and, where asked, the xVA. Each reduction\'s estimator, inputs, defaults',
        'and known limitations are specified on the [xVA](../theory/valuation_adjustments.md) page; the grids and the random',
        'numbers on [Calc Lifecycle](../developer/calc_lifecycle.md#grids).',
        '',
        'A profile is a curve $V(t)$ with values at a discrete set of future dates $0=t_0<t_1<...<t_m$. A',
        'deal is valued on its own dates and interpolated linearly onto the run\'s MTM grid, and is zero past',
        'its last date. The profile dates are the union of:',
        '',
        '- the dates of the **Time Grid** up to the book\'s last date,',
        '- each deal\'s own revaluation dates (payment, reset and exercise dates).',
        '',
        'Deal specific dates improve the accuracy of the profile by showing the effect of cashflows,',
        'exercises etc.',
        '',
        '### Aggregation',
        '',
        'If $U$ and $V$ are profiles, then $U+V$ is defined on the union of their dates. If $E$ is the',
        'credit exposure profile in the reporting currency (**Currency**), then:',
        '',
        '$$E = \\sum_{d} V_d$$',
        '',
        'where $V_d$ is the profile of the $d^{th}$ top-level deal or netting set, summed path by path.',
        'Everything in one run is netted together, so one run is one counterparty.',
        '',
        '#### Peak Exposure',
        '',
        'The simulated value $E(t)$ at each **Percentile** $q$ (in percent, comma-separated, default `95`),',
        'reported per date as `PFE_q`. It is a percentile of the signed value.',
        '',
        '#### Expected Exposure',
        '',
        'This is the profile defined by taking the average of the positive simulated exposures i.e. for each',
        'profile date $t$,',
        '',
        '$$\\bar E(t)=\\frac{1}{N}\\sum_{k=1}^N \\max(E_k(t),0),$$',
        '',
        'reported as `EE` beside the expected negative exposure `ENE`.',
        '',
        '#### Exposure Deflation',
        '',
        'Exposure at time $t$ is simulated in units of the time $t$ reporting currency. Exposure deflation',
        'converts this to time $0$ reporting currency i.e.',
        '',
        '$$V^*(t)=\\frac{V(t)}{\\beta(t)},\\qquad \\beta(t)=\\exp\\Big(\\int_0^t r(s)ds\\Big),$$',
        '',
        '$r$ the short rate of **Deflation Interest Rate**. With **Deflate Stochastically** `Yes` the run',
        'approximates it on the report rows by',
        '',
        '$$\\frac{1}{\\beta(t_i)}\\approx\\prod_{j=1}^{i}D(t_j,t_j+\\Delta_j),\\qquad \\Delta_j=t_j-t_{j-1},$$',
        '',
        'each discount factor read off the curve as simulated at $t_j$; with `No` it uses $D(0,t_i)$ from',
        'today\'s curve.',
        '',
        '### Credit Valuation Adjustment',
        '',
        'This represents the adjustment to the market value of the portfolio accounting for the risk of',
        'default. Only unilateral CVA (i.e accounting for the counterparty risk of default but ignoring the',
        'potential default of the investor) is calculated. It is given by:',
        '',
        '$$C=\\Bbb E(L(\\tau)),$$',
        '',
        'where the expectation is taken with respect to the risk-neutral measure, and',
        '',
        '$$L(t)=(1-R)\\max(E^*(t),0),$$',
        '',
        'with:',
        '',
        '- $R$ the counterparty recovery rate',
        '- $\\tau$ the counterparty time to default',
        '- $E^*(t)$ the exposure at time $t$ deflated by the money market account, zero past the end $T$ of',
        '  the portfolio\'s exposure.',
        '',
        'Now,',
        '',
        '$$\\Bbb E(L(\\tau))=\\Bbb E\\Big(\\int_0^T L(t)(-dH(t))\\Big),\\qquad H(t)=\\exp\\Big(-\\int_0^t h(u)du\\Big)$$',
        '',
        'where $h(t)$ is the hazard rate, and on the report rows $0=t_0<...<t_m=T$',
        '',
        '$$C\\approx\\sum_{i=1}^m \\Bbb E\\Big[\\Big(\\frac{L(t_{i-1})+L(t_i)}{2}\\Big)(H_{i-1}-H_i)\\Big].$$',
        '',
        'If **Stochastic Hazard Rates** is `No`, $H_i=S(0,t_i)$, the risk neutral survival probability to',
        '$t_i$, the same on every path. If it is `Yes`,',
        '',
        '$$H_i=\\prod_{j=1}^{i}S(t_j,t_j+\\Delta_j),$$',
        '',
        'each factor the survival probability as simulated at $t_j$. Note that the factor models used should',
        'be risk-neutral to give risk neutral simulations of $E^*(t)$.',
        '',
        '### Funding Valuation Adjustment',
        '',
        'Not posting (or receiving) collateral can imply a funding cost (or benefit) when there is a spread',
        'between a party\'s internal cost of funding and the rate that would be received should the',
        'counterparty place collateral. The discounted expectation of this cost (or benefit) summed across',
        'all time horizons and scenarios constitutes a funding value adjustment and can be expressed as:',
        '',
        '$$FCA=\\int_{0}^{T} \\Bbb{E}\\Big(\\max(V(t),0)[f_{fc,C}(t)-f_{rf,C}(t)]SP_C(t)\\Big)dt$$',
        '',
        '$$FBA=\\int_{0}^{T} \\Bbb{E}\\Big(\\min(V(t),0)[f_{fb,C}(t)-f_{rf,C}(t)]SP_C(t)\\Big)dt$$',
        '',
        'where',
        '',
        '- $T$ is the exposure horizon',
        '- $V(t)$ is the deflated funding profile at time $t$',
        '- $f_{fc,C}(t)$ and $f_{fb,C}(t)$ are the forward rates of **Funding Cost Interest Curve** and',
        '  **Funding Benefit Interest Curve**',
        '- $f_{rf,C}(t)$ is the forward rate of **Risk Free Curve**',
        '- $SP_C(t)=S(0,t)$ is the survival probability of the counterparty each netting set names under',
        '  **Credit Support Amounts**.',
        '',
        'FVA against the counterparty is then calculated as $FVA = FCA + FBA$, positive for a net cost. The',
        'run integrates on the report rows by the trapezoid rule on the exposure, each interval\'s spread',
        'accrued from the interval\'s start.',
        '',
        'Over several netting sets the survival weights apply set by set before the sum:',
        '',
        '$$FCA_{bank}=\\int_{0}^{T} \\Bbb{E}\\Bigg(\\max\\Big(\\sum_i V_i(t)SP_{C_i}(t),0\\Big)[f_{fc,C}(t)-f_{rf,C}(t)]\\Bigg)dt$$',
        '',
        '$$FBA_{bank}=\\int_{0}^{T} \\Bbb{E}\\Bigg(\\min\\Big(\\sum_i V_i(t)SP_{C_i}(t),0\\Big)[f_{fb,C}(t)-f_{rf,C}(t)]\\Bigg)dt$$',
        '',
        '$i$ the netting set: while FVA is on, every set\'s reported value, the exposure profile included,',
        'carries its survival weight.',
        '',
        'The collateral valuation adjustment (**Collateral Valuation Adjustment**) and the initial margin',
        '(**Initial Margin**) are experimental and are not specified.',
        '',
        'Calculation parameters are extended from the Base Valuation with these fields:',
        '',
        ' - **Deflation Interest Rate** - the interest rate price factor that PVs the exposure to today',
        ' - **Batch Size** - the number of simulations per batch (default 1024). Smaller batch sizes are more',
        '   likely to fit in memory; larger ones run quicker. Usually a power of 2.',
        ' - **Simulation Batches** - the number of batches to run (default 1). The total number of',
        '   simulations is **Simulation Batches** * **Batch Size**.',
        ' - **Antithetic** - each batch draws **Batch Size**/2 normals and uses them and their negatives,',
        '   so the number of simulations is unchanged: path $j$ and path $j$ + **Batch Size**/2 are a pair',
        '   (an even **Batch Size** is needed)',
        ' - **MCMC Simulations** - inner paths of a Monte Carlo pricer (default 2048, where the Base',
        '   Valuation declares 16,384)',
        ' - **Calc Scenarios** - return the simulated price factors: every path, or under **At_Percentile**',
        '   each factor and the netting set\'s mtm along the path nearest each **Percentile** of the profile',
        '   at every report date, for backtesting',
        ' - **Dynamic Scenario Dates** - generate scenarios not just on the **Time Grid**, but also on all',
        '   potential cashflow settlement dates (default `Yes`). Needed to accurately calculate liquidity and',
        '   settlement dynamics on collateralized portfolios; **At_Percentile** turns it on, every report',
        '   date being a scenario date',
        ' - **Generate Cashflows** - returns the simulated cashflows during the simulation period (default',
        '   `Yes`)',
        ' - **Credit Valuation Adjustment**, **Funding Valuation Adjustment** - one block per adjustment;',
        '   their fields and defaults are tabulated on the',
        '   [xVA](../theory/valuation_adjustments.md#inputs-and-defaults) page'
    ])

    fields = [
        F('Base_Date', 'Date', default=''),
        F('Currency', 'Text', default=REQUIRED),
        F('Time_Grid', 'Text', default='0d 2d 1w(1w) 3m(1m) 2y(3m)'),
        F('Deflation_Interest_Rate', 'Text', default='ZAR-SWAP'),
        F('Percentile', 'Text', default='95'),
        F('MCMC_Simulations', 'Integer', default=2048),
        F('Simulation_Batches', 'Integer', default=1),
        F('Batch_Size', 'Integer', default=1024),
        Calculation.RANDOM_SEED,
        Calculation.QUASI_SEED,
        Calculation.DETERMINISTIC_KERNELS,
        F('Tenor_Offset', 'Float', default=0.0,
          description='Years to shift every factor tenor by before the run'),
        F('Antithetic', 'Text', default='No', values=['Yes', 'No']),
        F('Calc_Scenarios', 'Text', default='No', values=['At_Percentile', 'All', 'No'],
          description='Report the simulated factors on every path, or each factor and the netting '
                      "set's mtm along the path nearest each Percentile of the profile at every "
                      'report date'),
        F('Dynamic_Scenario_Dates', 'Text', default='Yes', values=['Yes', 'No'],
          description='Simulate on every reset and settlement date beside the Time_Grid; Yes under '
                      'Calc_Scenarios At_Percentile'),
        F('Generate_Cashflows', 'Text', default='Yes', values=['Yes', 'No']),
        F('Keep_Tensor', 'Text', default='No', values=['Yes', 'No'],
          description='Keep the simulated mtm tensor on the device after the run'),
        F('NoModel', 'Text', default='Constant', values=['Constant', 'RiskNeutral'],
          description='How a factor with no model is read: Constant holds it at its value today, '
                      'RiskNeutral reads a static curve at each row as its forward from that row; '
                      'a spot or a price index with no model is constant under either'),
        F('Gradient_Variables', 'Text', default='All', values=['All', 'Factors', 'Implied'],
          description='Which leaves the sensitivity engine differentiates'),
        F('Boundary_AAD_Bandwidth', 'Float', default=0.01,
          description='Kernel bandwidth of the boundary correction assembled into backward()'),
        F('Boundary_AAD_Window_Touch', 'Text', default='Yes', values=['Yes', 'No'],
          description='Register the partial-time barrier\'s window-touch decision as a boundary '
                      'latch on the endpoint branch; `No` is the unregistered estimator'),
        F('Recompute_Inner_MC', 'Text', default='No', values=['Yes', 'No'],
          description='Re-simulate a Monte Carlo pricer\'s inner paths in backward() rather than '
                      'taping them; trades a second forward pass for the graph of every pricing'),
        F('Checkpoint_Outer_Walk', 'Text', default='Yes', values=['Yes', 'No'],
          description='Checkpoint the SCENARIO generator\'s own internal walk, recomputing each '
                      'segment\'s intermediates and redrawing its shocks in backward() instead of '
                      'taping them'),
        F('Credit_Valuation_Adjustment', 'Container',
          sub_fields=[
              F('Calculate', 'Text', default='No', values=['Yes', 'No']),
              F('Counterparty', 'Text', default=''),
              F('Bank', 'Text', default=''),
              F('Deflate_Stochastically', 'Text', default='Yes', values=['Yes', 'No']),
              F('Stochastic_Hazard_Rates', 'Text', default='No', values=['Yes', 'No']),
              F('Gradient', 'Text', default='No', values=['Yes', 'No']),
              F('Hessian', 'Text', default='No', values=['Yes', 'No'],
                description='Second derivatives of the CVA as well as the first, read only beside '
                            'Gradient: Yes'),
              F('CDS_Tenors', 'Container', default=[],
                description='Tenors in years to add to the survival curve so CDS rates can be '
                            'interpolated off it')]),
        F('Funding_Valuation_Adjustment', 'Container',
          sub_fields=[
              F('Calculate', 'Text', default='No', values=['Yes', 'No']),
              F('Counterparty', 'Text', default=''),
              F('Bank', 'Text', default=''),
              F('Risk_Free_Curve', 'Text', default=''),
              F('Funding_Cost_Interest_Curve', 'Text', default=''),
              F('Funding_Benefit_Interest_Curve', 'Text', default=''),
              F('Deflate_Stochastically', 'Text', default='Yes', values=['Yes', 'No']),
              F('Stochastic_Funding', 'Text', default='No', values=['Yes', 'No']),
              F('Gradient', 'Text', default='No', values=['Yes', 'No'])]),
        F('Collateral_Valuation_Adjustment', 'Container',
          sub_fields=[
              F('Calculate', 'Text', default='No', values=['Yes', 'No']),
              F('Gradient', 'Text', default='No', values=['Yes', 'No'])]),
        F('Initial_Margin', 'Container',
          description='The LCH-style initial margin add-on',
          sub_fields=[
              F('Calculate', 'Text', default='No', values=['Yes', 'No']),
              F('Liquidity_Weights', 'Text', default='',
                description='Path to the liquidity-weight csv, indexed by tenor bucket'),
              F('IRS_Weights', 'Text', default='',
                description='Path to the delta-weight csv, indexed by tenor bucket'),
              F('Local_Currency', 'Text', default='',
                description='The currency whose curves are charged the local weights'),
              F('IM_Currency', 'Text', default='',
                description='Currency the margin is reported in'),
              F('Delta_Factor', 'Float', default=1.0,
                description='Multiplier on the delta charge before the liquidity add-on')])
    ]

    def __init__(self, config, **kwargs):
        super(CreditMonteCarlo, self).__init__(config, **kwargs)
        self.reset_dates = None
        self.settlement_currencies = None
        self.jacobians = {}
        self.implied_factors = {}
        self.implied_var = {}

        self.all_var = None

    @classmethod
    def runner_keys(cls, block):
        """`Time_grid`, the block's `Time_Grid` or its declared default as text, and `Run_Date`,
        its `Base_Date` as `YYYY-MM-DD`."""
        return {'Time_grid': str(declared_defaults(cls, block)['Time_Grid']),
                'Run_Date': block['Base_Date'].strftime('%Y-%m-%d')}

    @classmethod
    def run(cls, context, overrides=None, runparallel=False, prec=torch.float32, device=None,
            job_id=0, num_jobs=1, res_queue=None, deterministic_batches=False):
        """Runs a credit Monte Carlo on `context`: `(calculation, output)`, the exposure profile
        under `output['Results']['exposure_profile']`, or shard `job_id` of `num_jobs`, its answer
        put on `res_queue`.

        `runparallel` runs `worker_count(runparallel)` such shards in processes of their own, each
        a contiguous range of the batches seeding every batch from its global index
        (`deterministic_batches`), so the answer does not depend on the worker count, and returns
        a dict of lists, one entry per worker in worker order under each of `Results`, `Stats`,
        `Params`, `Reference` and `Job`: pool the `mtm` and re-summarize, a `PFE_<p>` percentile
        not pooling by averaging. `device` overrides `utils.calculation_device`'s choice, a
        property of the box and not of the trade, as `prec` is.
        """
        if runparallel:
            count, queue = worker_count(runparallel), mp.Queue()
            # the queue hands them back in completion order, which is a race; the merge is keyed
            # by worker so a pooled reduction reads its operands in the same order every run
            answers = gathered([mp.Process(
                target=partial(shielded, cls.run, job, queue), args=(context, overrides), kwargs={
                    'prec': prec, 'device': device, 'job_id': job, 'num_jobs': count,
                    'res_queue': queue, 'deterministic_batches': True})
                for job in range(count)], queue)
            queue.close()
            merged = {}
            for answer in answers:
                for key, value in answer.items():
                    merged.setdefault(key, []).append(value)
            return merged

        device = utils.calculation_device(device, job_id)
        params = cls.judged(cls.parameters(context, overrides))
        if params.get('Credit_Valuation_Adjustment', {}).get('Calculate', 'No') == 'Yes':
            cva_sect = params['Credit_Valuation_Adjustment']
            if cva_sect.get('CDS_Tenors'):
                # the CDS dates join the survival curve's knots, its last hazard carried on past
                # them, so the CDS rates read a curve that runs to their dates - on the run's own
                # copy of the factor, the context it runs on being written to by nothing
                name = 'SurvivalProb.{}'.format(cva_sect['Counterparty'])
                survivalprob = context.params['Price Factors'][name]
                daycount = lambda time_in_days: utils.DayCount.accrual(
                    params['Base_Date'], time_in_days, utils.DayCount.ACT365)
                to_add = [daycount((x - params['Base_Date']).days) for x in utils.cds_dates(
                    params['Base_Date'], max(cva_sect.get('CDS_Tenors')) * 12)]
                new_terms = np.union1d(to_add, survivalprob['Curve'].array[:, 0])
                curve = utils.Curve(survivalprob['Curve'].meta, np.array(list(zip(
                    new_terms, riskfactors.SurvivalProb(survivalprob).current_value(new_terms)))))
                context = copy.copy(context)
                context.params = dict(context.params, **{'Price Factors': dict(
                    context.params['Price Factors'], **{name: dict(survivalprob, Curve=curve)})})

        calc = cls(context, device=device, prec=prec)
        out = calc.execute(params, job_id, num_jobs, deterministic_batches)
        calc.report_device_memory(out['Stats'])
        out['Results']['exposure_profile'] = summarize_data(
            out['Results']['mtm'], params.get('Percentile', '95').replace(' ', ''))

        # what this worker RAN, not what the document asked for: execute() rebinds params through
        # declared_defaults before `//= num_jobs`, so the division never reached this dict; the
        # identity where num_jobs is 1, which is every unsharded run
        params['Simulation_Batches'] = out['Stats']['Simulation_Batches']

        if res_queue is not None:
            res_queue.put({'Results': out['Results'], 'Stats': out['Stats'], 'Params': params,
                           'Reference': context.deals['Attributes']['Reference'], 'Job': job_id})

        return calc, out

    def update_factors(self, params, base_date, job_id, num_jobs):
        dependent_factors, stochastic_factors, implied_factors, reset_dates, settlement_currencies = \
            self.config.calculate_dependencies(params, base_date, self.input_time_grid)

        # a percentile's path is picked off the profile at every report date, so its state is wanted
        # on every one of them: the scenario grid is the mtm grid
        percentile = params['Calc_Scenarios'] == 'At_Percentile'
        if percentile and params['Dynamic_Scenario_Dates'] != 'Yes':
            logging.info('Calc_Scenarios At_Percentile simulates every report date: Dynamic_Scenario_Dates is Yes')
        self.update_time_grid(base_date, reset_dates, settlement_currencies,
                              dynamic_scenario_dates=percentile or params['Dynamic_Scenario_Dates'] == 'Yes')

        shared_mem = self._build_factor_state(
            dependent_factors, stochastic_factors, implied_factors, params, base_date, job_id, num_jobs)
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        return shared_mem

    def device_memory(self):
        """What this run held on its device at its peak, reserved, beside what the device has - or
        None on the host, which has nothing to page."""
        if self.device.type != 'cuda':
            return None
        _, total = torch.cuda.mem_get_info(self.device)
        return {'Peak_GB': torch.cuda.max_memory_reserved(self.device) / 2 ** 30,
                'Device_GB': total / 2 ** 30}

    def report_device_memory(self, stats):
        """`Device_Memory` into `stats`, with a warning where the peak passed the device: the driver
        pages such a batch over the bus rather than refuse it, at many times the wall."""
        memory = self.device_memory()
        if memory is None:
            return
        stats['Device_Memory'] = memory
        if memory['Peak_GB'] > memory['Device_GB']:
            logging.warning(
                'Batch_Size {} held {:.2f} GB at its peak on a {:.2f} GB device, so the driver paged '
                'it over the bus: a smaller Batch_Size with more Simulation_Batches runs the same '
                'paths in a fraction of the time'.format(
                    stats.get('Batch_Size'), memory['Peak_GB'], memory['Device_GB']))

    def _build_factor_state(self, dependent_factors, stochastic_factors, implied_factors,
                            params, base_date, job_id, num_jobs):
        """Construct the factor objects, tensors and shared memory, and precalculate the stochastic
        processes. Called by `update_factors` once the time grid and dependency sets are known;
        subclasses that build their own dependency sets can call it directly.

        IMPLIED-LEAF INVARIANT: a factor can be BOTH a static dependent factor and a spot process's
        implied factor. Minting a second leaf under the scope name the implied leaf already owns
        would split the gradient - the pricer and the scenario path would read different tensors -
        so the single implied leaf is reused and `backward()` sums both consumers into it.

        `_factor_precalc_args` caches each factor's `(ScenarioTimeGrid, implied_tensor)` so a
        consumer needing a per-path initial state can precalculate again.
        """
        self.stoch_factors.clear()

        for price_model, price_factor in stochastic_factors.items():
            factor_obj = construct_factor(
                price_factor, self.config.params['Price Factors'],
                self.config.params['Price Factor Interpolation'],
                base_date=base_date)
            implied_factor = implied_factors.get(price_model)
            try:
                if implied_factor:
                    implied_obj = construct_factor(
                        implied_factor, self.config.params['Price Factors'],
                        self.config.params['Price Factor Interpolation'],
                        base_date=base_date)
                    self.implied_factors[implied_factor] = implied_obj
                else:
                    implied_obj = None
            except KeyError as e:
                logging.error('Implied Factor {0} missing in market data file'.format(e.args))

            self.stoch_factors[price_factor] = construct_process(
                price_model.type, factor_obj,
                self.config.params['Price Models'].get(utils.check_tuple_name(price_model)),
                implied_obj)

        self.static_factors = {}
        for price_factor in set(dependent_factors).difference(stochastic_factors.values()):
            try:
                self.static_factors.setdefault(
                    price_factor, construct_factor(
                        price_factor, self.config.params['Price Factors'],
                        self.config.params['Price Factor Interpolation'],
                        base_date=base_date)
                )
            except KeyError as e:
                logging.warning('Price Factor {0} missing in market data file - skipping'.format(e.args))

        self.all_factors = self.stoch_factors.copy()
        self.all_factors.update(self.static_factors)
        self.all_factors.update(self.implied_factors)
        self.num_factors = sum([v.num_factors() for v in self.stoch_factors.values()])

        tenor_offset = params.get('Tenor_Offset', 0.0)
        greeks = bool(np.any([params[k].get('Gradient', 'No') == 'Yes' for k, v in params.items() if type(v) == dict]))
        sensitivities = params.get('Gradient_Variables', 'All')

        for key, value in self.stoch_factors.items():
            if key.type not in utils.DimensionLessFactors:
                if hasattr(value, 'implied'):
                    vars = {}
                    calc_grad = greeks and sensitivities in ['All', 'Implied']
                    for param_name, param_value in value.implied.current_value().items():
                        factor_name = utils.Factor(value.implied.__class__.__name__, key.name + (param_name,))
                        vars[factor_name] = self.factor_leaf(factor_name, param_value, calc_grad)
                    self.implied_var[key] = vars

                if tenor_offset:
                    factor_tenor_offset = utils.DayCount.accrual(
                        base_date, tenor_offset, value.factor.get_day_count() if hasattr(
                            value.factor, 'get_day_count') else utils.DayCount.ACT365)
                else:
                    factor_tenor_offset = 0.0

                current_val = value.factor.current_value(offset=factor_tenor_offset)
                calc_grad = greeks and sensitivities in ['All', 'Factors']
                self.stoch_var[key] = self.factor_leaf(
                    key, current_val, calc_grad, factor_tenor_offset)

        calc_grad = greeks and sensitivities in ['All', 'Factors']
        # reuse the single implied leaf (implied-leaf invariant) - never mint a second one here
        implied_leaves = {fk: t for vars in self.implied_var.values() for fk, t in vars.items()}
        for key, value in self.static_factors.items():
            if key.type not in utils.DimensionLessFactors:
                if tenor_offset:
                    factor_tenor_offset = utils.DayCount.accrual(
                        base_date, tenor_offset, value.get_day_count() if hasattr(
                            value, 'get_day_count') else utils.DayCount.ACT365)
                else:
                    factor_tenor_offset = 0.0
                for leaf, current_val in self.split(
                        key, value.current_value(offset=factor_tenor_offset)):
                    self.static_var[leaf] = implied_leaves[leaf] if leaf in implied_leaves else \
                        self.factor_leaf(leaf, current_val, calc_grad, factor_tenor_offset)

        # beside the seed, and set BOTH ways every run: the switch is process-global, so a job
        # that declared Yes must not leave the next job in the process pinned
        torch.use_deterministic_algorithms(params['Deterministic_Kernels'] == 'Yes', warn_only=True)
        quasi_seed = self.calc_stats['Quasi_Seed'] = int(params['Quasi_Seed'])
        shared_mem = self._init_shared_mem(
            int(params['Random_Seed']), quasi_seed, params['NoModel'],
            params['Currency'], params['MCMC_Simulations'],
            job_id, num_jobs, calc_greeks=sensitivities if greeks else None)
        shared_mem.calc_stats, shared_mem.unpriced = self.calc_stats, self.unpriced

        self.all_tenors = utils.update_tenors(self.base_date, self.all_factors)

        self._factor_precalc_args = {}
        for key, value in self.stoch_factors.items():
            if key.type not in utils.DimensionLessFactors:
                if key in self.implied_var:
                    implied_tensor = {k.name[-1]: v for k, v in self.implied_var[key].items()}
                    value.link_references(implied_tensor, self.implied_var, self.implied_factors)
                else:
                    implied_tensor = None
                # its own factor key, so it can publish auxiliaries to t_Scenario_Buffer under the
                # documented (factor_key, kind) convention
                value.factor_key = key
                scenario_grid = ScenarioTimeGrid(dependent_factors[key], self.time_grid, base_date)
                self._factor_precalc_args[key] = (scenario_grid, implied_tensor)
                value.precalculate(
                    base_date, scenario_grid,
                    self.stoch_var[key], shared_mem, self.process_ofs[key], implied_tensor=implied_tensor)
                if not value.params_ok:
                    logging.warning('Stochastic factor {} has been modified'.format(utils.check_scope_name(key)))
                # a process with no lognormal interval law returns None and does not appear, which
                # a pricer treats as "observe the endpoints"
                variance_rate = value.bridge_variance_rate
                if variance_rate is not None:
                    shared_mem.t_Bridge_Variance_Rate[key] = variance_rate

        for key, value in self.stoch_factors.items():
            if key.type not in utils.DimensionLessFactors:
                value.calc_references(key, self.static_factors, self.stoch_factors, self.all_tenors, self.all_factors)

        return shared_mem

    def update_time_grid(self, base_date, reset_dates, settlement_currencies, dynamic_scenario_dates=False):
        dynamic_dates = set([x for x in reset_dates if x > base_date])
        if not dynamic_dates:
            raise ValueError(
                'no deal of this book has a date after the base date {} - a simulation grid is '
                'grown from the dates the deals reach, and there are none to grow it from'.format(
                    base_date.strftime('%Y-%m-%d')))

        # we are repeating a period till the last reset date
        if self.input_time_grid.strip().endswith(')'):
            base_mtm_dates = self.config.parse_grid(base_date, max(dynamic_dates), self.input_time_grid)
            mtm_dates = base_mtm_dates.union(dynamic_dates)
        else:
            # we are only running the calc till the last period specified (and clipping everything else)
            max_date = min(max(dynamic_dates), base_date + self.config.periodparser.parseString(
                self.input_time_grid.strip().split(' ')[-1].upper())[0])
            base_mtm_dates = self.config.parse_grid(base_date, max_date, self.input_time_grid)
            reset_dates = [x for x in dynamic_dates if x <= max_date]
            mtm_dates = base_mtm_dates.union(reset_dates)

        if dynamic_scenario_dates:
            scenario_dates = mtm_dates
        else:
            scenario_dates = self.config.parse_grid(
                base_date, max(dynamic_dates), self.input_time_grid, past_max_date=True)

        self.time_grid = utils.TimeGrid(scenario_dates, mtm_dates, base_mtm_dates)
        self.base_date = base_date
        self.reset_dates = reset_dates
        self.time_grid.set_base_date(base_date)

        self.time_grid.set_currency_settlement(settlement_currencies)
        self.settlement_currencies = settlement_currencies

    def get_cholesky_decomp(self):
        correlation_matrix = np.eye(self.num_factors, dtype=np.float64)
        logging.root.name = self.config.deals['Attributes'].get('Reference', self.config.file_ref)
        correlation_factors = []
        self.process_ofs = {}
        for key, value in self.stoch_factors.items():
            proc_corr_type, proc_corr_factors = value.correlation_name
            # record the offset of this factor model (derived 0-factor processes get one
            # too — generate() ignores it, but the precalc plumbing indexes process_ofs)
            self.process_ofs.setdefault(key, len(correlation_factors))
            for sub_factors in proc_corr_factors:
                correlation_factors.append(utils.Factor(proc_corr_type, key.name + sub_factors))

        # a pair filed under a process name nothing simulates reads 0.0 like an undeclared one
        simulated = {utils.check_tuple_name(factor) for factor in correlation_factors}
        unread = sorted({name for pair in self.config.params['Correlations'] for name in pair}
                        - simulated)
        if unread:
            logging.info('Correlations declares {}, which no simulated factor answers to - every '
                         'pair filed under those names reads zero'.format(', '.join(unread)))

        for index1 in range(self.num_factors):
            for index2 in range(index1 + 1, self.num_factors):
                factor1, factor2 = utils.check_tuple_name(correlation_factors[index1]), utils.check_tuple_name(
                    correlation_factors[index2])
                key = (factor1, factor2) if (factor1, factor2) in self.config.params['Correlations'] else (
                    factor2, factor1)
                rho = self.config.params['Correlations'].get(key, 0.0) if factor1 != factor2 else 1.0
                correlation_matrix[index1, index2] = rho
                correlation_matrix[index2, index1] = rho

        raw_eigval, raw_eigvec = np.linalg.eig(correlation_matrix)
        eigval, eigvec = np.real(raw_eigval), np.real(raw_eigvec)
        while (eigval < 1e-8).any():
            # matrix not positive definite - find a close positive definite matrix
            if self.config.params['System Parameters']['Correlations_Healing_Method'] == 'Eigenvalue_Raising':
                logging.warning('Correlation matrix (size {0}) not positive definite - raising eigenvalues'.format(
                    correlation_matrix.shape))
                P_plus_B = eigvec.dot(np.diag(np.maximum(eigval, 1e-4))).dot(eigvec.T)
                diagonal_norm = np.diag(1.0 / np.sqrt(P_plus_B.diagonal()))
                new_correlation_matrix = diagonal_norm.dot(P_plus_B).dot(diagonal_norm)
            else:
                logging.warning('Correlation matrix (size {0}) not positive definite - alternating Projections'.format(
                    correlation_matrix.shape))

                C = correlation_matrix.astype(np.float64).copy()
                B = correlation_matrix.astype(np.float64).copy()

                # don't do more than 100 iterations - if we need to do this much, behaviour is undefined
                for k in range(100):
                    eigval, eigvec = np.linalg.eig(B)
                    P_plus_B = eigvec.dot(np.diag(np.maximum(eigval, 1e-4))).dot(eigvec.T)
                    nC = P_plus_B + np.diag(1.0 - P_plus_B.diagonal())
                    D = nC - P_plus_B
                    B += D

                    if np.abs(C - nC).max() < 1e-08 * np.abs(nC).max():
                        break

                    C = nC

                new_correlation_matrix = nC

            correlation_matrix = new_correlation_matrix
            raw_eigval, raw_eigvec = np.linalg.eig(correlation_matrix)
            eigval, eigvec = np.real(raw_eigval), np.real(raw_eigvec)

        correlation_matrix = torch.tensor(
            correlation_matrix, device=self.device, dtype=self.dtype, requires_grad=False)
        return torch.linalg.cholesky(correlation_matrix)

    def _init_shared_mem(self, seed, quasi_seed, nomodel, reporting_currency, mcmc_sim, job_id,
                         num_jobs, calc_greeks=None):
        """Allocate the `CMC_State` for this run (correlation cholesky, static buffer, reporting FX)
        and, when greeks are requested, build the flat AAD variable index over `calc_greeks`.

        `boundary_aad` has no JSON switch: wanting sensitivities IS the switch. The correction is
        worth exactly zero in the forward pass, so it can only ever change a derivative.
        """
        # single-underscore: HedgeMonteCarlo overrides this to build a CMC_State_Inner
        if calc_greeks is not None:
            implied_vars = list(itertools.chain(*[x.items() for x in self.implied_var.values()]))
            if calc_greeks == 'Implied':
                self.all_var = implied_vars
            elif calc_greeks == 'Factors':
                self.all_var = list(self.stoch_var.items()) + list(self.static_var.items())
            else:
                # a factor that is both a static dependent and an implied factor is ONE leaf
                # (the implied-leaf invariant), so the union must not report it twice
                self.all_var = list(dict(
                    implied_vars + list(self.stoch_var.items()) + list(self.static_var.items())).items())
            self.make_factor_index(self.all_var)

        scale_by_survival = (
            self.params['Funding_Valuation_Adjustment'].get('Calculate', 'No') == 'Yes')

        shared_mem = CMC_State(
            self.get_cholesky_decomp(), self.static_var, self.batch_size,
            torch.ones([1, 1], dtype=self.dtype, device=self.device), mcmc_sim, get_fxrate_factor(
                utils.check_rate_name(reporting_currency), self.static_factors, self.stoch_factors),
            seed, quasi_seed, job_id, num_jobs, scale_by_survival,
            nomodel=self.params.get('NoModel', 'Constant'),
            keep_tensor=self.params.get('Keep_Tensor', 'No') == 'Yes')
        shared_mem.boundary_aad = calc_greeks is not None
        # the kink kernel's Silverman width is sized from this job's WHOLE path count, not a batch's
        shared_mem.simulation_batches = self.params['Simulation_Batches']
        shared_mem.recompute_inner_mc = self.params.get('Recompute_Inner_MC', 'No') == 'Yes'
        shared_mem.checkpoint_outer_walk = self.params.get('Checkpoint_Outer_Walk', 'Yes') == 'Yes'
        # the one registration a document can decline by name; `execute` completed the block
        shared_mem.boundary_window_touch = self.params['Boundary_AAD_Window_Touch'] == 'Yes'
        return shared_mem

    def report(self, output):
        def valued(structure):
            # a structure the run skipped values nothing, nor does one whose deals and structures were
            return structure.obj.Instrument not in self.unpriced and (
                any(deal.Instrument not in self.unpriced for deal in structure.dependencies) or
                any(valued(x) for x in structure.sub_structures))

        skipped = ', '.join('{} {}'.format(count, name) for name, count in sorted(self.calc_stats.items())
                            if name.endswith('Skipped'))
        if skipped and not valued(self.netting_sets):
            raise utils.UnpriceableSchedule(
                'Nothing in the book was valued ({}), so the run has nothing to report: {}'.format(
                    skipped, '; '.join(sorted('{} - {}'.format(x.field.get('Reference'), x.unpriced_because)
                                              for x in self.unpriced))))
        for result, data in output.items():
            if result == 'scenarios':
                def frame(values, index, scenarios, key=None):
                    # rows by (tenor, scenario) over dates; a spot or an mtm has the one tenor
                    tenors = self.all_tenors[key][0].tenor if values.ndim == 3 else [0.0]
                    columns = pd.MultiIndex.from_product([tenors, scenarios], names=['tenor', 'scenario'])
                    return pd.DataFrame(values.reshape(len(index), -1), index=index, columns=columns).T

                scen = {}
                scenario_dates = pd.DatetimeIndex(sorted(self.time_grid.scenario_dates))
                if self.params['Calc_Scenarios'] == 'At_Percentile':
                    # the scenario grid is the mtm grid under this switch, so a report row is its
                    # scenario row; per report date, the path whose mtm sits nearest each percentile
                    rows = self.time_grid.report_index
                    mtms = np.concatenate(output['mtm'], axis=-1).astype(np.float64)
                    percentiles = self.params['Percentile'].replace(' ', '').split(',')
                    paths = [np.argmin(np.abs(mtms - np.percentile(mtms, float(x), axis=1)[:, np.newaxis]), axis=1)
                             for x in percentiles]
                    along = lambda values, at: np.stack([values[at, ..., path] for path in paths], axis=-1)
                    scen['mtm'] = frame(along(mtms, np.arange(len(rows))), scenario_dates[rows], percentiles)
                    for key, batches in data.items():
                        scen[utils.check_tuple_name(key)] = frame(
                            along(np.concatenate(batches, axis=-1), rows), scenario_dates[rows], percentiles, key)
                else:
                    for key, batches in data.items():
                        values = np.concatenate(batches, axis=-1)
                        scen[utils.check_tuple_name(key)] = frame(
                            values, scenario_dates[:values.shape[0]], np.arange(values.shape[-1]), key)
                self.output.setdefault('scenarios', scen)
            elif result == 'cashflows':
                self.output.setdefault('cashflows', {k: pd.concat(v, axis=1) for k, v in data.items()})
            elif result in ['cva', 'collva', 'fva', 'legacy_fva']:
                self.output.setdefault(result, np.array(data, dtype=np.float64).mean())
            elif result == 'collva_t':
                self.output.setdefault(result, np.array(data, dtype=np.float64).mean(axis=0))
            elif result in ['grad_cva', 'grad_collva', 'grad_fva', 'grad_legacy_fva']:
                grad = {}
                for k, v in data.items():
                    grad[k] = v.astype(np.float64) / self.params['Simulation_Batches']
                self.output.setdefault(result, self.gradients_as_df(grad, display_val=True))
            elif result == 'CS01':
                columns = pd.MultiIndex.from_arrays(
                    [data['Par_CDS'].keys(), data['Par_CDS'].values()], names=['Tenor', 'Par CDS'])
                self.output.setdefault(result, pd.DataFrame(columns=columns, index=data['Tenor'], data=np.transpose(
                    [x - data['Shifted_Log_Prob'][0] for x in data['Shifted_Log_Prob'][1:]])))
            elif result in ['grad_cva_hessian']:
                self.output.setdefault(result, self.gradients_as_df(
                    data.astype(np.float64) / self.params['Simulation_Batches']))
            elif result in ['mtm', 'collateral']:
                dates = np.array(sorted(self.time_grid.mtm_dates))[self.time_grid.report_index]
                self.output.setdefault(result, pd.DataFrame(
                    np.concatenate(data, axis=-1).astype(np.float64), index=dates))
            elif result == 'gross_mtm':
                dates = np.array(sorted(self.time_grid.mtm_dates))
                self.output.setdefault(result, pd.DataFrame(
                    np.concatenate(data, axis=-1).astype(np.float64), index=dates))
            else:
                self.output.setdefault(result, np.concatenate(data, axis=-1).astype(np.float64))

        return self.output

    def refuse_unframed(self):
        """Refuse a book this calculation would simulate nothing of, naming the shape it met.

        Read once the models and the deal structures are both known. Unframed, the batch loop
        reshapes a draw of no factors or reduces a structure holding no deal, and the run dies on
        the tensor rather than on the book. The third shape - no date after the base date - is
        refused where the grid is grown, `update_time_grid` having nothing to take a maximum over.
        """
        if not (self.calc_stats.get('Deals loaded', 0) + self.calc_stats.get('Structs loaded', 0)):
            raise ValueError(
                'every deal of this book was skipped or ignored - a credit Monte Carlo projects '
                'nothing over an empty book; /validate names what each deal is missing')
        if not self.num_factors:
            raise ValueError(
                'no factor of this book has a model - a credit Monte Carlo simulates nothing; '
                'declare a Price Models entry for a factor the deals reach, or run this book as '
                'a BaseValuation')

    def execute(self, params, job_id=0, num_jobs=1, deterministic_batches=False):
        """Run the batched exposure simulation plus whichever sub-calculations `params` enables
        (collateral, initial margin, CVA, FVA, scenarios, cashflows) and return the netting sets,
        stats and reports.

        DETERMINISTIC BATCHES (`deterministic_batches`, set only by the `runparallel` dispatch).
        Off - the default - the stream is seeded ONCE per worker in `CMC_State.__init__` and the
        batch loop consumes it sequentially, so batch b's draws depend on how many batches ran
        before it in this process: fine for one worker, fatal for n. On, each batch seeds its OWN
        stream from its GLOBAL index b (`batch_seed(Random_Seed, b)`) and workers take contiguous
        ranges `[j*k, (j+1)*k)`, so batch b draws the same numbers whichever worker runs it and the
        pooled result is bit-identical in n rather than merely close.

        BOTH STREAMS ARE ANCHORED, not just the generator. A world whose outer path draws quasi-
        random numbers reads a Sobol sequence that is reproducible but POSITION-dependent.
        `set_quasi_batch` gives it the same global index, and the anchored arm of
        `CMC_State.quasi_rng` takes batch b's draw from `QUASI_ANCHOR + b * sample_size` however
        many draws the asking worker has made - on an unsharded run, where the historical engine
        already stands. Two draws of one `(dimension, sample_size)` inside a batch is the one shape
        anchoring cannot carry, and refuses by name there.

        The CVA reduction keeps its grouping deliberately: a per-path vector cannot be reduced back
        to `mean over paths of a sum over time` in the same float order, and the reported number
        must not move by an ULP. `pricing.cva_per_scenario` is the same quantity for the
        counterfactuals.

        BOUNDARY AAD (CVA gradient): a hard transfer decision contributes a derivative the
        frozen-decision graph does not carry. The correction is worth exactly zero forward, so `cva`
        is untouched and only the differentiated scalar gains a term. `Boundary_AAD_Bandwidth`
        defaults to 0.01, which the declared 16,384 paths populate at the bottom of its measured plateau.

        THE CVA HESSIAN (`Hessian: 'Yes'`) rides the reported trapezoid through
        `pricing.exposure_kink_term`, so cva, the profile and grad_cva are untouched by
        construction. It restores the `delta(V) V_theta V_theta^T` term the double backward drops at
        the exposure relu, without which a LINEAR book reports a spot gamma of exactly zero. Two
        things refuse rather than report a plausible wrong matrix: a book that registered a boundary
        correction, and a reporting row whose bandwidth LADDER diverges.
        """
        # the declaration is the single source of an omitted field's default
        params = declared_defaults(type(self), params)
        credit = params['Credit_Valuation_Adjustment']
        if credit.get('Hessian', 'No') == 'Yes' and credit.get('Gradient', 'No') != 'Yes':
            raise ValueError(
                "Credit_Valuation_Adjustment Hessian: 'Yes' needs Gradient: 'Yes' beside it - the "
                'second derivatives are taken through the tape the first ones are reported off, '
                'which is built only where the gradient is asked for, so this run would report '
                'neither. Write Gradient: Yes for both blocks, or Hessian: No for the value alone')
        base_date = pd.Timestamp(params['Run_Date'])

        self.input_time_grid = params['Time_grid']
        # divide the batches across the jobs (multi-gpu)
        params['Simulation_Batches'] = params['Simulation_Batches'] // num_jobs
        # this worker's contiguous slice of the WHOLE job's batches: worker j owns [j*k, (j+1)*k).
        # Only `deterministic_batches` reads it, and at num_jobs == 1 it is zero.
        batch_offset = job_id * params['Simulation_Batches']
        self.batch_size = params['Batch_Size']
        self.numscenarios = self.batch_size * params['Simulation_Batches']

        self.params = params
        logging.root.name = self.config.deals['Attributes'].get('Reference', self.config.file_ref)

        self.calc_stats['Batch_Size'] = self.batch_size
        self.calc_stats['Simulation_Batches'] = self.params['Simulation_Batches']
        self.calc_stats['Random_Seed'] = params['Random_Seed']

        shared_mem = self.update_factors(params, base_date, job_id, num_jobs)

        self.netting_sets = DealStructure(Aggregation('root'), store_results=True)
        self.set_deal_structures(
            self.config.deals['Deals']['Children'], self.netting_sets, shared_mem.one,
            deal_level_mtm=params.get('DealLevel', False))
        self.netting_sets.finalize_struct(base_date, self.time_grid)
        self.refuse_unframed()

        output = defaultdict(list)
        tensors = {}
        execution_label = 'Tensor_Execution_Time ({})'.format(self.device.type)
        self.calc_stats[execution_label] = time.monotonic()
        base_ccy = get_fxrate_factor(
            utils.check_rate_name(self.config.params['System Parameters']['Base_Currency']),
            self.static_factors, self.stoch_factors)
        time_index = self.time_grid.report_index

        for run in range(self.params['Simulation_Batches']):

            if deterministic_batches:
                # batch b owns its streams, keyed by its GLOBAL index. BOTH streams: the torch
                # generator is reseeded and the quasi stream is repositioned onto the same batch
                torch.manual_seed(batch_seed(self.params['Random_Seed'], batch_offset + run))
                shared_mem.set_quasi_batch(batch_offset + run)

            shared_mem.reset(
                self.num_factors, self.time_grid, use_antithetic=params.get('Antithetic', 'No') == 'Yes')

            for key, value in self.stoch_factors.items():
                shared_mem.t_Scenario_Buffer[key] = value.generate(shared_mem)

            tensors['mtm'] = self.netting_sets.resolve_structure(shared_mem, self.time_grid)

            final_run = run == self.params['Simulation_Batches'] - 1
            # the mtm is in reporting currency - need to convert back to base currency
            fx_report = utils.calc_fx_cross(
                shared_mem.Report_Currency, base_ccy, self.time_grid.time_grid[time_index], shared_mem)

            if params['Collateral_Valuation_Adjustment'].get(
                    'Calculate', 'No') == 'Yes' and shared_mem.simulation_batch > 1:

                if params['Collateral_Valuation_Adjustment'].get('Gradient', 'No') == 'Yes':
                    tensors['collva_t'] = torch.mean(shared_mem.t_Credit['Funding'], dim=1)
                    tensors['collva'] = torch.sum(tensors['collva_t'])

                    sensitivity = SensitivitiesEstimator(tensors['collva'], self.all_var)

                    if final_run:
                        output['grad_collva'] = sensitivity.report_grad()
                        self.calc_stats['Gradient_Vector_Size'] = sensitivity.P

            if params['Initial_Margin'].get('Calculate', 'No') == 'Yes':
                def calc_buckets(liq_w, tenor):
                    liquidity = {}
                    for col in liq_w.T.iterrows():
                        left_limit = 0.0
                        right_limit = 0.0
                        series = col[1].dropna()
                        if '<=' in series.index[0]:
                            left_limit = 1.0
                            y = series.index.map(lambda x: float(x.replace('<=', '').replace('y', '')))
                        elif '>=' in series.index[-1]:
                            right_limit = 1.0
                            y = series.index.map(lambda x: float(x.replace('>=', '').replace('y', '')))
                        else:
                            y = series.index.map(lambda x: float(x.replace('y', '')))
                        liquidity[col[0]] = np.interp(tenor, y, series.values, left=left_limit, right=right_limit)
                    return liquidity

                def calc_max(liquidity_charge, tenor1, tenor2):
                    return torch.where(
                        liquidity_charge[tenor1]*liquidity_charge[tenor2] < 0,
                        torch.max(torch.abs(liquidity_charge[tenor1]), torch.abs(liquidity_charge[tenor2])),
                        torch.abs(liquidity_charge[tenor1])+torch.abs(liquidity_charge[tenor2]))

                liq_w = pd.read_csv(params['Initial_Margin']['Liquidity_Weights'], index_col=0)
                irs = pd.read_csv(params['Initial_Margin']['IRS_Weights'], index_col=0)
                local_curves = [k for k, v in self.all_factors.items() if len(
                    k.name) == 1 and k.type == 'InterestRate' and (
                                    v.factor.param if hasattr(v, 'factor') else v.param).get(
                    'Currency') == params['Initial_Margin']['Local_Currency']]
                IM_currency = get_fxrate_factor(
                    utils.check_rate_name(params['Initial_Margin']['IM_Currency']),
                    self.static_factors, self.stoch_factors)
                fx_IM_report = utils.calc_fx_cross(
                    base_ccy, IM_currency, self.time_grid.time_grid[time_index], shared_mem)
                if local_curves[0] in shared_mem.t_Scenario_Buffer:
                    scen_buf = shared_mem.t_Scenario_Buffer
                else:
                    scen_buf = shared_mem.t_Static_Buffer

                local_tenor = {k: self.all_tenors[k][0].tenor for k in local_curves}
                # round the tenor to 1 dp to ensure accurate bucket lookup
                local_shifts = {k: calc_buckets(liq_w, t.round(1)) for k, t in local_tenor.items()}

                all_shifts = {}
                liquidity_deltas = {}

                for d in local_shifts.values():
                    for key, value in d.items():
                        all_shifts.setdefault(key, []).append(value)

                shared_mem.t_Cashflows = None

                for tenor, shifts in all_shifts.items():
                    deltas = {}
                    for curvename, shift in zip(local_shifts.keys(), shifts):
                        deltas[curvename] = shared_mem.one.new_tensor(shift.reshape(1, -1, 1) * 0.01 * 0.01)
                        scen_buf[curvename] += deltas[curvename]

                    shared_mem.t_Buffer.clear()
                    # calc the liquidity change in base_currency - simple delta
                    liquidity_deltas[tenor] = (self.netting_sets.resolve_structure(
                        shared_mem, self.time_grid) - tensors['mtm']) * fx_report

                    for curvename, shift in zip(local_shifts.keys(), shifts):
                        scen_buf[curvename] -= deltas[curvename]

                liquidity_charge = {}
                for tenor, values in liquidity_deltas.items():
                    curve_tenor = utils.CurveTenor(irs[tenor].dropna().index.astype(np.float64).values)
                    curve_weights = shared_mem.one.new_tensor(irs[tenor].dropna().values)
                    index, index_next, alpha = curve_tenor.get_index(values)
                    liquidity_charge[tenor] = values * (
                            curve_weights[index] * (1 - alpha) + curve_weights[index_next] * alpha)

                IM_liquidity_charge = (calc_max(
                    liquidity_charge, '2y', '5y') + calc_max(liquidity_charge, '10y', '30y')) * fx_IM_report

                shared_mem.t_Buffer.clear()
                for int_rate in [k for k in shared_mem.t_Scenario_Buffer.keys() if
                                 k.type == 'InterestRate' and len(k.name) == 1]:
                    # calc pv01
                    shared_mem.t_Scenario_Buffer[int_rate] += 0.01 * 0.01
                for int_rate in [k for k in shared_mem.t_Static_Buffer if
                                 k.type == 'InterestRate' and len(k.name) == 1]:
                    # calc pv01
                    shared_mem.t_Static_Buffer[int_rate] += 0.01 * 0.01

                IM_delta_charge = (self.netting_sets.resolve_structure(
                    shared_mem, self.time_grid) - tensors['mtm']) * fx_report * fx_IM_report

                tensors['LCH_Margin'] = params['Initial_Margin']['Delta_Factor']*IM_delta_charge+IM_liquidity_charge

            if params['Funding_Valuation_Adjustment'].get('Calculate', 'No') == 'Yes':
                mtm_grid = self.time_grid.mtm_time_grid[time_index]

                funding_cost = get_interest_factor(
                    utils.check_rate_name(params['Funding_Valuation_Adjustment']['Funding_Cost_Interest_Curve']),
                    self.static_factors, self.stoch_factors, self.all_tenors)
                funding_benefit = get_interest_factor(
                    utils.check_rate_name(params['Funding_Valuation_Adjustment']['Funding_Benefit_Interest_Curve']),
                    self.static_factors, self.stoch_factors, self.all_tenors)
                riskfree = get_interest_factor(
                    utils.check_rate_name(params['Funding_Valuation_Adjustment']['Risk_Free_Curve']),
                    self.static_factors, self.stoch_factors, self.all_tenors)
                discount = get_interest_factor(
                    utils.check_rate_name(params['Deflation_Interest_Rate']),
                    self.static_factors, self.stoch_factors, self.all_tenors)

                # note that for FVA, we already scale the exposure matrix by the survival probability
                if params['Funding_Valuation_Adjustment'].get('Deflate_Stochastically', 'No') == 'Yes':
                    delta_scen_t = np.diff(mtm_grid).reshape(-1, 1)

                    deflation = utils.calc_time_grid_curve_rate(
                        discount, self.time_grid.time_grid[time_index], shared_mem)
                    DF_base = torch.exp(-torch.squeeze(deflation.gather_weighted_curve(
                        shared_mem, np.diff(np.append(0, mtm_grid)).reshape(-1, 1))).cumsum(dim=0))

                    discount_fund_cost = utils.calc_time_grid_curve_rate(
                        funding_cost, self.time_grid.time_grid[time_index[:-1]], shared_mem)
                    discount_fund_benefit = utils.calc_time_grid_curve_rate(
                        funding_benefit, self.time_grid.time_grid[time_index[:-1]], shared_mem)
                    discount_rf = utils.calc_time_grid_curve_rate(
                        riskfree, self.time_grid.time_grid[time_index[:-1]], shared_mem)

                    delta_fund_cost_rf = torch.squeeze(
                        torch.exp(discount_fund_cost.gather_weighted_curve(shared_mem, delta_scen_t)) -
                        torch.exp(discount_rf.gather_weighted_curve(shared_mem, delta_scen_t)), dim=1)

                    delta_fund_benefit_rf = torch.squeeze(
                        torch.exp(discount_fund_benefit.gather_weighted_curve(shared_mem, delta_scen_t)) -
                        torch.exp(discount_rf.gather_weighted_curve(shared_mem, delta_scen_t)), dim=1)
                else:
                    deflation = utils.calc_time_grid_curve_rate(discount, np.zeros((1, 3)), shared_mem)
                    DF_base = torch.squeeze(torch.exp(-deflation.gather_weighted_curve(
                        shared_mem, mtm_grid.reshape(1, -1))), dim=0)

                    zero_fund_cost = utils.calc_time_grid_curve_rate(funding_cost, np.zeros((1, 3)), shared_mem)
                    zero_fund_benefit = utils.calc_time_grid_curve_rate(funding_benefit, np.zeros((1, 3)), shared_mem)
                    zero_rf = utils.calc_time_grid_curve_rate(riskfree, np.zeros((1, 3)), shared_mem)

                    Dt_T_fund_cost = torch.squeeze(torch.exp(
                        -zero_fund_cost.gather_weighted_curve(shared_mem, mtm_grid.reshape(1, -1))), dim=0)
                    Dt_T_fund_benefit = torch.squeeze(torch.exp(
                        -zero_fund_benefit.gather_weighted_curve(shared_mem, mtm_grid.reshape(1, -1))), dim=0)
                    Dt_T_rf = torch.squeeze(torch.exp(
                        -zero_rf.gather_weighted_curve(shared_mem, mtm_grid.reshape(1, -1))), dim=0)

                    delta_fund_cost_rf = (
                        (Dt_T_fund_cost[:-1] / Dt_T_fund_cost[1:]) - (Dt_T_rf[:-1] / Dt_T_rf[1:]))
                    delta_fund_benefit_rf = (
                        (Dt_T_fund_benefit[:-1] / Dt_T_fund_benefit[1:]) - (Dt_T_rf[:-1] / Dt_T_rf[1:]))

                # tensors['mtm'] is in reporting currency - we need to convert back to base
                pv_exposure = (tensors['mtm'] * fx_report * DF_base) / fx_report[0]
                Vk_plus_ti = torch.relu(pv_exposure)
                Vk_minus_ti = torch.relu(-pv_exposure)
                Vk_star_ti_p = (Vk_plus_ti[1:] + Vk_plus_ti[:-1]) / 2
                Vk_star_ti_m = (Vk_minus_ti[1:] + Vk_minus_ti[:-1]) / 2

                FCA_t = torch.sum(delta_fund_cost_rf * Vk_star_ti_p, dim=0)
                FCA = torch.mean(FCA_t)
                FBA_t = torch.sum(delta_fund_benefit_rf * Vk_star_ti_m, dim=0)
                FBA = torch.mean(FBA_t)

                tensors['fva'] = FCA - FBA

                if params['Funding_Valuation_Adjustment'].get('Gradient', 'No') == 'Yes':
                    # FVA carries its own objective - the CVA section is deleted by the shipped
                    # batch job, so the correction assembled there cannot fire here
                    fva_for_aad = tensors['fva']
                    if shared_mem.boundary_sets:
                        fva_objective = lambda mtm: pricing.fva_per_scenario(
                            (mtm * fx_report * DF_base) / fx_report[0],
                            delta_fund_cost_rf, delta_fund_benefit_rf)
                        correction = pricing.boundary_correction(
                            shared_mem, fva_objective, tensors['mtm'],
                            float(params.get('Boundary_AAD_Bandwidth', 0.01)))
                        if correction is not None:
                            fva_for_aad = fva_for_aad + correction
                    sensitivity = SensitivitiesEstimator(fva_for_aad, self.all_var)

                    if final_run:
                        output['grad_fva'] = sensitivity.report_grad()
                        self.calc_stats['Gradient_Vector_Size'] = sensitivity.P

            if params['Credit_Valuation_Adjustment'].get('Calculate', 'No') == 'Yes':
                discount = get_interest_factor(
                    utils.check_rate_name(params['Deflation_Interest_Rate']),
                    self.static_factors, self.stoch_factors, self.all_tenors)
                survival = get_survival_factor(
                    utils.check_rate_name(params['Credit_Valuation_Adjustment']['Counterparty']),
                    self.static_factors, self.stoch_factors, self.all_tenors)
                recovery = get_survival_component(
                    utils.check_rate_name(params['Credit_Valuation_Adjustment']['Counterparty']),
                    self.all_factors).recovery_rate()

                # Calculates unilateral CVA with or without stochastic deflation.
                mtm_grid = self.time_grid.mtm_time_grid[time_index]
                delta_scen_t = np.hstack((0.0, np.diff(mtm_grid)))

                if params['Credit_Valuation_Adjustment']['Deflate_Stochastically'] == 'Yes':
                    zero = utils.calc_time_grid_curve_rate(
                        discount, self.time_grid.time_grid[time_index], shared_mem)
                    Dt_T = torch.exp(-torch.squeeze(zero.gather_weighted_curve(
                        shared_mem, delta_scen_t.reshape(-1, 1))).cumsum(dim=0))
                else:
                    zero = utils.calc_time_grid_curve_rate(discount, np.zeros((1, 3)), shared_mem)
                    Dt_T = torch.squeeze(torch.exp(
                        -zero.gather_weighted_curve(shared_mem, mtm_grid.reshape(1, -1))), dim=0)

                # FVA scales every set's reported MTM by survival (`CMC_State.scale_survival`), and
                # that factor is positive and deterministic per bucket, so `relu` commutes with it:
                # the CVA integrand takes it back out and reads the same either way
                unscale = torch.squeeze(torch.exp(utils.calc_time_grid_curve_rate(
                    survival, np.zeros((1, 3)), shared_mem).gather_weighted_curve(
                    shared_mem, mtm_grid.reshape(1, -1), multiply_by_time=False)), dim=0) \
                    if shared_mem.scale_survival else 1.0

                # tensors['mtm'] is in reporting currency - we need to convert back to base
                pv_exposure = torch.relu(tensors['mtm'] * unscale * fx_report * Dt_T) / fx_report[0]

                if params['Credit_Valuation_Adjustment']['Stochastic_Hazard_Rates'] == 'Yes':
                    surv = utils.calc_time_grid_curve_rate(
                        survival, self.time_grid.time_grid[time_index], shared_mem)
                    St_T = torch.exp(-torch.cumsum(torch.squeeze(surv.gather_weighted_curve(
                        shared_mem, delta_scen_t.reshape(-1, 1), multiply_by_time=False), dim=1),
                        dim=0))
                else:
                    surv = utils.calc_time_grid_curve_rate(survival, np.zeros((1, 3)), shared_mem)
                    St_T = torch.squeeze(torch.exp(-surv.gather_weighted_curve(
                        shared_mem, mtm_grid.reshape(1, -1), multiply_by_time=False)), dim=0)

                prob = St_T[:-1] - St_T[1:]
                # this grouping is float-order load-bearing - do not reduce it differently
                tensors['cva'] = (1.0 - recovery) * (
                        0.5 * (pv_exposure[1:] + pv_exposure[:-1]) * prob).mean(axis=1).sum()

                if params['Credit_Valuation_Adjustment'].get('Gradient', 'No') == 'Yes':
                    base_ir_curves = [x for x in self.stoch_var.keys() if
                                      x.type == 'InterestRate' and len(x.name) == 1]
                    self.jacobians = {}
                    for ir_factor in base_ir_curves:
                        jacobian_factor = utils.Factor('InterestRateJacobian', ir_factor.name)
                        ir_curve = self.stoch_factors[utils.Factor('InterestRate', ir_factor.name)].factor
                        var_name = 'Stochastic_Input/{0}:0'.format(utils.check_tuple_name(ir_factor))
                        try:
                            jac = construct_factor(
                                jacobian_factor, self.config.params['Price Factors'],
                                self.config.params['Price Factor Interpolation'])
                            jac.update(ir_curve)
                            self.jacobians[var_name] = jac.current_value()
                            logging.info('jacobian present for {0} - will attempt inverse bootstrap'.format(
                                utils.check_tuple_name(ir_factor)))
                        except KeyError:
                            pass

                    hessian = params['Credit_Valuation_Adjustment'].get('Hessian', 'No') == 'Yes'
                    if hessian and shared_mem.boundary_sets:
                        raise utils.SecondOrderRefused(
                            "Credit_Valuation_Adjustment Hessian: 'Yes' is refused - this book "
                            'takes decisions on simulated state and registered a boundary '
                            'correction: {}. That correction is what makes their FIRST derivative '
                            'right and it is a FIRST-ORDER estimator - (gap - gap.detach()) times '
                            'a DETACHED coefficient - so a second derivative through it comes back '
                            'with the density-DERIVATIVE flux block silently missing: a '
                            'plausible-looking cross-gamma rather than a failure, which is the '
                            'failure mode this refusal exists to prevent. Two remedies: drop '
                            'Hessian and keep grad_cva, which is unaffected, or ask for the '
                            'second-order block on a book without decision products.'.format(
                                ', '.join(sorted({str(b.deal) for b in shared_mem.boundary_sets}))))
                    cva_for_aad = tensors['cva']
                    if shared_mem.boundary_sets:
                        objective = lambda mtm: pricing.cva_per_scenario(
                            torch.relu(mtm * unscale * fx_report * Dt_T) / fx_report[0],
                            prob, recovery)
                        correction = pricing.boundary_correction(
                            shared_mem, objective, tensors['mtm'],
                            float(params.get('Boundary_AAD_Bandwidth', 0.01)))
                        if correction is not None:
                            cva_for_aad = cva_for_aad + correction
                    if hessian:
                        # the relu's argument above, SIGNED; built only when second order is asked
                        kink = pricing.exposure_kink_term(
                            shared_mem, tensors['mtm'] * unscale * fx_report * Dt_T / fx_report[0])
                        # mirrors the reported reduction exactly: same trapezoid, prob and recovery,
                        # so the (exact-zero) term rides the objective's own weights
                        cva_for_aad = cva_for_aad + (1.0 - recovery) * (
                                0.5 * (kink[1:] + kink[:-1]) * prob).mean(axis=1).sum()
                    sensitivity = SensitivitiesEstimator(
                        cva_for_aad, self.all_var, create_graph=hessian)

                    if final_run:
                        output['grad_cva'] = sensitivity.report_grad()
                        self.calc_stats['Gradient_Vector_Size'] = sensitivity.P

                        CDS_tenors = params['Credit_Valuation_Adjustment'].get('CDS_Tenors')
                        if CDS_tenors and recovery < 1.0:
                            CDS_rates, shifted_tenor, shifted_curves = utils.calc_cds_rates(
                                recovery, survival[0], discount[0], self.params['Base_Date'],
                                CDS_tenors, self.all_factors)

                            output['CS01'] = {
                                'Par_CDS': CDS_rates,
                                'Tenor': shifted_tenor,
                                'Shifted_Log_Prob': shifted_curves
                            }

                        if hessian:
                            # calculate the hessian matrix - warning - make sure you have enough memory
                            output['grad_cva_hessian'] = sensitivity.report_hessian()

            for k, v in tensors.items():
                output[k].append(v.cpu().detach().numpy())

            if self.params['Generate_Cashflows'] == 'Yes':
                shared_mem.save_cashflows(output, self.time_grid)

            if self.params.get('Calc_Scenarios', 'No') != 'No':
                for key, value in self.stoch_factors.items():
                    output.setdefault('scenarios', {}).setdefault(key, []).append(
                        shared_mem.t_Scenario_Buffer[key].cpu().detach().numpy())

        self.calc_stats[execution_label] = time.monotonic() - self.calc_stats[execution_label]

        results = {'Netting': self.netting_sets, 'Stats': self.calc_stats, 'Jacobians': self.jacobians}
        results['Results'] = self.report(output)

        return results


class Base_Reval_State(utils.Calculation_State):
    def __init__(self, static_buffer, one, mcmc_sims, report_currency, calc_greeks, gamma, quasi_seed,
                 nomodel='Constant'):
        """Single-date, single-scenario valuation state (base MtM and its greeks).

        `boundary_aad` follows the same contract as `CMC_State`: a decision taken on simulated state
        is recorded forward so its derivative can be restored before the reverse sweep. Base
        valuation has one date and one scenario, but a Monte Carlo pricer still runs a full inner
        simulation underneath it - where a TARF's knock-in is decided - so the defect and the
        estimator are the same ones and only the objective is simpler.
        """
        super(Base_Reval_State, self).__init__(
            static_buffer, one, mcmc_sims, report_currency, nomodel, 1, False, quasi_seed)
        self.calc_greeks = calc_greeks
        self.gamma = gamma
        self.boundary_aad = calc_greeks is not None
        self.boundary_sets = []

    @staticmethod
    def save_results(output, tensors):
        for k, v in tensors.items():
            output[k] = np.float64(v) if isinstance(v, float) else v.detach().cpu().numpy().astype(np.float64)


class BaseValuation(Calculation):
    """Simple deal revaluation - use this to reconcile with the source system.

    SECOND DERIVATIVES LIVE HERE AND NOWHERE ELSE. `Greeks: 'All'` asks the reverse sweep for
    `create_graph` and reports the result as `Greeks_Second` beside the first-order `Greeks_First`,
    which always accompanies it because the row labels are built off it.

    The shape is the FULL Hessian rather than a Hessian-vector product, because the number is a
    cross-gamma REPORT (spot-vol, spot-curve) that no caller arrives at with a direction to contract
    along. One date, one scenario and P = the factor knots the portfolio depends on keep that
    affordable - `report_hessian` runs P double-backward passes, 0.9x to 10x the first-order pass.
    An exposure-sized P would flip that argument, and exposure has its own CVA-Hessian route.

    The frame is the Hessian's SUPPORT: identically-zero rows and columns are dropped, so the matrix
    is square and symmetric but smaller than P, indexed on both axes by (Rate, Tenor, Tenor2,
    Tenor3) with the reporting reference as the columns' outer level.

    Two things refuse rather than report a plausible wrong number: a deal that registered a
    `BoundarySet` (`execute` below) and `Recompute_Inner_MC` (`pricing.InnerMCRecompute.backward`).
    `Branch_And_Weight` answers the first, declared here and on `ScenarioValuation`, which values
    as this does - the smooth estimator removes the decision instead of correcting it, so a deal
    priced under the switch registers nothing and the full Hessian flows.
    """
    documentation = ('Calculations', [
        'This applies the valuation models mentioned earlier to the portfolio per deal, at one date and on',
        'today\'s market (no factor is simulated).',
        '',
        'The inputs are:',
        '',
        '- **Currency** of the output.',
        '- **Base Date** at which the marketdata should be applied (i.e. $t_0$); the runner passes it on as',
        '  `Run_Date`, which an override may set.',
        '- **MCMC Simulations** the number of inner Monte Carlo paths a deal that requires Monte Carlo',
        '  pricing (e.g. Autocalls, TARF\'s etc.) spends per reporting row, mirrored $(z,-z)$ (default 16,384)',
        '- **Quasi Seed** the scramble seed of the inner paths, which every trade draws as its own',
        '  whatever is booked beside it; **Deterministic Kernels** pins the GPU backward pass to',
        '  deterministic kernels',
        '- **Greeks** `First` calculates all first order sensitivities (partial derivatives) of the portfolio',
        '  with respect to the relevant Price Factors; `All` adds the second order block (the full factor',
        '  Hessian, reported as `Greeks_Second`). Default `No`.',
        '- **Branch And Weight** (default `Yes`) prices the fixing-observed knockouts (TARF, accumulator,',
        '  discrete barrier, autocall) with the smooth estimator; `No` is the crisp estimator',
        '- **Boundary AAD Bandwidth** (default 0.01) the kernel width of the boundary correction a decision',
        '  on simulated state adds to the gradient',
        '  ([Calc Lifecycle](../developer/calc_lifecycle.md#boundary-corrections-the-sensitivity-subsystem))',
        '- **Recompute Inner MC** (default `No`) re-simulates a Monte Carlo pricer\'s inner paths in the',
        '  backward pass rather than taping them',
        '- **Correlation Bump** (default 0.025) the half-width of the common-random-number bump that reports',
        '  a quanto, compo or mid-curve correlation delta; 0 turns it off',
        '',
        'The output is a dictionary containing the DealStructure and the calculation computation statistics.'
    ])

    #: The smooth estimator's switch, on the calculations that value as this one does.
    BRANCH_AND_WEIGHT = F(
        'Branch_And_Weight', 'Text', default='Yes', values=['Yes', 'No'],
        description='Price fixing-observed knockouts (TARF, accumulator, discrete barrier, '
                    'autocall) with the SMOOTH estimator, each fixing\'s fired branch integrated '
                    'analytically; `No` is the crisp estimator')
    #: The state a pricing runs on.
    state = Base_Reval_State

    fields = [
        F('Base_Date', 'Date', default=''),
        F('Currency', 'Text', default=REQUIRED),
        F('MCMC_Simulations', 'Integer', default=1 << 14,
          description='Inner Monte Carlo paths a simulated pricer spends per reporting row, '
                      'mirrored (z, -z); an analytic deal never reads it'),
        Calculation.QUASI_SEED,
        Calculation.DETERMINISTIC_KERNELS,
        F('Greeks', 'Text', default='No', values=['All', 'First', 'No'],
          description='First order factor sensitivities, or `All` for the second order block '
                      '(`Greeks_Second`) as well'),
        F('Boundary_AAD_Bandwidth', 'Float', default=0.01,
          description='Kernel bandwidth of the boundary correction assembled into backward()'),
        F('Recompute_Inner_MC', 'Text', default='No', values=['Yes', 'No'],
          description='Re-simulate a Monte Carlo pricer\'s inner paths in backward() rather than '
                      'taping them; trades a second forward pass for the graph of every pricing'),
        BRANCH_AND_WEIGHT,
        F('Correlation_Bump', 'Float', default=0.025,
          description='Half-width of the CRN bump reporting a quanto, compo or mid-curve correlation '
                      'delta; 0 turns it off')
    ]

    def __init__(self, config, **kwargs):
        super(BaseValuation, self).__init__(config, **kwargs)
        self.base_date = None
        self.correlation_deltas = {}

        self.shared_memClass = namedtuple('shared_mem',
                                          't_Buffer t_Static_Buffer t_Feed_dict t_Cashflows calc_greeks \
                                          gpus riskneutral precision simulation_batch Report_Currency')

        self.static_var = {}

    @classmethod
    def runner_keys(cls, block):
        """`Run_Date`, the block's `Base_Date` as `YYYY-MM-DD`."""
        return {'Run_Date': block['Base_Date'].strftime('%Y-%m-%d')}

    @classmethod
    def run(cls, context, overrides=None, prec=torch.float64):
        """Values `context`'s book on the device `utils.calculation_device` picks, at `prec`, its
        block completed by `parameters` and judged: `(calculation, output)`."""
        calc = cls(context, device=utils.calculation_device(), prec=prec)
        return calc, calc.execute(calc.judged(cls.parameters(context, overrides)))

    def update_factors(self, params, base_date):
        dependent_factors, stochastic_factors, implied_factors, reset_dates, settlement_currencies = \
            self.config.calculate_dependencies(params, base_date, '0d', False)

        self.update_time_grid(base_date)

        self.static_factors = {}
        for price_factor in dependent_factors:
            try:
                self.static_factors.setdefault(
                    price_factor, construct_factor(
                        price_factor, self.config.params['Price Factors'],
                        self.config.params['Price Factor Interpolation'],
                        base_date=base_date))
            except KeyError as e:
                logging.warning('Price Factor {0} missing in market data file - skipping'.format(e.args))

        self.all_factors = self.static_factors

        calc_grad = params.get('Greeks', 'No') != 'No'
        for key, value in self.static_factors.items():
            if key.type not in utils.DimensionLessFactors:
                for leaf, current_val in self.split(key, value.current_value()):
                    self.static_var[leaf] = self.factor_leaf(leaf, current_val, calc_grad)

        quasi_seed = self.calc_stats['Quasi_Seed'] = int(params['Quasi_Seed'])
        shared_mem = self.__init_shared_mem(
            params['Currency'], params['MCMC_Simulations'], calc_grad, quasi_seed)
        shared_mem.calc_stats, shared_mem.unpriced = self.calc_stats, self.unpriced

        self.all_tenors = utils.update_tenors(self.base_date, self.all_factors)

        return shared_mem

    def correlation_bump(self, params, width):
        """Every quanto, compo or mid-curve correlation a priced deal read, as a CENTRAL CRN bump of
        the marked value: the book is re-compiled and re-valued once each side on the job's own
        draws, so the difference is the bump and nothing else.

        `Correlation` is a `DimensionLessFactor` - it mints no leaf, so this delta has no tape to
        come off and the alternative to a bump is reporting zero for a number the engine has. The
        marked market is put back and re-read before the report, which is what leaves the row's
        displayed value the desk's own.
        """
        named = {id(value): key for key, value in self.all_factors.items()}
        wanted, rows, market = set(), {}, self.config.params['Price Factors']
        stack = [self.netting_sets]
        while stack:
            node = stack.pop()
            stack.extend(node.sub_structures)
            # a container's own block too: a swaption prices its legs and reads its correlation itself
            for deal in [node.obj] + node.dependencies:
                wanted.update(named[id(deal.Factor_dep[key])] for key in
                              ('QuantoImpliedCorrelation', 'CompoImpliedCorrelation', 'Mid_Curve_Correlation')
                              if id((deal.Factor_dep or {}).get(key)) in named)
        for factor in sorted(wanted, key=utils.check_tuple_name):
            marked = market[utils.check_tuple_name(factor)]
            base, sides, keep = marked['Value'], [], self.netting_sets
            # BOTH sides have to fit inside [-1, 1] or the central difference is a half-width lie
            h = min(width, 1.0 - abs(base))
            if h <= 0.0:
                continue
            try:
                for side in (h, -h):
                    marked['Value'] = base + side
                    shared = self.update_factors(dict(params, Greeks='No'), self.base_date)
                    self.netting_sets = DealStructure(Aggregation('root'), store_results=True)
                    self.set_deal_structures(self.config.deals['Deals']['Children'],
                                             self.netting_sets, shared.one, deal_level_mtm=True)
                    self.netting_sets.resolve_structure(shared, self.time_grid)
                    sides.append(sum(x.obj.Calc_res['Value'].item()
                                     for x in self.netting_sets.sub_structures))
            finally:
                marked['Value'], self.netting_sets = base, keep
            rows[utils.check_scope_name(factor)] = (h, (sides[0] - sides[1]) / (2.0 * h))
        if rows:
            self.update_factors(params, self.base_date)
        return rows

    def update_time_grid(self, base_date, day=None):
        """The grid of the one date valued, `day`, the base date where None."""
        day = base_date if day is None else day
        self.time_grid = utils.TimeGrid({day}, {day}, {day})
        self.base_date = base_date
        self.time_grid.set_base_date(base_date)
        # the one date IS the reporting date - a boundary registration reads report_index to
        # know the grid its counterfactual lands on, and reads its absence as "not reportable"
        self.time_grid.set_report_dates(base_date, {day})

    def __init_shared_mem(self, reporting_currency, mcmc_sim, calc_greeks, quasi_seed):
        # set BOTH ways every run - the switch is process-global
        torch.use_deterministic_algorithms(
            self.params['Deterministic_Kernels'] == 'Yes', warn_only=True)

        base_currency = utils.Factor(
            'FxRate', (self.config.params['System Parameters']['Base_Currency'],))

        all_vars_concat = None
        if calc_greeks:
            all_vars_concat = [x for x in self.static_var.items() if x[0] != base_currency]
            self.make_factor_index(list(self.static_var.items()))

        shared_mem = self.state(
            self.static_var, torch.ones([1, 1], dtype=self.dtype, device=self.device),
            mcmc_sim, get_fxrate_factor(utils.check_rate_name(reporting_currency), self.static_factors, {}),
            all_vars_concat, self.params['Greeks'] == 'All', quasi_seed)
        shared_mem.recompute_inner_mc = self.params.get('Recompute_Inner_MC', 'No') == 'Yes'
        # the SMOOTH estimator, declared on the calculations valuing as this one does; `execute`
        # has completed the block, so the key is present and the read direct
        shared_mem.branch_and_weight = self.params['Branch_And_Weight'] == 'Yes'
        # the reserve line a calibration WROTE on its factor, which nothing on the tensor side
        # carries: `pricing.skew_reserve` composes it with the portfolio's own two derivatives
        shared_mem.reserve_line = {
            key: {name: factor.declared[name] for name in utils.LogVar2FJ.RESERVE_LINE}
            for key, factor in self.static_factors.items()
            if all(name in getattr(factor, 'declared', {}) for name in utils.LogVar2FJ.RESERVE_LINE)}
        # a PER-DEAL reserve needs each deal's own value still on the tape, and nothing else here
        # does - so a document carrying no reserve line pays nothing for it
        shared_mem.keep_tensor = bool(shared_mem.reserve_line) and calc_greeks is not None
        return shared_mem

    def report(self):

        def check_prices(n, parent):

            def format_row(deal, data, val, greeks):
                data['Deal Currency'] = (deal.Factor_dep or {}).get(
                    'Local_Currency', deal.Instrument.field.get('Currency', self.params['Currency']))
                try:
                    data['Ref_MTM'] = float(deal.Instrument.field.get('MtM', 0.0))
                except ValueError:
                    data['Ref_MTM'] = 0.0
                for k, v in val.items():
                    if k.startswith('Greeks'):
                        greeks.setdefault(k, []).append(
                            self.gradients_as_df(v, header=deal.Instrument.field.get('Reference'), display_val=True))
                    elif k == 'Value':
                        data[k] = v.item()
                    elif k == 'Skew_Reserve':
                        data[k] = v

            block = []
            greeks = {}

            if 'Greeks_First' in n.obj.Calc_res:
                format_row(n.obj, dict(parent), n.obj.Calc_res, greeks)

            for sub_struct in n.sub_structures:
                data = dict(parent + [(field, sub_struct.obj.Instrument.field.get(field, 'Root'))
                                      for field in ['Reference', 'Object']])
                format_row(sub_struct.obj, data, sub_struct.obj.Calc_res, greeks)
                block.append(data)
                sub_block, sub_greeks = check_prices(
                    sub_struct, [('Parent', data['Reference'])])
                block.extend(sub_block)
                # aggregate the sub structure greeks
                for k, v in sub_greeks.items():
                    greeks.setdefault(k, []).extend(v)

            valuations = [deal.Calc_res for deal in n.dependencies]
            for deal, val in zip(n.dependencies, valuations):
                data = dict(parent + [(field, deal.Instrument.field.get(field, '?'))
                                      for field in ['Reference', 'Object']])
                format_row(deal, data, val, greeks)
                block.append(data)

            return block, greeks

        self.output = {}
        mtm, greeks = check_prices(
            self.netting_sets, [('Parent', self.netting_sets.obj.Instrument.field.get('Reference'))])

        data = dict(
            [(field, self.netting_sets.obj.Instrument.field.get(field, 'Root')) for field in ['Reference', 'Object']])
        # a structure the run could not value is skipped and counted, never summed
        data['Value'] = sum([x.obj.Calc_res['Value'].item() for x in self.netting_sets.sub_structures
                             if 'Value' in x.obj.Calc_res and x.obj.Instrument not in self.unpriced])
        # the forward-skew reserve is the PORTFOLIO's, as the gradient it is composed from is, so it
        # sits on the row that carries the portfolio's value
        if 'Skew_Reserve' in self.netting_sets.obj.Calc_res:
            data['Skew_Reserve'] = self.netting_sets.obj.Calc_res['Skew_Reserve']
        mtm.insert(0, data)

        self.output['mtm'] = pd.DataFrame(mtm)
        for greek_name, greek_val in greeks.items():
            # the transposed spelling is pandas' own migration off groupby(axis=1) (removed in
            # pandas 3) and runs identically back to 1.x
            if greek_name == 'Greeks_Second':
                summary = pd.concat(greek_val, axis=1).T.groupby(level=[0, 1, 2, 3, 4]).sum(skipna=False).T
            elif greek_name == 'Greeks_First':
                # a NaN gradient is reported as NaN, never summed away into a zero
                summary = pd.concat(greek_val, axis=1).T.groupby(level=0).sum(skipna=False).T
            else:
                raise Exception('Unknown Greek requested', greek_name)
            self.output.setdefault(greek_name, summary)

        if self.correlation_deltas:
            self.output['Correlation_Bump'] = pd.DataFrame(
                [{'Rate': name, 'Bump': width, 'Gradient': value,
                  'Source': 'CRN bump of the marked correlation - NOT ON THE TAPE'}
                 for name, (width, value) in self.correlation_deltas.items()]).set_index('Rate')

        return self.output

    def execute(self, params):
        # the declaration is the single source of an omitted field's default
        params = declared_defaults(type(self), params)
        base_date = pd.Timestamp(params['Run_Date'])
        self.params = params
        shared_mem = self.update_factors(params, base_date)
        logging.root.name = self.config.deals['Attributes'].get('Reference', self.config.file_ref)
        # a mark carries the flag its own factors were written with: a parameter the box holds
        # is not a fitted one, and the reading it feeds inherits that
        guarded = {utils.check_scope_name(key): factor.declared['On_Guard']
                   for key, factor in self.static_factors.items()
                   if getattr(factor, 'declared', {}).get('On_Guard')}
        if guarded:
            self.calc_stats['On_Guard'] = guarded
        self.calc_stats['Deal_Setup_Time'] = time.monotonic()
        self.netting_sets = DealStructure(Aggregation('root'), store_results=True)
        self.set_deal_structures(
            self.config.deals['Deals']['Children'], self.netting_sets, shared_mem.one, deal_level_mtm=True)

        self.calc_stats['Deal_Setup_Time'] = time.monotonic() - self.calc_stats['Deal_Setup_Time']
        self.calc_stats['Graph_Setup_Time'] = time.monotonic()

        # now ask the netting set to construct each deal - no looping required (just 1 timepoint)
        mtm = self.netting_sets.resolve_structure(shared_mem, self.time_grid)
        self.calc_stats['Graph_Setup_Time'] = time.monotonic() - self.calc_stats['Graph_Setup_Time']
        ns_obj = self.netting_sets.obj
        if ns_obj.Instrument.field.get('Reference') is None:
            ns_obj.Instrument.field['Reference'] = self.config.deals['Attributes'].get(
                'Reference', self.config.file_ref)

        if shared_mem.calc_greeks is not None:
            self.calc_stats['Greek_Execution_Time'] = time.monotonic()
            # each deal's own boundary terms, for the per-deal reserve alone
            by_deal = {} if shared_mem.keep_tensor else None
            if shared_mem.boundary_sets:
                if shared_mem.gamma:
                    raise utils.SecondOrderRefused(
                        "Greeks: 'All' is refused - these deals take a decision on simulated "
                        'state and registered a boundary correction: {}. That correction is what '
                        'makes their FIRST derivative right, and it is (gap - gap.detach()) times '
                        'a DETACHED coefficient - differentiate it a second time and the '
                        'coefficient cannot move, so what comes back is the smooth part of the '
                        'second derivative with the density-derivative term silently missing: a '
                        'plausible wrong gamma rather than a failure. The honest route for these '
                        'is to bump the ADJOINT under common random numbers - re-run '
                        "Greeks: 'First' on ONE seed at S+h and S-h and difference the reported "
                        "delta. Ask for 'All' on a portfolio without them.".format(
                            ', '.join(sorted({str(b.deal) for b in shared_mem.boundary_sets}))))
                # the portfolio value IS the objective - one scenario, whose mean is the reported
                # number. Worth zero forward, so only the tape gains a term
                correction = pricing.boundary_correction(
                    shared_mem, lambda value: value.sum(axis=0), mtm,
                    float(params.get('Boundary_AAD_Bandwidth', 0.01)), by_deal)
                if correction is not None:
                    mtm = mtm + correction
            pricing.greeks(shared_mem, ns_obj, mtm)
            self.calc_stats['Greek_Execution_Time'] = time.monotonic() - self.calc_stats['Greek_Execution_Time']
            if shared_mem.keep_tensor:
                started = time.monotonic()
                pricing.deal_reserves(shared_mem, self.netting_sets.deals(), by_deal)
                self.calc_stats['Deal_Reserve_Time'] = time.monotonic() - started

        # the quanto correlation delta, which no leaf carries. `Greeks: 'All'` is excluded because
        # the second-order report labels its axes off the first-order index
        if params['Greeks'] == 'First' and float(params['Correlation_Bump']) > 0.0:
            self.correlation_deltas = self.correlation_bump(
                params, float(params['Correlation_Bump']))
            for name, (width, delta) in self.correlation_deltas.items():
                ns_obj.Calc_res['Greeks_First'][name] = np.array([delta])
                self.gradient_index[name] = (np.zeros((1, 3), dtype=np.int64), 1)

        return {'Netting': self.netting_sets, 'Stats': self.calc_stats, 'Results': self.report()}


class Diary(BaseValuation):
    """The diary - every payment, fixing and expiry the book's own compile announces, as JSON rows.

    THE COMPILE'S SCHEDULE RE-EMITTED. The deals are constructed and their schedules bound exactly
    as a valuation binds them and then READ rather than priced, reached by `utils.walk_schedules` -
    the one walk the binding itself takes - so the diary and the pricer cannot disagree about a
    payment: they are the same schedule. Nothing here learns about users, workflow or storage: a
    row names its deal's `Reference`, its `deal_path` and the `netting_set` it sits in, and
    `spine.diary` stamps the instrument booked at that path and its derived key.

    A NULL AMOUNT IS NEVER WRITTEN AS 0.0. A floating leg's amount is not determined until its
    resets fix, and a settlement file carrying 0.0 for one is an instruction to pay nothing - so the
    row says `determined: false`, and `export_settlements` refuses it by name rather than exporting
    a zero.
    """

    #: What a row announces. A `barrier` row is a monitoring date rather than a due thing: what the
    #: level did is read off the fixing that satisfies it, and a close never waits on one.
    PAYMENT, FIXING, EXPIRY, BARRIER = 'payment', 'fixing', 'expiry', 'barrier'

    #: What a deal the compile could not read announces. Not a due thing - a reading of the book
    #: that says the book could not be read - and a close is never legal while one stands.
    UNREADABLE = 'unreadable'

    #: The book READ rather than valued: that row is how a reading refuses, so the compile skips an
    #: unreadable deal whatever `Exclude_Deals_With_Missing_Market_Data` says.
    valuation = False

    #: Where a row stands. The compile answers these three; `settled` is the record's own answer
    #: and is stamped by whoever holds the log.
    DUE, OBSERVED, EXPIRED = 'due', 'observed', 'expired'

    #: The legs that are not a schedule: a day the deal's declarations settle one payment on, and
    #: its last day.
    SETTLEMENT, EXPIRY_LEG = 'Settlement', 'Expiry'

    #: A physically settled option DELIVERS, and delivering is an act somebody takes - so its expiry
    #: vests a choice and the diary says so.
    PHYSICAL = 'Physical'
    ELECTION = 'election'

    def execute(self, params):
        """Every payment, fixing and expiry this job's deals carry, as rows naming each deal by its
        `Reference`, its positional `deal_path` and the netting set it sits in, unsorted:
        `BaseValuation.execute`'s compile half, READ rather than priced.

        The loaded tree is walked once and each leaf read off what the compile made of it, the
        compiled deal being the leaf itself or the deal its history resolved it to. A deal the
        compile LEFT OUT is judged as the compile judged it, as what its history made it, and is
        expired, announcing what a live one of that type announces - a knocked-out barrier its
        rebate - or unreadable, and announces why.
        """
        params = declared_defaults(type(self), self.judged(params))
        base_date = pd.Timestamp(params['Run_Date'])
        self.params = params
        shared = self.update_factors(params, base_date)
        self.netting_sets = DealStructure(Aggregation('root'), store_results=True)
        self.set_deal_structures(self.config.deals['Deals']['Children'], self.netting_sets,
                                 shared.one, deal_level_mtm=True)
        rows, compiled = [], {held.Instrument.resolved_from or held.Instrument: held
                              for held in self.netting_sets.deals()}
        options = self.config.params.get('Valuation Configuration', {})
        for path, netting_set, deal in self._leaves(self.config.deals['Deals']['Children']):
            held = compiled.get(deal)
            # a deal carrying the compile's verdict is read by it, which resolving again could raise
            became = deal if held is not None or hasattr(deal, 'unpriced_because') else (
                deal.resolve_history(base_date, self.config.holidays, options))
            rows.extend(dict(row, deal_path=path, netting_set=netting_set) for row in (
                self._deal_rows(held.Instrument, held.Factor_dep, base_date) if held is not None
                else self._dropped_rows(became) if self._expired(base_date, became)
                else [self._unreadable_row(base_date, deal)]))
        return rows

    @staticmethod
    def _within(deal, netting_set):
        """The netting set a deal's children sit in: its own `Reference` where it is one."""
        return deal.field.get('Reference') if deal.field.get('Object') == 'NettingCollateralSet' \
            else netting_set

    @classmethod
    def _leaves(cls, children, netting_set=None, path=()):
        """Every LEAF deal of a compiled config's tree, as `(deal path, netting set, deal)`, the set
        the `Reference` of the one it sits in and None outside one. A container is its structure
        and carries no schedule of its own."""
        for position, node in enumerate(children):
            deal, at = node.get('Instrument'), path + (position,)
            if node.get('Ignore') == 'True' or not isinstance(deal, Deal):
                continue
            if node.get('Children'):
                yield from cls._leaves(node['Children'], cls._within(deal, netting_set), at)
            else:
                yield '/'.join(map(str, at)), netting_set, deal

    def _expired(self, base_date, deal):
        """Whether the compile left this deal out because it has EXPIRED - its last date behind the
        base date, or the compile's own verdict, `utils.InstrumentExpired` - the engine's own
        expiry answer, and the only one used."""
        return isinstance(getattr(deal, 'unpriced_because', None), utils.InstrumentExpired) or (
            DealStructure.calc_time_dependency(base_date, deal, self.time_grid) is None)

    @classmethod
    def _dropped_rows(cls, deal):
        """What a deal the compile dropped as EXPIRED still announces: its expiry, and the fixing
        and the last settlement its type declares.

        An option whose expiry is already behind the base date is exactly the deal a catch-up rule
        exists for - one that fell due and nobody cleared - and the declared rows are field reads
        that need no schedule, so the expired branch says what the live one last said.
        """
        fields = deal.field
        terms = schema.OBSERVES.get(fields.get('Object'))
        reference = fields.get('Reference')
        rows = [cls._expiry_row(deal, cls.EXPIRED)]
        rows.extend(cls._expiry_fixing(fields, terms, index_named(fields, terms, deal.base_currency),
                                       reference, rows))
        settled = cls._settled_rows(fields, type(deal), reference)
        last = max((row['due_date'] for row in settled), default=None)
        rows.extend(row for row in settled if row['due_date'] == last)
        return rows

    def _unreadable_row(self, base_date, deal):
        """The one row a deal the compile COULD NOT READ leaves: what the engine said.

        A deal skipped for a price factor the market data has no block for has no schedule, no
        expiry and, without this, no trace at all - and a close check answering `legal` on a book
        it could not read is a clean bill nobody earned. The dependency build is asked once more,
        for this deal alone, so the row carries the engine's own sentence rather than a paraphrase
        of it.
        """
        try:
            deal.calc_dependencies(base_date, self.static_factors, self.stoch_factors,
                                   self.all_factors, self.all_tenors, self.time_grid,
                                   self.config.holidays)
            said = 'the compile left this deal out and reading it again raised nothing'
        except Exception as error:
            said = '{}'.format(error)
        return self._row(self.UNREADABLE, deal.field.get('Reference'), None, 0, None,
                         state=self.UNREADABLE, reason='{} ({}): {}'.format(
                             deal.field.get('Reference'), deal.field.get('Object'), said))

    @classmethod
    def _deal_rows(cls, deal, compiled, base_date):
        """One deal's whole diary: its schedules, its monitoring table, the payments its
        declarations settle, and its expiry."""
        fields = deal.field
        terms = schema.OBSERVES.get(fields.get('Object'))
        reference = fields.get('Reference')
        indices = index_named(fields, terms, deal.base_currency)
        settlement = deal.get_settlement_currencies()
        # a type declaring its observation table fixes on that table's days, which a print fills: a
        # reset schedule it compiles off the table on any other day is a coupon's or a settlement's
        table = fields.get(terms.table) if terms is not None and terms.table else None
        days = {pd.Timestamp(row[0] if isinstance(row, (list, tuple)) else row).date().isoformat()
                for row in table} if table else None
        rows = []
        for leg, schedule in walk_schedules(compiled):
            currency = cls._currency(fields, settlement)
            if isinstance(schedule, TensorCashFlows):
                rows.extend(cls._payment_rows(schedule, cls._compiled_at(compiled, leg), leg,
                                              base_date, currency, reference))
                if schedule.Resets is not None:
                    rows.extend(cls._fixing_rows(schedule.Resets, leg + '.Resets', base_date,
                                                 indices, reference))
            elif isinstance(schedule, TensorResets):
                rows.extend(row for row in cls._fixing_rows(schedule, leg, base_date, indices,
                                                            reference)
                            if days is None or row['due_date'] in days)
        rows.extend(cls._barrier_rows(fields, terms, indices, reference))
        rows.extend(cls._expiry_fixing(fields, terms, indices, reference, rows))
        rows.extend(cls._settled_rows(fields, type(deal), reference, since=base_date))
        rows.append(cls._expiry_row(deal, cls.DUE))
        return rows

    @classmethod
    def _payment_rows(cls, schedule, compiled, leg, base_date, currency, reference):
        """A leg's PAYMENTS - one per pay day, not one per schedule row.

        The amount is `pricing.fixed_payments`', the line the pricer discounts: the rate coupon and
        the fixed amount of every row sharing that day, summed, compounded where the leg's terms
        compound. A schedule carrying RESETS is not determined and says so - its amount is a
        floating pricer's and the diary does not spell a second one.
        """
        determined = schedule.Resets is None
        rows = schedule.schedule
        days, index, counts = np.unique(rows[:, CASHFLOW_INDEX_Pay_Day], return_index=True,
                                        return_counts=True)
        amounts = fixed_payments(
            rows[:, CASHFLOW_INDEX_FixedRate] * rows[:, CASHFLOW_INDEX_Year_Frac],
            rows[:, CASHFLOW_INDEX_Nominal], rows[:, CASHFLOW_INDEX_FixedAmt], index, counts,
            bool(compiled.get('Compounding', False)))
        return [cls._row(cls.PAYMENT, reference, leg, position,
                         base_date + pd.Timedelta(days=int(day)), currency=currency,
                         notional=float(rows[index[position], CASHFLOW_INDEX_Nominal]),
                         amount=float(amounts[position]) if determined else None,
                         determined=determined)
                for position, day in enumerate(days)]

    @staticmethod
    def _compiled_at(compiled, leg):
        """The compiled block a leg's schedule sits in, which is where its `Compounding` flag is."""
        for step in leg.split('.')[:-1]:
            compiled = compiled[step] if isinstance(compiled, dict) else compiled[int(step)]
        return compiled if isinstance(compiled, dict) else {}

    @classmethod
    def _asked(cls, row, indices):
        """`row` once per index the deal names, `{index: power}`: under its own leg where it names
        one, under the leg spelt with each where a cross names two, so every row keeps a key of its
        own. The level a row carries is its index's only where that index is the level itself."""
        if len(indices) < 2 and all(power > 0 for power in indices.values()):
            return [dict(row, index=next(iter(indices), None))]
        return [dict(row, index=index, observed=None, state=cls.DUE,
                     leg=row['leg'] if len(indices) == 1 else '{}.{}'.format(row['leg'], index))
                for index in indices]

    @classmethod
    def _fixing_rows(cls, resets, leg, base_date, indices, reference):
        """A reset schedule's observations, each one already fixed where its value is on the row."""
        rows = []
        for position, row in enumerate(resets.schedule):
            observed = float(row[RESET_INDEX_Value])
            due = base_date + pd.Timedelta(days=int(row[RESET_INDEX_Reset_Day]))
            rows.extend(cls._asked(cls._row(cls.FIXING, reference, leg, position, due,
                                            observed=observed or None,
                                            state=cls.OBSERVED if observed else cls.DUE), indices))
        return rows

    @classmethod
    def _barrier_rows(cls, fields, terms, indices, reference):
        """The deal's own monitoring table: a date and the close it fixed at, through the one
        reader that already tolerates both shapes of the row, a 0.0 placeholder read unfixed as the
        fixing rows read it. Only a table its type declares `monitors` has barrier days - one it
        averages or settles off has none."""
        if terms is None or not terms.monitors:
            return []
        return [asked for position, (date, observed)
                in enumerate(barrier_monitoring_rows(fields.get(terms.table) or []))
                for asked in cls._asked(cls._row(
                    cls.BARRIER, reference, terms.table, position, date, observed=observed or None,
                    state=cls.OBSERVED if observed else cls.DUE), indices)]

    @classmethod
    def _settled_rows(cls, fields, deal_type, reference, since=None):
        """The payments a deal's own declarations settle (`schema.declared_settlements`): one per
        `Cash` on every date a field or a table's column marks from `since` on - one behind the base
        date paid, as a schedule's paid row is dropped - in the currency it names, a blank one the
        deal's own `Currency`, and at the amount its terms state.

        `get_settlement_currencies()` is the reval-date accumulator - a barrier registers its
        monitoring days in it - so it is not a payment ladder and is never read as one. AN OPTION'S
        PAYOFF IS NOT IN A FIELD (`Units` times an intrinsic nobody has fixed), so its row is due
        with `amount: null`: the money still moves, and a close waits for the transition that moved
        it. A day settling one payment is the `Settlement` leg; one settling several names each by
        its amount.
        """
        declared, rows = declared_settlements(deal_type), []
        for key, column, cash, position, day in settlement_days(fields, declared):
            legs = sum(other[:2] == (key, column) for other in declared)
            if day is None or since is not None and pd.Timestamp(day) < since:
                continue
            stated = fields.get(cash.amount) if cash.amount else None
            amount = None if stated is None else cash.sign * float(stated)
            rows.append(cls._row(cls.PAYMENT, reference, cls.SETTLEMENT if legs == 1
                                 else cash.amount or cash.currency, position, day,
                                 currency=fields.get(cash.currency) or fields.get('Currency'),
                                 amount=amount, determined=amount is not None))
        return rows

    @classmethod
    def _expiry_fixing(cls, fields, terms, indices, reference, rows):
        """The observation a deal's EXPIRY needs, where the compile builds no reset schedule for it.

        A European option's payoff is its underlying's print on the expiry day, and nothing in the
        compiled tree announces it - so a close after expiry would wait for nothing at all. A deal
        whose own schedules already announce a fixing on that day announces it once.
        """
        day = fields.get(terms.expires) if terms is not None and terms.expires else None
        if day is None or not indices:
            return []
        date = pd.Timestamp(day).date().isoformat()
        if any(row['kind'] == cls.FIXING and row['due_date'] == date for row in rows):
            return []
        return cls._asked(cls._row(cls.FIXING, reference, cls.EXPIRY_LEG, 0, day), indices)

    @classmethod
    def _expiry_row(cls, deal, state):
        """One row per deal at the last day it can pay, carrying what its expiry leaves to an actor:
        `election` where the terms vest a choice, and null where a fixing determines the payoff."""
        fields = deal.field
        terms = schema.OBSERVES.get(fields.get('Object'))
        dates = deal.get_reval_dates()
        elects = terms is not None and terms.elects and fields.get(terms.elects) == cls.PHYSICAL
        return cls._row(cls.EXPIRY, fields.get('Reference'), cls.EXPIRY_LEG, 0,
                        max(dates) if dates else None, needs=cls.ELECTION if elects else None,
                        state=state)

    @classmethod
    def _row(cls, kind, reference, leg, position, due, **named):
        """One diary row: every declared field present, its deal named by its `reference`."""
        date = None if due is None else pd.Timestamp(due).date().isoformat()
        row = {'reference': reference, 'leg': leg, 'schedule_index': position, 'kind': kind,
               'currency': None, 'amount': None, 'determined': False, 'notional': None,
               'index': None, 'source': None, 'observed': None, 'needs': None, 'reason': None,
               'state': cls.DUE, 'due_date': date}
        row.update(named)
        return row

    @staticmethod
    def _currency(fields, settlement):
        """The currency a deal settles in: its OWN field, else the one currency its reval dates
        hold.

        Never a guess at which currency's date set carries the day: a two-currency deal's legs share
        a maturity, so that guess labels one leg with the other's currency and a settlement file
        nets the two. A deal whose legs really do settle apart is a container whose children are
        the legs, and each child names its own.
        """
        currency = fields.get('Currency')
        if isinstance(currency, str) and currency:
            return currency
        return next(iter(settlement)) if len(settlement) == 1 else None


class Scenario_State(Base_Reval_State):
    """A base valuation's state over a batch of given markets: each pass the values its
    calculation supplies per leaf of a simulated factor (`t_Supplied`), which the factors'
    processes publish (`t_Scenario_Buffer`), one path a scenario."""

    def __init__(self, *args, **kwargs):
        super(Scenario_State, self).__init__(*args, **kwargs)
        self.t_Supplied = {}
        self.t_Scenario_Buffer = {}

    def reset(self, supplied, scenarios):
        """A pass of `scenarios` markets, `supplied` the value of each simulated leaf on each,
        the last pass's memo, published paths and decisions cleared."""
        self.resize(scenarios)
        self.t_Supplied = supplied
        self.t_Scenario_Buffer.clear()
        self.t_Buffer.clear()
        self.boundary_sets.clear()


class ScenarioValuation(BaseValuation):
    """The book valued on one date against each of a batch of given markets: a base valuation
    whose scenarios are data, which `SIMM` fills from its moves.

    A scenario is the price factors it states over the job's own, `{name: block}`. Where there are
    scenarios, every factor whose type carries a scenario axis (`SuppliedValuesModel.factor_types`)
    but the base currency's price is simulated, a process publishing its value on each scenario of
    a pass, one path a scenario; the rest are static, and a scenario stating one of those - a
    model's parameters - is priced in a pass of its own, the factor written into its leaf and put
    back. With none it compiles and prices as a base valuation does. One pass of reset, generate and
    `resolve_structure` values every deal on every scenario of the pass as a base valuation of that
    market does, a Monte Carlo deal's inner paths the canonical block broadcast over them; the
    deals are aged to `Valuation_Date` as a credit Monte Carlo ages them to a later grid point.
    """
    documentation = ('Calculations', [
        'This values the portfolio on one date against each of a batch of given markets, every',
        'scenario a market as a base valuation reads one, priced together in one pass of the engine.',
        '',
        'The inputs are:',
        '',
        '- **Currency** of the output.',
        '- **Base Date** of the market data, and **Valuation Date** the date the deals are valued on:',
        '  the base date where blank, a later one aging every deal to it',
        '- **Scenarios** the markets, each the price factors it states over the job\'s own, by name;',
        '  a curve, a spot, a survival, inflation or forward curve, a basis and a vol surface or',
        '  space carry one value a scenario, and a scenario stating a model\'s parameters is priced',
        '  alone',
        '- **Batch Size** the scenarios priced in one pass, as many as half the device\'s free memory',
        '  holds where 0',
        '- **MCMC Simulations**, **Quasi Seed**, **Deterministic Kernels** and **Branch And Weight**',
        '  as a base valuation reads them, every scenario drawing the same inner paths',
        '',
        'The output `mtm` holds each deal\'s and each structure\'s value on the job\'s own market and',
        'on each scenario, one column a scenario.'
    ])

    #: `MCMC_Simulations`, declared once for the calculations valuing on given markets.
    MCMC_SIMULATIONS = F('MCMC_Simulations', 'Integer', default=1 << 14,
                         description='Inner Monte Carlo paths of each pricing, as a BaseValuation '
                                     'reads them')
    #: `Batch_Size`, declared once for the calculations valuing on given markets.
    BATCH_SIZE = F('Batch_Size', 'Integer', default=0,
                   description='Scenarios priced in one pass, as many as half the device\'s free '
                               'memory holds where 0')
    state = Scenario_State

    fields = [
        F('Base_Date', 'Date', default=''),
        F('Currency', 'Text', default=REQUIRED),
        F('Valuation_Date', 'Date', default='', convention=True,
          description='The date the deals are valued on, each scenario a market there; the base '
                      'date where blank'),
        F('Scenarios', 'Container', default=[],
          description='The markets valued, each the price factors it states over the job\'s own, '
                      'by name'),
        BATCH_SIZE,
        MCMC_SIMULATIONS,
        Calculation.QUASI_SEED,
        Calculation.DETERMINISTIC_KERNELS,
        F('Greeks', 'Text', default='No', values=['All', 'First', 'No'],
          description='First values on the tape, each deal\'s result block keeping its value on '
                      'every scenario as the tensor a reader differentiates, All to second order'),
        BaseValuation.BRANCH_AND_WEIGHT
    ]

    def __init__(self, config, **kwargs):
        super(ScenarioValuation, self).__init__(config, **kwargs)
        #: the device memory a scenario took in the first pass priced, in bytes
        self.scenario_bytes = None

    @classmethod
    def run(cls, context, overrides=None, prec=torch.float64):
        """Values `context`'s book on its market and on each scenario on the device
        `utils.calculation_device` picks, at `prec`, its block completed by `parameters` and
        judged: `(calculation, output)`."""
        calc = cls(context, device=utils.calculation_device(), prec=prec)
        return calc, calc.execute(calc.judged(cls.parameters(context, overrides)))

    def update_time_grid(self, base_date, day=None):
        """The grid of `Valuation_Date`, the base date where blank."""
        super(ScenarioValuation, self).update_time_grid(
            base_date, self.params['Valuation_Date'] or None)

    def update_factors(self, params, base_date):
        """The book's price factors compiled as a base valuation compiles them, then, where there
        are scenarios, each whose type carries a scenario axis handed to a process publishing them,
        its compiled leaves the values it stands at, and the report currency read as the factors
        then stand. Where the leaves are differentiated each result block keeps its taped value
        (`tensor`)."""
        shared = super(ScenarioValuation, self).update_factors(params, base_date)
        # the base currency's price is identically one, as no model simulates it either
        one = utils.Factor('FxRate', (self.config.params['System Parameters']['Base_Currency'],))
        for key in [key for key in self.static_factors if key.type in
                    SuppliedValuesModel.factor_types and key != one] if params['Scenarios'] else []:
            process = self.stoch_factors[key] = SuppliedValuesModel(self.static_factors.pop(key), None)
            process.leaves = [leaf for leaf, _ in self.split(key, process.factor.current_value())]
            self.stoch_var.update((leaf, self.static_var.pop(leaf)) for leaf in process.leaves)
        self.all_factors = {**self.stoch_factors, **self.static_factors}
        shared.Report_Currency = get_fxrate_factor(
            utils.check_rate_name(params['Currency']), self.static_factors, self.stoch_factors)
        shared.keep_tensor = shared.calc_greeks is not None
        return shared

    def nodes(self, structure=None):
        """`(parent reference, node)` for the root and every structure and deal beneath it,
        depth first, a structure before what it holds."""
        if structure is None:
            yield None, self.netting_sets.obj
            structure = self.netting_sets
        reference = structure.obj.Instrument.field.get('Reference')
        for sub in structure.sub_structures:
            yield reference, sub.obj
            yield from self.nodes(sub)
        for deal in structure.dependencies:
            yield reference, deal

    def passes(self, shared, scenarios):
        """`([[position]], [position])` - the passes `scenarios` are priced in: those stating no
        static factor `Batch_Size` at a time, or as many as half the device's free memory holds at
        what a scenario took in the first pass, and those stating one, each alone, a static factor
        taking one value a pass."""
        static = {utils.check_tuple_name(key) for key in self.static_factors}
        alone = [at for at, scenario in enumerate(scenarios) if static.intersection(scenario)]
        batched = sorted(set(range(len(scenarios))).difference(alone))
        budget = pricing.device_budget(shared) if shared.one.is_cuda else None
        size = max(1, self.params['Batch_Size'] or (
            int(budget // max(self.scenario_bytes or 0.0, 1.0)) if budget else len(batched)))
        self.calc_stats['Batch_Size'] = max(self.calc_stats.get('Batch_Size', 0), size)
        logging.debug('SCENARIO_BATCH %d scenarios %d a pass, %s bytes a scenario, budget %s',
                      len(batched), size, self.scenario_bytes, budget)
        return [batched[at:at + size] for at in range(0, len(batched), size)], alone

    def priced(self, shared, scenarios):
        """Every deal and structure valued on each of `scenarios` in one pass, its result block's
        `Value` one column a scenario: each simulated factor's process publishes its value on each,
        as the scenario states it or where it does not as its leaf stands, and a static factor the
        pass states is written into its leaf and put back after."""
        columns = [dict(self.leaves(scenario)) for scenario in scenarios]
        static = {key: value for column in columns for key, value in column.items()
                  if key in self.static_var}
        written = {key: self.static_var[key].detach().clone() for key in static}
        with torch.no_grad():
            for key, value in static.items():
                self.static_var[key].copy_(written[key].new_tensor(value))
        shared.reset({key: torch.stack([leaf.new_tensor(column[key]) if key in column else leaf
                                        for column in columns], dim=-1)
                      for key, leaf in self.stoch_var.items()}, len(scenarios))
        for _, node in self.nodes():
            node.Calc_res.clear()
        for process in self.stoch_factors.values():
            shared.t_Scenario_Buffer.update(process.generate(shared))
        # the first pass measures what a scenario takes, the peak from it on the run's own
        probe = shared.one.is_cuda and self.scenario_bytes is None
        if probe:
            held = torch.cuda.memory_allocated(shared.one.device)
            torch.cuda.reset_peak_memory_stats(shared.one.device)
        self.netting_sets.resolve_structure(shared, self.time_grid)
        if probe:
            self.scenario_bytes = (torch.cuda.max_memory_allocated(shared.one.device) - held) / len(
                scenarios)
        with torch.no_grad():
            for key, leaf in written.items():
                self.static_var[key].copy_(leaf)

    def execute(self, params):
        """Every deal's and structure's value on the job's market and on each of its `Scenarios`,
        the first priced alone to measure what a scenario takes."""
        params = self.params = declared_defaults(type(self), params)
        shared = self.update_factors(params, pd.Timestamp(params['Run_Date']))
        self.netting_sets = DealStructure(Aggregation('root'), store_results=True)
        self.set_deal_structures(self.config.deals['Deals']['Children'], self.netting_sets,
                                 shared.one, deal_level_mtm=True)
        nodes, scenarios = list(self.nodes()), params['Scenarios']
        table = np.full((len(nodes), len(scenarios) + 1), np.nan)

        def recorded(columns, markets):
            self.priced(shared, markets)
            for row, (_, node) in enumerate(nodes):
                if 'Value' in node.Calc_res:
                    table[row, columns] = node.Calc_res['Value'].reshape(-1)

        recorded([0], [{}])
        batched, alone = self.passes(shared, scenarios)
        for chunk in batched + [[at] for at in alone]:
            recorded([at + 1 for at in chunk], [scenarios[at] for at in chunk])
        frame = pd.DataFrame([{'Parent': parent, 'Reference': node.Instrument.field.get('Reference'),
                               'Object': node.Instrument.field.get('Object', 'Root')}
                              for parent, node in nodes])
        frame = pd.concat([frame, pd.DataFrame(table, columns=['Value'] + list(range(len(scenarios))))],
                          axis=1)
        return {'Netting': self.netting_sets, 'Stats': self.calc_stats, 'Results': {'mtm': frame}}


class HedgeMonteCarlo(CreditMonteCarlo):
    documentation = ('Calculations', [
        'A specialisation of `CreditMonteCarlo` that wires the same simulated scenario',
        'engine into a dynamic-hedging solver. Instead of producing exposure profiles, the',
        'calculation solves for a hedge of a portfolio of liabilities, trading a configured',
        'set of futures (or other instruments) over time. The Monte Carlo engine is unchanged',
        '— the additions are:',
        '',
        '- A **bundle** built per simulation batch containing the trajectories the solver',
        '  needs (tradable prices, liability MtM, factor history, leg metadata, AAD-derived',
        '  hedge ratios) plus an on-demand inner-MC fork (`Bundle.inner_mc`).',
        '- A **runtime** dict normalised from the JSON `Hedging_Problem` block (tradables,',
        '  position limits, cash accounts, objective, solver config).',
        '- A **differential-ML solver** (`DiffSolver`): a backward-DP value function fit by',
        '  the Huge–Savine twin loss (value + AAD pathwise gradient) under an asymmetric',
        '  utility objective, with `HindsightDpSolver` (clairvoyant upper bound) and the',
        '  textbook averaging hedge (lower bound) as benchmark tracks.',
        '',
        'The configuration contract is documented in the',
        '[Hedging_Problem](../json/index.md#calculation) section of the JSON reference.',
        '',
        '### Inner-MC Calculation fields',
        '',
        'The nested simulation is configured alongside the outer `Batch_Size`:',
        '`Inner_MC_Enabled` (`Yes`/`No`, required by `solve_hedge`), `Inner_Sub_Batch` (inner draws',
        'per outer path), `Inner_Antithetic` (`Yes` mirrors the Sobol draws as (z, −z) pairs — needs',
        'an even `Inner_Sub_Batch`), and `Inner_Draws` (`sobol` default, or `random` for iid',
        'Gaussians).',
        '',
        'Each fork runs in ONE pass at `Batch_Size x Inner_Sub_Batch` flat samples, so peak memory',
        'is a function of those two fields and nothing else — there is no cap, no partition and no',
        'host-dependent knob. A config too wide for the card raises CUDA OOM naming the fork: the',
        'config is the contract, and results follow the config rather than the machine.',
        '',
        '### Recommended operating point (measured, 24 GiB card)',
        '',
        'Peak scales LINEARLY in flat samples, at a rate that depends on the world (how many',
        'factors, how long the grid, how heavy the liability). Measured per flat sample:',
        '**~118 kB** on the test fixture, **~229 kB** on a larger walk-forward world',
        '(GARCH spot + observed basis + carry + a 31-tenor SOFR curve, 125-step grid). Size against',
        'YOUR world, not against a remembered number:',
        '',
        '| Batch_Size x Inner_Sub_Batch | fixture peak | larger world peak | s / fit step |',
        '| --- | --- | --- | --- |',
        '| 1024 x 64 | 7.4 GiB | 14.3 GiB | 0.74 |',
        '| **1280 x 64** | **9.3 GiB** | **18.3 GiB** | **0.75** |',
        '| 1536 x 64 | 11.1 GiB | 22.8 GiB | 0.83 |',
        '| 2048 x 64 | 14.8 GiB | OOM | 0.99 |',
        '',
        '`Batch_Size=1280, Inner_Sub_Batch=64` is the recommended operating point: 18.3 GiB of the',
        '23.6 GiB usable, ~5 GiB of headroom for world variation, and 0.75 s per fit step (1707',
        'paths/s — 1.3x the throughput of 1024 at the same step cost, since the step is launch- not',
        'bandwidth-bound below ~1300). 1536 fits but leaves only 0.8 GiB and is not recommended;',
        '2048 x 64 does NOT fit single-pass on the larger world and raises OOM inside the',
        "liability's cashflow pricing.",
        '',
        '### A solve is a stream (`Simulation_Batches`)',
        '',
        'A bundle is built per simulation batch inside the simulation loop and handed straight to',
        'a persistent solver: warmup on batch 1, step on each later batch, finish on the final',
        'batch, which is never trained on. What follows from that:',
        '',
        '- `Simulation_Batches` is a STREAM LENGTH here and a path MULTIPLIER under',
        '  `simulate_only`. Trained paths = `(Simulation_Batches - 1) x Batch_Size`; the last',
        '  batch is the held-out world the verdict and the benchmark tracks are measured on.',
        '  Minimum 2. `derivus_batch` divides it by the job count before that check.',
        '- Inner-MC fork width follows `Batch_Size` alone, so peak fork memory is set by',
        '  `Batch_Size x Inner_Sub_Batch` however long the stream is.',
        '- Every fit step sees paths no earlier step did, so overfitting to the simulated set is',
        '  not structurally possible.',
        '- The **frame is locked on the warmup batch**: the utility scale `c`, the market/wealth',
        '  standardization stats, and the per-t trust region are computed on batch 1 and frozen.',
        '  Later batches report how often their fitted targets fall outside the frozen region',
        '  (an INFO line per step) rather than re-fitting it.',
        '- A loaded checkpoint (`DiffV2_Load_Value_Fn`) is a frozen EVALUATION: it fits nothing,',
        '  so it is the degenerate stream of length one and `Simulation_Batches` must be 1 — that',
        '  single batch is the held-out world, since frozen nets saw none of it.',
        '- Checkpoints carry a `frame_stamp` (scale, z-frame, trust-region envelope); an ensemble',
        '  that MIXES frame provenances is refused.',
        '',
        'The walk-forward smoke gate reproduces this end to end — `gates/wf_smoke_gate.sh` at',
        '`--batch 512 --batches 5 --fit-iters 40`, seed 7 — pinning the training utility, the',
        'value function at the start, the greedy and no-hedge P&L, the churn and the',
        'policy-independent bound. It reads a walk-forward archive the repository does not carry.',
        '',
        '### Execution modes',
        '',
        '- `Execution_Mode = "solve_hedge"` — run the configured `Solver.Object` (DiffSolver).',
        '  Returns the fitted value-function artifact, the greedy-policy verdict, and the',
        '  benchmark comparison table + acceptance ladder.',
        '- `Execution_Mode = "simulate_only"` — build the bundle and run the no-trade baseline.',
        '  The result exposes `create_stepper()` to drive the simulator day-by-day with any',
        '  explicit policy (e.g. the textbook hedge). Useful for offline analysis and reporting.',
        '',
        '### Output',
        '',
        '`out[\'Results\']` contains the solver artifact, the simulated bundle, the normalized',
        'runtime, and an evaluation summary with terminal P&L statistics and a position-limit',
        'audit. Wallclock and device statistics are in `out[\'Stats\']`.',
        '',
        '### Reusing the inherited Credit Monte Carlo simulator',
        '',
        'Because this class inherits from `CreditMonteCarlo`, the underlying scenario',
        'engine — random factor generation, instrument valuation across paths, calendar',
        'handling, AAD graph construction — is identical. Anything that can be priced for',
        'a credit-exposure run can be priced as a hedging-problem leg or tradable. The only',
        'difference is what we do with the simulated MtMs: aggregate into exposures, or',
        'feed into a learning loop.'
    ])

    fields = [
        F('Base_Date', 'Date', default=''),
        F('Currency', 'Text', default=REQUIRED),
        F('Time_Grid', 'Text', default='0d 1d(1d) 4m'),
        F('Calendar', 'Text', default='', convention=True,
          description='Holiday calendar naming the business day the roll steps on'),
        F('Simulation_Batches', 'Integer', default=1),
        F('Batch_Size', 'Integer', default=1024),
        Calculation.RANDOM_SEED,
        Calculation.QUASI_SEED,
        Calculation.DETERMINISTIC_KERNELS,
        F('Tenor_Offset', 'Float', default=0.0,
          description='Years to shift every factor tenor by before the run'),
        F('MCMC_Simulations', 'Integer', default=2048),
        F('NoModel', 'Text', default='Constant', values=['Constant', 'RiskNeutral'],
          description='How a factor with no model is read: Constant holds it at its value today, '
                      'RiskNeutral reads a static curve at each row as its forward from that row; '
                      'a spot or a price index with no model is constant under either'),
        F('Antithetic', 'Text', default='No', values=['Yes', 'No']),
        F('Execution_Mode', 'Text', default='simulate_only',
          values=['simulate_only', 'solve_hedge']),
        F('Inner_MC_Enabled', 'Text', default='No', values=['Yes', 'No'],
          description='Required by solve_hedge - the nested simulation the DP sweep prices on'),
        F('Inner_Sub_Batch', 'Integer', default=0,
          description='Inner draws per outer path; peak memory is Batch_Size x this'),
        F('Inner_Antithetic', 'Text', default='No', values=['Yes', 'No'],
          description='Mirror the inner Sobol draws as (z, -z) pairs - needs an even '
                      'Inner_Sub_Batch'),
        F('Inner_Draws', 'Text', default='sobol', values=['sobol', 'random']),
        F('Scenario_Factors', 'Container', default=[],
          description='Price factors no deal reaches but the scenario set must simulate, named '
                      'as `type.name` strings'),
        F('Observed_Scenario', 'Text', default='', convention=True,
          description='Path to an npz of realized factor paths that replace the simulated draw'),
        F('Hedging_Problem', 'Container',
          description='The hedging problem itself - tradables, liabilities, objective, solver',
          sub_fields=[
              F('History_Lookback_Business_Days', 'Integer', default=30),
              F('Randomize_Initial_State', 'Text', default='No', values=['Yes', 'No']),
              F('Inner_Belief_Filter', 'Text', default='Yes', values=['Yes', 'No'],
                description='Publish a one-step filtered belief into the inner fork instead of '
                            'the privileged true-regime one-hot'),
              F('Tradable_Instruments', 'Container', default={},
                description='Deal blocks keyed by Object then by Reference'),
              F('Liabilities', 'Container', default={},
                description='Deal blocks keyed by Object then by Reference'),
              F('Portfolio_State', 'Container', default={},
                description='Opening positions, cash balances and posted margin'),
              F('Objective', 'Container',
                description='The utility SHAPE and its parameters, dispatched on Object',
                sub_fields=[
                    F('Object', 'Text', default=REQUIRED,
                      values=['AsymmetricUtility_Symlog', 'AsymmetricUtility_Huber',
                              'AsymmetricUtility_CARA', 'LogWealth'],
                      description='The utility shape the DP recursion works in; LogWealth is '
                                  'the per-step growth objective (reward = log(W1/W0), '
                                  'scale-free, implies Running_Wealth)'),
                    F('Utility_Scale_Mode', 'Text', default='vol_scaled_notional',
                      values=['vol_scaled_notional', 'conditional_sim'],
                      description='How the utility scale c is derived from the book; '
                                  'conditional_sim measures a per-decision-step schedule off the '
                                  'warmup batch instead of one number'),
                    F('Utility_Scale_Explicit', 'Float', default=None,
                      description='Literal dollar c, overriding the formula'),
                    F('Utility_Scale_Floor_Frac', 'Float', default=0.05,
                      description='Floor of the conditional_sim schedule, as a fraction of its '
                                  'terminal entry; inert under every other scale mode'),
                    F('Reference_Mode', 'Text', default='Fixed',
                      values=['Fixed', 'Running_Wealth'],
                      description='What the utility is applied to: TERMINAL wealth, or the '
                                  "day's wealth increment (a per-step reward the DP sums)"),
                    F('Reference_Wealth', 'Float', default=0.0,
                      description='Benchmark wealth in DOLLARS the utility is measured against; '
                                  'every shape works in x = (W - this) / c'),
                    F('Huber_Aversion', 'Float', default=2.5,
                      description='Curvature of the quadratic loss arm, in units of c'),
                    F('Huber_Delta', 'Float', default=1.0,
                      description='Knee beyond which the loss arm goes linear, in units of c'),
                    F('Up_Aversion', 'Float', default=0.0,
                      description='Curvature of the quadratic GAIN arm, in units of c; 0 leaves '
                                  'gains exactly linear'),
                    F('Up_Knee', 'Float', default=0.15,
                      description='Knee beyond which the gain arm goes linear, in units of c'),
                    F('CARA_Gamma', 'Float', default=1.0,
                      description='Absolute risk aversion of u = (1-exp(-gamma x))/gamma')]),
              F('Evaluator', 'Container',
                description='Accounting mode and cash instruments, dispatched on Object',
                sub_fields=[
                    F('Accounting_Mode', 'Text', default='futures',
                      values=['futures', 'cash_account'],
                      description='Whether a rebalance settles variation margin or moves cash'),
                    F('Transaction_Cost_Per_Unit', 'Float', default=0.0,
                      description='Flat turnover cost per contract, before the spread charge'),
                    F('Bid_Offer_Spread_Bps', 'Float', default=0.0,
                      description='Half-spread bps on notional; a spec dict may replace the '
                                  'scalar for maturity- and vol-dependent spreads'),
                    F('Roll_As_Calendar_Spread', 'Text', default='No', values=['Yes', 'No'],
                      description='Charge a matched roll one calendar half-cost instead of two '
                                  'outright half-spreads'),
                    F('Calendar_Spread_Bps', 'Float', default=None,
                      description='Half-spread bps of the calendar roll leg; setting it arms the '
                                  'matched-leg pricing everywhere'),
                    F('IM_Funding_Spread_Bps', 'Float', default=0.0,
                      description='Spread paid to fund initial margin; 0 switches the term off'),
                    F('IM_Vol_Multiplier', 'Float', default=0.0,
                      description='Initial margin as a multiple of notional at the reference vol'),
                    F('IM_Ref_Vol', 'Float', default=1.0,
                      description='Vol the margin multiplier is quoted at'),
                    F('Force_Flat_At_End', 'Text', default='Yes', values=['Yes', 'No'],
                      description='Close any residual book at the liability terminal'),
                    F('Total_Position_Abs_Limit', 'Float', default=0.0,
                      description='Cap on the absolute signed book total; 0 = uncapped'),
                    F('Max_Trade_Per_Step', 'Float', default=0.0,
                      description='Per-leg cap on |position change| per decision step at the '
                                  'argmax; 0 = uncapped'),
                    F('Decision_Deadband_Sigma', 'Float', default=0.0,
                      description='Standard errors of the paired inner-draw difference by which '
                                  'the argmax must beat HOLDING the book before it trades; 0 = '
                                  'trade on any improvement'),
                    F('Total_Position_Schedule', 'Table', default=None,
                      row=Row([F('Step', 'Integer', default=0),
                               F('Min_Total', 'Float', default=0.0),
                               F('Max_Total', 'Float', default=0.0)]),
                      description='Piecewise-constant corridor on the signed book total, by '
                                  'decision step'),
                    F('Allocation_Mode', 'Text', default='Exposure',
                      values=['Exposure', 'Carry_Variance'],
                      description='How the net cover splits across the hedge legs: Exposure = '
                                  'the declared Allocation_Weights table; Carry_Variance = '
                                  'per-step weights the solver DERIVES from the warmup sims'),
                    F('Allocation_Weights', 'Table', default=None,
                      row=Row([F('Step', 'Integer', default=0),
                               F('Instrument', 'Text', default=''),
                               F('Weight', 'Float', default=0.0)]),
                      description='Piecewise-constant split of the NET cover across the hedge '
                                  'legs, by decision step; absent, the argmax searches the product '
                                  'of per-leg levels')]),
              F('Solver', 'Container',
                description='The value-function solver and its schedule, dispatched on Object',
                sub_fields=[
                    F('Object', 'Text', default=REQUIRED,
                      values=['DiffSolver', 'DiffSolverV2', 'HindsightDpSolver'],
                      description='The value-function solver; solve_hedge requires DiffSolver '
                                  '(DiffSolver is the legacy spelling of the same solver)'),
                    F('Multi_Seed_Count', 'Integer', default=1,
                      description='Independent training seeds the artifact is selected across'),
                    F('T_Min', 'Integer', default=0,
                      description='Earliest step the backward sweep fits; 0 = full sweep'),
                    F('Training_Action_Grid_Levels_Per_Axis', 'Integer', default=11,
                      description='Levels per hedge axis in the greedy action grid, or rungs on '
                                  'the NET cover under Evaluator.Allocation_Weights'),
                    F('Training_Action_Chunk_Size', 'Integer', default=64,
                      description='Actions scored per batched argmax pass'),
                    F('Use_Advantage_Decomp', 'Text', default='Yes', values=['Yes', 'No'],
                      description='Fit the NN residual over the bounded-utility anchor rather '
                                  'than the continuation itself'),
                    F('DiffV2_Fit_Iters', 'Integer', default=150,
                      description='Adam iterations per residual net'),
                    F('DiffV2_LR', 'Float', default=2.0e-3, description='Adam learning rate'),
                    F('DiffV2_Bank_Noise_Frac', 'Float', default=0.15,
                      description='Bank q-exploration noise as a fraction of each [Min,Max] range'),
                    F('DiffV2_Risk_Aversion', 'Float', default=1.0,
                      description='The backward DP aversion, dividing the capital line in the '
                                  'LogWealth reward (1.0 neutral; higher = less capital at risk = '
                                  'more averse)'),
                    F('DiffV2_Drift_Threshold_Sigmas', 'Float', default=3.0,
                      description='The drift tripwire tail: how many validation-measured null '
                                  'sigmas the inference CUSUM must exceed to trip'),
                    F('DiffV2_Drift_Beta', 'Float', default=0.0,
                      description='On-trip correction strength: the forecast used by the '
                                  'ranking is biased toward the REALIZED drift by beta times '
                                  'the observed average residual; 0 = report-only'),
                    F('DiffV2_Load_Horizon_Pad', 'Text', default='No', values=['Yes', 'No'],
                      description='Yes loads a value-function checkpoint fitted on another '
                                  'decision horizon, clamped to its saved range with the last '
                                  'fitted step repeated; No refuses the mismatch'),
                    F('DiffV2_Returns_State', 'Text', default='No', values=['Yes', 'No'],
                      description='Yes makes the value state dimensionless: price columns as '
                                  'log-returns vs the calibrated t0 spot, basis columns as '
                                  'fractions of it, wealth as a fraction of the t0 book '
                                  'notional'),
                    F('DiffV2_Weight_Decay', 'Float', default=0.0,
                      description='Residual-net weight decay'),
                    F('DiffV2_Hidden', 'Integer', default=32,
                      description='Hidden width of each residual net'),
                    F('DiffV2_Lambda_Grad', 'Float', default=1.0,
                      description='Twin-loss weight on the pathwise-gradient term'),
                    F('DiffV2_Risk_Kappa', 'Float', default=0.0,
                      description='Downside semideviation penalty at the argmax; 0 = plain E[C]'),
                    F('DiffV2_Cost_Aware_Argmax', 'Text', default='No', values=['Yes', 'No'],
                      description='Charge the L1 repositioning cost at the verdict argmax'),
                    F('DiffV2_Fit_Tol', 'Float', default=0.001,
                      description='Relative loss-plateau tolerance at which an INHERITED '
                                  'net\'s fit stops early; 0 = never stop early'),
                    F('DiffV2_Temporal_Proximity', 'Float', default=0.0,
                      description='Weight pulling each net\'s parameters toward its fitted '
                                  'successor\'s during the fit; 0 = off'),
                    F('DiffV2_Churn_Lambda', 'Float', default=0.0,
                      description='Quadratic repositioning charge in currency per contract^2, '
                                  'taken off the wealth entering the continuation at the argmax '
                                  'and the training-label argmax; 0 = off'),
                    F('DiffV2_Position_State', 'Text', default='No', values=['Yes', 'No'],
                      description='Frictional Bellman: the signed net book fraction becomes a '
                                  'state coordinate of the fitted value and the repositioning '
                                  'charge enters the regressed target'),
                    F('DiffV2_Wealth_Free_Value', 'Text', default='No', values=['Yes', 'No'],
                      description='Drop the wealth column from the value net\'s inputs, so the '
                                  'fitted residual reads market state (and the position, under '
                                  'DiffV2_Position_State) alone'),
                    F('DiffV2_Stepper_Rollout', 'Text', default='No', values=['Yes', 'No'],
                      description='Roll a frozen policy day-by-day through the real accounting'),
                    F('DiffV2_Decision_Curve_Dump', 'Text', default='',
                      description='Path a per-decision CSV of the stepper rollout\'s FULL '
                                  'ranking curve is written to; empty = off'),
                    F('DiffV2_Per_Column_Grad_Norm', 'Text', default='Yes', values=['Yes', 'No'],
                      description='Normalize twin-loss greeks per input column; No = pooled'),
                    F('DiffV2_Save_Value_Fn', 'Text', default='',
                      description='Path the fitted nets and their frame are written to'),
                    F('DiffV2_Load_Value_Fn', 'Text', default='',
                      description='Checkpoint to evaluate frozen; a LIST loads an ensemble'),
                    F('Run_Hindsight_Diagnostic', 'Text', default='No', values=['Yes', 'No'],
                      description='Assemble the clairvoyant upper-bound benchmark track'),
                    F('Run_Textbook_Benchmark', 'Text', default='No', values=['Yes', 'No'],
                      description='Assemble the averaging-hedge lower-bound benchmark track')])])
    ]

    @classmethod
    def run(cls, context, overrides=None, prec=torch.float32):
        """Runs the hedge on `context` on the device `utils.calculation_device` picks, at `prec`,
        its block completed by `parameters`: `(calculation, HedgeRuntimeExecutionResult)`."""
        calc = cls(context, device=utils.calculation_device(), prec=prec)
        return calc, calc.execute(cls.parameters(context, overrides))

    @staticmethod
    def _factor_bundle_key(factor_key):
        return utils.check_tuple_name(factor_key) if hasattr(factor_key, 'type') and hasattr(factor_key, 'name') else factor_key

    def _init_shared_mem(self, seed, quasi_seed, nomodel, reporting_currency, mcmc_sim, job_id,
                         num_jobs, calc_greeks=None):
        """Build the `CMC_State_Inner` this calculation runs on, so one shared_mem hosts both
        the outer mode (inherited `reset()`) and the inner one (`reset_inner()`).
        HedgeMonteCarlo computes neither greeks nor FVA, so the parent's factor-index and
        survival-scaling setup is skipped."""
        return CMC_State_Inner(
            self.get_cholesky_decomp(), self.static_var, self.batch_size,
            torch.ones([1, 1], dtype=self.dtype, device=self.device), mcmc_sim, get_fxrate_factor(
                utils.check_rate_name(reporting_currency), self.static_factors, self.stoch_factors),
            seed, quasi_seed, job_id, num_jobs,
            simulation_sub_batch=int(self.params.get('Inner_Sub_Batch', 0)),
            keep_tensor=True)  # hedge stepper replays need the kept tensor, unconditionally

    @staticmethod
    def _require_all_compiled(declared, structure, role):
        """Raise unless every declared deal compiled into `structure`.

        The pricing walk's skip-and-continue is the right contract for a reporting book, but here a
        skipped tradable shrinks the solver's menu and a skipped liability shrinks the target it is
        hedging, so the solve reports a confident answer to a different problem."""
        loaded = ({d.Instrument.field.get('Reference') for d in structure.dependencies
                   if d.Factor_dep is not None} |
                  {s.obj.Instrument.field.get('Reference') for s in structure.sub_structures})
        missing = [n['Instrument'].field.get('Reference') for n in declared
                   if n['Instrument'].field.get('Reference') not in loaded]
        if missing:
            raise Exception(f'HedgeMonteCarlo: {role} legs failed to compile and were skipped: '
                            f'{missing} — a hedge book prices whole or not at all')

    def update_factors(self, params, base_date, job_id, num_jobs, end_date):
        """Build the factor state from the deal-driven dependencies plus the calculation's explicit
        `Scenario_Factors` - factors no deal reaches through a schema edge (a basis consumed only by
        a composed spot).

        The horizon is the latest tradable reset date capped at `end_date` (the liability terminal):
        hedge maturities past liability end leave the simulation horizon, while the hedges price
        through liability end and any residual position closes out at fair value there."""
        dependent_factors, stochastic_factors, _, reset_dates, settlement_currencies = self.config.calculate_dependencies(
            params, base_date, self.input_time_grid)
        for name in params.get('Scenario_Factors', []):
            factor_type, factor_name = name.split('.', 1)
            dependent_factors.setdefault(utils.Factor(factor_type, utils.check_rate_name(factor_name)), [])

        # horizon = max tradable reset date, capped at the liability terminal
        max_expiry = min(max(reset_dates), end_date)
        reset_dates = self.config.parse_grid(base_date, max_expiry, self.input_time_grid, past_max_date=True)
        reset_dates.update({base_date, max_expiry})
        self.update_time_grid(base_date, reset_dates, settlement_currencies, dynamic_scenario_dates=True)

        # Use the last scenario grid date so ScenarioTimeGrid covers the extra step from past_max_date=True
        last_scen_date = base_date + pd.DateOffset(days=int(self.time_grid.scen_time_grid[-1]))
        dependent_factors = {k: last_scen_date for k in dependent_factors}
        stochastic_factors, additional_factors = self.config.find_models(dependent_factors)

        shared_mem = self._build_factor_state(
            dependent_factors, stochastic_factors, additional_factors, params, base_date, job_id, num_jobs)
        return shared_mem

    def _liability_schedule_scalars(self):
        """`(total_leg_volume, last_payment_day)` off the cashflow schedules - the summed |notional|
        across all liability legs and the latest payment day as a day offset from base_date, which
        the symlog utility-scale needs. `Bundle.from_batch` maps that day onto the
        (history-prefixed) bundle time grid to recover `last_settlement_index`."""
        return self.liabilities.aggregate_leg_descriptors()

    def _reveal_transform(self, key, block, kind):
        """Returns-state coordinate map (identity when the switch is off): a CommodityPrice's
        CONTINUOUS segment becomes log(x / calibrated t0 spot); an ObservedBasis segment becomes
        x / that spot. Sufficient statistics (log h, beliefs) and rate curves pass through."""
        if not self._returns_state:
            return block
        if key.type == 'CommodityPrice' and kind == REVEAL_CONTINUOUS:
            return (block / self._state_spot0).log()
        if key.type == 'ObservedBasis':
            return block / self._state_spot0
        return block

    def execute(self, params, job_id=0, num_jobs=1):
        """Simulate the scenario engine over batches, building the tensor bundle (tradable prices,
        liability MtM, factor paths), and return a `HedgeRuntimeExecutionResult`.

        A SOLVE IS A STREAM: `solve_hedge` builds one Bundle per batch inside the batch loop and
        hands it to a persistent solver as it is built - warmup on batch 1, step on each later
        batch, finish on a held-out final batch - so the inner-MC forks are only ever `Batch_Size`
        wide and every fit step sees fresh paths. `simulate_only` instead accumulates every batch
        into one bundle, for which `Simulation_Batches` is a path multiplier rather than a stream
        length.

        LIABILITY-DRIVEN TIME-GRID CAP: the global grid stops at the liability's last cashflow /
        reval date. Past it there is nothing to hedge, and any residual position closes out at fair
        value.

        INNER-MC SETUP: the process copies are forked only after outer setup precalc has populated
        `factor_key` / `spot0`, so inner-mode precalc cannot clobber outer attrs.

        `Randomize_Initial_State='Yes'` gives the diff-ML boundary label the variance in z_0 it
        needs, via a per-batch burn-in: run each process once from the calibrated t=0, snapshot the
        terminal state per path, re-precalculate from that snapshot. Batch k+1 REWINDS to the
        calibrated t=0 first, or the batch sequence random-walks away from it instead of drawing N
        independent worlds (measured over 5 batches: the symlog scale drifted +94%, to NaN sweeps
        some months). Single-batch runs never rewind, so every `Simulation_Batches=1` job is
        bit-identical.

        `Observed_Scenario` (walk-forward backtest) replaces the simulated draw with grid-aligned
        realized paths from an .npz keyed by factor name; all preparation lives in the driver.

        The declared underlying(s) are leafed so the base-delta / conditional-feature pass can read
        d(value)/d(spot) via AAD. The diff-ML solver differentiates the continuation off its own
        fresh state-at-t leaves (`Bundle.inner_mc_grad`), so it needs no outer leaf.
        """
        # the declaration is the single source of an omitted field's default
        params = declared_defaults(type(self), self.judged(params))
        # `.get` with no fallback on purpose: HedgeMonteCarlo declares no `Greeks` field, so there
        # is no default to publish - it only refuses the value that would silently do nothing
        if params.get('Greeks') == 'All':
            raise Exception(
                "Greeks: 'All' is not a HedgeMonteCarlo parameter - this calculation reports a "
                'hedge, not a sensitivity block, and nothing here reads Greeks at all, so the key '
                'would be silently ignored rather than honoured. The second-order block is '
                "BaseValuation's ('Greeks': 'All' there); the AAD this calculation does run is the "
                'solver\'s own pathwise gradient, configured under Hedging_Problem.')
        base_date = pd.Timestamp(params['Run_Date'])
        self.input_time_grid = params['Time_Grid']
        params['Simulation_Batches'] = params['Simulation_Batches'] // num_jobs
        self.batch_size = params['Batch_Size']
        self.numscenarios = self.batch_size * params['Simulation_Batches']
        self.params = params

        logging.root.name = self.config.deals['Attributes'].get('Reference', self.config.file_ref)
        self.calc_stats['Batch_Size'] = self.batch_size
        self.calc_stats['Simulation_Batches'] = params['Simulation_Batches']
        self.calc_stats['Random_Seed'] = params['Random_Seed']

        execution_mode = params.get('Execution_Mode', 'simulate_only')
        hedging_problem = params.get('Hedging_Problem', {})
        # forwarded to each process's `reseed_inner_state` OPAQUELY - the calc never reads a model
        # switch out of it; a process owns which keys mean what to it
        self._inner_state_opts = hedging_problem

        instruments = self.config.deals_from_object_map(hedging_problem.get('Tradable_Instruments', {}))
        liabilities = self.config.deals_from_object_map(hedging_problem.get('Liabilities', {}))
        self.config.set_calculation_children(instruments + liabilities)
        # cap the grid at the liability terminal - past it there is nothing to hedge
        end_date = DealStructure.max_settlement_date(liabilities, self.config.holidays)
        shared_mem = self.update_factors(params, base_date, job_id, num_jobs, end_date=end_date)
        # the valuation structure first; the hedging runtime consumes the live factor and
        # instrument tensors this same loop produces
        self.netting_sets = DealStructure(Aggregation('root'), store_results=True)
        self.set_deal_structures(instruments, self.netting_sets, shared_mem.one, deal_level_mtm=True)
        self._require_all_compiled(instruments, self.netting_sets, 'tradable')
        self.netting_sets.finalize_struct(base_date, self.time_grid)

        self.liabilities = DealStructure(Aggregation('contracts'), store_results=False)
        # the inner-MC fork windows `Time_dep` and shares `Factor_dep` by reference on the same
        # `shared_mem`, so its copies price off the schedules bound here
        self.set_deal_structures(liabilities, self.liabilities, shared_mem.one, deal_level_mtm=False)
        self._require_all_compiled(liabilities, self.liabilities, 'liability')
        self.liabilities.finalize_struct(base_date, self.time_grid)

        t_days_arr = self.time_grid.scenario_grid[:, utils.TIME_GRID_MTM]  # [T]
        execution_label = 'Tensor_Execution_Time ({})'.format(self.device.type)
        self.calc_stats[execution_label] = time.monotonic()

        normalized_runtime = construct_hedge_runtime(
            params, stoch_factors=self.stoch_factors,
        )
        # canonical underlying (commodity-spot) factor-name set, derived once by the runtime; the
        # spot-leaf pass and `_find_spot_key` read it rather than re-deriving it
        self._underlying_names = set(normalized_runtime['referenced_commodities'])

        # Inner-MC setup; the copies below are forked only after outer setup precalc has run
        inner_mc_enabled = params.get('Inner_MC_Enabled', 'No') == 'Yes'
        tradable_refs = sorted(normalized_runtime['names']['hedges']) if inner_mc_enabled else ()

        # solve_hedge: inner MC runs in the backward DP/MPC sweep, not the outer loop, so the
        # per-batch outer scenario buffer is cached for it to fork on demand
        solve_hedge_mode = str(execution_mode).lower() == 'solve_hedge'
        if inner_mc_enabled:
            self.stoch_factors_inner = {k: proc.copy() for k, proc in self.stoch_factors.items()}
        streaming_solve = StreamingSolve(normalized_runtime) if solve_hedge_mode else None
        held_out = None

        # Per-batch tensor accumulators. `simulate_only` appends every batch and concatenates once
        # at the end; a solve re-inits them each batch (one block per key).
        def _new_blocks():
            return ({self._factor_bundle_key(key): [] for key in self.stoch_factors},
                    defaultdict(list),
                    {'mtm': [], 'realized_cashflows': defaultdict(list)},
                    defaultdict(list))

        # `privileged_factor_blocks` is keyed by (factor_name, factor_attr) — whatever the process
        # exposes via `privileged_factors()`.
        (factor_tensor_blocks, tradable_blocks, hedge_profile_blocks,
         privileged_factor_blocks) = _new_blocks()
        total_leg_volume, last_payment_day = self._liability_schedule_scalars()
        bus_day = self.config.holidays.get(
            self.params['Calendar'], {'businessday': pd.offsets.BDay(1)})['businessday']
        # per-batch burn-in: variance in z_0 for the diff-ML boundary label
        randomize_t0 = hedging_problem.get('Randomize_Initial_State', 'No') == 'Yes'
        # returns-state coordinates: dimensionless market columns off one constant divisor
        # per run, so outer rows and fork reveals share a coordinate system across epochs
        self._returns_state = (hedging_problem.get('Solver') or {}).get(
            'DiffV2_Returns_State', 'No') == 'Yes'
        self._state_spot0 = None
        if self._returns_state:
            price_keys = [k for k in self.stoch_factors if k.type == 'CommodityPrice']
            if len(price_keys) != 1:
                raise ValueError(
                    f'DiffV2_Returns_State=Yes needs exactly one simulated CommodityPrice to '
                    f'anchor the coordinates; this world has {len(price_keys)}')
            self._state_spot0 = float(
                self.stoch_var[price_keys[0]].detach().reshape(-1)[0])
            normalized_runtime['objective']['state_notional'] = (
                self._state_spot0 * total_leg_volume)
        # walk-forward replay: substitute observed paths; the driver owns all the prep
        observed = None
        if params.get('Observed_Scenario'):
            npz = np.load(params['Observed_Scenario'])
            by_name = {utils.check_tuple_name(k): k for k in self.stoch_factors}
            observed = {by_name[n]: shared_mem.one.new_tensor(npz[n]).unsqueeze(-1)
                        for n in npz.files if n in by_name}
            unmatched = [n for n in npz.files if n not in by_name]
            if unmatched:
                raise ValueError(
                    f'Observed_Scenario keys matched no simulated factor: {unmatched}; '
                    f'available={sorted(by_name)}')
            logging.info('Observed_Scenario substituted %d factor(s): %s', len(observed),
                         [utils.check_tuple_name(k) for k in observed])

        for run in range(params['Simulation_Batches']):
            shared_mem.reset(
                self.num_factors, self.time_grid,
                use_antithetic=params.get('Antithetic', 'No') == 'Yes')

            # session-print conditioning from STATE: the calibrated t0 values ARE this batch's first
            # step, so a process informing another off a print there publishes its shift here
            t0_state = {k: v.reshape(1, 1, -1) for k, v in self.stoch_var.items()}
            print_keys = self._publish_print_seeds(
                self.stoch_factors.items(), t0_state, 0, shared_mem)

            if randomize_t0:
                # REWIND to the calibrated t=0 first, or the batch sequence random-walks away from
                # it (measured: symlog scale 592k -> 1.15M over 5 batches, NaN sweeps some months)
                if run:
                    for key, proc in self.stoch_factors.items():
                        scenario_grid, implied_tensor = self._factor_precalc_args[key]
                        proc.precalculate(
                            self.base_date, scenario_grid, self.stoch_var[key], shared_mem,
                            self.process_ofs[key], implied_tensor=implied_tensor)
                # burn-in for every factor; `simulated[-1]` is the per-path shape precalculate
                # accepts - the same contract `_run_inner_mc_at_t` forks on
                initial_t0 = {}
                outer_reseeds = {}
                for key, proc in self.stoch_factors.items():
                    simulated = proc.generate(shared_mem)
                    # publish as we go: linked factors read their underlying's path here (topo order)
                    shared_mem.t_Scenario_Buffer[key] = simulated
                    initial_t0[key] = simulated[-1].detach()
                    # each process owns its t=0 seed for the next run; detached now so it survives
                    # the buffer-clearing reset below
                    outer_reseeds.update(proc.outer_reseed())
                # independent innovation stream for the main run
                shared_mem.reset(
                    self.num_factors, self.time_grid,
                    use_antithetic=params.get('Antithetic', 'No') == 'Yes')
                for key, init_state in initial_t0.items():
                    scenario_grid, implied_tensor = self._factor_precalc_args[key]
                    self.stoch_factors[key].precalculate(
                        self.base_date, scenario_grid, init_state,
                        shared_mem, self.process_ofs[key],
                        implied_tensor=implied_tensor)
                for seed_key, seed_val in outer_reseeds.items():
                    shared_mem.t_Scenario_Buffer[seed_key] = seed_val
                # the restart's state carries each path's OWN prints - condition on them per path,
                # exactly as a fork conditions on its day's prints
                restart_state = {k: v.reshape(1, 1, -1) for k, v in initial_t0.items()}
                print_keys |= self._publish_print_seeds(
                    self.stoch_factors.items(), restart_state, 0, shared_mem)

            for key, proc in self.stoch_factors.items():
                simulated = proc.generate(shared_mem)
                if observed is not None and key in observed:
                    # the driver supplies a dense daily path from the base date; take the
                    # sim-grid-length prefix (Time_Grid is daily) and broadcast to the batch
                    simulated = observed[key][:simulated.shape[0]].expand(
                        *simulated.shape).contiguous()
                    # the process republishes its own path-dependent revealed state along the
                    # replayed path (belief, log-variance); base processes no-op
                    proc.reseed_from_path(simulated, shared_mem)
                # leaf the declared underlying(s) for the base-delta / conditional-feature pass
                if utils.check_tuple_name(key) in self._underlying_names:
                    simulated = simulated.detach().requires_grad_(True)
                shared_mem.t_Scenario_Buffer[key] = simulated
                # each process owns its privileged-factor surface; the default returns {}, so a
                # process opts in by overriding `privileged_factors`
                priv = proc.privileged_factors(simulated)
                if priv:
                    factor_name = key.name[0] if key.name else str(key)
                    for attr_name, tensor in priv.items():
                        privileged_factor_blocks[(factor_name, attr_name)].append(tensor.detach().clone())

            # print seeds are CONSUMED state, not path series: leaving them in the buffer would make
            # the fork's republication filter mistake a per-path scalar for a path
            for k in print_keys:
                shared_mem.t_Scenario_Buffer.pop(k, None)

            # solve_hedge: snapshot this batch's outer scenario buffer (factor paths + every
            # per-process aux key each generate() published) — the forks run against THIS batch.
            batch_buffer = ({key: tensor.detach().clone()
                             for key, tensor in shared_mem.t_Scenario_Buffer.items()}
                            if solve_hedge_mode else None)

            _ = self.netting_sets.resolve_structure(shared_mem, self.time_grid)
            # clear hedge cashflows so t_Cashflows after the next call holds only liability cashflows
            shared_mem.reset_cashflows(self.time_grid)
            # the liability mark, post-process-free (no per-batch GPU->CPU save_results copy)
            mtm = self.liabilities.resolve_hedge_structure(shared_mem, self.time_grid).get('mtm')
            if mtm is not None:
                hedge_profile_blocks['mtm'].append(mtm.detach().clone())

            mtm_grid_size = self.time_grid.mtm_time_grid.size
            for currency, by_time in (shared_mem.t_Cashflows or {}).items():
                dense = shared_mem.one.new_zeros(mtm_grid_size, shared_mem.simulation_batch)
                for t_idx, payoff in by_time.items():
                    dense[int(t_idx)] = payoff
                hedge_profile_blocks['realized_cashflows'][str(currency)].append(dense.detach().clone())

            if factor_tensor_blocks is not None:
                for key in self.stoch_factors:
                    factor_tensor_blocks[self._factor_bundle_key(key)].append(
                        shared_mem.t_Scenario_Buffer[key].detach().clone()
                    )

            trade_tensors = self.netting_sets.tensor_marks()

            if tradable_blocks is not None:
                for instrument_name, instrument_tensor in trade_tensors.items():
                    tradable_blocks[instrument_name].append(instrument_tensor.detach().clone())

            shared_mem.t_Buffer.clear()

            if solve_hedge_mode:
                # this batch IS a bundle: build it, attach its forks, hand it to the solver. The
                # last batch is never fitted - it is the held-out world `finish` measures
                bundle = Bundle.from_batch(
                    base_date, bus_day, shared_mem.one.new_tensor(t_days_arr),
                    tradable_blocks, factor_tensor_blocks, hedge_profile_blocks, 1,
                    self.stoch_factors, normalized_runtime,
                    privileged_factor_blocks=privileged_factor_blocks,
                    total_leg_volume=total_leg_volume, last_payment_day=last_payment_day)
                self._attach_inner_mc(bundle, batch_buffer, shared_mem, base_date, tradable_refs)
                if run == 0:
                    streaming_solve.warmup(bundle)
                elif run < params['Simulation_Batches'] - 1:
                    streaming_solve.step(bundle)
                # the LAST batch is the held-out world. At Simulation_Batches == 1 - a frozen policy,
                # which fits nothing - warmup's batch is also that world
                if run == params['Simulation_Batches'] - 1:
                    held_out = bundle
                # fresh accumulators - this batch is consumed. The solver's forks borrowed
                # `shared_mem` and handed it back as found, so outer generation resumes unaffected
                (factor_tensor_blocks, tradable_blocks, hedge_profile_blocks,
                 privileged_factor_blocks) = _new_blocks()

        self.calc_stats[execution_label] = time.monotonic() - self.calc_stats[execution_label]

        bundle = held_out if solve_hedge_mode else Bundle.from_batch(
            base_date,
            bus_day,
            shared_mem.one.new_tensor(t_days_arr),
            tradable_blocks,
            factor_tensor_blocks,
            hedge_profile_blocks,
            params['Simulation_Batches'],
            self.stoch_factors,
            normalized_runtime,
            privileged_factor_blocks=privileged_factor_blocks,
            total_leg_volume=total_leg_volume,
            last_payment_day=last_payment_day,
        )

        evaluation_summary = None
        optimizer_diagnostics = None
        policy_artifact = None
        runtime_present = False
        runtime_diagnostics = {}
        # A solve already ran batch by batch; all that is left is the held-out verdict + the
        # tracks. `simulate_only` rolls its accumulated bundle with zero trades.
        optimization_result = (streaming_solve.finish(held_out) if solve_hedge_mode
                               else run_hedge_execution(bundle, normalized_runtime))
        if optimization_result is not None:
            evaluation_summary = optimization_result['evaluation_output']
            optimizer_diagnostics = optimization_result['optimizer_diagnostics']
            policy_artifact = optimization_result['policy_artifact']
            runtime_present = True
            runtime_diagnostics = {
                'num_episodes': int(evaluation_summary.get('diagnostics', {}).get('num_episodes', 0)),
                'derivus_simulation_pricing_time_seconds': float(self.calc_stats.get(execution_label, 0.0)),
                'accounting_mode': normalized_runtime.get('accounting_mode'),
                'tradable_names': tuple(normalized_runtime.get('names', {}).get('tradables', ())),
                'cash_account_names': tuple(normalized_runtime.get('names', {}).get('cash_accounts', ())),
            }

        if evaluation_summary is not None and bundle is not None:
            # world-before-solver tripwire: each hedge leg's expected drift over the live window,
            # in the sim the policy trains on, so a conditional-mean seam prints HERE
            t_live = bundle.last_live_mtm_index
            drift = {h: round(float((bundle.tradables_sim[h][t_live]
                                     - bundle.tradables_sim[h][0]).mean()), 4)
                     for h in normalized_runtime.get('names', {}).get('hedges', ())}
            evaluation_summary.setdefault('diagnostics', {})['hedge_drift_usd'] = drift
            logging.info('hedge window drift E[dF] per leg (USD): %s', drift)

        return HedgeRuntimeExecutionResult(
            bundle=bundle,
            runtime=normalized_runtime,
            evaluation_summary=evaluation_summary,
            optimizer_diagnostics=optimizer_diagnostics,
            policy_artifact=policy_artifact,
            metadata={
                'execution_mode': execution_mode,
                'bundle_present': bundle is not None,
                'num_batches': params['Simulation_Batches'],
                'num_paths': self.numscenarios,
                'optimizer_diagnostics_present': optimizer_diagnostics is not None,
                'runtime_present': runtime_present,
                'runtime_diagnostics': runtime_diagnostics,
            },
        )

    # Inner-MC subsystem: forks the simulator from each outer-path state at an outer timestep.
    # Outer process instances are never touched - inner runs on shallow copies, so per-instance
    # precalc state cannot bleed across the outer/inner boundary.

    def _find_spot_key(self):
        """The unique underlying (commodity-spot) factor key, mapping the runtime-owned underlying
        name set back to the live key object. Where more than one CommodityPrice is simulated,
        prefers the martingale primary - the spot exposing a revealed sufficient statistic
        (non-empty `privileged_layout`). Raises unless exactly one remains."""
        spots = [k for k in self.stoch_factors
                 if utils.check_tuple_name(k) in self._underlying_names]
        primaries = [k for k in spots if self.stoch_factors[k].privileged_layout(self.stoch_factors[k].param)]
        spot_keys = primaries or spots
        if len(spot_keys) != 1:
            raise ValueError(
                f'Inner MC expects exactly one underlying spot factor; found {len(spot_keys)}: {spot_keys}'
            )
        return spot_keys[0]

    def _restricted_struct(self, outer_struct, cutoff_mtm_idx, window_end_idx=None):
        """Mirror `outer_struct` with each deal's `Time_dep` restricted to events at mtm positions
        >= `cutoff_mtm_idx`, or windowed to [`cutoff_mtm_idx`, `window_end_idx`] when that is given
        (the one-step fork prices exactly {t, t+1}). `Factor_dep` is shared by reference and `Calc_res` is fresh, so
        inner pricing cannot clobber outer storage; deals entirely in the past are dropped. Does not
        recurse into sub_structures.

        Aggregation storage is off while the per-deal `Calc_res` stays: the fork harvests on the
        DEVICE via `tensor_marks()` and `resolve_hedge_structure()`, and the aggregate's stored
        'Value' D2H copy was 93% of the fork's host egress."""
        inner = DealStructure(outer_struct.obj.Instrument, store_results=False)
        for dd in outer_struct.dependencies:
            keep = dd.Time_dep.deal_time_grid >= cutoff_mtm_idx
            if window_end_idx is not None:
                keep &= dd.Time_dep.deal_time_grid <= window_end_idx
            new_td = dd.Time_dep.restricted(keep)
            if new_td is None:
                continue
            inner.dependencies.append(utils.DealDataType(
                Instrument=dd.Instrument,
                Factor_dep=dd.Factor_dep,
                Time_dep=new_td,
                Calc_res={} if outer_struct.store_results else None,
            ))
        return inner

    def _attach_inner_mc(self, bundle, outer_buffer, shared_mem, base_date, tradable_refs):
        """Attach the on-demand inner-MC forks to `bundle` as closures, so the solver can fork
        without a calc handle. Every fork is windowed to {t, t+1} - 2-row generation and a real
        2-row pricing pass - giving the exact per-tradable F_t1 and L_t / L_t1 the diff-ML bootstrap
        reads. `outer_rows` lets the solver run the GRAD fork in outer-path sub-slices.

        The outer buffer is passed in rather than read off the calc: under streaming there is one
        bundle per batch, and each forks from ITS OWN batch."""
        bundle.inner_mc = lambda t: self._run_inner_mc_at_t(
            t, outer_buffer, shared_mem, base_date, tradable_refs)
        bundle.inner_mc_grad = lambda t, outer_rows=None: self._run_inner_mc_at_t(
            t, outer_buffer, shared_mem, base_date, tradable_refs,
            with_grad=True, outer_rows=outer_rows)

    def _publish_print_seeds(self, procs, state, t, shared_mem):
        """Publish every process's `print_seed` off `state` (factor key -> row-indexable snapshot)
        into the buffer, and return the published keys. Those keys are CONSUMED state, not path
        series - the caller drops them once the run has generated - and every seed is published
        before any generate."""
        keys = set()
        for key, proc in procs:
            for seed_key, seed_val in proc.print_seed(key, state, t).items():
                shared_mem.t_Scenario_Buffer[seed_key] = seed_val
                keys.add(seed_key)
        return keys

    def _run_inner_mc_at_t(self, t, outer_scenario_buffer, shared_mem, base_date,
                           tradable_refs, with_grad=False, outer_rows=None):
        """Run inner MC at a single outer timestep `t`, forking from `outer_scenario_buffer` - a
        snapshot of the outer `t_Scenario_Buffer` (factor keys plus every per-process aux key, batch
        dim B_outer). `outer_rows=(lo, hi)` forks only that contiguous outer-path range; labels are
        per-outer-path, so row slices are independent. The DP/MPC backward sweep calls this on
        demand outside the outer loop, via `Bundle.inner_mc`.

        Returns the inner samples the solver bootstraps from:
            F_t1          {ref: (B_outer, B_inner)}       futures price at outer t+1
            L_t, L_t1     (B_outer, B_inner)              liability MTM at outer t and t+1
            market_t1     (B_outer, B_inner, market_dim)  inner market state at outer t+1
            market_t      (B_outer, market_dim)           outer-realised market state at t
            t, cutoff_idx
        `market_t` / `market_t1` are every simulated factor's revealed segments concatenated in
        reveal order; each process owns its packing via `reveal_state_at`, so the solvers consume
        the column block without knowing what the factors are.

        SINGLE PASS: generation, stuffing, pricing and extraction all run at
        `B_outer x Inner_Sub_Batch` flat, so peak memory is a function of two JSON fields
        (`Batch_Size`, `Inner_Sub_Batch`) and nothing else. Too wide for the card raises CUDA OOM
        naming this fork; that is the contract, not a knob.

        THE WINDOW: the inner grid and every deal's `Time_dep` are truncated to {t, t+1}, so the
        pricing chain runs for real on 2 rows - exact per-tradable F_t1 and liability L_t / L_t1,
        and correct on mixed strips, where a market-only short-circuit would broadcast spot as every
        F_t1. One forward step is also what bounds the AAD tape.

        TWO COORDINATE SYSTEMS: processes generate against the shifted-base `inner_time_grid`, while
        pricers run against the full outer `self.time_grid` with each deal's `Time_dep` restricted.
        Buffer stuffing prepends the outer-realized past so path-dependent payoffs see the realized
        fixings, published as its own `ScenarioBlock` carrying a `past_columns` index rather than
        materialized B_inner times. Every path series goes through that publication.

        PER-PROCESS HOOKS, no isinstance branch anywhere: `inner_fork_seed` supplies the
        per-outer-path t=0 privileged state, `reseed_inner_state` republishes whatever revealed
        state `reveal_state_at` needs at t+1 and returns differentiable leaves for the twin loss
        (`self._inner_state_opts` is forwarded opaquely), and `reveal_state_at` yields each factor's
        informative segments - empty where a factor has no revealed sufficient statistic. Under
        `with_grad`, `state_t_leaf_widths` pairs each leaf with the market_t width it occupies.

        FAIL LOUDLY on both halves: a tradable live in this fork but missing from `tensor_marks()`
        reads downstream as an expired contract, and a liability swallowed by the canonical deal
        guard silently corrupts the solver's LABELS.

        The fork BORROWS `shared_mem` and the `finally` restores it on ANY exit - without that a
        mid-fork raise leaves the state flat-sized and the next t-step fails on shapes rather than
        the real cause, and clears the fork's `t_Buffer` and `t_Scenario_Buffer`.
        """
        spot_key = self._find_spot_key()
        if outer_rows is not None:
            lo, hi = outer_rows
            outer_scenario_buffer = {k: v[..., lo:hi] for k, v in outer_scenario_buffer.items()}
        B_outer = outer_scenario_buffer[spot_key].shape[-1]
        B_inner = shared_mem.simulation_sub_batch
        t_days = int(self.time_grid.scen_time_grid[t])
        inner_time_grid = self.time_grid.truncate_to(base_date, t_days)

        # terminal / past-end: no inner horizon, so nothing to price. The DP sweep uses the
        # closed-form V_T there
        if inner_time_grid.scen_time_grid.size < 2:
            return dict(t=t, cutoff_idx=t, L_T=None, market_t=None, market_t1=None, F_t1={})

        # In HedgeMonteCarlo scen_time_grid == mtm_time_grid (dynamic_scenario_dates),
        # so the same `t` indexes both the scenario buffer and the mtm grid.
        cutoff_idx = t
        inner_base_date = base_date + pd.Timedelta(days=t_days)

        # THE WINDOW: inner grid truncated to {t, t+1}, so the AAD tape stays bounded
        window_end_idx = min(cutoff_idx + 1, self.time_grid.mtm_time_grid.size - 1)
        if inner_time_grid.scen_time_grid.size > 2:
            kept = set(sorted(inner_time_grid.scenario_dates)[:2])
            inner_time_grid = utils.TimeGrid(
                kept,
                kept & set(inner_time_grid.mtm_dates),
                kept & set(inner_time_grid.base_MTM_dates))
            inner_time_grid.set_base_date(inner_base_date)

        # generation, stuffing, pricing and extraction all at `B_outer x B_inner` flat, ONE pass
        B_flat = B_outer * B_inner

        grad_ctx = (torch.enable_grad() if with_grad else torch.no_grad())
        # per-process initial-state leaves, exposed via the result dict so the caller can
        # `.backward()` from any function of the inner outputs and read `.grad` per path
        state_t_leaves = {} if with_grad else None
        # what the fork BORROWS, to give back exactly as found (see the finally below): a fork over
        # an outer-path SLICE would otherwise leave the state slice-sized
        borrowed_batch, borrowed_fill = shared_mem.simulation_batch, shared_mem.fillvalue
        with grad_ctx:
            try:
                shared_mem.simulation_batch = B_outer
                shared_mem.reset_inner(self.num_factors, inner_time_grid,
                                       use_antithetic=self.params.get('Inner_Antithetic', 'No') == 'Yes',
                                       use_random=str(self.params.get('Inner_Draws', 'sobol')).lower() == 'random')

                market_t1_parts = []
                # the fork BORROWS this buffer, so an entry it does not rewrite is still the outer
                # run's - references only, read once by the publication below
                outer_entries = dict(shared_mem.t_Scenario_Buffer)
                live_procs = []
                fork_state = {}
                for key, proc_inner in self.stoch_factors_inner.items():
                    if key.type in utils.DimensionLessFactors:
                        continue
                    # raw per-path init state for this factor's inner-MC precalculate fork
                    init_state = outer_scenario_buffer[key][t]
                    if with_grad:
                        # leaf with grad: differentiates inner-sim + pricing back to state_t
                        init_state = init_state.detach().clone().requires_grad_(True)
                        state_t_leaves[key] = init_state
                    proc_inner.precalculate(
                        inner_base_date, inner_time_grid,
                        init_state,
                        shared_mem, self.process_ofs[key],
                        implied_tensor=self._factor_precalc_args[key][1],
                    )
                    # the day-t state snapshot the print seeds read; under `with_grad` this IS
                    # the leaf, so conditioning derived from a print stays on the tape
                    fork_state[key] = init_state.unsqueeze(0)
                    live_procs.append((key, proc_inner))
                # EVERY fork seed before ANY generate: a seed reads only the OUTER state, and a
                # process may seed a factor that generates before it in topological order
                for key, proc_inner in live_procs:
                    for seed_key, seed_val in proc_inner.inner_fork_seed(
                            key, outer_scenario_buffer, t).items():
                        shared_mem.t_Scenario_Buffer[seed_key] = seed_val
                self._publish_print_seeds(live_procs, fork_state, 0, shared_mem)
                for key, proc_inner in live_procs:
                    simulated = proc_inner.generate(shared_mem)
                    shared_mem.t_Scenario_Buffer[key] = simulated
                    # post-generate coherence: publish revealed state at t+1, return twin-loss leaves
                    inner_leaves = proc_inner.reseed_inner_state(
                        key, simulated, outer_scenario_buffer, t, shared_mem, self._inner_state_opts, with_grad)
                    if with_grad:
                        state_t_leaves.update(inner_leaves)
                    # market state at outer t+1 (inner-time index 1), in reveal order
                    for block, _kind in proc_inner.reveal_state_at(1, shared_mem.t_Scenario_Buffer):
                        market_t1_parts.append(
                            self._reveal_transform(key, block, _kind).reshape(-1, B_outer, B_inner))

                # publish past-then-forked rows; the past keeps ONE outer column per B_inner flat
                # columns, handed over as data rather than re-derived downstream
                past_columns = torch.arange(
                    B_flat, device=shared_mem.one.device) // B_inner
                # every path series this fork WROTE that the outer path also carries. Each half of
                # the test excludes one thing: an entry the fork never rewrote, and its own seed
                for key in [k for k, v in shared_mem.t_Scenario_Buffer.items()
                            if v is not outer_entries.get(k) and k in outer_scenario_buffer]:
                    inner_path = shared_mem.t_Scenario_Buffer[key]                  # (T_inner, ..., B, SB)
                    past = [utils.ScenarioBlock(outer_scenario_buffer[key][:cutoff_idx],
                                                batch_index=past_columns)] if cutoff_idx else []
                    shared_mem.t_Scenario_Buffer[key] = utils.ScenarioSource(
                        *past, utils.ScenarioBlock(
                            inner_path.reshape(*inner_path.shape[:-2], B_flat),
                            first_row=cutoff_idx))

                shared_mem.t_Buffer.clear()
                shared_mem.resize(B_flat)
                # restricted DealStructures: same instruments and Factor_dep, fresh Time_dep
                # windowed to [t, t+1], fresh Calc_res
                inner_netting_sets = self._restricted_struct(self.netting_sets, cutoff_idx, window_end_idx)
                inner_liabilities = self._restricted_struct(self.liabilities, cutoff_idx, window_end_idx)
                inner_netting_sets.resolve_structure(shared_mem, self.time_grid)
                # TRADABLE half of the fail-loudly guard: live in this fork but no mark => skipped
                priced = set(inner_netting_sets.tensor_marks())
                skipped = sorted(
                    dd.Instrument.field['Reference'] for dd in inner_netting_sets.dependencies
                    if dd.Instrument.field['Reference'] in tradable_refs
                    and dd.Instrument.field['Reference'] not in priced)
                if skipped:
                    raise RuntimeError(
                        f'inner-fork tradable pricing failed for {skipped} at t={t} — the deal is '
                        f'live in this fork but produced no mark; see the CRITICAL log above for '
                        f'the cause.')
                shared_mem.reset_cashflows(self.time_grid)
                mtm_flat = inner_liabilities.resolve_hedge_structure(
                    shared_mem, self.time_grid)['mtm']
                if mtm_flat.dim() < 2 or mtm_flat.shape[-1] != B_outer * B_inner:
                    # the canonical deal guard swallows exceptions into a scalar-0 mark, which
                    # inside a fork silently corrupts the solver's LABELS - fail loudly here
                    raise RuntimeError(
                        f'inner-fork liability pricing degenerated (shape '
                        f'{tuple(mtm_flat.shape)}, expected (*, {B_outer * B_inner})) — a deal '
                        f'was skipped inside the fork; see the CRITICAL log above for the cause.')
                inner_mtm = mtm_flat.reshape(*mtm_flat.shape[:-1], B_outer, B_inner)

                def _fan(t):
                    # a STATIC tradable (a cash account off an unsimulated curve) marks with a
                    # 1-wide batch, which is a legitimate mark - broadcast it. Any other width is a
                    # shape error and expand fails loud.
                    if t.shape[-1] == B_outer * B_inner:
                        return t.reshape(*t.shape[:-1], B_outer, B_inner)
                    core = t.squeeze(-1) if t.shape[-1] == 1 else t
                    return core.reshape(*core.shape, 1, 1).expand(*core.shape, B_outer, B_inner)

                inner_trade_tensors = {
                    ref: _fan(t) for ref, t in inner_netting_sets.tensor_marks().items()}
                # the window prices {t, t+1}, so no terminal row exists to read: `L_T` is None and
                # the horizon stats are not produced
                result = {}
                F_t1 = {}
                zero_bs = inner_mtm[-2].new_zeros(inner_mtm[-2].shape)   # (B_outer, B_inner)
                for ref in tradable_refs:
                    td = inner_trade_tensors.get(ref)
                    if td is None:
                        # Tradable expired before this fork — zero moves, no position.
                        F_t1[ref] = zero_bs
                        continue
                    td = td[cutoff_idx:]                                # (T_inner, B_outer, B_inner)
                    # td has < 2 time points when the tradable's last deal event is at t
                    # (it expires this step) — no t+1 slice; freeze it (dF == 0).
                    F_t1[ref] = (td[1] if td.shape[0] >= 2 else td[-1]).clone()
                # market state - every simulated factor's informative state concatenated;
                # factor order is the `stoch_factors_inner` order, identical for t and t+1
                market_t1 = torch.cat(market_t1_parts, dim=0).permute(1, 2, 0).contiguous()
                market_t_parts = []
                market_t_widths = []
                for key in self.stoch_factors_inner:
                    if key.type in utils.DimensionLessFactors:
                        continue
                    proc_inner = self.stoch_factors_inner[key]
                    width = 0
                    for block, _kind in proc_inner.reveal_state_at(t, outer_scenario_buffer):
                        b = self._reveal_transform(key, block, _kind).reshape(-1, B_outer)
                        market_t_parts.append(b)
                        width += b.shape[0]
                    # the process's differentiable-state-leaf suffixes, so the solver's label
                    # projection maps leaf grads to market columns with no model concept
                    market_t_widths.append((key, width, tuple(proc_inner.diff_state_leaves())))
                market_t = torch.cat(market_t_parts, dim=0).permute(1, 0).contiguous()
                # exact liability MTM at outer-t and outer-t+1 on the inner draws (same time
                # indexing as F_t1) - not a Jacobian linearization
                mtm_fwd = inner_mtm[cutoff_idx:]                            # (T_inner, B_outer, B_inner)
                L_t_inner = mtm_fwd[0].clone()                             # outer-t (shared across draws)
                L_t1_inner = (mtm_fwd[1] if mtm_fwd.shape[0] >= 2
                              else mtm_fwd[-1]).clone()                     # outer-t+1 per inner draw
                result.update(
                    t=t, cutoff_idx=cutoff_idx, L_T=None,
                    L_t=L_t_inner, L_t1=L_t1_inner, F_t1=F_t1,
                    market_t=market_t, market_t1=market_t1)
                if with_grad:
                    result['state_t_leaves'] = state_t_leaves
                    result['state_t_leaf_widths'] = market_t_widths
            finally:
                # restore on ANY exit - a mid-fork raise must not leave the state flat-sized
                shared_mem.simulation_batch = borrowed_batch
                shared_mem.fillvalue = borrowed_fill
                shared_mem.t_Buffer.clear()
                shared_mem.t_Scenario_Buffer.clear()

        return result


class NettingSetCalculation(Calculation):
    """A calculation over the trades a book holds under its netting sets: the block judged, its
    `Mapping` read and the trades listed under their sets, and each trade valued off its own
    structure by one scenario valuation of the book compiled once, a trade its compile refuses set
    aside (`base`), which `value` prices over given markets. `SIMM` and `SACCR` are its two."""

    #: `Quasi_Seed`, declared once for the calculations over netting sets.
    QUASI_SEED = F('Quasi_Seed', 'Integer', default=schema.QUASI_SEED, convention=True,
                   description="The scramble seed every pricing's quasi-random draws run on, each "
                               "trade's inner paths its own whatever is booked beside it")

    def setup(self, params):
        """The block judged and completed, its `Mapping` read, the base valuation it prices by
        stated and the book listed - what the run and each of its workers share - and the
        `Parameters` file as it reads, returned."""
        params = self.params = declared_defaults(type(self), self.judged(params))
        with open(params['Mapping'], 'rt', encoding='utf-8') as f:
            self.mapping = json.load(f)
        with open(params['Parameters'], 'rt', encoding='utf-8') as f:
            parameters = json.load(f)
        self.calc_stats['Parameters'] = parameters.get('version', '')
        self.calc_stats['Quasi_Seed'] = params['Quasi_Seed']
        self.valuation = {
            'Object': 'ScenarioValuation', 'Base_Date': params['Base_Date'],
            'Currency': params['Currency'], 'MCMC_Simulations': params['MCMC_Simulations'],
            'Quasi_Seed': params['Quasi_Seed'], 'Greeks': 'No', 'Deterministic_Kernels': 'Yes',
            'Scenarios': []}
        book = self.config.deals['Deals']['Children']
        loose = [node['Instrument'].field.get('Reference') for node in book
                 if node['Instrument'].field['Object'] != 'NettingCollateralSet']
        if loose:
            raise ValueError('{}: {} sit under no netting set, and the calculation is a netting '
                             'set\'s'.format(type(self).__name__, ', '.join(map(str, loose))))
        self.trades = [(agreement, node) for agreement in book
                       for node in agreement.get('Children', [])]
        return parameters

    @contextlib.contextmanager
    def prepared(self):
        """The config as the run's compile reads it - the valuation calculation and its date - put
        back as it stood on the way out."""
        config = self.config
        calculation = config.deals['Calculation']
        try:
            config.deals['Calculation'] = self.valuation
            config.params['System Parameters']['Base_Date'] = self.valuation['Base_Date']
            yield
        finally:
            config.deals['Calculation'] = calculation

    def base(self, over=None):
        """`[(value, skipped)]` per trade: the book compiled once as a scenario valuation compiles
        it over the valuation's `Scenarios`, each trade alone under its set, on the price factors
        `over` states in place of the standing ones, and priced as it stands, with the engine's
        sentences for what it skipped inside each trade, the distinct ones spoken at its deals. A
        trade whose own compile refuses its document by name (`utils.UnpriceableSchedule`) is set
        aside unvalued, its sentence among them."""
        factors = self.config.params['Price Factors']
        calc = self.compiled = ScenarioValuation(self.config, prec=torch.float64,
                                                 device=utils.calculation_device())
        calc.calc_stats = self.calc_stats
        calc.params = declared_defaults(ScenarioValuation, self.valuation)
        heard = utils.CapturedErrors()
        logging.getLogger().addHandler(heard)
        try:
            self.config.params['Price Factors'] = dict(factors, **(over or {}))
            self.shared = calc.update_factors(calc.params, self.valuation['Base_Date'])
            calc.netting_sets = DealStructure(Aggregation('root'), store_results=True)
            for agreement, node in self.trades:
                try:
                    calc.set_deal_structures([dict(agreement, Children=[node])], calc.netting_sets,
                                             self.shared.one, deal_level_mtm=True)
                except utils.UnpriceableSchedule:
                    # a census names the trade and values the rest: its compile attached nothing
                    # and stamped and counted the deal it refused
                    continue
            held = {deal.Instrument.resolved_from or deal.Instrument: deal.Calc_res for deal in [
                structure.obj for structure in calc.netting_sets.structures()] + list(
                calc.netting_sets.deals())}
            self.results = [held.get(node['Instrument'], {}) for _, node in self.trades]
            values = self.value([{}])[0]
        finally:
            logging.getLogger().removeHandler(heard)
            self.config.params['Price Factors'] = factors
        # the engine says so in so many words, wherever it skips a deal or a structure
        skipped = [(deal, said) for deal, said in heard.said if 'skipp' in said.lower()]
        found = []
        for (_, node), value in zip(self.trades, values):
            references = {deal['Instrument'].field.get('Reference') for _, deal in schema.walk([node])}
            mine = list(dict.fromkeys(said for deal, said in skipped if deal in references))
            found.append((value, mine if value is not None else
                          mine + ['the trade itself was not valued']))
        return found

    def value(self, scenarios):
        """Per scenario of `scenarios`, the markets of one pass (`ScenarioValuation.priced`), each
        trade's value in the calculation currency off its own structure, None where it was not
        valued."""
        self.compiled.priced(self.shared, scenarios)
        return [[result['Value'][..., column].item() if 'Value' in result else None
                 for result in self.results] for column in range(len(scenarios))]


class SIMM(NettingSetCalculation):
    """ISDA SIMM sensitivities by bump and revaluation: the CRIF of a book of netting sets, one
    compile and its market moves the scenarios of one valuation (`ScenarioValuation`).

    MARKET PRICES -> PRICE FACTORS -> MTM. A trade is a booked structure under a netting set, never
    its legs. The set is compiled once, each market input a trade's own dependency walk names moved
    by ISDA's definition of the sensitivity to it, and the moves are scenarios priced in passes,
    each beside the standing market, each trade read off its own structure: a trade's change over
    the shift is a CRIF row. A Monte Carlo pricer draws a trade's inner paths off `Quasi_Seed`
    alone, whatever is booked or priced beside it, so each trade draws the same paths on the
    standing market and on every move and each change is one of common random numbers. A moved
    benchmark quote rides its curve family's calibration Jacobian, `theta* + J (q - q0)`, off one
    solve per coupled set at the run's bootstrap, exact to first order in the move.

    How a factor moves is the methodology and lives here: a basis point on each benchmark quote of
    a curve, split linearly between the vertices either side of its maturity, the survival curves
    solved on the curve re-solved on it; a basis point of spread on each quote of a survival curve,
    its block re-solved, split between the credit vertices either side of its tenor; a basis point
    on every node of a cross-currency basis; a basis point on the breakevens an inflation curve is
    defined by, the quotes of the block that solves it re-solved, else its nodes' own, one row; a
    percent of a spot, of a currency's price in the calculation currency, or of every price factor
    of a commodity at once, its spot and each delivery of its forward curve, the factors a mapping
    states under one qualifier being one commodity; and a point of vol in a TENT on each vertex -
    the surface given a pillar at every vertex expiry it lacks, read off its own interpolation, the
    set compiled on it so the base and every move share one grid, then each vertex alone moved, so
    an option's vega lands on its own expiry and the rows sum to the parallel move. A Skew or an
    SVI surface gains a pillar only past its ends (`Factor2D.augmented`), so a vertex moves the
    pillars its tent reaches and an option expiring between two pillars has its vega land on the
    vertices whose tents reach them. A vol space gains a slice at each vertex expiry it lacks, every
    node of its other two columns (`Factor3D.augmented`), so a swaption's or a caplet's vega lands
    on its own expiry, the underlying tenor summed over, and a rate vertex moves each vol of its
    slice a percent of itself, its row ISDA's vol-weighted vega. Where the rows land - risk type,
    qualifier, bucket, labels - is the bank's `Mapping`, held to `utils.FactorRiskClass`; a trade's
    product class is read off its walk, the one class among Credit, Equity and Commodity its
    factors are of, RatesFX where they are of none, and its own `Product_Class` where they are of
    more than one (`product_class`); the rate and vol vertices are the regime's `Parameters`, the credit
    vertices the methodology's; the regimes margin is posted and collected under are the
    agreement's own terms, declared on the netting set. A shift other than ISDA's is scaled to it.

    `Method: Graph` moves nothing: the set is priced once on a tape and each trade differentiated
    by one backward pass, each of its moves a row as the derivative of the move at a zero shift
    (`tangent`), the limit of the difference `Bump` reads.

    MARKET PRICES ARE A CONDITION: an interest rate or a survival curve a trade reads that no
    configured family bootstraps, and that is no child of a curve one does, refuses by name. A leg
    the engine skips is reported per trade in its own sentence and the trade's rows written
    regardless; a trade whose own compile refuses its document by name is set aside, reported
    unvalued in that sentence and writing no row (`base`).
    """
    #: The CRIF risk types the run moves: the risk class each belongs to and how its factor moves -
    #: a basis point on each benchmark quote of the curve's block, on every node of a basis or on
    #: the breakevens an inflation curve is defined by, a basis point of spread on each quote of a
    #: survival curve's block, a percent of a currency's price in the calculation currency, of a
    #: spot or of a commodity's price factors together, a point of vol per vertex, or on a rate
    #: space each vol of the vertex's slice a percent of itself, in whichever convention the space
    #: declares (ISDA's vol-weighted vega, `weighted`); the curve benchmarks quoted as a price,
    #: which no basis point moves; and per `Surface_Type` the block a point of vega moves and the
    #: column its expiry is in, the level being the last, a vol space stating none and read as
    #: Explicit, its expiry the second of its four columns - a Skew smile is added to its ATM level,
    #: so moving the level holds its skew, and the surface owns what a point adds to a level
    #: (`Factor2D.vol_point`). ISDA's Risk_InflationVol and Risk_CreditVol are not here, the engine
    #: having no inflation or credit vol factor type, so a mapping naming either is refused as one
    #: naming any type not here is.
    MOVES = {'Risk_IRCurve': ('InterestRate', 'quote'),
             'Risk_XCcyBasis': ('InterestRate', 'parallel'),
             'Risk_Inflation': ('InterestRate', 'inflation'),
             'Risk_FX': ('FX', 'fx'), 'Risk_Equity': ('Equity', 'spot'),
             'Risk_Commodity': ('Commodity', 'commodity'),
             'Risk_IRVol': ('InterestRate', 'weighted'), 'Risk_FXVol': ('FX', 'vega'),
             'Risk_EquityVol': ('Equity', 'vega'), 'Risk_CommodityVol': ('Commodity', 'vega'),
             'Risk_CreditQ': ('Credit', 'cds'), 'Risk_CreditNonQ': ('Credit', 'cds')}
    #: The vertices a credit spread's rows land on, the methodology's: the parameters file weighs
    #: credit by bucket and names none.
    CREDIT_VERTICES = ('1y', '2y', '3y', '5y', '10y')
    #: ISDA's product class of each risk class of the engine's; a cross-class factor rides the
    #: factor it is read beside and names none.
    PRODUCT_CLASSES = {'InterestRate': 'RatesFX', 'FX': 'RatesFX', 'Credit': 'Credit',
                       'Equity': 'Equity', 'Commodity': 'Commodity'}
    OUTRIGHTS = ('FXForwardDeal',)
    VOL_PILLARS = {'Skew': ('ATM_Vol', 0), 'Malz': ('Surface', 1), 'Explicit': ('Surface', 1),
                   'SVI': ('a', 0)}
    #: The moves of a vol surface: a point of vol on a vertex, or a percent of each vol there.
    VEGAS = ('vega', 'weighted')
    #: What the margin calculation reads off a CRIF, in the order it is written, `Amount` in the
    #: calculation currency.
    CRIF_COLUMNS = ('TradeID', 'Counterparty', 'PostRegulations', 'CollectRegulations',
                    'ProductClass', 'RiskType', 'Qualifier', 'Bucket', 'Label1', 'Label2',
                    'Amount', 'AmountUSD')
    #: One move of one market input: the factor its rows are reported under, that factor's
    #: coordinates, `[(Label1, weight)]` its amount is split by and the move itself - a key `move`
    #: turns into price factors.
    Bump = namedtuple('Bump', 'factor coordinates split move')
    #: The methodology's risk classes the margin aggregates: per class its section of the
    #: parameters file and the keys read there, the name the class correlations know it by, and the
    #: risk types of its delta and of its vega, whose rows margin its curvature too.
    RiskClass = namedtuple('RiskClass', 'section keys name deltas vega')
    RISK_CLASSES = {
        'InterestRate': RiskClass(
            'ir', ('rw', 'ccy', 'con', 'rho', 'rho_inter', 'hvr', 'bucket_vol_map'), 'IR',
            ('Risk_IRCurve', 'Risk_Inflation', 'Risk_XCcyBasis'), 'Risk_IRVol'),
        'CreditQ': RiskClass('crq', ('rw', 'con', 'rho_intra', 'rho_inter'), 'CSR (Q)',
                             ('Risk_CreditQ',), None),
        'CreditNonQ': RiskClass('crnq', ('rw', 'con', 'rho_intra', 'rho_inter'), 'CSR (NQ)',
                                ('Risk_CreditNonQ',), None),
        'Equity': RiskClass('eq', ('rw', 'con', 'rho_intra', 'rho_inter', 'hvr'), 'Equity',
                            ('Risk_Equity',), 'Risk_EquityVol'),
        'Commodity': RiskClass('cm', ('rw', 'con', 'rho_intra', 'rho_inter', 'hvr'), 'Commodity',
                               ('Risk_Commodity',), 'Risk_CommodityVol'),
        'FX': RiskClass('fx', ('rw', 'ccy', 'con', 'rho', 'hvr', 'high'), 'FX', ('Risk_FX',),
                        'Risk_FXVol')}
    #: The margins of a risk class, and the directions a set's margin is read in, each under the
    #: regime its agreement declares: posted on the rows' negation, collected on them as written.
    MARGINS = ('Delta', 'Vega', 'Curvature')
    DIRECTIONS = (('Posted', 'PostRegulations', -1.0), ('Collected', 'CollectRegulations', 1.0))
    #: The methodology's constants: a vega's vol is its delta weight over a fortnight at the 99th
    #: percentile, a curvature scales an expiry by half a fortnight's share of it and its lambda
    #: reads the 99.5th, a residual bucket aggregates apart, and an equity volatility index takes
    #: no curvature.
    FORTNIGHT, RESIDUAL, VOLATILITY_INDICES = 14.0, 'Residual', '12'
    Q99, Q995 = scipy.stats.norm.ppf([0.99, 0.995])
    fields = [
        F('Base_Date', 'Date', default=''),
        F('Currency', 'Text', default=REQUIRED,
          description='The calculation currency the CRIF amounts are in'),
        F('Mapping', 'Text', default=REQUIRED,
          description='Path to the regime mapping: per price factor its SIMM coordinates, or null '
                      'where ISDA names no risk factor for it'),
        F('Parameters', 'Text', default=REQUIRED,
          description='Path to the ISDA SIMM parameters file, whose interest-rate weights name the '
                      'vertices the rate and vol rows land on'),
        F('Horizon', 'Text', default='Horizon_10d',
          description='The horizon of the parameters file the vertices and the margin are read '
                      'under'),
        F('Method', 'Text', default='Bump', values=['Bump', 'Graph'],
          description='Bump reads each row off the set revalued once per move, Graph as the '
                      'move\'s derivative off one backward pass per trade'),
        F('Workers', 'Integer', default=1,
          description='Processes the moves, or under Graph the trades, are dealt over, each on a '
                      'context of its own; 1 values them in this one'),
        F('Rate_Shift', 'Float', default=1.0,
          description='Basis points a benchmark quote, a curve node, a basis, a credit spread or a '
                      'breakeven moves by, ISDA\'s one; another is scaled to it'),
        F('Spot_Shift', 'Float', default=1.0,
          description='Percent a spot, a currency\'s price or a commodity\'s prices move by, '
                      'relative, ISDA\'s one; another is scaled to it'),
        F('Vol_Shift', 'Float', default=1.0,
          description='Points of implied vol a vertex moves by, or on a rate space percent of each '
                      'vol, ISDA\'s one; another is scaled to it'),
        ScenarioValuation.BATCH_SIZE,
        ScenarioValuation.MCMC_SIMULATIONS,
        NettingSetCalculation.QUASI_SEED
    ]

    def setup(self, params):
        """The run's inputs read and its book listed - what the run and each of its workers share."""
        from .bootstrappers import SurvivalProb, bootstrap_writers, market_prices_for

        parameters = super(SIMM, self).setup(params)
        params = self.params
        self.graph = params['Method'] == 'Graph'
        self.valuation['Batch_Size'] = params['Batch_Size']
        if self.graph:
            # the compile's leaves on the tape, each trade's taped value kept (`base`)
            self.valuation['Greeks'] = 'First'
        missing = ['a non-zero ' + name for name in ('Rate_Shift', 'Spot_Shift', 'Vol_Shift')
                   if not params[name]]
        if missing:
            raise ValueError('SIMM: the calculation states no {}'.format(', '.join(missing)))
        try:
            horizon = self.parameters = parameters['horizons'][params['Horizon']]
            weights = next(iter(horizon['ir']['rw']['Delta']['Volatility'].values()))['Weight']
        except (AttributeError, KeyError, StopIteration, TypeError):
            raise ValueError('SIMM: {} carries no interest-rate delta weights by tenor under {}, '
                             'which name the vertices'.format(params['Parameters'],
                                                              params['Horizon']))
        missing = [] if 'class_rho' in horizon else ['class_rho']
        missing += ['{}.{}'.format(stated.section, key) for stated in self.RISK_CLASSES.values()
                    for key in stated.keys if key not in horizon.get(stated.section, {})]
        if missing:
            raise ValueError('SIMM: {} carries no {} under {}, which the margin reads'.format(
                params['Parameters'], ', '.join(missing), params['Horizon']))
        self.vertices = sorted(((label, self.years(label)) for label in weights),
                               key=lambda vertex: vertex[1])
        self.credit_vertices = [(label, self.years(label)) for label in self.CREDIT_VERTICES]
        self.shifts = {'quote': params['Rate_Shift'], 'parallel': params['Rate_Shift'],
                       'inflation': params['Rate_Shift'], 'cds': params['Rate_Shift'],
                       'spot': params['Spot_Shift'], 'fx': params['Spot_Shift'],
                       'commodity': params['Spot_Shift'], 'vega': params['Vol_Shift'],
                       'weighted': params['Vol_Shift'] * 0.01}
        self.universe = set(self.config.factor_universe()['resolved'])
        prices = self.config.params['Market Prices']
        # what a configured family solves, and nothing a family the configuration omits carries
        self.claimed = {name for family, entry in self.config.params[
            'Bootstrapper Configuration'].items() for name in market_prices_for(
            family, prices, declared=entry.get(schema.PRICES_KEY) if isinstance(entry, dict) else None)}
        # the survival curves the book reads, which a move of a curve they solve on re-solves
        self.credit = {name for name in bootstrap_writers(prices, self.universe)
                       if utils.check_rate_name(name)[0] == SurvivalProb.market_factor_type}

    @staticmethod
    def years(label):
        """A tenor `label` - a vertex, a credit quote's `Tenor` - in years of 365 days, a month
        365/12 of them, as the methodology counts."""
        return int(label[:-1]) * {'w': 7.0, 'm': 365 / 12.0, 'y': 365.0}[label[-1].lower()] / 365.0

    @classmethod
    def run(cls, context, overrides=None):
        """Runs SIMM on `context`, the book compiled once and its market moves the scenarios of one
        valuation of it: `(calculation, output)`, the CRIF under `output['Results']['CRIF']`.
        `Workers` above one values the book here and deals what it reaches in turn over that many
        processes, each compiling it on a context of its own (`shard`); the CRIF is the same to
        the bit either way."""
        params = cls.parameters(context, overrides)
        calc = cls(context)
        calc.setup(params)
        count = worker_count(calc.params['Workers'])
        if count == 1:
            return calc, calc.execute(params)
        # the base alone is valued here, and a calibration's tape crosses no process
        calc.valuation['Greeks'] = 'No'
        with calc.prepared():
            base = calc.base()
            work, queue = calc.reach(base), mp.Queue()
            values = dict(pair for answer in gathered([mp.Process(
                target=partial(shielded, cls.shard, job, queue), args=(
                    context, calc.params, work[job::count], job, queue))
                for job in range(min(count, len(work)))], queue)
                for pair in answer['Values'].items())
            return calc, calc.report(base, values)

    @classmethod
    def shard(cls, context, params, work, job, queue):
        """A worker's share of what a run reaches, valued on a context of its own calibrated and
        compiled once over its moves, an artifact crossing no process: `{(trade, move): change}`
        (`values`), or under Graph `{(trade, move): derivative}` (`derivatives`)."""
        calc = cls(context)
        calc.setup(params)
        with calc.prepared():
            calc.base(() if calc.graph else [move for move, _ in work])
            queue.put({'Job': job, 'Values': calc.values(work)})

    def execute(self, params):
        """The run in this process alone; `run` deals what it reaches over workers."""
        self.setup(params)
        with self.prepared():
            base = self.base(() if self.graph else dict.fromkeys(
                bump.move for bumps in self.bumped for bump in bumps))
            return self.report(base, self.values(self.reach(base)))

    @contextlib.contextmanager
    def prepared(self):
        """The config as the run's compile reads it - the valuation calculation, its date, the
        curves its market prices solve (`calibrate`), each trade's moves, classes and agreements -
        put back as it stood on the way out."""
        authored = dict(self.config.params['Market Prices'])
        with super(SIMM, self).prepared():
            try:
                self.calibrate()
                self.bumped, self.classes = [], []
                for index, (agreement, node) in enumerate(self.trades):
                    walk = self.walk(index)
                    self.bumped.append(self.bumps(index, walk))
                    self.classes.append(self.product_class(walk, node))
                    self.agreement(agreement['Instrument'].field.get('Reference'))
                yield
            finally:
                self.config.params['Market Prices'].update(authored)

    def calibrate(self):
        """The curves the book reads solved from their quotes, each coupled set publishing its
        calibration for a quote move to ride (`requoted`) whatever its blocks declare, which then
        declare no ride, so the compile mints the curves the bootstrap solved. Where the valuation
        is differentiated (Graph) every curve, survival and inflation block keeps its solve on the
        tape (`Quote_Sensitivity`), the re-solve a move differences being its family's Jacobian
        there."""
        from . import bootstrappers

        prices = self.config.params['Market Prices']
        if self.valuation['Greeks'] != 'No':
            taped = {family.market_factor_type for family in (
                bootstrappers.InterestRate, bootstrappers.SurvivalProb,
                bootstrappers.InflationRate)}
            prices.update({name: dict(block, instrument=dict(
                block['instrument'], Quote_Sensitivity='Yes')) for name, block in prices.items()
                if utils.check_rate_name(name)[0] in taped})
        curves = {name: block for name, block in prices.items()
                  if utils.check_rate_name(name)[0] == bootstrappers.InterestRate.market_factor_type}

        def declaring(propagation):
            prices.update({name: dict(block, instrument=dict(
                block['instrument'], Quote_Propagation=propagation))
                for name, block in curves.items()})

        # every set's operator must stand at once for the moves to ride, whatever the store held
        self.config.artifacts.size = max(self.config.artifacts.size, len(curves))
        declaring('Linear')
        self.config.bootstrap(wanted=self.universe)
        # a ride to the quotes standing is the solved curve at the cost of a mispricing per set
        declaring('No')

    def report(self, base, values):
        """The run's results off the book's `base` and the changes its moves made (`values`), or
        under Graph their derivatives: the CRIF, a row per trade and the margin off the CRIF
        (`margins`)."""
        trades = [self.trade(index, base, values) for index in range(len(self.trades))]
        crif = self.frame(trades)
        self.output = dict({'CRIF': crif, 'Trades': pd.DataFrame(
            [dict({key: value for key, value in trade.items() if key != 'Rows'},
                  Skipped='; '.join(trade['Skipped']), Written=len(trade['Rows']))
             for trade in trades],
            columns=['TradeID', 'NettingSet', 'ProductClass', 'Value', 'Moves', 'Written',
                     'Skipped'])}, **self.margins(crif))
        return {'Stats': self.calc_stats, 'Results': self.output}

    def coordinates(self, factor, trade):
        """`factor`'s SIMM coordinates off the mapping, None where it carries no risk factor, held
        to the engine's risk class of its type, a cross-class type taking its risk type's, and a
        vega risk type to a vol surface."""
        if factor not in self.mapping['Factors']:
            raise ValueError('SIMM: trade {} reads {}, and the mapping states nothing for it - '
                             'state its SIMM coordinates, or null where ISDA names no risk factor '
                             'for it'.format(trade, factor))
        stated = self.mapping['Factors'][factor]['SIMM']
        if stated is None:
            return None
        kind = self.MOVES.get(stated.get('RiskType'))
        if kind is None:
            raise ValueError('SIMM: {} maps to {}, and the risk types moved here are {}'.format(
                factor, stated.get('RiskType'), ', '.join(sorted(self.MOVES))))
        factor_type = utils.check_rate_name(factor)[0]
        risk_class = utils.FactorRiskClass[factor_type]
        # a cross-class factor's class is inherited: a basis's is the commodity's it rides
        if risk_class not in (kind[0], 'CrossClass'):
            raise ValueError('SIMM: {} is a factor of the {} class and maps to {}, a risk type of '
                             'the {} class'.format(factor, risk_class, stated['RiskType'], kind[0]))
        if kind[1] in self.VEGAS and factor_type not in (
                utils.TwoDimensionalFactors + utils.ThreeDimensionalFactors):
            raise ValueError('SIMM: {} maps to {}, and a point of vega moves a vol surface - a '
                             'model\'s parameters move by their family\'s fit, which a move here '
                             're-solves for curves and credit alone'.format(
                                 factor, stated['RiskType']))
        return dict({'Qualifier': '', 'Bucket': '', 'Label1': '', 'Label2': ''}, **stated)

    def product_class(self, walk, node):
        """A trade's product class off the factors its `walk` reads (`PRODUCT_CLASSES`): the one
        class among Credit, Equity and Commodity they are of, RatesFX where they are of none, its
        curves and currencies being every trade's, or where they are of more than one the
        `Product_Class` the trade states, refused by name where it states none or one the
        methodology does not name."""
        field = node['Instrument'].field
        classes = sorted({self.PRODUCT_CLASSES[utils.FactorRiskClass[kind]] for kind in {
            utils.check_rate_name(factor)[0] for factor in walk}
            if utils.FactorRiskClass[kind] in self.PRODUCT_CLASSES} - {'RatesFX'})
        if len(classes) < 2:
            return classes[0] if classes else 'RatesFX'
        stated = field['Product_Class']
        if stated not in set(self.PRODUCT_CLASSES.values()):
            raise ValueError('SIMM: trade {} reads factors of the {} classes{} - state its '
                             'Product_Class, one of {}'.format(
                                 field.get('Reference'), ' and '.join(classes),
                                 ' and states {}'.format(stated) if stated else '',
                                 ', '.join(sorted(set(self.PRODUCT_CLASSES.values())))))
        return stated

    def agreement(self, reference):
        """What the CRIF carries of the agreement a trade sits under: the netting set's reference
        as its counterparty - the id a margin is reconciled by - and the regimes it declares margin
        posted and collected under, each a term of the agreement a blank one refuses by name."""
        terms = next(agreement['Instrument'].field for agreement, _ in self.trades
                     if agreement['Instrument'].field.get('Reference') == reference)
        regimes = {key: terms.get(field) or '' for key, field in (
            ('PostRegulations', 'Post_Regulations'), ('CollectRegulations', 'Collect_Regulations'))}
        blank = [field for field, regime in regimes.items() if not regime]
        if blank:
            raise ValueError('SIMM: the netting set {} declares no {} - the regimes margin is '
                             'posted and collected under are terms of the agreement'.format(
                                 reference, ' or '.join(blank)))
        return dict(regimes, Counterparty=reference)

    def walk(self, index):
        """The price factors trade `index` reads, booked alone under its netting set, refused by
        name where one is missing or an interest rate or a survival curve no configured family
        bootstraps - a curve pasted in as nodes is no margin."""
        from . import bootstrappers

        agreement, node = self.trades[index]
        reference, deals = node['Instrument'].field.get('Reference'), self.config.deals['Deals']
        self.config.deals['Deals'] = {'Children': [dict(agreement, Children=[node])]}
        try:
            universe = self.config.factor_universe()
        finally:
            self.config.deals['Deals'] = deals
        if universe['missing']:
            raise ValueError('SIMM: trade {} reads {}, which the market data does not '
                             'carry'.format(reference, ', '.join(universe['missing'])))
        families = {family.__name__: family.market_factor_type + '.' for family in (
            bootstrappers.InterestRate, bootstrappers.SurvivalProb)}

        def quoted(rate):
            return families[rate[0]] + '.'.join(rate[1:]) in self.claimed or (
                rate[0] == 'InterestRate' and len(rate) > 2 and quoted(rate[:-1]))

        unquoted = [name for name in universe['resolved'] if utils.check_rate_name(name)[0] in
                    families and not quoted(utils.check_rate_name(name))]
        if unquoted:
            raise ValueError(
                'SIMM: trade {} reads {}, which no InterestRatePrices or SurvivalProbPrices '
                'block a configured family claims bootstraps and which is no spread over a curve '
                'one does. Market prices are a condition of the calculation - quote the curve '
                '(bootstrappers.knot_prices authors an interest rate block on its own knots) and '
                'configure its family'.format(reference, ', '.join(unquoted)))
        return universe['resolved']

    def bumps(self, index, walk):
        """Every move ISDA defines a sensitivity of trade `index` by, in a fixed order: each
        benchmark quote of every curve and survival block its `walk` stands on, then each other
        mapped factor, the price factors of a commodity in one."""
        from . import bootstrappers
        from .bootstrappers import bootstrap_precedents, bootstrap_writers, quote_nodes

        reference = self.trades[index][1]['Instrument'].field.get('Reference')
        prices, factors = self.config.params['Market Prices'], self.config.params['Price Factors']
        curves = bootstrappers.InterestRate
        families = {family.market_factor_type: family for family in (
            curves, bootstrappers.SurvivalProb)}
        # an inflation curve moves on all its breakevens at once, one row, so its block lists none
        admitted = set(families) | {bootstrappers.InflationRate.market_factor_type}
        writers = bootstrap_writers(prices, walk)
        others = sorted(name for name in writers if utils.check_rate_name(name)[0] not in admitted)
        if others:
            raise ValueError('SIMM: trade {} reads what {} writes, and a move here re-solves the '
                             'curve and credit families alone'.format(reference, ', '.join(others)))
        moves, base_date = [], self.config.params['System Parameters']['Base_Date']
        for block in sorted(name for name in bootstrap_precedents(prices, writers)
                            if utils.check_rate_name(name)[0] in families):
            rate = utils.check_rate_name(block)
            family = families[rate[0]]
            factor = utils.check_tuple_name(utils.Factor(family.__name__, rate[1:]))
            stated = self.coordinates(factor, reference)
            instrument = declared_defaults(family, prices[block]['instrument'])
            used = family.used_quotes(instrument, block)
            # what moves a price factor in place re-solves no block, so no block may read one
            read = sorted(curve for curve in {instrument['Discount_Rate']} |
                          curves.benchmark_curves(instrument, block)
                          if curve and curves.market_factor_type + '.' + curve not in prices
                          ) if family is curves else []
            if read or any(point['DealType'] in self.OUTRIGHTS for point in used) or (
                    stated is not None and self.MOVES[stated['RiskType']][1] not in (
                        'quote', 'cds')):
                raise ValueError(
                    'SIMM: {} quotes {} in {}{}{} - a basis point moves a curve quoted in rates, '
                    'priced off curves quoted so and mapped Risk_IRCurve'.format(
                        block, factor, ', '.join(sorted({point['DealType'] for point in used})),
                        ' priced off {}'.format(', '.join(read)) if read else '',
                        ' and mapped {}'.format(stated['RiskType']) if stated else ''))
            if stated is None:
                continue
            if family is curves:
                discount = instrument['Discount_Rate'] or '.'.join(rate[1:])
                vertices, tenors = self.vertices, curves.quote_knots(
                    quote_nodes(used, discount), base_date, 'ACT_365', self.config.holidays)
            else:
                # a credit quote's sensitivity is at its tenor's vertex, as ISDA quotes them
                vertices = self.credit_vertices
                tenors = [self.years(point['Tenor']) for point in used]
            positions = [position for position, point in enumerate(instrument['Points'])
                         if any(point is quote for quote in used)]
            moves.extend(self.Bump(factor, stated, utils.vertex_weights(tenor, vertices),
                                   (self.MOVES[stated['RiskType']][1], block, position))
                         for position, tenor in zip(positions, tenors))

        currency = self.valuation['Currency']
        for factor in walk:
            stated = self.coordinates(factor, reference)
            kind = None if stated is None else self.MOVES[stated['RiskType']][1]
            rate = utils.check_rate_name(factor)
            if kind == 'quote' and not any(move.factor == factor for move in moves):
                raise ValueError('SIMM: {} maps to Risk_IRCurve, which moves a curve by its '
                                 'quotes, and no InterestRatePrices block quotes it - map it null '
                                 'where it rides on its parent'.format(factor))
            elif kind == 'inflation':
                # the breakevens a curve is defined by are its block's quotes where one solves it
                moves.append(self.Bump(factor, stated, [('', 1.0)], (kind, factor) + tuple(
                    sorted(bootstrap_writers(prices, {factor}) & self.claimed))))
            elif kind in ('parallel', 'spot') or (kind == 'fx' and rate[1] != currency):
                moves.append(self.Bump(factor, stated, [('', 1.0)], (kind, factor)))
            elif kind == 'commodity' and not any(move.coordinates == stated for move in moves):
                # the price factors a mapping states under one qualifier are one commodity
                moves.append(self.Bump(factor, stated, [('', 1.0)], (kind,) + tuple(sorted(
                    name for name in walk if self.coordinates(name, reference) == stated))))
            elif kind in self.VEGAS:
                surface = factors[factor].get('Surface_Type', 'Explicit')
                if surface not in self.VOL_PILLARS:
                    raise ValueError('SIMM: {} is a {} surface, and a point of vega moves the vols '
                                     'of a {} one'.format(factor, surface,
                                                          ', '.join(self.VOL_PILLARS)))
                moves.extend(self.Bump(factor, stated, [(label, 1.0)], (kind, factor, label))
                             for label, _ in self.vertices)
        return moves

    def move(self, move, tangent=False):
        """The price factors `move` writes over the standing ones, a vol surface's over the one the
        set is compiled on (`surfaces`); with `tangent`, what it writes to first order in its
        shift, a curve quote's ride alone, the survival curves solved on it chained on their tape
        (`calibrate`)."""
        factors, kind, name = self.config.params['Price Factors'], move[0], move[1]
        base = self.config.params['System Parameters']['Base_Currency']
        spot = 1.0 + self.params['Spot_Shift'] / 100.0
        if kind == 'quote':
            return self.requoted(name, move[2], tangent)
        if kind == 'cds':
            return self.resolved({name}, {}, move[2])
        if kind == 'parallel':
            curve = factors[name]['Curve']
            return {name: dict(factors[name], Curve=utils.Curve(
                curve.meta, curve.array + np.array([0.0, self.params['Rate_Shift'] * 1e-4])))}
        if kind == 'inflation' and move[2:]:
            return self.reinflated(name, move[2])
        if kind == 'inflation':
            # each node is a growth rate r whose breakeven, exp(r) - 1, the basis point moves
            curve = factors[name]['Curve']
            tenors, rates = curve.array.T
            step = self.params['Rate_Shift'] * 1e-4 * np.exp(-rates)
            return {name: dict(factors[name], Curve=utils.Curve(curve.meta, np.column_stack((
                tenors, rates + (step if tangent else np.log1p(step))))))}
        if kind == 'commodity':
            moved = {}
            for other in move[1:]:
                type_name = utils.check_rate_name(other)[0]
                structural, values = schema.partition_factor(type_name, factors[other])
                # a commodity's price is every value its factors hold, a spot's and each delivery's
                moved[other] = schema.apply_values(type_name, structural, {
                    key: np.multiply(value, spot) for key, value in values.items()})
            return moved
        if kind == 'fx' and utils.check_rate_name(name)[1] == base:
            # a currency's price in the calculation currency is its spot over that currency's,
            # so the base currency's - whose own spot is one - rises by every other falling
            return {other: dict(block, Spot=block['Spot'] * (2.0 - spot) if tangent else
                                block['Spot'] / spot)
                    for other, block in factors.items()
                    if utils.check_rate_name(other)[0] == 'FxRate'
                    and utils.check_rate_name(other)[1] != base}
        if kind in ('fx', 'spot'):
            return {name: dict(factors[name], Spot=factors[name]['Spot'] * spot)}
        block = self.surfaces[name]
        held, column = self.VOL_PILLARS[block.get('Surface_Type', 'Explicit')]
        rows = block[held].array.copy()
        point = self.params['Vol_Shift'] * 0.01 * np.array([dict(
            utils.vertex_weights(t, self.vertices)).get(move[2], 0.0) for t in rows[:, column]])
        # a rate vertex moves each vol a percent of itself, the one rule in either convention
        rows[:, -1] += rows[:, -1] * point if kind == 'weighted' else riskfactors.Factor2D.vol_point(
            block, point, tangent)
        return {name: dict(block, **{held: utils.Curve(block[held].meta, rows)})}

    def requoted(self, block, position, tangent=False):
        """Every curve of `block`'s coupled set with a basis point on its quote `position`, ridden
        off the calibration `calibrate` published, `theta* + J (q - q0)`, and written as the
        family writes a solved curve, with the survival curves solved on one re-solved on them
        (`resolved`), or with `tangent` the ride alone - nothing on the config written."""
        from . import bootstrappers

        config, family = self.config, bootstrappers.InterestRate
        prices, factors = config.params['Market Prices'], config.params['Price Factors']
        standing = prices[block]
        points = [dict(point, Quoted_Market_Value=point['Quoted_Market_Value'] +
                       self.params['Rate_Shift'] * 0.01) if at == position else point
                  for at, point in enumerate(standing['instrument']['Points'])]
        artifact = family.artifact(
            config.artifacts, utils.Factor(family.__name__, utils.check_rate_name(block)[1:]),
            prices, config.params['Price Factor Interpolation'],
            config.params['System Parameters']['Base_Date'])
        theta = artifact.ride(family.quotes(artifact, dict(prices, **{block: dict(
            standing, instrument=dict(standing['instrument'], Points=points))})))
        ridden = {(name := utils.check_tuple_name(curve)): dict(
            factors[name], Curve=family.solved_curve(artifact.benchmarks, theta, curve))
            for curve in artifact.factors}
        return ridden if tangent else dict(ridden, **self.resolved(
            bootstrappers.bootstrap_dependents(prices, artifact.members), ridden))

    def survival(self, block):
        """`(solver, instrument)` - the survival family as configured, and the survival block
        `block`'s instrument as that family completes it."""
        from . import bootstrappers

        name, config = bootstrappers.SurvivalProb.__name__, self.config.params
        solver = bootstrappers.construct_bootstrapper(
            name, config['Bootstrapper Configuration'][name])
        return solver, dict(solver.param, **config['Market Prices'][block]['instrument'])

    def spread(self, block, instrument, position, found, shift):
        """Quote `position` of the survival block `block`, its `instrument` completed, a basis point
        of spread `shift` over on the price factors `found`: a `Par_Spread` by the point, an
        `Upfront` to what its contract quotes at a coupon the point under its own, the value being
        affine in the coupon."""
        from . import bootstrappers

        config, point = self.config, instrument['Points'][position]
        if point['Quote_Type'] == 'Par_Spread':
            return point['Quoted_Market_Value'] + shift
        base_date = config.params['System Parameters']['Base_Date']
        contract = bootstrappers.SurvivalProb.contracts(dict(instrument, Points=[point]), block,
                                                        base_date)[0]
        return float(bootstrappers.par_quotes([dict(contract, Deal=dict(
            contract['Deal'], Pay_Rate=utils.Basis(
                1e4 * float(contract['Deal']['Pay_Rate']) - shift)))],
            instrument['Discount_Rate'], instrument['Currency'], found,
            config.params['Price Factor Interpolation'], base_date, config.holidays)[0])

    def resolved(self, blocks, curves, position=None):
        """The survival curves the book reads that `blocks` write, solved by their family on the
        standing curves with `curves` over them and with each block's quote `position` moved by a
        basis point of spread (`spread`) on the solved curve - nothing on the config written."""
        from . import bootstrappers

        config, family = self.config, bootstrappers.SurvivalProb
        interp = config.params['Price Factor Interpolation']
        base_date = config.params['System Parameters']['Base_Date']
        found, solved = dict(config.params['Price Factors'], **curves), []
        for block in sorted(self.credit.intersection(blocks)):
            solver, instrument = self.survival(block)
            if position is not None:
                points = instrument['Points'] = list(instrument['Points'])
                points[position] = dict(points[position], Quoted_Market_Value=self.spread(
                    block, instrument, position, found, self.params['Rate_Shift']))
            solver.solve(block, instrument, found, interp, base_date, config.holidays)
            solved.append(utils.check_tuple_name(utils.Factor(
                family.__name__, utils.check_rate_name(block)[1:])))
        return {name: found[name] for name in solved}

    def reinflated(self, name, block):
        """The inflation curve `name` that `block` solves, solved by its family on the standing
        curves with every breakeven a basis point up - nothing on the config written."""
        from . import bootstrappers

        config, family = self.config, bootstrappers.InflationRate
        solver = bootstrappers.construct_bootstrapper(
            family.__name__, config.params['Bootstrapper Configuration'][family.__name__])
        instrument = dict(solver.param, **config.params['Market Prices'][block]['instrument'])
        instrument['Points'] = [dict(point, Quoted_Market_Value=point['Quoted_Market_Value'] +
                                     self.params['Rate_Shift'] * 0.01)
                                for point in instrument['Points']]
        found = dict(config.params['Price Factors'])
        solver.solve(block, instrument, found, config.params['Price Factor Interpolation'],
                     config.params['System Parameters']['Base_Date'], config.holidays)
        return {name: found[name]}

    def base(self, moves=()):
        """`[(value, skipped)]` per trade off the book compiled with each surface a vega move
        reaches given its vertex pillars (`surfaces`), over the markets `moves` write as its
        scenarios (`scenarios`)."""
        factors = self.config.params['Price Factors']
        self.surfaces = {name: getattr(riskfactors, utils.check_rate_name(name)[0]).augmented(
            factors[name], [t for _, t in self.vertices]) for name in {
            bump.move[1] for bumps in self.bumped for bump in bumps if bump.move[0] in self.VEGAS}}
        self.scenarios = {move: self.move(move) for move in moves}
        self.valuation['Scenarios'] = list(self.scenarios.values())
        return super(SIMM, self).base(self.surfaces)

    def reach(self, base):
        """`[(move, [index])]` - every move the trades valued in `base` take, in a fixed order,
        and the trades whose rows it writes; under Graph `[index]`, each trade valued."""
        if self.graph:
            return [index for index, (value, _) in enumerate(base) if value is not None]
        reach = {}
        for index, bumps in enumerate(self.bumped):
            for bump in bumps if base[index][0] is not None else []:
                reach.setdefault(bump.move, []).append(index)
        return list(reach.items())

    def values(self, work):
        """`{(index, move): change}` - each `(move, indices)` of `work` a scenario, its market the
        move's, priced in passes (`ScenarioValuation.passes`) beside the standing market, and each
        trade of `indices` read off its value there less on the standing market in the same pass,
        so the two share its arithmetic; a move states no factor a scenario axis does not carry. A
        trade valued to nothing finite is refused by name; under Graph `{(index, move):
        derivative}` off each trade of `work` (`derivatives`)."""
        if self.graph:
            return self.derivatives(work)
        scenarios, found = [self.scenarios[move] for move, _ in work], {}
        batched, _ = self.compiled.passes(self.shared, scenarios)

        def priced(chunk):
            """Per position of `chunk`, each trade's value on its market and on the standing one."""
            valued = self.value([{}] + [scenarios[at] for at in chunk])
            return [(column, valued[0]) for column in valued[1:]]

        def finite(value):
            return value is not None and math.isfinite(value)

        for chunk in batched:
            columns = priced(chunk)
            if len(chunk) > 1 and not all(finite(column[index]) and finite(base[index]) for (
                    column, base), at in zip(columns, chunk) for index in work[at][1]):
                # a pricer refusing one market leaves its trade unvalued on all, so each is priced
                # alone and the refusal names its own move
                columns = [priced([at])[0] for at in chunk]
            for (column, base), at in zip(columns, chunk):
                move, indices = work[at]
                for index in indices:
                    if not finite(column[index]):
                        raise ValueError(
                            '{}: moving {} values the trade to {}, off which no sensitivity can be '
                            'read'.format(self.trades[index][1]['Instrument'].field.get('Reference'),
                                          '/'.join(map(str, move)), column[index]))
                    found[index, move] = column[index] - base[index]
        return found

    def derivatives(self, indices):
        """`{(index, move): derivative}` - each trade of `indices` differentiated by one backward
        pass of its taped value, each move it takes read off that gradient along the move's
        `tangent`. A boundary correction is worth nothing here: a base valuation's one scenario
        gives a latched decision no kernel width, and the smooth estimator it declares registers
        no inner path's."""
        config = self.config
        tape = {**self.compiled.static_var, **{
            ('Calibrated', factor): theta for factor, theta in config.calibrated_factors.items()},
            **{name: quotes for name, (_, quotes) in config.quote_leaves.items()}}
        tangents, found = {}, {}
        for index in indices:
            moves = [bump.move for bump in self.bumped[index]]
            for move in moves:
                if move not in tangents:
                    tangents[move] = self.tangent(move)
            keys = list(dict.fromkeys(key for move in moves for key in tangents[move]))
            value = self.results[index]['tensor'].sum()
            gradient = dict(zip(keys, torch.autograd.grad(
                [value], [tape[key] for key in keys], allow_unused=True,
                retain_graph=True) if value.requires_grad and keys else ()))
            for move in moves:
                found[index, move] = sum(
                    float(np.dot(direction, gradient[key].detach().cpu().numpy().ravel()))
                    for key, direction in tangents[move].items() if gradient.get(key) is not None)
        return found

    def tangent(self, move):
        """`move`'s derivative at a zero shift, per unit of ISDA's shift: `{key: direction}` along
        the tensors `derivatives` differentiates against - the compiled leaves the move writes, to
        first order in it (`move`); the calibrated curves a quote rides, whose tape carries the
        survival curves solved on them; and the quotes of a block the move re-solves, the
        re-solve's derivative being its family's Jacobian on the tape (`calibrate`)."""
        from .bootstrappers import SurvivalProb

        kind, name, shift = move[0], move[1], self.shifts[move[0]]
        if kind == 'cds':
            _, instrument = self.survival(name)
            used = SurvivalProb.used_quotes(instrument, name)
            standing, direction = self.config.params['Price Factors'], np.zeros(len(used))
            # a par spread moves by the point, an upfront by what a point of coupon is worth
            direction[[quote is instrument['Points'][move[2]] for quote in used].index(True)] = (
                self.spread(name, instrument, move[2], standing, shift) -
                self.spread(name, instrument, move[2], standing, 0.0)) / shift
            return {name: direction}
        if kind == 'inflation' and move[2:]:
            # every breakeven a basis point up, a hundredth of the percent it is quoted in
            return {move[2]: np.full(self.config.quote_leaves[move[2]][1].numel(), 0.01)}
        moved = self.move(move, tangent=True)
        if kind == 'quote':
            curves = [utils.Factor(rate[0], rate[1:]) for rate in map(utils.check_rate_name, moved)]
            standing = dict(self.compiled.leaves(self.config.params['Price Factors'], curves))
            return {('Calibrated', key): (value - standing[key]) / shift
                    for key, value in self.compiled.leaves(moved, curves)}
        return {key: (value - self.compiled.static_var[key].detach().cpu().numpy()) / shift
                for key, value in self.compiled.leaves(moved)}

    def trade(self, index, base, values):
        """Trade `index`'s CRIF rows off the book's `base` and the changes its moves made
        (`values`), or under Graph their derivatives, its base value, the moves it took and what the
        engine skipped inside it - its rows written regardless."""
        agreement, node = self.trades[index]
        value, skipped = base[index]
        rows = {}
        for bump in self.bumped[index] if value is not None else []:
            delta = values[index, bump.move] if self.graph else (
                values[index, bump.move] / self.shifts[bump.move[0]])
            stated = bump.coordinates
            for label, weight in bump.split:
                key = (stated['RiskType'], stated['Qualifier'], stated['Bucket'],
                       label or stated['Label1'], stated['Label2'])
                rows[key] = rows.get(key, 0.0) + float(weight) * delta
        return {'TradeID': node['Instrument'].field.get('Reference'),
                'NettingSet': agreement['Instrument'].field.get('Reference'),
                'ProductClass': self.classes[index],
                'Value': value, 'Moves': len(self.bumped[index]), 'Skipped': skipped,
                'Rows': [key + (amount,) for key, amount in rows.items() if amount != 0.0]}

    def frame(self, trades):
        """The CRIF of `trades`, in trade order and then by coordinates, a vertex by its place on
        the grid; a move that left a trade's value exactly where it stood writes no row."""
        usd = self.usd()
        order = {label: position for position, (label, _) in enumerate(self.vertices)}
        rows = []
        for trade in trades:
            agreement = self.agreement(trade['NettingSet'])
            for risk_type, qualifier, bucket, label1, label2, amount in sorted(
                    trade['Rows'], key=lambda row: row[:3] + (order.get(row[3], -1),) + row[3:5]):
                rows.append(dict(agreement, TradeID=trade['TradeID'],
                                 ProductClass=trade['ProductClass'], RiskType=risk_type,
                                 Qualifier=qualifier, Bucket=bucket, Label1=label1, Label2=label2,
                                 Amount=amount, AmountUSD=amount * usd))
        return pd.DataFrame(rows, columns=list(self.CRIF_COLUMNS))

    def usd(self):
        """A unit of the calculation currency in USD, off the `FxRate` factors."""
        factors, currency = self.config.params['Price Factors'], self.valuation['Currency']
        return 1.0 if currency == 'USD' else factors['FxRate.' + currency]['Spot'] / factors[
            'FxRate.USD']['Spot']

    def margins(self, crif):
        """`{'Margin': ..., 'Margin_Rows': ...}` off the CRIF `crif`: each netting set's margin in
        each direction under the regime its agreement declares, by product class, risk class and
        margin type with each level's `Total`, in the calculation currency and USD (`margin`); and
        per row and direction the derivative of its set's margin in the row's amount and its share,
        the derivative times the amount."""
        usd, margins, shares = self.usd(), [], []
        for counterparty in dict.fromkeys(
                agreement['Instrument'].field.get('Reference') for agreement, _ in self.trades):
            rows = crif[crif['Counterparty'] == counterparty]
            agreement = self.agreement(counterparty)
            for direction, regime, sign in self.DIRECTIONS:
                amounts = torch.tensor(rows['AmountUSD'].to_numpy(np.float64), requires_grad=True)
                found = self.margin(rows, sign * amounts)
                total = found['Total', 'Total', 'Total']
                derivative = torch.autograd.grad(total, amounts)[0].numpy() if (
                    total.requires_grad) else np.zeros(len(rows))
                margins.extend(dict(
                    Counterparty=counterparty, Direction=direction, Regime=agreement[regime],
                    ProductClass=product, RiskClass=risk_class, MarginType=margin,
                    Amount=value.item() / usd, AmountUSD=value.item())
                    for (product, risk_class, margin), value in found.items())
                shares.extend(dict(
                    row, Direction=direction, Regime=agreement[regime], Derivative=slope,
                    Share=slope * row['Amount'], ShareUSD=slope * row['AmountUSD'])
                    for row, slope in zip(rows.to_dict('records'), derivative))
        return {'Margin': pd.DataFrame(margins, columns=[
                    'Counterparty', 'Direction', 'Regime', 'ProductClass', 'RiskClass',
                    'MarginType', 'Amount', 'AmountUSD']),
                'Margin_Rows': pd.DataFrame(shares, columns=[
                    'TradeID', 'Counterparty', 'Direction', 'Regime', 'ProductClass', 'RiskType',
                    'Qualifier', 'Bucket', 'Label1', 'Label2', 'Derivative', 'Share', 'ShareUSD'])}

    def margin(self, rows, amounts):
        """The methodology's margin of one netting set's CRIF `rows` as one torch function of
        `amounts`, their USD amounts as one direction reads them: `{(product class, risk class,
        margin type): margin}` in USD, each risk class's three margins summed as its `Total`, a
        product class's risk classes correlated by the class correlations as its `Total` and the
        product classes summed as the set's."""
        correlations, found = self.parameters['class_rho']['Correlation'], {}
        for product in rows['ProductClass'].unique():
            within, classes = (rows['ProductClass'] == product).to_numpy(), {}
            for risk_class, stated in self.RISK_CLASSES.items():
                held = {margin: within & rows['RiskType'].isin(
                    stated.deltas if margin == 'Delta' else (stated.vega,)).to_numpy()
                    for margin in self.MARGINS}
                if not any(mask.any() for mask in held.values()):
                    continue
                for margin, mask in held.items():
                    found[product, risk_class, margin] = amounts.new_zeros(()) if not mask.any() \
                        else self.aggregated(risk_class, margin, rows[mask],
                                             amounts[torch.as_tensor(mask)])
                classes[risk_class] = found[product, risk_class, 'Total'] = sum(
                    found[product, risk_class, margin] for margin in self.MARGINS)
            names = [self.RISK_CLASSES[risk_class].name for risk_class in classes]
            psi = amounts.new_tensor([[self.correlation(correlations, a, b) for b in names]
                                      for a in names])
            margin = torch.stack(list(classes.values()))
            found[product, 'Total', 'Total'] = utils.sqrt_or_zero(margin @ psi @ margin)
        found['Total', 'Total', 'Total'] = sum(
            (value for (_, risk_class, _), value in found.items() if risk_class == 'Total'),
            amounts.new_zeros(()))
        return found

    def aggregated(self, risk_class, margin, rows, amounts):
        """`risk_class`'s `margin` - Delta, Vega or Curvature - of the CRIF `rows`, `amounts` their
        USD amounts as one direction reads them: each factor's exposure `x` its rows' amounts
        scaled, weighted and concentrated, `CR = sqrt(max(|sum| / T, 1))` over its group; a
        bucket's `K = sqrt(sum_kl rho_kl f_kl x_k x_l)` and `S` its exposures' sum within `[-K,
        K]`; the margin `sqrt(sum K^2 + sum_{b != c} gamma_bc g_bc S_b S_c)` over the non-residual
        buckets, plus the residual's `K` - or for curvature, on `rho^2` and `gamma^2`, each part's
        `max(sum x + lambda K, 0)`, `lambda` off `theta = min(sum x / sum |x|, 0)`, a rate's over
        its `HVR^2`. A norm at zero is zero, and a tie takes the branch a strict comparison
        selects."""
        found, correlation, across, inner, outer = self.exposures(risk_class, margin, rows)
        curvature = margin == 'Curvature'
        power, tensor = 2 if curvature else 1, partial(torch.as_tensor, dtype=amounts.dtype)
        described = {factor: rest for factor, *rest in found}
        factors = list(described)
        buckets = list(dict.fromkeys(described[factor][0] for factor in factors))
        groups = list(dict.fromkeys(described[factor][1] for factor in factors
                                    if described[factor][1] is not None))
        thresholds = tensor([next(described[factor][2] for factor in factors
                                  if described[factor][1] == group) for group in groups])
        residual = np.array([bucket == self.RESIDUAL for bucket in buckets])
        rho = tensor([[1.0 if k == l else correlation(k, l, described[k][0]) ** power
                       if described[k][0] == described[l][0] else 0.0 for l in factors]
                      for k in factors])
        gamma = tensor([[0.0 if b == c or residual[i] or residual[j] else across(b, c) ** power
                         for j, c in enumerate(buckets)] for i, b in enumerate(buckets)])
        member = tensor(np.eye(len(buckets))[[buckets.index(described[k][0]) for k in factors]])
        grouped = tensor([[float(described[k][1] == group) for group in groups] for k in factors])
        own = tensor([[float(bucket == group) for group in groups] for bucket in buckets])

        exposure = amounts.new_zeros(len(factors)).index_add(
            0, torch.as_tensor([factors.index(row[0]) for row in found]),
            amounts * tensor([row[5] for row in found]))
        level = (exposure @ grouped).abs() / thresholds
        group = torch.sqrt(torch.where(level > 1.0, level, torch.ones_like(level)))
        # a factor or a bucket in no group, a basis or a curvature, is unconcentrated
        concentration = grouped @ group + 1.0 - grouped.sum(1)
        x = tensor([described[k][3] for k in factors]) * exposure * concentration
        if inner:
            rho = rho * self.ratios(concentration)
        if outer:
            gamma = gamma * self.ratios(own @ group + 1.0 - own.sum(1))
        square = member.T @ (x * (rho @ x))
        K, summed = utils.sqrt_or_zero(square), member.T @ x
        S = torch.where(summed > K, K, torch.where(summed < -K, -K, summed))
        apart, inside = tensor(residual.astype(float)), tensor((~residual).astype(float))
        aggregate = utils.sqrt_or_zero((square * inside).sum() + S @ gamma @ S)
        if not curvature:
            return aggregate + (K * apart).sum()

        def floored(part, root):
            total, absolute = (summed * part).sum(), (x.abs() * (member @ part)).sum()
            theta = torch.where(total < 0.0, total / torch.where(
                absolute > 0.0, absolute, torch.ones_like(absolute)), torch.zeros_like(total))
            value = total + ((self.Q995 ** 2 - 1.0) * (1.0 + theta) - theta) * root
            return torch.where(value > 0.0, value, torch.zeros_like(value))

        floor = floored(inside, aggregate) + floored(apart, (K * apart).sum())
        return floor / self.parameters['ir']['hvr'] ** 2 if risk_class == 'InterestRate' else floor

    def exposures(self, risk_class, margin, rows):
        """`(found, correlation, across, inner, outer)` of the CRIF `rows` of `risk_class` to its
        `margin`, as the methodology and the parameters file state them: per row `(factor, bucket,
        group, threshold, weight, scale)` - the factor it nets into, the factor's bucket, its
        concentration group, None where nothing concentrates it, the group's threshold, the
        factor's weight and what the row's amount is scaled by there; the correlation of two
        factors of a bucket and of two buckets; and whether the concentration ratio scales each."""
        stated = self.parameters[self.RISK_CLASSES[risk_class].section]
        rw, delta, curvature = stated['rw'], margin == 'Delta', margin == 'Curvature'
        con = None if curvature else stated['con'][margin]['Threshold']
        rows = list(zip(*(rows[column] for column in (
            'RiskType', 'Qualifier', 'Bucket', 'Label1', 'Label2'))))

        def scale(expiry):
            # an expiry in days of the methodology's year, its scale half a fortnight's share of it
            return 0.5 * min(1.0, self.FORTNIGHT / (365.0 * self.years(expiry)))

        def vol(weight):
            # the vol a vega is weighed by, its delta weight over a fortnight at the 99th percentile
            return weight * np.sqrt(365.0 / self.FORTNIGHT) / self.Q99

        if risk_class == 'InterestRate':
            rho, groups = stated['rho'], stated['ccy']

            def group(currency, kind):
                return groups[kind]['Currency'].get(currency, groups[kind]['default'])

            def weight(risk_type, currency, bucket, tenor):
                weights = rw['Delta']['Volatility']
                return weights[stated['bucket_vol_map'][bucket]]['Weight'][tenor] if (
                    risk_type == 'Risk_IRCurve') else weights[group(currency, 'Weights')][{
                        'Risk_Inflation': 'InflationWeight',
                        'Risk_XCcyBasis': 'CrossCurrencyBasisWeight'}[risk_type]]

            def correlation(k, l, bucket):
                kinds = {k[1], l[1]}
                return rho['CrossCurrencyBasisCorrelation'] if 'Risk_XCcyBasis' in kinds else \
                    rho['InflationCorrelation'] if 'Risk_Inflation' in kinds else self.correlation(
                        rho['Correlation'], k[2], l[2]) * (
                        rho['BasisRiskCorrelation'] if k[3] != l[3] else 1.0)

            # a currency concentrates its rates and inflation alone, never its basis
            found = [((currency, risk_type, tenor, curve), currency,
                      None if curvature or risk_type == 'Risk_XCcyBasis' else currency,
                      None if curvature else con[group(currency, 'ConcentrationThresholds')],
                      weight(risk_type, currency, bucket, tenor) if delta else 1.0 if curvature
                      else rw['Vega'], scale(tenor) if curvature else 1.0)
                     for risk_type, currency, bucket, tenor, curve in rows]
            return found, correlation, lambda b, c: stated['rho_inter'], False, not curvature
        if risk_class == 'FX':
            calculation = self.params['Currency']

            def volatility(*currencies):
                return ''.join('High' if currency in stated['high'] else 'Regular'
                               for currency in currencies)

            def category(currency):
                return stated['ccy']['Currency'].get(currency, stated['ccy']['default'])

            # a pair's vol is the delta weight its two currencies' groups state, in order
            vols = {pair: vol(rw['Delta'][volatility(pair[:3], pair[3:])])
                    for _, pair, _, _, _ in rows if not delta}
            found = [(qualifier, '', qualifier, con[category(qualifier)],
                      rw['Delta'][volatility(qualifier, calculation)], 1.0) if delta else
                     (qualifier, '', None if curvature else qualifier,
                      None if curvature else con[category(qualifier[:3])][category(qualifier[3:])],
                      1.0 if curvature else rw['Vega'],
                      vols[qualifier] * (scale(expiry) if curvature else stated['hvr']))
                     for _, qualifier, _, expiry, _ in rows]
            correlations = stated['rho']['Correlation'][
                'CalculationCurrency{}Volatility'.format(volatility(calculation))]
            return found, (lambda k, l, bucket: correlations[volatility(k, l)]) if delta else (
                lambda k, l, bucket: stated['rho']['CorrelationVol']), None, not curvature, False
        if risk_class in ('CreditQ', 'CreditNonQ'):
            # an issuer's vertices are its factors, concentrated together, alike or not by name
            found = [((issuer, tenor, label), bucket, issuer, con[bucket],
                      rw['Delta']['Weight'][bucket], 1.0)
                     for _, issuer, bucket, tenor, label in rows]
            return found, lambda k, l, bucket: stated['rho_intra'][
                'Residual' if bucket == self.RESIDUAL else 'NonResidual'][
                'SameName' if k[0] == l[0] else 'DiffName'], lambda b, c: self.correlation(
                stated['rho_inter']['Correlation'], b, c), True, False
        vegas = rw['Vega'].get('Weight', {})
        found = [(qualifier, bucket, None if curvature else qualifier,
                  None if curvature else con[bucket],
                  rw['Delta']['Weight'][bucket] if delta else 1.0 if curvature else vegas.get(
                      bucket, rw['Vega']['default']),
                  1.0 if delta else vol(rw['Delta']['Weight'][bucket]) * (
                      scale(expiry) * (risk_class != 'Equity' or bucket != self.VOLATILITY_INDICES)
                      if curvature else stated['hvr']))
                 for _, qualifier, bucket, expiry, _ in rows]
        return found, lambda k, l, bucket: stated['rho_intra']['Correlation'][bucket], \
            lambda b, c: self.correlation(stated['rho_inter']['Correlation'], b, c), \
            not curvature, False

    @staticmethod
    def ratios(concentrations):
        """`min / max` of each pair of `concentrations`, the methodology's ratios f and g."""
        return torch.minimum(concentrations[:, None], concentrations) / torch.maximum(
            concentrations[:, None], concentrations)

    @staticmethod
    def correlation(table, a, b):
        """The correlation between `a` and `b` the upper triangle `table` of the parameters file
        states, either way round, one where they are one."""
        return 1.0 if a == b else table[a]['Correlation'][b] if b in table.get(a, {}).get(
            'Correlation', {}) else table[b]['Correlation'][a]

    @classmethod
    def write(cls, frame, path):
        """A CRIF as the interchange carries it and the margin calculation reads it, tab
        separated."""
        frame.to_csv(path, sep='\t', index=False, columns=list(cls.CRIF_COLUMNS))


class SACCR(NettingSetCalculation):
    """The standardised approach to counterparty credit risk (CRE52): each netting set's exposure
    at default, `alpha (RC + PFE)` (52.1), off one base valuation of the book and each trade's
    declared terms.

    A TRADE IS READ THROUGH ITS TYPE'S DECLARATION. Its asset class is the `Mapping`'s
    `SACCR.Asset_Class` for its type, one class a trade (52.26-52.27; 52.28's allocation to several
    is not built), and what its exposure is read off is its type's own `schema.Exposure`; a type
    the mapping classes in none, or declaring none, refuses by name. A credit entity's rating is the
    mapping's `SACCR.Rating` under its survival curve, AAA to CCC a single name and IG or SG an
    index; an equity is an index where its price's SIMM bucket is 11 or 12, Risk_Equity's index
    buckets; a commodity's type is its price's SIMM bucket, whose `commodity_sets` row in the
    `Parameters` names its hedging set and subclass.

    The netting set's terms, symbol by symbol:

    - margined (52.2): `Collateralized` True.
    - V (52.10): its trades' values in the calculation currency, each off its own structure.
    - C (52.11, 52.18): a margined set's `Opening_Balance` as the engine's balance counts it.
    - MPOR (52.50-52.51): `Liquidation_Period` in business days, floored as the file floors it.
    - TH (52.18): `Received_Threshold` at its first value, crossed from `Agreement_Currency`.
    - MTA (52.18): `Minimum_Received` at its first value, crossed from `Agreement_Currency`.
    - NICA (52.17): `Independent_Amount` at its first value, received positive, crossed likewise.

    A trade's dates count in years of 30/360 from the base date, a whole number of months exact as
    the methodology's examples count a tenor: S its declared start, zero once started, E and M the
    last day its type's tenor fields name (`schema.tenor_fields`), T its declared expiry (52.31).
    Its d is an IR or credit notional crossed times SD, averaged over its life where it amortises,
    an FX trade's foreign leg, the larger where both are, or an equity or commodity trade's units at
    a unit's price (52.34-52.37); its delta plus or minus one by its long side and the amount's
    sign, or an option's by Black on the supervisory vol at P, its underlying's forward at T off the
    compiled factors, a rate option's P and K shifted by its currency's `option_shift`
    (52.38-52.40); its MF sqrt(min(M, 1)), M floored at ten business days, or under a margined set
    1.5 sqrt(MPOR / 250) (52.48, 52.52).

    Not built: CDO tranches (52.41); basis and volatility hedging sets (52.46-52.47), every trade
    sitting in its ordinary set; the non-daily, cleared, illiquid and disputed MPOR floors
    (52.50(2)-(3), 52.51(2)-(3)), which a netting set declares nothing to read off; several margin
    agreements to a set or sets to an agreement (52.74-52.76).
    """
    documentation = ('Calculations', [
        'The exposure at default of each netting set under the standardised approach to counterparty',
        'credit risk [@bcbscre52], $\\mathrm{EAD}=\\alpha(\\mathrm{RC}+\\mathrm{PFE})$ (CRE52.1), off one base',
        'valuation of the book and each trade\'s terms as its type declares them. The inputs are:',
        '',
        '- **Currency** the exposures are in, and **Base Date** they are read at.',
        '- **Mapping**: per instrument type its asset class (`SACCR.Asset_Class`), per credit factor its',
        '  rating (`Factors.<name>.SACCR.Rating`, AAA to CCC a single name, IG or SG an index), and the',
        '  SIMM bucket of an equity, which is an index in buckets 11 and 12, and of a commodity, which is',
        '  its type.',
        '- **Parameters**: Table 2 of CRE52.72, alpha, the multiplier\'s floor, the maturity buckets and',
        '  correlations of CRE52.57, the maturity and MPOR floors of CRE52.48-52.52, each commodity',
        '  bucket\'s hedging set and subclass and a currency\'s option shift, each beside its paragraph.',
        '- **MCMC Simulations** and **Quasi Seed**, as a base valuation reads them.',
        '',
        'Per trade, its dates in years of 30/360 from the base date: the adjusted notional $d$',
        '(CRE52.34-52.37), an IR or credit trade\'s notional, averaged over time where it amortises,',
        'times $SD=(e^{-0.05S}-e^{-0.05E})/0.05$ floored at ten business days, an FX trade\'s foreign leg,',
        'and an equity or commodity trade\'s units at a unit\'s price; the supervisory delta $\\delta$',
        '(CRE52.38-52.40), $\\pm1$ by its long side or an option\'s Black delta on Table 2\'s vol at its',
        'underlying\'s forward $P$; the maturity factor (CRE52.48, CRE52.52), $\\sqrt{\\min(M,1)}$ with $M$',
        'floored at ten business days, or $1.5\\sqrt{\\mathrm{MPOR}/250}$ under a margined set; and',
        '$D=d\\,\\delta\\,\\mathrm{MF}$. The add-ons (CRE52.55-52.70): IR by currency over three maturity',
        'buckets, $\\mathrm{EN}=(D_1^2+D_2^2+D_3^2+1.4D_1D_2+1.4D_2D_3+0.6D_1D_3)^{1/2}$; FX by pair,',
        '$|\\sum D|$; credit and equity by entity and commodity by type within its hedging set, each',
        '$A_k$ its supervisory factor times $\\sum D$, as $\\big((\\sum\\rho_kA_k)^2+\\sum(1-\\rho_k^2)',
        'A_k^2\\big)^{1/2}$; summed over the classes (CRE52.25). Then',
        '',
        '$$\\mathrm{multiplier}=\\min\\Big(1,\\,F+(1-F)\\exp\\frac{V-C}{2(1-F)\\,\\mathrm{AddOn}}\\Big),',
        '\\qquad\\mathrm{PFE}=\\mathrm{multiplier}\\cdot\\mathrm{AddOn}$$',
        '',
        'with $F$ 5% (CRE52.23); RC is $\\max(V-C,0)$ unmargined and $\\max(V-C,\\,TH+MTA-NICA,\\,0)$',
        'margined (CRE52.10, CRE52.18), a margined set\'s EAD capped at its unmargined one (CRE52.2).',
        '',
        'Not built: CDO tranches (CRE52.41); basis and volatility hedging sets (CRE52.46-52.47); a trade\'s',
        'allocation to several classes (CRE52.28); the non-daily, cleared, illiquid and disputed MPOR',
        'floors (CRE52.50-52.51); several margin agreements to a set, or sets to one (CRE52.74-52.76). A',
        'type declaring no exposure refuses the run by name.'])
    #: Table 2's rows (52.72) by asset class, each subclass a factor, correlation and option vol:
    #: a credit index rated IG or SG, a single name AAA to CCC.
    SUBCLASSES = {'IR': ('',), 'FX': ('',),
                  'Credit': ('AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'IG', 'SG'),
                  'Equity': ('Single', 'Index'),
                  'Commodity': ('Electricity', 'Oil/Gas', 'Metals', 'Agricultural', 'Other')}
    #: Risk_Equity's index buckets, which make an equity an index.
    EQUITY_INDICES = ('11', '12')
    #: The columns of the netting sets frame and of the trades frame.
    SETS = ('Counterparty', 'Margined', 'MPOR', 'V', 'C', 'RC') + tuple(
        'AddOn_' + asset for asset in SUBCLASSES) + (
        'AddOn_Aggregate', 'Multiplier', 'PFE', 'EAD', 'Skipped')
    TRADES = ('TradeID', 'NettingSet', 'AssetClass', 'HedgingSet', 'Subclass', 'Bucket',
              'SupervisoryFactor', 'AdjustedNotional', 'Delta', 'MaturityFactor',
              'EffectiveNotional', 'Value')
    fields = [
        F('Base_Date', 'Date', default=''),
        F('Currency', 'Text', default=REQUIRED,
          description='The calculation currency the exposures are in'),
        F('Mapping', 'Text', default=REQUIRED,
          description='Path to the regime mapping: per instrument type its SA-CCR asset class, per '
                      'credit factor its rating and per price factor its SIMM bucket'),
        F('Parameters', 'Text', default=REQUIRED,
          description='Path to the SA-CCR parameters file, the supervisory numbers of CRE52 each '
                      'beside the paragraph it is read from'),
        ScenarioValuation.MCMC_SIMULATIONS,
        NettingSetCalculation.QUASI_SEED
    ]

    def setup(self, params):
        """The block and the book read as every calculation over netting sets reads them, and the
        `Parameters` file's numbers, refused by name where one CRE52 reads is missing or
        malformed."""
        parameters = super(SACCR, self).setup(params)
        try:
            self.alpha, self.floor = float(parameters['alpha']), float(parameters['multiplier_floor'])
            self.supervisory = {(asset, subclass): {key: float(row[key] or 0.0) for key in (
                'factor', 'correlation', 'option_volatility')}
                for asset, subclasses in self.SUBCLASSES.items() for subclass, row in (
                    (subclass, parameters['supervisory'][asset][subclass]) for subclass in subclasses)}
            rates, maturity = parameters['interest_rate'], parameters['maturity']
            self.duration_rate = float(rates['supervisory_duration'])
            self.buckets = tuple(float(edge) for edge in rates['buckets'])
            self.correlation = np.array(rates['correlations'], dtype=np.float64).reshape(3, 3)
            self.year, self.margined_scale = float(maturity['business_days']), float(
                maturity['margined'])
            self.horizon = float(maturity['floor']) / self.year
            floors = maturity['mpor']
            self.floors = (float(floors['bilateral']), float(floors['large_set']),
                           int(floors['large_set_trades']))
            self.sets = {str(bucket): (hedging, subclass)
                         for bucket, (hedging, subclass) in parameters['commodity_sets'].items()}
            unknown = {subclass for _, subclass in self.sets.values()}.difference(
                self.SUBCLASSES['Commodity'])
            if unknown:
                raise KeyError('a commodity_sets subclass of Table 2: {}'.format(
                    ', '.join(sorted(unknown))))
            self.shifts = {currency: float(shift)
                           for currency, shift in parameters['option_shift'].items()}
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('SACCR: {} does not state {} as CRE52 reads it'.format(
                self.params['Parameters'], error))

    def execute(self, params):
        """The run: the book valued once and each netting set's exposure read off it."""
        self.setup(params)
        with self.prepared():
            return self.report(self.base())

    def years(self, day):
        """`day` in years of 30/360 from the base date, nothing once it has passed."""
        base = self.valuation['Base_Date']
        return max(utils.DayCount.accrual(base, (day - base).days, utils.DayCount.ACT30_360), 0.0)

    def fx(self, currency):
        """What one unit of `currency` is worth in the calculation currency at the base date, off
        the compiled spots."""
        spots = self.compiled.all_factors
        return spots[utils.Factor('FxRate', utils.check_rate_name(currency))].current_value()[
            0] / spots[utils.Factor('FxRate', utils.check_rate_name(self.params['Currency']))
                       ].current_value()[0]

    def stated(self, reference, name, regime, key):
        """What the mapping states under `regime`'s `key` for the price factor `name` trade
        `reference` reads, refused by name where it states nothing."""
        found = ((self.mapping['Factors'].get(name) or {}).get(regime) or {}).get(key)
        if not found:
            raise ValueError('SACCR: trade {} reads {}, and the mapping states no {}.{} for '
                             'it'.format(reference, name, regime, key))
        return str(found)

    def price(self, deal, field):
        """`(price, currency)` of one unit of what `field` names at the base date (52.36): a spot
        as compiled, or a reference price's forward curve at the base date's delivery, its first
        where the curve starts later."""
        found = self.compiled.all_factors[utils.Factor(
            type(deal).factor_fields[field][0], utils.check_rate_name(deal.field[field]))]
        if isinstance(found, riskfactors.ReferencePrice):
            found = self.compiled.all_factors[utils.Factor('ForwardPrice', found.get_forwardprice())]
            return found.current_value(
                [(self.valuation['Base_Date'] - utils.excel_offset).days])[0], found.get_currency()
        return found.current_value()[0], found.get_currency()

    def averaged(self, principal, steps, end):
        """52.37(2): `principal` less the amortisation `steps` it has reached, averaged over time
        from the base date to `end`."""
        held, area, last = principal, 0.0, 0.0
        for day, step in sorted(steps.data.items()):
            reached = min(self.years(day), end)
            area, held, last = area + held * (reached - last), held - step, reached
        return (area + held * (end - last)) / end

    def duration(self, start, end):
        """SD (52.34), `(exp(-r S) - exp(-r E)) / r` at the supervisory rate, floored at ten
        business days."""
        rate = self.duration_rate
        return max(math.exp(-rate * start) * -math.expm1(-rate * (end - start)) / rate,
                   self.horizon)

    def swap_rate(self, deal, terms):
        """A swaption's forward swap rate in percent, its strike's unit: the par rate of the legs it
        is written on, on the curve it forecasts on, as the compiled legs read it."""
        structure = next(structure for structure in self.compiled.netting_sets.structures() if (
            structure.obj.Instrument.resolved_from or structure.obj.Instrument) is deal)
        curve = self.compiled.all_factors[structure.obj.Factor_dep['Forward'][0][
            utils.FACTOR_INDEX_Offset]]
        legs = [leg.Factor_dep['Cashflows'] for leg in structure.dependencies]
        floating = next(leg for leg in legs if leg.Resets is not None)
        fixed = next(leg for leg in legs if leg.Resets is None)
        rate, annuity = floating.get_par_swap_rate(self.valuation['Base_Date'], curve)
        return 100.0 * rate * annuity / fixed.get_par_swap_rate(self.valuation['Base_Date'], curve)

    def fx_forward(self, deal, terms):
        """An FX option's outright to its expiry, a unit of its amount's currency in the other one,
        as the engine carries a pair (`utils.calc_fx_forward`)."""
        calc = self.compiled
        codes = [instruments.get_fx_and_zero_rate_factor(
            utils.check_rate_name(deal.field[key]), calc.static_factors, calc.stoch_factors,
            calc.all_tenors, calc.all_factors) for key in (terms.currency, terms.underlying)]
        return utils.calc_fx_forward(
            *codes, (deal.field[terms.expiry] - self.valuation['Base_Date']).days,
            calc.time_grid.time_grid[:1], self.shared).item()

    #: How an option's P is read off the compiled factors, by its asset class.
    FORWARDS = {'IR': swap_rate, 'FX': fx_forward}

    def exposure(self, index, mpor):
        """Trade `index`'s row of the trades frame, its maturity factor its set's (`mpor` None
        where unmargined), and its effective notional were the set unmargined, `Unmargined`."""
        agreement, node = self.trades[index]
        deal = node['Instrument']
        fields, kind, reference = deal.field, deal.field['Object'], deal.field.get('Reference')
        terms, asset = schema.EXPOSURES.get(kind), self.mapping['SACCR']['Asset_Class'].get(kind)
        if terms is None or asset is None:
            raise ValueError('SACCR: trade {} is a {}, {}'.format(
                reference, kind, 'which declares no exposure CRE52 reads' if terms is None else
                'and the mapping classes it under no SACCR.Asset_Class'))
        end = self.years(max(day for *_, day in schema.settlement_days(
            fields, schema.tenor_fields(type(deal))) if day is not None))
        start = self.years(fields[terms.start]) if terms.start else 0.0
        amount = fields[terms.amount]
        side = math.copysign(1.0, amount) * (
            1.0 if terms.long is None or fields[terms.long[0]] == terms.long[1] else -1.0)
        name = terms.underlying and utils.check_tuple_name(utils.Factor(
            type(deal).factor_fields[terms.underlying][0],
            utils.check_rate_name(fields[terms.underlying])))
        bucket = ''
        if asset == 'IR':
            hedging, subclass = utils.check_rate_name(fields[terms.currency])[0], ''
            bucket = str(1 + (end >= self.buckets[0]) + (end > self.buckets[1]))
        elif asset == 'FX':
            own = utils.check_rate_name(fields[terms.currency])[0]
            pair = sorted((own, utils.check_rate_name(fields[terms.underlying])[0]))
            # long the pair as it is ordered, the other way up short
            hedging, subclass, side = '/'.join(pair), '', side * (1.0 if own == pair[0] else -1.0)
        elif asset == 'Credit':
            hedging, subclass = name, self.stated(reference, name, 'SACCR', 'Rating')
        elif asset == 'Equity':
            hedging, subclass = name, 'Index' if self.stated(
                reference, name, 'SIMM', 'Bucket') in self.EQUITY_INDICES else 'Single'
        else:
            bucket = self.stated(reference, name, 'SIMM', 'Bucket')
            hedging, subclass = self.sets[bucket]
        supervisory = self.supervisory[asset, subclass]

        size = abs(amount)
        if asset in ('IR', 'Credit'):
            steps = terms.amortisation and fields[terms.amortisation]
            notional = (self.averaged(size, steps, end) if steps else size) * self.fx(
                fields[terms.currency]) * self.duration(start, end)
        elif asset == 'FX':
            legs = [(size, fields[terms.currency])] + (
                [(size * fields[terms.strike], fields[terms.underlying])] if terms.strike else [])
            notional = max(leg * self.fx(currency) for leg, currency in legs
                           if utils.check_rate_name(currency) != utils.check_rate_name(
                               self.params['Currency']))
        else:
            price, currency = self.price(deal, terms.price or terms.underlying)
            notional = size * price * self.fx(currency)

        delta = side
        if terms.strike:
            expiry, vol = self.years(fields[terms.expiry]), supervisory['option_volatility']
            shift = self.shifts.get(hedging, 0.0) if asset == 'IR' else 0.0
            d1 = (math.log((self.FORWARDS[asset](self, deal, terms) + shift) / (
                fields[terms.strike] + shift)) + 0.5 * vol * vol * expiry) / (vol * math.sqrt(expiry))
            delta = side * (scipy.stats.norm.cdf(d1) if fields[terms.call[0]] == terms.call[1]
                            else -scipy.stats.norm.cdf(-d1))
        unmargined = math.sqrt(min(max(end, self.horizon), 1.0))
        factor = unmargined if mpor is None else self.margined_scale * math.sqrt(mpor / self.year)
        return {'TradeID': reference, 'NettingSet': agreement['Instrument'].field.get('Reference'),
                'AssetClass': asset, 'HedgingSet': hedging, 'Subclass': subclass, 'Bucket': bucket,
                'SupervisoryFactor': supervisory['factor'], 'Correlation': supervisory['correlation'],
                'AdjustedNotional': notional, 'Delta': delta, 'MaturityFactor': factor,
                'EffectiveNotional': notional * delta * factor,
                'Unmargined': notional * delta * unmargined}

    def add_ons(self, rows, column):
        """`[add-on]` per asset class over a set's trade `rows`, each effective notional its
        `column`: IR by currency over its three maturity buckets (52.57), FX by pair (52.59),
        credit and equity by entity (52.61, 52.66), commodity by type within its hedging set
        (52.70)."""
        found = []
        for asset in self.SUBCLASSES:
            summed = defaultdict(float)
            for row in rows:
                if row['AssetClass'] == asset:
                    summed[row['HedgingSet'], row['Bucket'], row['SupervisoryFactor'],
                           row['Correlation']] += row[column]
            if asset == 'IR':
                buckets = defaultdict(lambda: np.zeros(3))
                for (currency, bucket, factor, _), notional in summed.items():
                    buckets[currency, factor][int(bucket) - 1] += notional
                found.append(sum(factor * math.sqrt(notional @ self.correlation @ notional)
                                 for (_, factor), notional in buckets.items()))
            elif asset == 'FX':
                found.append(sum(factor * abs(notional) for (_, _, factor, _), notional in
                                 summed.items()))
            else:
                # one single-factor sum per set: credit's and equity's one, commodity's four
                sets = defaultdict(list)
                for (hedging, _, factor, rho), notional in summed.items():
                    sets[hedging if asset == 'Commodity' else ''].append((factor * notional, rho))
                found.append(sum(math.sqrt(sum(rho * add_on for add_on, rho in entities) ** 2 + sum(
                    (1.0 - rho * rho) * add_on * add_on for add_on, rho in entities))
                    for entities in sets.values()))
        return found

    def potential(self, v, c, add_on):
        """`(multiplier, PFE)` (52.20-52.23) of a set at `v` less `c` over its aggregate `add_on`:
        the multiplier one where `v - c` is not negative, falling to its floor as it falls."""
        excess = min(v - c, 0.0)
        decay = math.exp(excess / (2.0 * (1.0 - self.floor) * add_on)) if add_on else float(
            excess == 0.0)
        multiplier = self.floor + (1.0 - self.floor) * decay
        return multiplier, multiplier * add_on

    def netting_set(self, agreement, indices, base):
        """The set's row of the netting sets frame and its valued trades' rows: RC (52.10, 52.18),
        PFE (52.20) and EAD (52.1), a margined set's capped at its unmargined one (52.2)."""
        terms = agreement['Instrument'].field
        margined = terms['Collateralized'] == 'True'
        mpor = max(terms['Liquidation_Period'], self.floors[1] if len(indices) > self.floors[2]
                   else self.floors[0]) if margined else None
        rows = [dict(self.exposure(index, mpor), Value=base[index][0]) for index in indices
                if base[index][0] is not None]
        v = sum(row['Value'] for row in rows)
        c, margin = self.collateral(terms) if margined else (0.0, 0.0)
        add_ons = self.add_ons(rows, 'EffectiveNotional')
        multiplier, pfe = self.potential(v, c, sum(add_ons))
        unmargined = max(v - c, 0.0)
        rc = max(unmargined, margin) if margined else unmargined
        ead = self.alpha * (rc + pfe)
        if margined:
            ead = min(ead, self.alpha * (unmargined + self.potential(v, c, sum(self.add_ons(
                rows, 'Unmargined')))[1]))
        skipped = ['{}: {}'.format(self.trades[index][1]['Instrument'].field.get('Reference'), said)
                   for index in indices for said in base[index][1]]
        return dict(zip(self.SETS, (
            terms.get('Reference'), margined, mpor, v, c, rc, *add_ons, sum(add_ons), multiplier,
            pfe, ead, '; '.join(skipped)))), rows

    def collateral(self, terms):
        """`(C, TH + MTA - NICA)` of a margined set at the base date in the calculation currency:
        its opening balance as the engine's balance counts it, held at one less its received
        haircut and posted at face over one less its posted one (`instruments.collateral_credit`),
        and its credit support amounts at their first values, crossed from its agreement's
        currency."""
        support, agreement = terms['Credit_Support_Amounts'], self.fx(terms['Agreement_Currency'])
        margin = agreement * (support['Received_Threshold'].value() + support[
            'Minimum_Received'].value() - support['Independent_Amount'].value())
        assets = terms['Collateral_Assets']
        if assets.get('Bond_Collateral') or assets.get('Equity_Collateral'):
            raise ValueError('SACCR: netting set {} holds bond or equity collateral, whose value at '
                             'the base date the calculation does not price'.format(
                                 terms.get('Reference')))
        balance_currency = terms.read('Balance_Currency', terms['Agreement_Currency'])
        rows = assets.get('Cash_Collateral') or [{'Currency': balance_currency}]
        worth = [(1.0 if row.get('Amount') is None else row['Amount']) * self.fx(row['Currency'])
                 for row in rows]
        cuts = [instruments.NettingCollateralSet.haircuts(row) for row in rows]
        face = sum(worth)
        posted, held = (sum(unit * cut[key] for unit, cut in zip(worth, cuts))
                        for key in ('Haircut_Posted', 'Haircut_Received'))
        opening = terms['Opening_Balance'] * self.fx(balance_currency)
        balance = opening / (face - posted * (opening < 0))
        return instruments.collateral_credit(balance, face, held), margin

    def report(self, base):
        """The run's results off the book's `base`: a row per netting set and per trade it
        values, in book order."""
        sets, trades = [], []
        for agreement in self.config.deals['Deals']['Children']:
            row, rows = self.netting_set(agreement, [
                index for index, (held, _) in enumerate(self.trades) if held is agreement], base)
            sets.append(row)
            trades.extend(rows)
        self.output = {'SACCR': {'netting_sets': pd.DataFrame(sets, columns=list(self.SETS)),
                                 'trades': pd.DataFrame(trades, columns=list(self.TRADES))}}
        return {'Stats': self.calc_stats, 'Results': self.output}


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


def worker_count(runparallel):
    """How many workers `runparallel` asks for - the knob is that argument, and there is no second.

    `True` keeps the historical meaning, one worker per visible CUDA device, but falls back to a
    SINGLE worker where there is no CUDA rather than to zero: `torch.cuda.device_count()` returns 0
    on a CPU box, and zero workers would leave the parent waiting on a queue nothing ever writes to.
    An int asks for exactly that many and is deliberately NOT capped by the device count - workers
    share devices through `job_id % device_count`, so a four-way shard runs on a two-device box and
    an n-way shard runs on a box with no device at all.

    `isinstance(True, int)` is True in Python, so the bool is matched by identity first.
    """
    if runparallel is True:
        return torch.cuda.device_count() or 1
    count = int(runparallel)
    if count < 1:
        raise ValueError(
            'runparallel is a worker count: True for one per device, or an int >= 1, got '
            '{!r}'.format(runparallel))
    return count


def shielded(target, job, queue, *args, **kwargs):
    """A worker's body: `target`, which answers on `queue` itself, and on a raise the error put
    there as worker `job`'s answer instead - so the parent re-raises it rather than waiting."""
    try:
        target(*args, **kwargs)
    except Exception as error:
        try:
            pickle.loads(pickle.dumps(error))
        except Exception:
            # an error that cannot cross the process boundary crosses as its own traceback
            error = RuntimeError(traceback.format_exc())
        queue.put({'Job': job, 'Error': error})


def gathered(workers, queue):
    """Start `workers` and take one answer each off `queue`, returned in worker order though the
    queue hands them back in completion order: a worker's error is raised here with the others
    stopped, and a worker dead without answering is named rather than waited on."""
    answers = {}
    for worker in workers:
        worker.start()
    try:
        while len(answers) < len(workers):
            try:
                answer = queue.get(timeout=1.0)
            except Empty:
                silent = ['{} (exit {})'.format(job, worker.exitcode)
                          for job, worker in enumerate(workers)
                          if worker.exitcode not in (None, 0) and job not in answers]
                if silent:
                    raise RuntimeError('worker {} died without answering'.format(
                        ', '.join(silent)))
                continue
            if 'Error' in answer:
                raise answer['Error']
            answers[answer['Job']] = answer
    finally:
        for worker in workers:
            if len(answers) < len(workers) and worker.is_alive():
                worker.terminate()
            worker.join()
    return [answers[job] for job in range(len(workers))]


def calculation_class(name):
    """The calculation a job's `Object` names - the class of that name the Calculation store
    publishes - refused by name where it publishes none."""
    types = schema.mapping['Calculation']['types']
    if name not in types:
        raise ValueError('{!r} is no calculation - this engine runs {}'.format(
            name, ', '.join(sorted(types))))
    return globals()[name]


if __name__ == '__main__':
    pass
