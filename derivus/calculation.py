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


# import standard libraries

import time
import logging
import itertools
import pandas as pd
import numpy as np
import torch
from functools import reduce

# load up some useful data types
from collections import namedtuple, defaultdict
# import the risk factors (also known as price factors)
from .riskfactors import construct_factor
# import the stochastic processes
from .stochasticprocess import construct_process
# import the currency/curve lookup factors 
from .instruments import get_fxrate_factor, get_survival_component, get_interest_factor, get_survival_factor
# import the hessian function
from .pricing import SensitivitiesEstimator
# import the documentation and utils modules
from . import utils, pricing
from .schema import F, REQUIRED, Row, declared_defaults
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
        # Honour store_results=False (Calc_res is None) exactly as pricing.interpolate does.
        if deal_data.Calc_res is not None:
            shared.save_results(deal_data.Calc_res, {'Value': accum})
        return accum

class DealStructure(object):
    def __init__(self, obj, store_results=False):
        self.obj = utils.DealDataType(
            Instrument=obj, Factor_dep={}, Time_dep=None, Calc_res={} if store_results else None)
        # gather a list of deal dependencies
        self.dependencies = []
        # maintain a list of container objects
        self.sub_structures = []
        # Do we want to store each deal level MTM explicitly?
        self.store_results = store_results

    @staticmethod
    def calc_time_dependency(base_date, deal, time_grid):
        # calculate the additional (dynamic) dates that this instrument needs to be evaluated at
        deal_time_dep = None
        try:
            reval_dates = deal.get_reval_dates(clip_expiry=True)
            if len(time_grid.scenario_dates) == 1:
                if len(reval_dates) > 0 and max(reval_dates) < base_date:
                    raise utils.InstrumentExpired(deal.field.get('Reference', 'Unknown Instrument Reference'))
                deal_time_dep = time_grid.calc_deal_grid({base_date})
            else:
                deal_time_dep = time_grid.calc_deal_grid(reval_dates)
        except utils.InstrumentExpired as e:
            logging.warning('skipping expired deal {0}'.format(e.args))

        return deal_time_dep

    def add_deal_to_structure(self, base_date, deal, static_offsets, stochastic_offsets,
                              all_factors, all_tenors, time_grid, calendars, stats, unit):
        """
        The logic is as follows: a structure contains deals - a set of deals are netted off and then the rules that the
        structure itself contains is applied.
        """
        deal_time_dep = self.calc_time_dependency(base_date, deal, time_grid)
        if deal_time_dep is not None:
            # calculate dependencies based on field names
            try:
                self.dependencies.append(
                    utils.DealDataType(Instrument=deal,
                                       Factor_dep=utils.bind_schedules(deal.calc_dependencies(
                                           base_date, static_offsets, stochastic_offsets,
                                           all_factors, all_tenors, time_grid, calendars), unit),
                                       Time_dep=deal_time_dep,
                                       Calc_res={} if self.store_results else None))
                stats['Deals loaded'] = stats.setdefault('Deals loaded', 0) + 1
            except Exception as e:
                logging.error('{0} {1} - Skipped'.format(deal.field['Object'], e.args))
                stats['Deals Skipped'] = stats.setdefault('Deals Skipped', 0) + 1

    def finalize_struct(self, base_date, time_grid):
        all_report_dates = [set(
            x.obj.Instrument.get_report_dates(time_grid, base_date)) for x in self.sub_structures]
        self.obj.Instrument.set_report_dates(
            reduce(set.union, all_report_dates) if all_report_dates else time_grid.mtm_dates)
        # copy across the reporting dates to the time_grid
        time_grid.set_report_dates(base_date, self.obj.Instrument.get_report_dates())

    def add_structure_to_structure(self, struct, base_date, static_offsets, stochastic_offsets,
                                   all_factors, all_tenors, time_grid, calendars, stats, unit):
        # get the dependencies
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
                # Structure object representing a netted set of cashflows
                self.sub_structures.append(struct)
                stats['Structs loaded'] = stats.setdefault('Structs loaded', 0) + 1
            except Exception as e:
                logging.error('{0} {1} - Skipped'.format(struct.obj.Instrument.field['Object'], e.args))
                stats['Structs Skipped'] = stats.setdefault('Structs Skipped', 0) + 1

    def resolve_structure(self, shared, time_grid):
        """
        Resolves the Structure
        """

        accum = 0.0 * shared.one
        # Anything registered from here on is a decision taken BENEATH this structure, so the tail
        # past this mark is exactly what its post_process gets to speak for. It cannot be found
        # any later: post_process runs only once the children have been priced.
        mark = len(shared.boundary_sets) if getattr(shared, 'boundary_aad', False) else None

        if self.sub_structures:
            # process sub structures
            for structure in self.sub_structures:
                logging.root.name = structure.obj.Instrument.field.get('Reference', 'root')
                # reset cashflows if this structure accumulates its dependencies (e.g. netting sets)
                if structure.obj.Instrument.accum_dependencies and hasattr(shared, 'reset_cashflows'):
                    shared.reset_cashflows(time_grid)
                struct = structure.resolve_structure(shared, time_grid)
                if (struct != struct).any():
                    logging.critical('Netting set contains NANS! Please Investigate! - skipping for now')
                    continue
                if structure.obj.Instrument.accum_dependencies and hasattr(shared, 'save_cashflows'):
                    shared.save_cashflows(structure.obj.Calc_res, time_grid)
                if structure.obj.Instrument.field.get('Reference', 'root').startswith('FLIP'):
                    logging.warning('Netting set starts with FLIP - inverting MTM')
                    struct = -struct
                accum = accum + struct

        if self.dependencies and self.obj.Instrument.accum_dependencies:
            # accumulate the mtm's
            deal_tensors = 0.0

            for deal_data in self.dependencies:
                logging.root.name = deal_data.Instrument.field.get('Reference', 'root')
                deal_mark = len(shared.boundary_sets) if mark is not None else None
                mtm = deal_data.Instrument.calculate(shared, time_grid, deal_data)
                if deal_mark is not None:
                    utils.stamp_boundary_sets(shared, deal_mark, logging.root.name)
                deal_tensors = deal_tensors + mtm

            accum = accum + deal_tensors

        # postprocessing code for working out the mtm of all deals, collateralization etc..
        if hasattr(self.obj.Instrument, 'post_process'):
            # the actual answer for this netting set
            logging.root.name = self.obj.Instrument.field.get('Reference', 'root')
            try:
                accum = self.obj.Instrument.post_process(accum, shared, time_grid, self.obj, self.dependencies)
            except RuntimeError as e:
                logging.error('Runtime error Deal skipped - {}'.format(e.args))
                raise
            except Exception as e:
                logging.critical('Deal skipped - {}'.format(e.args))
            finally:
                if mark is not None:
                    utils.claim_boundary_sets(shared, mark)
                    # whatever post_process itself registered (a margin call) has no deal to
                    # name it, so the structure does
                    utils.stamp_boundary_sets(shared, mark, logging.root.name)

        return accum

    def resolve_hedge_structure(self, shared, time_grid):
        """Accumulate the liability MTM across the structure and return `{'mtm': tensor}`.
        Unlike `resolve_structure` this skips `post_process`/`save_results` — i.e. no per-batch
        GPU->CPU copy of the mark — which is why the hedge sim loop and the inner MC use it for
        the liability. (Deal feature tensors were removed; the symlog utility-scale's two static
        descriptors now come from the cashflow schedule via `_liability_schedule_scalars`.)"""
        def merge_features(cumulative, new_features):
            new_mtm = new_features.get('mtm')
            if new_mtm is not None:
                cumulative['mtm'] = new_mtm if cumulative.get('mtm') is None else cumulative['mtm'] + new_mtm

        accum = {}

        if self.sub_structures:
            # process sub structures
            for structure in self.sub_structures:
                logging.root.name = structure.obj.Instrument.field.get('Reference', 'root')
                features = structure.resolve_hedge_structure(shared, time_grid)
                merge_features(accum, features)


        if self.dependencies and self.obj.Instrument.accum_dependencies:
            # accumulate the mtm's
            deal_features = {}

            for deal_data in self.dependencies:
                logging.root.name = deal_data.Instrument.field.get('Reference', 'root')
                features = deal_data.Instrument.build_features(shared, time_grid, deal_data)
                merge_features(deal_features, features)

            merge_features(accum, deal_features)

        return accum

    def aggregate_leg_descriptors(self):
        """Reduce the per-deal cashflow descriptors over this structure (recursing
        sub_structures the way `resolve_hedge_structure` does): summed |notional| across
        all legs and the latest pay-day. Legs with no schedule contribute (0.0, None)."""
        total_volume, last_payment_day = 0.0, None
        for vol, pay in ([dd.Instrument.leg_descriptors(dd) for dd in self.dependencies] +
                         [sub.aggregate_leg_descriptors() for sub in self.sub_structures]):
            total_volume += vol
            if pay is not None:
                last_payment_day = pay if last_payment_day is None else max(last_payment_day, pay)
        return total_volume, last_payment_day

    def tensor_marks(self):
        """Stored per-deal price series keyed by deal Reference, recursing sub_structures
        (mirrors `resolve_structure`'s walk). Only deals whose `Calc_res` holds a kept
        'tensor' (set by `pricing.interpolate` when `shared.keep_tensor`) are included."""
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
        """Latest clipped reval/settlement date across a set of (un-built) deal nodes — the
        liability-terminal horizon that caps the sim time grid. Resets each instrument from
        its `field` first (idempotent; `set_deal_structures` resets again downstream), since
        the structure isn't built yet when the horizon is needed."""
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
        self.time_grid_years = self.scen_time_grid / utils.DAYS_IN_YEAR
        self.scenario_grid = global_time_grid.scenario_grid[:offset]


class Calculation(object):

    def __init__(self, config, prec=torch.float32, device=torch.device('cpu')):
        """
        Construct a new calculation - all calculations must set up their own tensors.
        """

        self.config = config
        self.dtype = prec
        self.time_grid = None
        self.device = device

        # the risk factor data
        self.static_factors = {}
        self.static_var = {}
        self.stoch_factors = {}
        self.stoch_var = {}
        self.all_factors = {}
        self.all_tenors = {}

        self.base_date = None
        self.tenor_size = None
        self.tenor_offset = None

        # the deal structure
        self.netting_sets = None

        # performance and admin feedback
        self.calc_stats = {}
        # the calculation parameters (defined by calling execute)
        self.params = {}
        # Index for the gradients of this calculation (if requested)
        self.gradient_index = None
        # output of calc stored here
        self.output = {}

    def execute(self, params):
        pass

    def factor_leaf(self, factor, current_val, requires_grad, offset=0.0):
        """The AAD leaf for one factor - and the ONE seam a calibration upstream of it can reach.

        Every leaf the engine mints is `torch.tensor(factor.current_value(...))`, a fresh tensor
        built out of a numpy array, so anything that produced those numbers is severed by
        construction: it does not raise, it reports a zero gradient. A curve - or one named
        parameter of a calibrated model - the library BOOTSTRAPPED and kept the graph of is offered
        here instead, as

            leaf + (theta - theta.detach())

        which is the boundary correction's shape and is here for exactly its reason: worth zero in
        the forward pass, derivative one, so what reaches `backward()` changes and what is reported
        cannot. `leaf` stays a leaf, so the tensor is still the one the pricers read and
        `retain_grad` keeps `.grad` populated - the factor greek reported for this curve is the
        same number it always was, and dV/dq arrives in the same pass.

        A non-zero `Tenor_Offset` declines the attachment: the curve the calculation consumes is
        then a SHIFTED one, and dtheta_shifted/dq is not dtheta/dq. Quote sensitivities are t0
        risk; a tenor offset is a different curve, so it gets the leaf it always got.

        THE SAME SEAM CARRIES A PROPAGATED CALIBRATION, and that one changes the VALUE. A block
        asking for `Quote_Propagation` rides its last artifact to the quotes standing now -
        `theta* + dtheta/dq (q_now - q0)` - so the leaf is minted out of the ridden nodes instead
        of the ones the last bootstrap wrote. That is the point: a tick reaches a valuation without
        a re-solve. It is derived here rather than stored, so nothing about it survives the call.

        A `Tenor_Offset` REFUSES the ride rather than declining it. `current_value(offset)`
        interpolates off coefficients fitted on the numpy rate column at construction, so the
        shifted curve cannot be ridden without refitting them - and declining would silently price
        the STALE curve, which is a wrong number rather than a missing derivative.

        Every ride LEAVES ITS ARTIFACT'S ID IN `calc_stats`, so the run reports which calibration
        produced the curve it priced. Nothing else in the replay tuple can say: a ride and a refit
        of the same plan at the same quotes agree on `plan_hash`, `values_hash`, the version and
        the seed, and differ only in which artifact was in the store.
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

    def make_factor_index(self, tensors):
        # need to match the indices back
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
            # get the factor values from all_factors if necessary
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
                # store the actual factor value if required
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
            # sort both axis
            df = pd.DataFrame(values, index=multi_index[0], columns=multi_index[1]).sort_index(
                level=[0, 1, 2, 3], axis=0).sort_index(level=[0, 1, 2, 3], axis=1)

        return df

    def set_deal_structures(self, deals, output, unit, deal_level_mtm=False):
        """Compile the deal tree. `unit` is the calculation's dtype/device anchor: a deal's
        schedules are BOUND to it as they compile, so the tensor half's birthday is this walk."""
        for node in deals:
            # get the instrument
            instrument = node['Instrument']
            # should we skip it?
            if node.get('Ignore') == 'True':
                self.calc_stats['Ignored'] = self.calc_stats.setdefault('Ignored', 0) + 1
                continue

            # logging info
            logging.root.name = instrument.field.get('Reference', '<undefined>')
            if node.get('Children'):
                struct = DealStructure(instrument, store_results=deal_level_mtm)
                self.set_deal_structures(node['Children'], struct, unit, deal_level_mtm)
                output.add_structure_to_structure(
                    struct, self.base_date, self.static_factors, self.stoch_factors, self.all_factors,
                    self.all_tenors, self.time_grid, self.config.holidays, self.calc_stats, unit)
                continue

            output.add_deal_to_structure(
                self.base_date, instrument, self.static_factors, self.stoch_factors, self.all_factors,
                self.all_tenors, self.time_grid, self.config.holidays, self.calc_stats, unit)


class CMC_State(utils.Calculation_State):
    def __init__(self, cholesky, static_buffer, batch_size, one, mcmc_sims, report_currency,
                 seed, job_id, num_jobs, scale_survival=False, nomodel='Constant', keep_tensor=False):
        """Per-calculation Monte Carlo state: correlated random numbers, scenario buffers and the
        caches a batched exposure run needs on top of `Calculation_State`.

        The `t_PreCalc` memo is deliberately NOT on `Calculation_State`: `t_Buffer` is the
        per-batch eval cache that `reset` clears, and this is the per-CALCULATION one — which only
        earns its keep where a calculation spans many batches. Its presence is therefore the marker
        for "exposure-based", and pricers read it that way rather than inventing a switch.

        `t_Bridge_Variance_Rate` holds the per-factor annualized log-variance RATE, published once
        the processes are precalculated. A barrier is monitored continuously while a deal grid only
        observes its own dates, so a crossing in between is a conditional probability needing the
        variance of the interval it spans - and it must be the SIMULATION variance, not a pricing
        implied vol.
        """
        super(CMC_State, self).__init__(
            static_buffer, one, mcmc_sims, report_currency, nomodel, batch_size, keep_tensor=keep_tensor)
        # these are tensors
        # per-CALCULATION memo (vs the per-batch t_Buffer); its presence marks "exposure-based"
        self.t_PreCalc = {}
        # Discontinuous decisions recorded during the forward pass so their derivative can be
        # restored before the single reverse sweep. Per BATCH, like t_Buffer — backward() runs
        # once per batch, so a correction assembled from a previous batch's graph is stale.
        self.boundary_aad = False
        self.boundary_sets = []
        # per-factor annualized log-variance RATE, published once the processes are precalculated
        self.t_Bridge_Variance_Rate = {}
        self.t_cholesky = cholesky
        self.t_random_numbers = None
        self.t_Scenario_Buffer = {}
        # these are shared parameter states
        self.sobol = {}
        # idea is to reuse quasi rng numbers where applicable (but still using enough randomness)
        self.t_quasi_rng = {}
        self.t_quasi_rng_batch = {}
        # set the random seed - seed each job by its offset
        torch.manual_seed(seed + job_id)
        # needed if we are running across multiple gpu's
        self.job_id = job_id
        self.num_jobs = num_jobs
        # do we need to scale the mtm by the survival probability in the final answer?
        self.scale_survival = scale_survival

    def quasi_rng(self, dimension, sample_size):
        # may need to parameterize these
        seed = 1234
        fast_forward = 1024

        if dimension not in self.sobol:
            self.sobol[dimension] = torch.quasirandom.SobolEngine(dimension=dimension, scramble=True, seed=seed)
            # skip this many samples
            self.sobol[dimension].fast_forward(fast_forward)

        # hash the tensor
        batch_key = (dimension, sample_size)
        batch_num = self.t_quasi_rng_batch.setdefault(batch_key, 0)
        sample_key = (dimension, sample_size, batch_num)

        if sample_key not in self.t_quasi_rng:
            sample_sobol = self.sobol[dimension].draw(sample_size, dtype=self.one.dtype)
            margin = 1.0e-6
            u = sample_sobol.clamp(min=margin, max=1.0 - margin).to(self.one.device)
            self.t_quasi_rng[sample_key] = (utils.norm_icdf(u), u)

        # update the batch key
        self.t_quasi_rng_batch[batch_key] += 1
        # return the cached batch of quasi random numbers
        return self.t_quasi_rng[sample_key]

    def reset_qrg(self):
        self.t_quasi_rng_batch = {}
        
    def reset_cashflows(self, time_grid):
        # reset the cashflows
        self.t_Cashflows = {k: {t_i: self.one.new_zeros(self.simulation_batch)
                                for t_i in np.where(v >= 0)[0]} for k, v in time_grid.CurrencyMap.items()}

    def save_cashflows(self, output, time_grid):
        dates = np.array(sorted(time_grid.mtm_dates))
        for currency, values in self.t_Cashflows.items():
            cash_index = dates[sorted(values.keys())]
            output.setdefault('cashflows', {}).setdefault(currency, []).append(
                pd.DataFrame(
                    [v.cpu().detach().numpy() for _, v in sorted(values.items())], index=cash_index))

    @staticmethod
    def save_results(output, tensors):
        for k, v in tensors.items():
            output.setdefault(k, []).append(v.detach().cpu().numpy().astype(np.float64))

    def reset(self, num_factors, time_grid: utils.TimeGrid, use_antithetic=False):
        # update the random numbers
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

        # clear the buffers
        self.t_Buffer.clear()
        self.boundary_sets.clear()


class CMC_State_Inner(CMC_State):
    """Inner-MC variant of CMC_State for nested simulation. Each of `simulation_batch`
    outer-path states fans out into `simulation_sub_batch` (B2) independent forward
    sample paths. Base `reset()` is inherited unchanged so outer-mode usage of this
    state object is transparent. `reset_inner()` swaps in the inner-shape random
    numbers: `(num_factors, T, simulation_batch, simulation_sub_batch)` instead of
    the base `(num_factors, T, simulation_batch)`. Stochastic processes dispatch on
    `Z.ndim` to pick between outer and inner code paths. `use_antithetic=True`
    (`Inner_Antithetic='Yes'`) mirrors the Sobol draws as (z, -z) pairs on the inner
    axis. quasi_rng is inherited — callers handle any reshape."""

    def __init__(self, cholesky, static_buffer, batch_size, one, mcmc_sims, report_currency,
                 seed, job_id, num_jobs, simulation_sub_batch=0,
                 scale_survival=False, nomodel='Constant', keep_tensor=False):
        super().__init__(cholesky, static_buffer, batch_size, one, mcmc_sims, report_currency,
                         seed, job_id, num_jobs, scale_survival=scale_survival,
                         nomodel=nomodel, keep_tensor=keep_tensor)
        # 0 (default) = inner mode unused; base `reset()` works, `reset_inner()` raises.
        self.simulation_sub_batch = simulation_sub_batch

    def reset_inner(self, num_factors, time_grid: utils.TimeGrid, use_antithetic=False,
                    use_random=False):
        """Draw the inner-mode correlated Gaussians, shaped
        `(num_factors, T, simulation_batch, simulation_sub_batch)`, and clear the per-batch caches.

        `use_random` (`Inner_Draws='random'`) swaps the shared Sobol tensor for plain iid
        Gaussians. One low-discrepancy stream strided across (T,B,B2) loses its uniformity
        guarantees on the per-(t,b) B2-slices as B grows - the measured B=512 label/argmax
        degradation. iid draws have no cross-(B,B2) coupling, so per-fork label noise is
        B-independent.

        `use_antithetic` (`Inner_Antithetic='Yes'`) draws B2/2 quasi-normals per (t, outer-path)
        and mirrors them (z, -z) on the inner axis. This halves the label/argmax variance of the
        inner-MC E[C] estimate - the diff-ML winner's-curse lever validated in the toy - and stays
        unbiased because the emissions are symmetric in z. Only the symmetric emissions are folded:
        auxiliary streams (e.g. a discrete-state transition) draw from a separate quasi_rng stream
        and stay iid.
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
            # mirror B2/2 Sobol quasi-normals on the inner axis; auxiliary streams stay iid
            Z_normal, _ = self.quasi_rng(num_factors, T * B * (B2 // 2))
            half = torch.matmul(
                self.t_cholesky, Z_normal.transpose(0, 1)
            ).reshape(num_factors, T, B, B2 // 2)
            self.t_random_numbers = torch.cat([half, -half], dim=-1)
        else:
            Z_normal, _ = self.quasi_rng(num_factors, T * B * B2)                    # (T*B*B2, num_factors)
            self.t_random_numbers = torch.matmul(
                self.t_cholesky, Z_normal.transpose(0, 1)
            ).reshape(num_factors, T, B, B2)

        self.reset_cashflows(time_grid)
        self.t_Buffer.clear()


class Credit_Monte_Carlo(Calculation):
    documentation = ('Calculations', [
        'A profile is a curve $V(t)$ with values specified at a discrete set of future dates $0=t_0<t_1<...<t_m$ with',
        'values at other dates  obtained via linear interpolation or zero extrapolation i.e. if $t_{i-1}<t<t_i$ then',
        '$V(t)$ is a linear interpolation of $V(t_{i-1})$ and $V(t_i)$; otherwise $V(t)=0$.',
        '',
        'The valuation models described earlier are used to construct the profile. The profile dates $t_1,...,t_m$ are',
        'obtained by taking the following union:',
        '',
        '- The deal\'s maturity date.',
        '- The dates in the **Time Grid** up to the deal\'s maturity date.',
        '- Deal specific dates such as payment and exercise dates.',
        '',
        'Deal specific dates improve the accuracy of the profile by showing the effect of cashflows, exercises etc.',
        '',
        '### Aggregation',
        '',
        'If $U$ and $V$ are profiles, then the set $U+V$ is the union of profile dates $U$ and $V$. If $E$ is the',
        'credit exposure profile in reporting',
        'currency (**Currency**), then:',
        '',
        '$$E = \\sum_{d} V_d$$',
        '',
        'where $V_d$ is the valuation profile of the $d^{th}$ deal. Note that Netting is always assumed to be',
        '**True**.',
        '',
        '#### Peak Exposure',
        '',
        'This is the simulated exposure at **Percentile** $q$ where $0<q<1$ (typically q=.95 or .99).',
        '',
        '#### Expected Exposure',
        '',
        'This is the profile defined by taking the average of the positive simulated exposures i.e. for each profile',
        'date $t$,',
        '',
        '$$\\bar E(t)=\\frac{1}{N}\\sum_{k=1}^N \\max(E(t)(k),0).$$'
        '',
        '#### Exposure Deflation',
        '',
        'Exposure at time $t$ is simulated in units of the time $t$ reporting currency. Exposure deflation converts',
        'this to time $0$ reporting currency i.e.',
        '',
        '$$V^*(t)=\\frac{V(t)}{\\beta(t)}$$',
        '',
        'where',
        '',
        '$$\\beta(t)=\\exp\\Big(\\int_0^t r(s)ds\\Big).$$',
        '',
        'This can be approximated by:',
        '',
        '$$\\beta(t)=\\prod_{i=0}^n\\frac{1}{D(s_{i-1},s_i)}$$',
        '',
        'where $0=s_0<...<s_n=t$. The discrete set of dates $s_1,...,s_{n-1}$ are model-dependent.'
        '',
        '### Credit Valuation Adjustment',
        '',
        'This represents the adjustment to the market value of the portfolio accounting for the risk of default. Only',
        'unilateral CVA (i.e accounting for the counterparty risk of default but ignoring the potential default of',
        'the investor) is calculated. It is given by:',
        '',
        '$$C=\\Bbb E(L(\\tau)),$$',
        '',
        'where the expectation is taken with respect to the risk-neutral measure, and ',
        '',
        '$$L(t)=(1-R)\\max(E^*(t),0),$$',
        '',
        'with:',
        '',
        '- $R$ the counterparty recovery rate',
        '- $\\tau$ the counterparty time to default',
        '- $E^*(t)$ the exposure at time $t$ deflated by the money market account.',
        '',
        'If **Deflate Stochastically** is **No** then the deflated expected exposure is assumed to be deterministic',
        'i.e. $E^*(t)=E(t)D(0,t)$. Note that if $T$ is the end date of the portfolio exposure then $E^*(t)=0$ for',
        '$t>T$.',
        '',
        'Now,',
        '',
        '$$\\Bbb E(L(\\tau))=\\Bbb E\\Big(\\int_0^T L(t)(-dH(t))\\Big)$$',
        '',
        '$$H(t)=\\exp\\Big(-\\int_0^t h(u)du\\Big)$$',
        '',
        'where $h(t)$ is the stochastic hazard rate. There are two ways to calculate the expectation:',
        '',
        'If **Stochastic Hazard** is **No** then $H(t)=\\Bbb P(\\tau > t)=S(0,t)$, the risk neutral survival',
        'probability to time $t$ and',
        '',
        '$$C=\\int_0^T \\Bbb E(L(t))(-dH(t))\\approx\\sum_{i=1}^m C_i,$$',
        '',
        'with',
        '',
        '$$C_i=\\Big(\\frac{\\Bbb E(L(t_{i-1}))+\\Bbb E(L(t_i))}{2}\\Big)(S(0,t_{i-1})-S(0,t_i))$$',
        '',
        'and $0=t_0<...<t_m=T$ are the time points on the exposure profile. Note that the factor models used should',
        'be risk-neutral to give risk neutral simulations of $\\Bbb E^*(t)$.',
        '',
        'If **Stochastic Hazard** is **Yes** then $S(t,u)$ is the simulated survival probability at time $t$ for',
        'maturity $u$ and is related to $H$ by',
        '',
        '$$S(t,u)=\\Bbb E(\\frac{H(u)}{H(t)}\\vert\\mathcal F(t)).$$',
        '',
        'where $\\mathcal F$ is the filtration given by the risk factor processes. For small $u-t$, the approximation',
        '$H(u)\\approx H(t)S(t,u)$ is accurate so that',
        '',
        '$$C\\approx\\sum_{i=1}^m C_i$$',
        '',
        'and',
        '',
        '$$C_i=\\Bbb E\\Big[\\Big(\\frac{L(t_{i-1}))+L(t_i)}{2}\\Big)(H_{i-1}-H_i)\\Big],$$',
        '',
        'again, $0=t_0<...<t_m=T$ are the time points on the exposure profile and',
        '',
        '$$H_i=S(0,t_1)S(t_1,t_2)...S(t_{i-1},t_i).$$',
        '',
        '### Funding Valuation Adjustment',
        '',
        'Not Posting (or receiving) collateral can imply a funding cost (or benefit) when there is a spread between a',
        'party\'s interal cost of funding and the rate that would be recieved should the counterparty place collateral.'
        'The discounted expectation of this cost (or benefit) summed across all time horizons and scenarios constitutes',
        'a funding value adjustment and can be expressed as:',
        '',
        '$$FCA=\\int_{0}^{T} \\Bbb{E}\\Big(max(V(t),0)[f_{fc,C}(t)-f_{rf,C}(t)]SP_C(t)\\Big)dt$$',
        '',
        '$$FBA=\\int_{0}^{T} \\Bbb{E}\\Big(min(V(t),0)[f_{fb,C}(t)-f_{rf,C}(t)]SP_C(t)\\Big)dt$$',
        '',
        'where',
        '',
        '- $T$ is the exposure horizon',
        '- $V(t)$ is the deflated funding profile at time $t$',
        '- $f_{fc,C}(t)$ and $f_{fb,C}(t)$ is the funding cost and benefit spreads respectively',
        '- $f_{rf,C}(t)$ is the risk-free rate',
        '- $SP_C(t)$ is the survival probability of the Counterparty.',
        '',
        'FVA against the counterparty is then calculated as $FVA = FCA + FBA$',
        '',
        'At the bank wide level the $FCA$ and $FBA$ is calculated as:',
        '',
        '$$FCA_{bank}=\\int_{0}^{T} \\Bbb{E}\\Bigg(max\\Big(\\sum_i V(t)SP_{Ci},0\\Big)[f_{fc,C}(t)-f_{rf,C}(t)]\\Bigg)dt$$',
        '',
        '$$FBA_{bank}=\\int_{0}^{T} \\Bbb{E}\\Bigg(min\\Big(\\sum_i V(t)SP_{Ci},0\\Big)[f_{fb,C}(t)-f_{rf,C}(t)]\\Bigg)dt$$',
        '',
        'The idea is the same as defined above except that $i$, the counterparty index, sums over all uncollateralized ',
        'or partially collateralized counterparties (one-way CSA or high threshold CSA).',
        '',
        'Calculation parameters are extended from the Base Valuation with these new fields:',
        '',
        ' - **Deflation Interest Rate** - the interest rate price factor to PV the exposure to today',
        ' - **Batch Size** - The number of simulations per batch. Smaller Batch sizes are more likely to fit in memory.'
        '  This needs to be balanced with speed - larger batch sizes will run quicker.',
        ' - **Simulation Batches** - Number of batches to run. Total number of sumulations is **Simulation Batches** * '
        '**Batch Size**. Note that **Batch Size** is usually a power of 2.',
        ' - **Antithetic** - Use antithethic variables - we run twice the number of simulations using the negative of ',
        '  the random sample for the second run',
        ' - **Calc Scenarios** - return the simulated price factors used in the calculation ',
        ' - **Dynamic Scenario Dates** - Generate scenarios not just on the **Time Grid**, but also on all potential '
        'cashflow settlement dates. Needed to accurately calculate liquidity and settlement dynamics on collateralized ',
        'portfolios.',
        ' - **Generate Cashflows** - returns the simulated cashflows during the simulation period'
    ])

    calc_type = 'CreditMonteCarlo'
    fields = [
        F('Base_Date', 'Date', default=''),
        F('Currency', 'Text', default='ZAR'),
        F('Time_Grid', 'Text', default='0d 2d 1w(1w) 3m(1m) 2y(3m)'),
        F('Deflation_Interest_Rate', 'Text', default='ZAR-SWAP'),
        F('Percentile', 'Text', default='95'),
        F('MCMC_Simulations', 'Integer', default=2048),
        F('Simulation_Batches', 'Integer', default=1),
        F('Batch_Size', 'Integer', default=1024),
        F('Random_Seed', 'Integer', default=5120),
        F('Tenor_Offset', 'Float', default=0.0,
          description='Years to shift every factor tenor by before the run'),
        F('Antithetic', 'Text', default='No', values=['Yes', 'No']),
        F('Calc_Scenarios', 'Text', default='No', values=['At_Percentile', 'All', 'No']),
        F('Dynamic_Scenario_Dates', 'Text', default='Yes', values=['Yes', 'No']),
        F('Generate_Cashflows', 'Text', default='Yes', values=['Yes', 'No']),
        F('Keep_Tensor', 'Text', default='No', values=['Yes', 'No'],
          description='Keep the simulated mtm tensor on the device after the run'),
        F('NoModel', 'Text', default='Constant', values=['Constant', 'RiskNeutral'],
          description='How a factor with no stochastic process evolves'),
        F('Gradient_Variables', 'Text', default='All', values=['All', 'Factors', 'Implied'],
          description='Which leaves the sensitivity engine differentiates'),
        F('Boundary_AAD_Bandwidth', 'Float', default=0.01,
          description='Kernel bandwidth of the boundary correction assembled into backward()'),
        F('Recompute_Inner_MC', 'Text', default='No', values=['Yes', 'No'],
          description='Re-simulate a Monte Carlo pricer\'s inner paths in backward() rather than '
                      'taping them; trades a second forward pass for the graph of every pricing'),
        F('Credit_Valuation_Adjustment', 'Container',
          default={"Calculate": "No", "Counterparty": "", "Bank": "",
                   "Deflate_Stochastically": "Yes", "Stochastic_Hazard_Rates": "No",
                   "Gradient": "No"},
          sub_fields=[
              F('Calculate', 'Text', default='No', values=['Yes', 'No']),
              F('Counterparty', 'Text', default=''),
              F('Bank', 'Text', default=''),
              F('Deflate_Stochastically', 'Text', default='Yes', values=['Yes', 'No']),
              F('Stochastic_Hazard_Rates', 'Text', default='No', values=['Yes', 'No']),
              F('Gradient', 'Text', default='No', values=['Yes', 'No']),
              F('Hessian', 'Text', default='No', values=['Yes', 'No'],
                description='Second derivatives of the CVA as well as the first'),
              F('CDS_Tenors', 'Container', default=[],
                description='Tenors in years to add to the survival curve so CDS rates can be '
                            'interpolated off it')]),
        F('Funding_Valuation_Adjustment', 'Container',
          default={"Calculate": "No", "Counterparty": "", "Bank": "", "Risk_Free_Curve": "",
                   "Funding_Cost_Interest_Curve": "", "Funding_Benefit_Interest_Curve": "",
                   "Deflate_Stochastically": "Yes", "Stochastic_Funding": "No", "Gradient": "No"},
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
          default={"Calculate": "No", "Collateral_Curve": "", "Funding_Curve": "",
                   "Collateral_Spread": 0, "Funding_Spread": 0, "Gradient": "No"},
          sub_fields=[
              F('Calculate', 'Text', default='No', values=['Yes', 'No']),
              F('Collateral_Curve', 'Text', default=''),
              F('Funding_Curve', 'Text', default=''),
              F('Collateral_Spread', 'Integer', default=0),
              F('Funding_Spread', 'Integer', default=0),
              F('Gradient', 'Text', default='No', values=['Yes', 'No'])]),
        F('Initial_Margin', 'Container',
          default={"Calculate": "No", "Liquidity_Weights": "", "IRS_Weights": "",
                   "Local_Currency": "", "IM_Currency": "", "Delta_Factor": 1.0},
          description='The LCH-style initial margin add-on',
          sub_fields=[
              F('Calculate', 'Text', default='No', values=['Yes', 'No']),
              F('Liquidity_Weights', 'Text', default='',
                description='Path to the liquidity-weight csv, indexed by tenor bucket'),
              F('IRS_Weights', 'Text', default='',
                description='Path to the delta-weight csv, indexed by tenor bucket'),
              F('Local_Currency', 'Text', default='',
                description='The currency whose curves are charged the local weights'),
              F('IM_Currency', 'Text', default='', obj='Tuple',
                description='Currency the margin is reported in'),
              F('Delta_Factor', 'Float', default=1.0,
                description='Multiplier on the delta charge before the liquidity add-on')])
    ]

    def __init__(self, config, **kwargs):
        super(Credit_Monte_Carlo, self).__init__(config, **kwargs)
        self.reset_dates = None
        self.settlement_currencies = None
        # used to store any jacobian matrices
        self.jacobians = {}
        # implied factors
        self.implied_factors = {}
        # we represent the calc as a combination of static, stochastic and implied parameters
        self.implied_var = {}

        # potentially store the full list of variables
        self.all_var = None

    def update_factors(self, params, base_date, job_id, num_jobs):
        dependent_factors, stochastic_factors, implied_factors, reset_dates, settlement_currencies = \
            self.config.calculate_dependencies(params, base_date, self.input_time_grid)

        self.update_time_grid(base_date, reset_dates, settlement_currencies,
                              dynamic_scenario_dates=params['Dynamic_Scenario_Dates'] == 'Yes')

        return self._build_factor_state(
            dependent_factors, stochastic_factors, implied_factors, params, base_date, job_id, num_jobs)

    def _build_factor_state(self, dependent_factors, stochastic_factors, implied_factors,
                            params, base_date, job_id, num_jobs):
        """Construct factor objects, tensors, shared memory and precalculate processes.

        Called by update_factors after the time grid and dependency sets are known.
        Subclasses that build their own dependency sets (e.g. HedgeMonteCarlo) can
        call this directly instead of going through calculate_dependencies.

        IMPLIED-LEAF INVARIANT: a factor can be BOTH a static dependent factor (e.g. the OSS
        pricer's HestonNandiModelParameters, pulled in via the EquityPrice/FxRate conditional
        field) AND a spot process's implied factor (implied_var, e.g. HestonNandiImpliedSpotModel).
        With greeks on, minting a fresh static leaf for it would create a SECOND AAD leaf under the
        exact scope name the implied leaf already owns: the pricer (t_Static_Buffer) and the
        scenario path (implied_tensor) would then read different tensors, splitting the gradient
        and desyncing a bump. The single implied leaf is reused so one tensor serves both consumers
        and `value.backward()` sums both paths' sensitivities into it.

        `_factor_precalc_args` caches per-factor (ScenarioTimeGrid, implied_tensor) so consumers
        that need to re-precalculate with a per-path initial state (e.g. the diff-ML t=0 burn-in in
        HedgeMonteCarlo.execute) can call precalculate again without re-deriving the
        dependent_factors / time-grid plumbing.
        """
        # now construct the stochastic factors and static factors for the simulation
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

        # get the tenor offset (if any)
        tenor_offset = params.get('Tenor_Offset', 0.0)
        # check if we need gradients for any sub calc
        greeks = bool(np.any([params[k].get('Gradient', 'No') == 'Yes' for k, v in params.items() if type(v) == dict]))
        sensitivities = params.get('Gradient_Variables', 'All')

        # now get the stochastic risk factors ready - these will be generated from the price models
        for key, value in self.stoch_factors.items():
            if key.type not in utils.DimensionLessFactors:
                # check if there are any implied factors linked here
                if hasattr(value, 'implied'):
                    vars = {}
                    calc_grad = greeks and sensitivities in ['All', 'Implied']
                    for param_name, param_value in value.implied.current_value().items():
                        factor_name = utils.Factor(value.implied.__class__.__name__, key.name + (param_name,))
                        vars[factor_name] = self.factor_leaf(factor_name, param_value, calc_grad)
                    self.implied_var[key] = vars

                # check the daycount for the tenor_offset
                if tenor_offset:
                    factor_tenor_offset = utils.get_day_count_accrual(
                        base_date, tenor_offset, value.factor.get_day_count() if hasattr(
                            value.factor, 'get_day_count') else utils.DAYCOUNT_ACT365)
                else:
                    factor_tenor_offset = 0.0

                # record the offset of this risk factor
                current_val = value.factor.current_value(offset=factor_tenor_offset)
                calc_grad = greeks and sensitivities in ['All', 'Factors']
                self.stoch_var[key] = self.factor_leaf(
                    key, current_val, calc_grad, factor_tenor_offset)

        # and then get the static risk factors ready - these will just be looked up
        calc_grad = greeks and sensitivities in ['All', 'Factors']
        # reuse the single implied leaf (implied-leaf invariant) - never mint a second one here
        implied_leaves = {fk: t for vars in self.implied_var.values() for fk, t in vars.items()}
        for key, value in self.static_factors.items():
            if key.type not in utils.DimensionLessFactors:
                # check the daycount for the tenor_offset
                if tenor_offset:
                    factor_tenor_offset = utils.get_day_count_accrual(
                        base_date, tenor_offset, value.get_day_count() if hasattr(
                            value, 'get_day_count') else utils.DAYCOUNT_ACT365)
                else:
                    factor_tenor_offset = 0.0
                # record the offset of this risk factor
                current_val = value.current_value(offset=factor_tenor_offset)
                if isinstance(current_val, dict):
                    for k, v in current_val.items():
                        fkey = utils.Factor(key.type, key.name + (k,))
                        self.static_var[fkey] = implied_leaves[fkey] if fkey in implied_leaves else \
                            self.factor_leaf(fkey, v, calc_grad, factor_tenor_offset)
                else:
                    self.static_var[key] = implied_leaves[key] if key in implied_leaves else \
                        self.factor_leaf(key, current_val, calc_grad, factor_tenor_offset)

        # set up the device and allocate memory
        shared_mem = self._init_shared_mem(
            int(params['Random_Seed']), params['NoModel'],
            params['Currency'], params['MCMC_Simulations'],
            job_id, num_jobs, calc_greeks=sensitivities if greeks else None)

        # calculate a reverse lookup for the tenors and store the daycount code
        self.all_tenors = utils.update_tenors(self.base_date, self.all_factors)

        # now initialize all stochastic factors, caching the per-factor precalculate plumbing
        self._factor_precalc_args = {}
        for key, value in self.stoch_factors.items():
            if key.type not in utils.DimensionLessFactors:
                if key in self.implied_var:
                    implied_tensor = {k.name[-1]: v for k, v in self.implied_var[key].items()}
                    value.link_references(implied_tensor, self.implied_var, self.implied_factors)
                else:
                    implied_tensor = None
                # Hand the process its own factor key so it can publish auxiliaries to
                # t_Scenario_Buffer under the documented (factor_key, kind) convention.
                value.factor_key = key
                scenario_grid = ScenarioTimeGrid(dependent_factors[key], self.time_grid, base_date)
                self._factor_precalc_args[key] = (scenario_grid, implied_tensor)
                value.precalculate(
                    base_date, scenario_grid,
                    self.stoch_var[key], shared_mem, self.process_ofs[key], implied_tensor=implied_tensor)
                if not value.params_ok:
                    logging.warning('Stochastic factor {} has been modified'.format(utils.check_scope_name(key)))
                # Processes without a lognormal interval law return None and simply do not
                # appear, which is what a pricer treats as "observe the endpoints".
                variance_rate = value.bridge_variance_rate
                if variance_rate is not None:
                    shared_mem.t_Bridge_Variance_Rate[key] = variance_rate

        # now check if any of the stochastic processes depend on other processes
        for key, value in self.stoch_factors.items():
            if key.type not in utils.DimensionLessFactors:
                value.calc_references(key, self.static_factors, self.stoch_factors, self.all_tenors, self.all_factors)

        return shared_mem

    def update_time_grid(self, base_date, reset_dates, settlement_currencies, dynamic_scenario_dates=False):
        # work out the scenario and dynamic dates
        dynamic_dates = set([x for x in reset_dates if x > base_date])

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

        # set up the scenario and time grids
        self.time_grid = utils.TimeGrid(scenario_dates, mtm_dates, base_mtm_dates)
        self.base_date = base_date
        self.reset_dates = reset_dates
        self.time_grid.set_base_date(base_date)

        # Set the settlement dates
        self.time_grid.set_currency_settlement(settlement_currencies)
        self.settlement_currencies = settlement_currencies

    def get_cholesky_decomp(self):
        # create the correlation matrix
        correlation_matrix = np.eye(self.num_factors, dtype=np.float64)
        logging.root.name = self.config.deals['Attributes'].get('Reference', self.config.file_ref)
        # prepare the correlation matrix (and the offsets of each stochastic process)
        correlation_factors = []
        self.process_ofs = {}
        for key, value in self.stoch_factors.items():
            proc_corr_type, proc_corr_factors = value.correlation_name
            # record the offset of this factor model (derived 0-factor processes get one
            # too — generate() ignores it, but the precalc plumbing indexes process_ofs)
            self.process_ofs.setdefault(key, len(correlation_factors))
            for sub_factors in proc_corr_factors:
                # record the name of needed correlation lookup
                correlation_factors.append(utils.Factor(proc_corr_type, key.name + sub_factors))

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
        # need to do cholesky
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

                    # exit early
                    if np.abs(C - nC).max() < 1e-08 * np.abs(nC).max():
                        break

                    C = nC

                new_correlation_matrix = nC

            correlation_matrix = new_correlation_matrix
            # check again
            raw_eigval, raw_eigvec = np.linalg.eig(correlation_matrix)
            eigval, eigvec = np.real(raw_eigval), np.real(raw_eigvec)

        correlation_matrix = torch.tensor(
            correlation_matrix, device=self.device, dtype=self.dtype, requires_grad=False)
        # return the cholesky decomp
        return torch.linalg.cholesky(correlation_matrix)

    def _init_shared_mem(self, seed, nomodel, reporting_currency, mcmc_sim, job_id, num_jobs, calc_greeks=None):
        """Allocate the CMC_State for this run (correlation cholesky, static buffer, reporting FX)
        and, when greeks are requested, build the flat AAD variable index over `calc_greeks`.

        `boundary_aad` deliberately has no JSON switch: wanting sensitivities IS the switch. The
        correction is worth exactly zero in the forward pass, so it can only ever change a
        derivative - there is nothing a user could sensibly turn off, and recording events nobody
        differentiates would just be memory held across a batch. Without greeks this runs as it
        always did.
        """
        # Single-underscore (overridable): HedgeMonteCarlo overrides to construct
        # CMC_State_Inner with simulation_sub_batch from params.
        if calc_greeks is not None:
            implied_vars = list(itertools.chain(*[x.items() for x in self.implied_var.values()]))
            if calc_greeks == 'Implied':
                self.all_var = implied_vars
            elif calc_greeks == 'Factors':
                self.all_var = list(self.stoch_var.items()) + list(self.static_var.items())
            else:
                # A factor that is BOTH a static dependent and a spot process's implied factor is
                # ONE deduped leaf (the implied-leaf invariant), so it is reachable twice here and
                # the union must not report it twice.
                self.all_var = list(dict(
                    implied_vars + list(self.stoch_var.items()) + list(self.static_var.items())).items())
            self.make_factor_index(self.all_var)

        scale_by_survival = (
            self.params['Funding_Valuation_Adjustment'].get('Calculate', 'No') == 'Yes')

        shared_mem = CMC_State(
            self.get_cholesky_decomp(), self.static_var, self.batch_size,
            torch.ones([1, 1], dtype=self.dtype, device=self.device), mcmc_sim, get_fxrate_factor(
                utils.check_rate_name(reporting_currency), self.static_factors, self.stoch_factors),
            seed, job_id, num_jobs, scale_by_survival, nomodel=self.params.get('NoModel', 'Constant'),
            keep_tensor=self.params.get('Keep_Tensor', 'No') == 'Yes')
        # wanting sensitivities IS the switch (worth zero forward; only a derivative can move)
        shared_mem.boundary_aad = calc_greeks is not None
        shared_mem.recompute_inner_mc = self.params.get('Recompute_Inner_MC', 'No') == 'Yes'
        return shared_mem

    def report(self, output):
        for result, data in output.items():
            if result == 'scenarios':
                scen = {}
                scenario_date_index = pd.DatetimeIndex(sorted(self.time_grid.scenario_dates))
                if self.params['Calc_Scenarios'] == 'At_Percentile':
                    # calc pfe
                    dates = np.array(sorted(self.time_grid.mtm_dates))[self.time_grid.report_index]
                    mtms = pd.DataFrame(np.concatenate(output['mtm'], axis=-1).astype(np.float64), index=dates)
                    percentiles = self.params.get('Percentile', '95').replace(' ', '').split(',')
                    profiles = {x: np.percentile(mtms.values, float(x), axis=1) for x in percentiles}
                    index = {x: np.argmin(np.abs(mtms.values - profiles[x][:, np.newaxis]), axis=1) for x in percentiles}

                    # now only extract the scenarios at percentile points
                    for factor_key, factor_values in data.items():
                        factor_name = utils.check_tuple_name(factor_key)
                        values = np.concatenate(factor_values, axis=-1)  # Shape: (num_rows, num_scenarios)
                        value_len = values.shape[0]
                        if len(values.shape) == 2:
                            columns = pd.MultiIndex.from_product(
                                [[0.0], percentiles], names=['tenor', 'scenario'])
                            vals = np.dstack([values[np.arange(value_len), i[:value_len]] for i in index.values()])
                            scen[factor_name] = pd.DataFrame(
                                vals.reshape(value_len, -1), index=scenario_date_index[:value_len], columns=columns).T
                        else:
                            tenors = self.all_tenors[factor_key][0].tenor
                            columns = pd.MultiIndex.from_product(
                                [tenors, percentiles], names=['tenor', 'scenario'])
                            vals = np.dstack([values[np.arange(value_len), :, i[:value_len]] for i in index.values()])
                            scen[factor_name] = pd.DataFrame(
                                vals.reshape(value_len, -1),
                                index=scenario_date_index[:value_len], columns=columns).T
                else:
                    for k, v in data.items():
                        factor_name = utils.check_tuple_name(k)
                        values = np.concatenate(v, axis=-1)
                        if len(values.shape) == 2:
                            columns = pd.MultiIndex.from_product(
                                [[0.0], np.arange(values.shape[-1])], names=['tenor', 'scenario'])
                            scen[factor_name] = pd.DataFrame(
                                values, index=scenario_date_index[:values.shape[0]], columns=columns).T
                        else:
                            tenors = self.all_tenors[k][0].tenor
                            columns = pd.MultiIndex.from_product(
                                [tenors, np.arange(values.shape[-1])], names=['tenor', 'scenario'])
                            scen[factor_name] = pd.DataFrame(
                                values.reshape(values.shape[0], -1),
                                index=scenario_date_index[:values.shape[0]], columns=columns).T
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

    def execute(self, params, job_id=0, num_jobs=1):
        """Run the batched exposure simulation plus whichever sub-calculations `params` enables
        (collateral, CVA, FVA, scenarios, cashflows) and return netting sets, stats and reports.

        The CVA reduction is left in its original grouping deliberately: a per-path vector cannot
        be reduced back to `mean over paths of a sum over time` in the same float order, and the
        reported number must not move by even an ULP. `pricing.cva_per_scenario` is the same
        quantity for the COUNTERFACTUALS, where only internal consistency matters.

        BOUNDARY AAD (CVA gradient): a hard transfer decision contributes a derivative that the
        frozen-decision graph does not carry. The correction is worth exactly zero in the forward
        pass, so tensors['cva'] - the REPORTED number - is untouched; only the scalar being
        differentiated gains a term. `Boundary_AAD_Bandwidth` defaults to 0.01: the estimate
        converges monotonically as the bandwidth shrinks and is still stable across seeds there,
        where 0.05 upward is visibly biased. It needs enough paths for the near-boundary band to be
        populated - measured at 32768 - so a thin run should widen it and expect bias rather than
        noise.
        """
        # the declaration is the single source of an omitted field's default
        params = declared_defaults(type(self), params)
        # get the rundate
        base_date = pd.Timestamp(params['Run_Date'])

        # Define the base and scenario grids
        self.input_time_grid = params['Time_grid']
        # needed if we are using multiprocessing across gpu's
        params['Simulation_Batches'] = params['Simulation_Batches'] // num_jobs
        self.batch_size = params['Batch_Size']
        self.numscenarios = self.batch_size * params['Simulation_Batches']

        # store the params
        self.params = params
        # set the name of the root logger to this netting set (makes tracking errors easier)
        logging.root.name = self.config.deals['Attributes'].get('Reference', self.config.file_ref)

        # store the stats for the batches
        self.calc_stats['Batch_Size'] = self.batch_size
        self.calc_stats['Simulation_Batches'] = self.params['Simulation_Batches']
        self.calc_stats['Random_Seed'] = params['Random_Seed']

        # update the factors and obtain shared state
        shared_mem = self.update_factors(params, base_date, job_id, num_jobs)

        # set up the all instruments
        self.netting_sets = DealStructure(Aggregation('root'), store_results=True)
        self.set_deal_structures(
            self.config.deals['Deals']['Children'], self.netting_sets, shared_mem.one,
            deal_level_mtm=params.get('DealLevel', False))
        self.netting_sets.finalize_struct(base_date, self.time_grid)

        # clear the output
        output = defaultdict(list)
        # reset the tensors - used for storing simulation data
        tensors = {}
        # record how long it took to run the calc (python + pytorch)
        execution_label = 'Tensor_Execution_Time ({})'.format(self.device.type)
        self.calc_stats[execution_label] = time.monotonic()
        # record the base currency
        base_ccy = get_fxrate_factor(
            utils.check_rate_name(self.config.params['System Parameters']['Base_Currency']),
            self.static_factors, self.stoch_factors)
        # record the report_grid index
        time_index = self.time_grid.report_index

        for run in range(self.params['Simulation_Batches']):

            # need to refresh random numbers and zero out buffers
            shared_mem.reset(
                self.num_factors, self.time_grid, use_antithetic=params.get('Antithetic', 'No') == 'Yes')

            # simulate the price factors
            for key, value in self.stoch_factors.items():
                shared_mem.t_Scenario_Buffer[key] = value.generate(shared_mem)

            # construct the valuations

            # use these lines below to track down any issues that prevent gradients from flowing (debugging only)
            # with torch.autograd.detect_anomaly():
            tensors['mtm'] = self.netting_sets.resolve_structure(shared_mem, self.time_grid)
            #    m = tensors['mtm'].mean()
            #    m.backward()

            # is this the final run?
            final_run = run == self.params['Simulation_Batches'] - 1
            # the mtm is in reporting currency - need to convert back to base currency
            fx_report = utils.calc_fx_cross(
                shared_mem.Report_Currency, base_ccy, self.time_grid.time_grid[time_index], shared_mem)

            # now calculate all the valuation adjustments (if necessary)
            if params['Collateral_Valuation_Adjustment'].get(
                    'Calculate', 'No') == 'Yes' and shared_mem.simulation_batch > 1:

                if params['Collateral_Valuation_Adjustment'].get('Gradient', 'No') == 'Yes':
                    tensors['collva_t'] = torch.mean(shared_mem.t_Credit['Funding'], dim=1)
                    tensors['collva'] = torch.sum(tensors['collva_t'])

                    # calculate all the derivatives of fva
                    sensitivity = SensitivitiesEstimator(tensors['collva'], self.all_var)

                    if final_run:
                        output['grad_collva'] = sensitivity.report_grad()
                        # store the size of the Gradient
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

                for d in local_shifts.values():  # you can list as many input dicts as you want here
                    for key, value in d.items():
                        all_shifts.setdefault(key, []).append(value)

                # switch off cashflows
                shared_mem.t_Cashflows = None

                for tenor, shifts in all_shifts.items():
                    # bump the scenarios
                    deltas = {}
                    for curvename, shift in zip(local_shifts.keys(), shifts):
                        deltas[curvename] = shared_mem.one.new_tensor(shift.reshape(1, -1, 1) * 0.01 * 0.01)
                        scen_buf[curvename] += deltas[curvename]

                    # reset the cache
                    shared_mem.t_Buffer.clear()
                    # calc the liquidity change in base_currency - simple delta
                    liquidity_deltas[tenor] = (self.netting_sets.resolve_structure(
                        shared_mem, self.time_grid) - tensors['mtm']) * fx_report

                    # unbump the scenarios
                    for curvename, shift in zip(local_shifts.keys(), shifts):
                        scen_buf[curvename] -= deltas[curvename]

                liquidity_charge = {}
                for tenor, values in liquidity_deltas.items():
                    curve_tenor = utils.tenor_diff(irs[tenor].dropna().index.astype(np.float64).values)
                    curve_weights = shared_mem.one.new_tensor(irs[tenor].dropna().values)
                    index, index_next, alpha = curve_tenor.get_index(values)
                    liquidity_charge[tenor] = values * (
                            curve_weights[index] * (1 - alpha) + curve_weights[index_next] * alpha)

                IM_liquidity_charge = (calc_max(
                    liquidity_charge, '2y', '5y') + calc_max(liquidity_charge, '10y', '30y')) * fx_IM_report

                shared_mem.t_Buffer.clear()
                for int_rate in [k for k in shared_mem.t_Scenario_Buffer.keys() if
                                 k.type == 'InterestRate' and len(k.name) == 1]:
                    #calc pv01
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
                    # calculate all the derivatives of fva
                    # The shipped batch job DELETES the CVA section, so the correction assembled
                    # over there could never fire for FVA - it carries its own objective.
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
                        # store the size of the Gradient
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

                # tensors['mtm'] is in reporting currency - we need to convert back to base
                pv_exposure = torch.relu(tensors['mtm'] * fx_report * Dt_T) / fx_report[0]

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
                    # potentially fetch ir jacobian matrices for base curves
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
                        except KeyError as e:
                            pass

                    # calculate all the derivatives of cva
                    hessian = params['Credit_Valuation_Adjustment'].get('Hessian', 'No') == 'Yes'
                    # boundary correction is zero forward; only the differentiated scalar gains a
                    # term. Bandwidth 0.01 (see docstring) needs ~32768 paths to be noise, not bias
                    cva_for_aad = tensors['cva']
                    if shared_mem.boundary_sets:
                        objective = lambda mtm: pricing.cva_per_scenario(
                            torch.relu(mtm * fx_report * Dt_T) / fx_report[0], prob, recovery)
                        correction = pricing.boundary_correction(
                            shared_mem, objective, tensors['mtm'],
                            float(params.get('Boundary_AAD_Bandwidth', 0.01)))
                        if correction is not None:
                            cva_for_aad = cva_for_aad + correction
                    sensitivity = SensitivitiesEstimator(
                        cva_for_aad, self.all_var, create_graph=hessian)

                    if final_run:
                        output['grad_cva'] = sensitivity.report_grad()
                        # store the size of the Gradient
                        self.calc_stats['Gradient_Vector_Size'] = sensitivity.P

                        # now fetch the CDS tenors and calculate the CDS spreads
                        CDS_tenors = params['Credit_Valuation_Adjustment'].get('CDS_Tenors')
                        if CDS_tenors and recovery < 1.0:
                            # calculate cds sensitivities
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

            # store all output tensors
            for k, v in tensors.items():
                output[k].append(v.cpu().detach().numpy())

            # fetch cashflows if necessary
            if self.params['Generate_Cashflows'] == 'Yes':
                dates = np.array(sorted(self.time_grid.mtm_dates))
                for currency, values in shared_mem.t_Cashflows.items():
                    cash_index = dates[sorted(values.keys())]
                    output.setdefault('cashflows', {}).setdefault(currency, []).append(
                        pd.DataFrame(
                            [v.cpu().detach().numpy() for _, v in sorted(values.items())], index=cash_index))

            # add any scenarios if necessary
            if self.params.get('Calc_Scenarios', 'No') != 'No':
                for key, value in self.stoch_factors.items():
                    output.setdefault('scenarios', {}).setdefault(key, []).append(
                        shared_mem.t_Scenario_Buffer[key].cpu().detach().numpy())

        self.calc_stats[execution_label] = time.monotonic() - self.calc_stats[execution_label]

        # store the results
        results = {'Netting': self.netting_sets, 'Stats': self.calc_stats, 'Jacobians': self.jacobians}
        results['Results'] = self.report(output)

        return results


class Base_Reval_State(utils.Calculation_State):
    def __init__(self, static_buffer, one, mcmc_sims, report_currency, calc_greeks, gamma, nomodel='Constant'):
        """Single-date, single-scenario valuation state (base MtM and its greeks).

        `boundary_aad` follows the same contract as CMC_State: a decision taken on simulated state
        is recorded during the forward pass so its derivative can be restored before the reverse
        sweep. Base valuation has one date and one scenario, but a Monte Carlo pricer still runs a
        full INNER simulation underneath it - which is where a TARF's knock-in is decided - so the
        defect and the estimator are the same ones, only the objective is simpler.
        """
        super(Base_Reval_State, self).__init__(
            static_buffer, one, mcmc_sims, report_currency, nomodel, 1, False)
        self.calc_greeks = calc_greeks
        self.gamma = gamma
        # same boundary-AAD contract as CMC_State - the inner MC is where decisions are taken
        self.boundary_aad = calc_greeks is not None
        self.boundary_sets = []

    @staticmethod
    def save_results(output, tensors):
        for k, v in tensors.items():
            output[k] = np.float64(v) if isinstance(v, float) else v.detach().cpu().numpy().astype(np.float64)


class Base_Revaluation(Calculation):
    """Simple deal revaluation - Use this to reconcile with the source system.

    SECOND DERIVATIVES LIVE HERE AND NOWHERE ELSE. `Greeks: 'All'` asks the reverse sweep for
    `create_graph`, and what comes back is reported as `out['Results']['Greeks_Second']` beside
    the first-order `Greeks_First` - a stable key, always accompanied by the first-order block
    because the row labels are built off it.

    THE SHAPE IS THE FULL HESSIAN, not a Hessian-vector product, and the reason is what the
    number is FOR. `Greeks_Second` is a report - a cross-gamma matrix a risk system reads, whose
    off-diagonal is the whole point (spot-vol, spot-curve) - so no caller arrives with a
    direction to contract along, and an HVP interface would only mean forming the same matrix a
    column at a time outside the engine. The cost is what makes that affordable: one date, one
    scenario, and P = the number of factor KNOTS the portfolio depends on (5-11 across this
    repo's fixtures), against which `report_hessian` runs P double-backward passes - measured at
    0.9x to 10x the first-order pass, 0.002s to 0.07s. An exposure-sized P is what would flip
    that argument, and exposure does not come here - `Credit_Monte_Carlo` has its own
    CVA-Hessian route.

    The frame is the Hessian's SUPPORT: rows and columns that are identically zero are dropped
    (a factor the portfolio does not touch at second order), so it is square and symmetric but
    smaller than P, indexed on both axes by (Rate, Tenor, Tenor2, Tenor3) - the columns carrying
    the reporting reference as an outer level, the way `Greeks_First` does.

    TWO THINGS REFUSE RATHER THAN REPORT, both because the failure would otherwise be a plausible
    number: a deal that registered a `BoundarySet` (`execute` below) and `Recompute_Inner_MC`
    (`pricing.InnerMCRecompute.backward`).
    """
    documentation = ('Calculations',
                     ['This applies the valuation models mentioned earlier to the portfolio per deal.',
                      '',
                      'The inputs are:',
                      '',
                      '- **Currency** of the output.',
                      '- **Run_Date** at which the marketdata should be applied (i.e. $t_0$)',
                      '- **MCMC Simulations** the number of Monte Carlo simulations to use for deals that require ',
                      '  Monte Carlo pricing (e.g. Autocalls, TARF\'s etc.)',
                      '- **Random Seed** the seed for the Monte Carlo Pricer',
                      '- **Greeks** `First` calculates all first order sensitivities (partial derivatives) of ',
                      '  the portfolio with respect to the relevant Price Factors; `All` adds the second order ',
                      '  block (the full factor Hessian, reported as `Greeks_Second`). Default is neither.',
                      '',
                      'The output is a dictionary containing the DealStructure and the calculation computation ',
                      'statistics.'
                      ])

    calc_type = 'BaseValuation'
    fields = [
        F('Base_Date', 'Date', default=''),
        F('Currency', 'Text', default='ZAR'),
        # 4096 * 8 is what run_baseval always injected while the store said 2048 and nothing read
        # it; the declaration records the behaviour production has, so no realized number moves
        F('MCMC_Simulations', 'Integer', default=4096 * 8),
        F('Random_Seed', 'Integer', default=5120),
        F('Greeks', 'Text', default='No', values=['All', 'First', 'No'],
          description='First order factor sensitivities, or `All` for the second order block '
                      '(`Greeks_Second`) as well - see the class docstring for its shape'),
        F('Boundary_AAD_Bandwidth', 'Float', default=0.01,
          description='Kernel bandwidth of the boundary correction assembled into backward()'),
        F('Recompute_Inner_MC', 'Text', default='No', values=['Yes', 'No'],
          description='Re-simulate a Monte Carlo pricer\'s inner paths in backward() rather than '
                      'taping them; trades a second forward pass for the graph of every pricing')
    ]

    def __init__(self, config, **kwargs):
        super(Base_Revaluation, self).__init__(config, **kwargs)
        self.base_date = None

        # Cuda related variables to store the state of the device between calculations
        self.shared_memClass = namedtuple('shared_mem',
                                          't_Buffer t_Static_Buffer t_Feed_dict t_Cashflows calc_greeks \
                                          gpus riskneutral precision simulation_batch Report_Currency')

        # prepare the risk factor output matrix . .
        self.static_var = {}

    def update_factors(self, params, base_date):
        dependent_factors, stochastic_factors, implied_factors, reset_dates, settlement_currencies = \
            self.config.calculate_dependencies(params, base_date, '0d', False)

        # update the time grid
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
        # and then get the static risk factors ready - these will just be looked up
        for key, value in self.static_factors.items():
            if key.type not in utils.DimensionLessFactors:
                current_val = value.current_value()
                if isinstance(current_val, dict):
                    for k, v in current_val.items():
                        fkey = utils.Factor(key.type, key.name + (k,))
                        self.static_var[fkey] = self.factor_leaf(fkey, v, calc_grad)
                else:
                    self.static_var[key] = self.factor_leaf(key, current_val, calc_grad)

        # set up the device and allocate memory
        shared_mem = self.__init_shared_mem(
            params['Currency'], params['MCMC_Simulations'], calc_grad, params['Random_Seed'])

        # calculate a reverse lookup for the tenors and store the daycount code
        self.all_tenors = utils.update_tenors(self.base_date, self.all_factors)

        return shared_mem

    def update_time_grid(self, base_date):
        # set up the scenario and time grids
        self.time_grid = utils.TimeGrid({base_date}, {base_date}, {base_date})
        self.base_date = base_date
        self.time_grid.set_base_date(base_date)
        # The one date IS the reporting date. Exposure calculations get this from finalize_struct;
        # here nothing else needs it, but a boundary registration reads report_index to know the
        # grid its counterfactual has to land on, and treats its absence as "not reportable".
        self.time_grid.set_report_dates(base_date, {base_date})

    def __init_shared_mem(self, reporting_currency, mcmc_sim, calc_greeks, random_seed):
        # fix the seed if we need to price mc instruments
        torch.manual_seed(random_seed)

        # name of the base currency
        base_currency = utils.Factor(
            'FxRate', (self.config.params['System Parameters']['Base_Currency'],))

        # now decide what we want to calculate greeks with respect to
        all_vars_concat = None
        if calc_greeks:
            all_vars_concat = [x for x in self.static_var.items() if x[0] != base_currency]
            self.make_factor_index(list(self.static_var.items()))

        # allocate memory on the device
        shared_mem = Base_Reval_State(
            self.static_var, torch.ones([1, 1], dtype=self.dtype, device=self.device),
            mcmc_sim, get_fxrate_factor(utils.check_rate_name(reporting_currency), self.static_factors, {}),
            all_vars_concat, self.params['Greeks'] == 'All')
        shared_mem.recompute_inner_mc = self.params.get('Recompute_Inner_MC', 'No') == 'Yes'
        return shared_mem

    def report(self):

        def check_prices(n, parent):

            def format_row(deal, data, val, greeks):
                data['Deal Currency'] = deal.Factor_dep.get(
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
                        data[k] = float(v)
                # update any tags
                if deal.Instrument.field.get('Tags'):
                    data.update(dict(zip(tag_titles, deal.Instrument.field['Tags'][0].split(','))))

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

        # clear the output
        self.output = {}
        # load any tag titles
        tag_titles = self.config.deals['Attributes'].get('Tag_Titles', '').split(',')
        mtm, greeks = check_prices(
            self.netting_sets, [('Parent', self.netting_sets.obj.Instrument.field.get('Reference'))])

        # calculate the grand total
        data = dict(
            [(field, self.netting_sets.obj.Instrument.field.get(field, 'Root')) for field in ['Reference', 'Object']])
        data['Value'] = sum([float(x.obj.Calc_res['Value']) for x in self.netting_sets.sub_structures])
        mtm.insert(0, data)

        # write out the dataframe
        self.output['mtm'] = pd.DataFrame(mtm)
        for greek_name, greek_val in greeks.items():
            # this guarantees that the multiindex is uniquely defined when we write it out
            if greek_name == 'Greeks_Second':
                summary = pd.concat(greek_val, axis=1).groupby(level=[0, 1, 2, 3, 4], axis=1).sum()
            elif greek_name == 'Greeks_First':
                summary = pd.concat(greek_val, axis=1).groupby(level=0, axis=1).sum()
            else:
                raise Exception('Unknown Greek requested', greek_name)
            self.output.setdefault(greek_name, summary)

        return self.output

    def execute(self, params):
        # the declaration is the single source of an omitted field's default
        params = declared_defaults(type(self), params)
        # get the rundate
        base_date = pd.Timestamp(params['Run_Date'])
        # store the params
        self.params = params
        # update the factors
        shared_mem = self.update_factors(params, base_date)
        # set the logging name
        logging.root.name = self.config.deals['Attributes'].get('Reference', self.config.file_ref)
        self.calc_stats['Deal_Setup_Time'] = time.monotonic()
        self.netting_sets = DealStructure(Aggregation('root'), store_results=True)
        self.set_deal_structures(
            self.config.deals['Deals']['Children'], self.netting_sets, shared_mem.one, deal_level_mtm=True)

        # record the (pure python) dependency setup time
        self.calc_stats['Deal_Setup_Time'] = time.monotonic() - self.calc_stats['Deal_Setup_Time']
        self.calc_stats['Graph_Setup_Time'] = time.monotonic()

        # now ask the netting set to construct each deal - no looping required (just 1 timepoint)
        mtm = self.netting_sets.resolve_structure(shared_mem, self.time_grid)
        # record the graph loading time
        self.calc_stats['Graph_Setup_Time'] = time.monotonic() - self.calc_stats['Graph_Setup_Time']
        # populate the mtm at the netting set
        ns_obj = self.netting_sets.obj
        # make sure the netting set object has a reference and a mtm
        if ns_obj.Instrument.field.get('Reference') is None:
            ns_obj.Instrument.field['Reference'] = self.config.deals['Attributes'].get(
                'Reference', self.config.file_ref)

        if shared_mem.calc_greeks is not None:
            # record the cuda execution stats
            self.calc_stats['Greek_Execution_Time'] = time.monotonic()
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
                # The portfolio value IS the objective, so the per-scenario vector is the value
                # itself - one scenario, whose mean is the reported number. Worth exactly zero in
                # the forward pass, so `mtm` here is untouched and only the tape gains a term.
                correction = pricing.boundary_correction(
                    shared_mem, lambda value: value.sum(axis=0), mtm,
                    float(params.get('Boundary_AAD_Bandwidth', 0.01)))
                if correction is not None:
                    mtm = mtm + correction
            pricing.greeks(shared_mem, ns_obj, mtm)
            self.calc_stats['Greek_Execution_Time'] = time.monotonic() - self.calc_stats['Greek_Execution_Time']

        # return a dictionary of output
        return {'Netting': self.netting_sets, 'Stats': self.calc_stats, 'Results': self.report()}


class HedgeMonteCarlo(Credit_Monte_Carlo):
    documentation = ('Calculations', [
        'A specialisation of `Credit_Monte_Carlo` that wires the same simulated scenario',
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
        '- A **differential-ML solver** (`DiffSolverV2`): a backward-DP value function fit by',
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
        '**~118 kB** on the test fixture, **~229 kB** on the production platinum walk-forward world',
        '(GARCH spot + observed basis + carry + a 31-tenor SOFR curve, 125-step grid). Size against',
        'YOUR world, not against a remembered number:',
        '',
        '| Batch_Size x Inner_Sub_Batch | fixture peak | production peak | s / fit step |',
        '| --- | --- | --- | --- |',
        '| 1024 x 64 | 7.4 GiB | 14.3 GiB | 0.74 |',
        '| **1280 x 64** | **9.3 GiB** | **18.3 GiB** | **0.75** |',
        '| 1536 x 64 | 11.1 GiB | 22.8 GiB | 0.83 |',
        '| 2048 x 64 | 14.8 GiB | OOM | 0.99 |',
        '',
        '`Batch_Size=1280, Inner_Sub_Batch=64` is the recommended production point: 18.3 GiB of the',
        '23.6 GiB usable, ~5 GiB of headroom for world variation, and 0.75 s per fit step (1707',
        'paths/s — 1.3x the throughput of 1024 at the same step cost, since the step is launch- not',
        'bandwidth-bound below ~1300). 1536 fits but leaves only 0.8 GiB and is not recommended;',
        '2048 x 64 does NOT fit single-pass on the production world and raises OOM inside the',
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
        'The walk-forward smoke gate reproduces this end to end — `gates/wf_smoke_gate.sh`, trade',
        '202001 at `--batch 512 --batches 5 --fit-iters 40`, seed 7 — pinning train_u',
        '-0.5006, V_0 -0.1082737073302269, greedy -104.71 $/oz, churn 193.8 and the',
        'policy-independent nohedge -194.35 / pf_bound 810.1.',
        '',
        '### Execution modes',
        '',
        '- `Execution_Mode = "solve_hedge"` — run the configured `Solver.Object` (DiffSolverV2).',
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
        'Because this class inherits from `Credit_Monte_Carlo`, the underlying scenario',
        'engine — random factor generation, instrument valuation across paths, calendar',
        'handling, AAD graph construction — is identical. Anything that can be priced for',
        'a credit-exposure run can be priced as a hedging-problem leg or tradable. The only',
        'difference is what we do with the simulated MtMs: aggregate into exposures, or',
        'feed into a learning loop.'
    ])

    calc_type = 'HedgeMonteCarlo'
    fields = [
        F('Base_Date', 'Date', default=''),
        F('Currency', 'Text', default='ZAR'),
        F('Time_Grid', 'Text', default='0d 1d(1d) 4m'),
        F('Calendar', 'Text', default='',
          description='Holiday calendar naming the business day the roll steps on'),
        F('Simulation_Batches', 'Integer', default=1),
        F('Batch_Size', 'Integer', default=1024),
        F('Random_Seed', 'Integer', default=5120),
        F('Tenor_Offset', 'Float', default=0.0,
          description='Years to shift every factor tenor by before the run'),
        F('MCMC_Simulations', 'Integer', default=2048),
        F('NoModel', 'Text', default='Constant', values=['Constant', 'RiskNeutral'],
          description='How a factor with no stochastic process evolves'),
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
        F('Observed_Scenario', 'Text', default='',
          description='Path to an npz of realized factor paths that replace the simulated draw'),
        F('Hedging_Problem', 'Container', default={},
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
              F('Objective', 'Container', default={},
                description='The utility SHAPE and its parameters, dispatched on Object',
                sub_fields=[
                    F('Object', 'Text', default=REQUIRED,
                      values=['AsymmetricUtility_Symlog', 'AsymmetricUtility_Huber',
                              'AsymmetricUtility_CARA'],
                      description='The utility shape the DP recursion works in'),
                    F('Utility_Scale_Mode', 'Text', default='vol_scaled_notional',
                      values=['vol_scaled_notional'],
                      description='How the utility scale c is derived from the book'),
                    F('Utility_Scale_Explicit', 'Float', default=None,
                      description='Literal dollar c, overriding the formula'),
                    F('Huber_Aversion', 'Float', default=2.5,
                      description='Curvature of the quadratic loss arm, in units of c'),
                    F('Huber_Delta', 'Float', default=1.0,
                      description='Knee beyond which the loss arm goes linear, in units of c'),
                    F('CARA_Gamma', 'Float', default=1.0,
                      description='Absolute risk aversion of u = (1-exp(-gamma x))/gamma')]),
              F('Evaluator', 'Container', default={},
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
                      description='Half-spread bps of the calendar roll leg'),
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
                                  'argmax; 0 = uncapped. Execution policy only - training is '
                                  'unaffected, so a trained policy can be re-rolled under it'),
                    F('Total_Position_Schedule', 'Table', default=None,
                      row=Row([F('Step', 'Integer', default=0),
                               F('Min_Total', 'Float', default=0.0),
                               F('Max_Total', 'Float', default=0.0)]),
                      description='Piecewise-constant corridor on the signed book total, by '
                                  'decision step')]),
              F('Solver', 'Container', default={},
                description='The value-function solver and its schedule, dispatched on Object',
                sub_fields=[
                    F('Object', 'Text', default=REQUIRED,
                      values=['DiffSolverV2', 'HindsightDpSolver'],
                      description='The value-function solver; solve_hedge requires DiffSolverV2'),
                    F('Multi_Seed_Count', 'Integer', default=1,
                      description='Independent training seeds the artifact is selected across'),
                    F('T_Min', 'Integer', default=0,
                      description='Earliest step the backward sweep fits; 0 = full sweep'),
                    F('Training_Action_Grid_Levels_Per_Axis', 'Integer', default=11,
                      description='Levels per hedge axis in the greedy action grid'),
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
                    F('DiffV2_Weight_Decay', 'Float', default=0.0,
                      description='Residual-net weight decay; a crutch for path-starved problems'),
                    F('DiffV2_Hidden', 'Integer', default=32,
                      description='Hidden width of each residual net'),
                    F('DiffV2_Lambda_Grad', 'Float', default=1.0,
                      description='Twin-loss weight on the pathwise-gradient term'),
                    F('DiffV2_Risk_Kappa', 'Float', default=0.0,
                      description='Downside semideviation penalty at the argmax; 0 = plain E[C]'),
                    F('DiffV2_Cost_Aware_Argmax', 'Text', default='No', values=['Yes', 'No'],
                      description='Charge the L1 repositioning cost at the verdict argmax'),
                    F('DiffV2_Stepper_Rollout', 'Text', default='No', values=['Yes', 'No'],
                      description='Roll a frozen policy day-by-day through the real accounting'),
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

    @staticmethod
    def _factor_bundle_key(factor_key):
        return utils.check_tuple_name(factor_key) if hasattr(factor_key, 'type') and hasattr(factor_key, 'name') else factor_key

    def _init_shared_mem(self, seed, nomodel, reporting_currency, mcmc_sim, job_id, num_jobs, calc_greeks=None):
        """Override: HedgeMonteCarlo doesn't compute greeks or FVA, so skip the parent's
        make_factor_index / scale_survival setup. Build CMC_State_Inner directly so the
        same shared_mem hosts outer (inherited `reset()`) and inner (`reset_inner()`) modes."""
        return CMC_State_Inner(
            self.get_cholesky_decomp(), self.static_var, self.batch_size,
            torch.ones([1, 1], dtype=self.dtype, device=self.device), mcmc_sim, get_fxrate_factor(
                utils.check_rate_name(reporting_currency), self.static_factors, self.stoch_factors),
            seed, job_id, num_jobs,
            simulation_sub_batch=int(self.params.get('Inner_Sub_Batch', 0)),
            keep_tensor=self.params.get('Keep_Tensor', 'No') == 'Yes')

    @staticmethod
    def _require_all_compiled(declared, structure, role):
        """A hedge book must compile WHOLE. The pricing walk's skip-and-continue is the right
        contract for a reporting book — one broken deal should not lose the run — but here a
        skipped tradable silently shrinks the solver's menu and a skipped liability halves the
        target it is hedging, and the solve then reports a confident answer to a different
        problem. Measured: an APS leg whose basis law failed to compile dropped n* from −44.8
        to −22.1 with nothing but an ERROR log."""
        loaded = ({d.Instrument.field.get('Reference') for d in structure.dependencies} |
                  {s.obj.Instrument.field.get('Reference') for s in structure.sub_structures})
        missing = [n['Instrument'].field.get('Reference') for n in declared
                   if n['Instrument'].field.get('Reference') not in loaded]
        if missing:
            raise Exception(f'HedgeMonteCarlo: {role} legs failed to compile and were skipped: '
                            f'{missing} — a hedge book prices whole or not at all')

    def update_factors(self, params, base_date, job_id, num_jobs, end_date):
        """Override: deal-driven dependencies plus the calc's explicit Scenario_Factors list —
        factors no deal reaches directly (e.g. a basis consumed only by a composed spot)
        are declared in the JSON, not discovered through schema edges.

        The horizon is the max tradable reset date capped at `end_date` (the liability terminal):
        hedge maturities past liability end are dropped from the simulation horizon; the hedges
        themselves are priced through liability end and any residual position closes out at fair
        value there."""
        dependent_factors, stochastic_factors, _, reset_dates, settlement_currencies = self.config.calculate_dependencies(
            params, base_date, self.input_time_grid)
        for name in params.get('Scenario_Factors', []):
            factor_type, factor_name = name.split('.', 1)
            dependent_factors.setdefault(utils.Factor(factor_type, utils.check_rate_name(factor_name)), [])

        # horizon = max tradable reset date, capped at the liability terminal
        max_expiry = min(max(reset_dates), end_date)
        reset_dates = self.config.parse_grid(base_date, max_expiry, self.input_time_grid, past_max_date=True)
        reset_dates.update({base_date, max_expiry})
        # generate scerarios at each grid date
        self.update_time_grid(base_date, reset_dates, settlement_currencies, dynamic_scenario_dates=True)

        # Use the last scenario grid date so ScenarioTimeGrid covers the extra step from past_max_date=True
        last_scen_date = base_date + pd.DateOffset(days=int(self.time_grid.scen_time_grid[-1]))
        dependent_factors = {k: last_scen_date for k in dependent_factors}
        stochastic_factors, additional_factors = self.config.find_models(dependent_factors)

        shared_mem = self._build_factor_state(
            dependent_factors, stochastic_factors, additional_factors, params, base_date, job_id, num_jobs)
        return shared_mem

    def _liability_schedule_scalars(self):
        """Static (batch-independent) liability descriptors the symlog utility-scale needs, read
        straight from the cashflow schedules — no per-batch leg pass. Returns
        `(total_leg_volume, last_payment_day)`: the summed |notional| across all liability legs
        and the latest payment day (offset in days from base_date). `Bundle.from_batch` maps the
        payment day onto the (history-prefixed) bundle time grid to recover
        `last_settlement_index`."""
        return self.liabilities.aggregate_leg_descriptors()

    def execute(self, params, job_id=0, num_jobs=1):
        """Simulate the scenario engine over batches, building the tensor bundle (tradable
        prices, liability MtM, factor paths). Returns a HedgeRuntimeExecutionResult.

        A SOLVE IS A STREAM: `solve_hedge` builds a Bundle PER BATCH inside the batch loop and
        hands it to a persistent solver as it is built (warmup on batch 1, step on each later
        batch, finish on a held-out final batch), so the inner-MC forks are only ever `Batch_Size`
        wide and every fit step sees fresh paths. `simulate_only` instead accumulates every batch
        into one bundle and exposes it for stepping, for which Simulation_Batches is a path
        multiplier rather than a stream length.

        LIABILITY-DRIVEN TIME-GRID CAP (design choice, not a bug): historically the simulator
        priced every hedge instrument to its own maturity, which extended the time grid to the
        latest hedge expiry. For hedge-MC that is wasteful - past the liability terminal there is
        nothing to hedge, and any residual hedge position is closed out at fair value at that
        point. The global grid is capped at the liability's last cashflow / reval date so outer and
        inner sim both stop there (`max_settlement_date` resets each liability from its `field`
        first, which is idempotent with the reset `set_deal_structures` does).

        INNER-MC SETUP: the process copies are forked only after outer setup precalc has populated
        `factor_key` / `spot0` / etc., so inner-mode precalc on the copies cannot clobber
        outer-instance attrs that outer generate reads each batch. `shared_mem` is a
        CMC_State_Inner: outer batches use the inherited `reset()`, inner uses `reset_inner()`.

        `Randomize_Initial_State='Yes'`: Huge-Savine diff-ML needs variance in z_0 for the
        differential label at the boundary to be well-posed, obtained here via a per-batch burn-in
        - run each process once from the calibrated t=0, snapshot the terminal state per path, then
        re-precalculate with that snapshot as the new t=0. The designer distribution is the
        process's own T-step pushforward, so there is no separate sampler. Every factor gets the
        burn-in, in the same iteration order as the main generate loop; each process's
        `simulated[-1]` is the per-path shape its precalculate accepts as `tensor` (a curve process
        gets an (n_tenors, B) snapshot, a spot a (B,) one) - the identical contract
        `_run_inner_mc_at_t` forks on. Paths are published to the buffer as they are generated
        because linked factors (e.g. BasisLinkedSpotModel) read their underlying's path out of
        t_Scenario_Buffer during their own generate, and stoch_factors is in topological order.

        Batch k+1 REWINDS to the calibrated t=0 before its own burn-in: the burn-in leaves each
        process precalculated from ITS OWN terminal state, so without the rewind the batch sequence
        becomes a random walk away from the calibrated world instead of N independent draws from
        it. Measured before the rewind, over 5 streaming batches of one walk-forward month, the
        symlog scale drifted 592k -> 1.15M (+94%), so later batches trained on a materially
        different world than the one the frame was locked on, and on some months the drift ran
        until the sweep went NaN. Single-batch runs never rewind anything (`run == 0`), so every
        Simulation_Batches=1 job is bit-identical.

        `Observed_Scenario` (walk-forward backtest): a driver prepares grid-aligned realized paths
        (an .npz keyed by factor name) and the simulated draw is replaced by the observed one - the
        deal pricers then produce the realized tradable / liability marks the stepper replays. All
        preparation (archive read, interpolation, state source) lives in the driver; here we only
        substitute and let each process reseed its own state.

        The declared underlying(s) are leafed so the base-delta / conditional-feature pass can read
        d(value)/d(spot) via AAD. The diff-ML solver differentiates the continuation inside the
        inner MC off its own fresh state-at-t leaves (see Bundle.inner_mc_grad), so it needs no
        outer leaf."""
        # the declaration is the single source of an omitted field's default
        params = declared_defaults(type(self), params)
        # `.get` with no fallback on purpose: HedgeMonteCarlo declares no `Greeks` field and has no
        # default to publish for one - it only refuses the value that would silently do nothing
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
        # keep the mtm
        self.params['Keep_Tensor'] = 'Yes'

        logging.root.name = self.config.deals['Attributes'].get('Reference', self.config.file_ref)
        self.calc_stats['Batch_Size'] = self.batch_size
        self.calc_stats['Simulation_Batches'] = params['Simulation_Batches']
        self.calc_stats['Random_Seed'] = params['Random_Seed']

        execution_mode = params.get('Execution_Mode', 'simulate_only')
        hedging_problem = params.get('Hedging_Problem', {})
        # The hedging-problem cfg is forwarded to each process's `reseed_inner_state` OPAQUELY —
        # the calc never reads a model switch out of it; a process owns which keys mean what to it.
        self._inner_state_opts = hedging_problem

        instruments = self.config.deals_from_object_map(hedging_problem.get('Tradable_Instruments', {}))
        liabilities = self.config.deals_from_object_map(hedging_problem.get('Liabilities', {}))
        # store it away for deal resolution
        self.config.set_calculation_children(instruments + liabilities)
        # cap the grid at the liability terminal - past it there is nothing to hedge
        end_date = DealStructure.max_settlement_date(liabilities, self.config.holidays)
        shared_mem = self.update_factors(params, base_date, job_id, num_jobs, end_date=end_date)
        # Build the valuation structure first; the hedging runtime will consume
        # the live factor and instrument tensors produced by this same loop.
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
        # Canonical underlying (commodity-spot) factor-name set, derived once by the runtime
        # from the live CommodityPrice factors. Read by the spot-leaf pass and `_find_spot_key`
        # (mapping back to the live key object via stoch_factors) — no divergent re-derivation.
        self._underlying_names = set(normalized_runtime['referenced_commodities'])

        # Inner-MC setup; the copies below are forked only after outer setup precalc has run
        inner_mc_enabled = params.get('Inner_MC_Enabled', 'No') == 'Yes'
        tradable_refs = sorted(normalized_runtime['names']['hedges']) if inner_mc_enabled else ()

        # solve_hedge: inner MC runs in the backward DP/MPC sweep, not the outer loop.
        # Cache the per-batch outer scenario buffer so inner MC can fork on demand later.
        solve_hedge_mode = str(execution_mode).lower() == 'solve_hedge'
        if inner_mc_enabled:
            self.stoch_factors_inner = {k: proc.copy() for k, proc in self.stoch_factors.items()}
        # A SOLVE IS A STREAM: one Bundle per batch, handed to a persistent solver in-loop
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
        # Static liability descriptors (read off the cashflow schedules, batch-independent).
        total_leg_volume, last_payment_day = self._liability_schedule_scalars()
        # get the calendar for business day
        bus_day = self.config.holidays.get(
            self.params['Calendar'], {'businessday': pd.offsets.BDay(1)})['businessday']
        # per-batch burn-in: variance in z_0 for the diff-ML boundary label
        randomize_t0 = hedging_problem.get('Randomize_Initial_State', 'No') == 'Yes'
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
                    # Each process owns its t=0 seed for the next run (regime / variance / none);
                    # captured now (detached) so it survives the buffer-clearing reset below.
                    outer_reseeds.update(proc.outer_reseed())
                # Independent innovation stream for the main run (regenerates
                # t_random_numbers via torch.randn; quasi-rng auto-advances).
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

            for key, proc in self.stoch_factors.items():
                simulated = proc.generate(shared_mem)
                if observed is not None and key in observed:
                    # Driver supplies a dense daily path from the base date; take the
                    # sim-grid-length prefix (Time_Grid is daily) and broadcast to the batch.
                    simulated = observed[key][:simulated.shape[0]].expand(
                        *simulated.shape).contiguous()
                    # The process re-derives & publishes its own path-dependent revealed state
                    # along the replayed path (belief, log-variance, …); base processes no-op.
                    proc.reseed_from_path(simulated, shared_mem)
                # leaf the declared underlying(s) for the base-delta / conditional-feature pass
                if utils.check_tuple_name(key) in self._underlying_names:
                    simulated = simulated.detach().requires_grad_(True)
                shared_mem.t_Scenario_Buffer[key] = simulated
                # Each process owns its privileged-factor surface; ask it what to expose. Default
                # implementation returns {} so processes opt in by overriding privileged_factors.
                priv = proc.privileged_factors(simulated)
                if priv:
                    factor_name = key.name[0] if key.name else str(key)
                    for attr_name, tensor in priv.items():
                        privileged_factor_blocks[(factor_name, attr_name)].append(tensor.detach().clone())

            # solve_hedge: snapshot this batch's outer scenario buffer (factor paths + every
            # per-process aux key each generate() published) — the forks run against THIS batch.
            batch_buffer = ({key: tensor.detach().clone()
                             for key, tensor in shared_mem.t_Scenario_Buffer.items()}
                            if solve_hedge_mode else None)

            _ = self.netting_sets.resolve_structure(shared_mem, self.time_grid)
            # clear hedge cashflows so t_Cashflows after the next call holds only liability cashflows
            shared_mem.reset_cashflows(self.time_grid)
            # grab the liability mark — post-process-free (no per-batch GPU->CPU save_results copy).
            # The feature tensor is gone; the symlog scale's two static descriptors come from the
            # cashflow schedule (`_liability_schedule_scalars`), not a per-batch leg pass.
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

            # grab the simulated instruments and collect them into a generic bundle
            trade_tensors = self.netting_sets.tensor_marks()

            if tradable_blocks is not None:
                for instrument_name, instrument_tensor in trade_tensors.items():
                    tradable_blocks[instrument_name].append(instrument_tensor.detach().clone())

            shared_mem.t_Buffer.clear()

            if solve_hedge_mode:
                # This batch IS a bundle: build it, attach its own forks, and hand it to the
                # persistent solver. The last batch is reserved — never fitted — as the held-out
                # world `finish` measures the verdict and the benchmark tracks on.
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
                # The LAST batch is the held-out world. At Simulation_Batches == 1 — a frozen
                # policy, which fits nothing — warmup's batch is also that world.
                if run == params['Simulation_Batches'] - 1:
                    held_out = bundle
                # Fresh accumulators for the next batch — this one has been consumed. (The forks
                # the solver just ran borrowed `shared_mem`; `_run_inner_mc_at_t` hands it back as
                # it found it, so outer generation resumes unaffected.)
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

    # ------------------------------------------------------------------
    # Inner-MC subsystem
    #
    # Parallel to the outer-loop body inlined in `execute()`. Forks the simulator
    # from each outer-path state at each outer timestep, runs inner MC to terminal
    # under `no_grad`, reduces to conditional features per outer path. Outer process
    # instances are not touched — inner uses shallow copies (`StochasticProcess.copy`)
    # so per-instance precalc state (spot0, scenario_horizon, z_offset, ...) doesn't
    # bleed across the outer/inner boundary.
    # ------------------------------------------------------------------

    def _find_spot_key(self):
        """Return the unique underlying (commodity-spot) factor key. Its name comes from the
        runtime-owned underlying set (`self._underlying_names`); map back to the live key
        object via stoch_factors. The sufficient statistic (HMM regime/belief, GARCH log-
        variance) lives on the martingale primary — prefer the spot exposing a revealed
        sufficient statistic (non-empty `privileged_layout`) when more than one CommodityPrice
        factor is simulated. Raises unless exactly one."""
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
        """Build a fresh DealStructure mirroring outer_struct but with each deal's
        Time_dep restricted to events at mtm positions >= cutoff_mtm_idx via
        `DealTimeDependencies.copy_restricted` — or, when `window_end_idx` is given,
        to the window [cutoff_mtm_idx, window_end_idx] via `copy_window` (the one-step
        fork prices at exactly {t, t+1}). Factor_dep is shared by reference
        (static factor lookups, time-grid-independent for the deal types used in
        inner-MC); Calc_res is fresh so inner pricing doesn't clobber outer storage.
        Returns a DealStructure with possibly fewer dependencies (deals fully in
        the past are dropped). Does not recurse into sub_structures — inner-MC use
        case has a flat dependency list.

        AGGREGATION STORAGE IS OFF (`store_results=False`) while the per-deal `Calc_res`
        below stays. The fork harvests on the DEVICE — `tensor_marks()` for the tradables
        (which needs the per-deal dict, since `pricing.interpolate` stashes 'tensor' there)
        and `resolve_hedge_structure()` for the liability — and nothing reads the
        aggregate's stored 'Value'. Storing it cost a pageable D2H copy of the FULL-width
        mtm grid (127 x B_outer*B_inner fp32 = 16.6 MB on the wf-gate world) per fork, 93%
        of the fork's host egress, for a number the fork discards."""
        inner = DealStructure(outer_struct.obj.Instrument, store_results=False)
        for dd in outer_struct.dependencies:
            new_td = (dd.Time_dep.copy_restricted(cutoff_mtm_idx) if window_end_idx is None
                      else dd.Time_dep.copy_window(cutoff_mtm_idx, window_end_idx))
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
        """Attach the on-demand inner-MC forks to a bundle. The closures let the solver fork
        without a calc handle — they capture `self` (the inner-MC machinery), the outer scenario
        snapshot they fork FROM, and shared_mem. Every fork is windowed to {t, t+1}: 2-row
        generation AND a real 2-row pricing pass, giving exact per-tradable F_t1 and exact
        L_t/L_t1 — the only fields the diff-ML bootstrap reads. `outer_rows` lets the solver run
        the GRAD fork in outer-path sub-slices at large B_outer (per-slice tapes; the tape covers
        2 rows, so the flat cap binds instead of the cells cap and slices get wide).

        One bundle per batch under streaming, so the buffer is passed in rather than read off the
        calc: each bundle forks from ITS OWN batch."""
        bundle.inner_mc = lambda t: self._run_inner_mc_at_t(
            t, outer_buffer, shared_mem, base_date, tradable_refs)
        bundle.inner_mc_grad = lambda t, outer_rows=None: self._run_inner_mc_at_t(
            t, outer_buffer, shared_mem, base_date, tradable_refs,
            with_grad=True, outer_rows=outer_rows)

    def _run_inner_mc_at_t(self, t, outer_scenario_buffer, shared_mem, base_date,
                           tradable_refs, with_grad=False, outer_rows=None):
        """Run inner MC at a single outer timestep `t`, forking from `outer_scenario_buffer`
        — a snapshot of the outer `t_Scenario_Buffer` (factor keys plus every per-process aux
        key, batch dim B_outer).

        `outer_rows=(lo, hi)` forks only that contiguous outer-path range — the row window a
        caller uses to bound the AAD tape (labels are per-outer-path, so row slices are
        independent). The solver no longer needs it: its forks are single-pass at Batch_Size.

        The DP/MPC backward sweep calls this on demand outside the outer loop (via the
        closure `Bundle.inner_mc`), forking inner MC at the requested `t`.

        Returns the inner samples the solver bootstraps from:
            F_t1          {ref: (B_outer, B_inner)}      futures price at outer t+1
            L_t, L_t1     (B_outer, B_inner)             liability MTM at outer t and t+1
            market_t1     (B_outer, B_inner, market_dim)  inner market state at outer t+1
            market_t      (B_outer, market_dim)           outer-realised market state at t
            t, cutoff_idx
        `market_t`/`market_t1` are every simulated factor's revealed segments concatenated —
        the column block is generic (each process owns its packing via `reveal_state_at`),
        so the DP/MPC solvers consume it without knowing what the factors are.

        SINGLE PASS. The whole fork — generation, stuffing, pricing, extraction — runs at
        `B_outer x Inner_Sub_Batch` flat in one go, so peak memory is a function of two JSON
        fields (`Batch_Size`, `Inner_Sub_Batch`) and nothing else. A config too wide for the
        card raises CUDA OOM naming this fork; that is the contract, not a knob. The old
        outer-path chunk loop is gone: the stream caps fork width at
        Batch_Size rather than the whole simulation — see the doc attr for the measured
        operating point.

        THE WINDOW. The inner grid is truncated to {t, t+1} and every deal's Time_dep windowed to
        those two rows (`copy_window` via `window_end_idx`), so the pricing chain runs for real on
        a 2-row grid - exact per-tradable F_t1 and liability L_t / L_t1, which is every field the
        diff-ML bootstrap reads, and correct on mixed strips (each future prices its own
        basis+carry; the old market-only short-circuit broadcast SPOT as every F_t1). Restricting
        the AAD tape to a single forward step is what keeps its memory bounded - a full t->T_dec
        horizon multiplies tape and pricing by the remaining rows for no use. The scenario buffer
        only reaches row t+1, and the windowed Time_dep rebuilds `interp` up to its last kept
        event, so nothing indexes past it.

        TWO COORDINATE SYSTEMS: processes generate against the shifted-base `inner_time_grid`;
        pricers run against the full outer `self.time_grid` with each deal's Time_dep restricted
        via `copy_restricted`. Buffer stuffing prepends the outer-realized past (broadcast across
        B_inner) so path-dependent payoffs see the realized fixings. That past is a slice of the
        outer snapshot, already resident at B_outer, and every one of its rows is identical across
        the B_inner draws - the `cat` that used to join them wrote it out B_inner times (98% of the
        stuffed buffer at 1280x64, dragging a same-shaped slab of Hermite g,c with it), so it is
        published as its own `ScenarioBlock` carrying the `past_columns` index instead.
        `ScenarioSource` is the same sequence-of-row-blocks the outer loop publishes with one
        block, so the pricer reads both through one mechanism; a fork at t=0 has no past and
        publishes one block. EVERY path series goes through that publication, not every factor: a
        process's own `(key, kind)` series is read through the same seam as a factor (a pricer
        cannot tell them apart), so one the outer snapshot also carries is per-path state that
        forked with the path and gets the same logical grid. The fork's own seeds are excluded by
        that rule rather than by a name test - the outer path does not carry them.

        PER-PROCESS HOOKS, no isinstance branch anywhere - a factor without a revealed sufficient
        statistic returns an empty dict and the forker's single uniform loop covers every model
        world. `inner_fork_seed` supplies the per-outer-path t=0 privileged-state seed the inner
        generate reads (regime for the HMM, conditional variance h0 for GARCH).
        `reseed_inner_state` restores post-generate coherence: the process publishes whatever
        path-dependent revealed state its `reveal_state_at` needs at t+1 (e.g. a filtered belief)
        and returns differentiable leaves for the twin loss; `self._inner_state_opts` is forwarded
        to it opaquely, and base / GARCH processes are no-ops because their revealed state is
        already published by generate, or detached by design. `reveal_state_at` then yields each
        factor's informative segments from the live buffer (factor path plus any aux just
        published); this method owns the (factor_flat, B, SB) reshape and concatenates in reveal
        order.

        `L_t` / `L_t1` are the `resolve_hedge_structure` marks themselves, time-indexed exactly as
        F_t1 (mtm[cutoff_idx:][0] is outer-t, [1] is outer-t+1). They replace the Jacobian
        linearization of the liability in the diff-ML one-step bootstrap, so the bootstrap value
        marks the liability EXACTLY at each inner draw.

        Under `with_grad`, `state_t_leaf_widths` pairs each leaf with the market_t column width it
        occupies: the differential-label projection in the diff-ML solver needs this to write
        per-leaf gradients into the right deep-state columns without re-deriving factor widths,
        which would silently drift if a process's `reveal_state_at` packing changed.

        FAIL LOUDLY, both halves. `_restricted_struct` drops fully-expired deals, so a tradable
        still in `dependencies` but missing from `tensor_marks()` can only mean its pricing was
        skipped; that reads downstream as F_t1 = 0, i.e. an expired contract, and the solver's
        `live` mask retires it from the hedge set - a wrong number, not a crash. The liability half
        is the same defect arriving through the canonical deal guard, which swallows exceptions
        (e.g. CUDA OOM) into a scalar-0 mark and so silently corrupts the solver's LABELS.

        The fork BORROWS `shared_mem` (`borrowed_batch` / `borrowed_fill`), so the `finally`
        restores on ANY exit: without it a mid-fork raise (CUDA OOM, a degenerate-pricing
        RuntimeError) left the state flat-sized and the NEXT t-step failed on shapes instead of the
        real cause. The Sobol sample cache is dropped there too - it is keyed by sample_size and
        would otherwise grow unbounded across t-steps. Each fork re-draws a fresh, independent
        quasi-MC stream (the engine advances); the pricer's per-pass `reset_qrg` caching is intact
        within a fork, only cleared between them."""
        spot_key = self._find_spot_key()
        if outer_rows is not None:
            lo, hi = outer_rows
            outer_scenario_buffer = {k: v[..., lo:hi] for k, v in outer_scenario_buffer.items()}
        B_outer = outer_scenario_buffer[spot_key].shape[-1]
        B_inner = shared_mem.simulation_sub_batch
        t_days = int(self.time_grid.scen_time_grid[t])
        inner_time_grid = self.time_grid.truncate_to(base_date, t_days)

        # Terminal / past-end — no inner horizon, so nothing to price. The DP sweep does not call
        # here at terminal (it uses the closed-form V_T); a caller querying `inner_mc` at or past
        # terminal does, hence the guard.
        if inner_time_grid.scen_time_grid.size < 2:
            return dict(t=t, cutoff_idx=t, L_T=None, market_t=None, market_t1=None, F_t1={})

        # In HedgeMonteCarlo scen_time_grid == mtm_time_grid (dynamic_scenario_dates),
        # so the same `t` indexes both the scenario buffer and the mtm grid.
        cutoff_idx = t
        inner_base_date = base_date + pd.Timedelta(days=t_days)

        # THE WINDOW: inner grid + every deal's Time_dep truncated to {t, t+1}, so the pricing
        # chain runs for real on 2 rows and the AAD tape stays bounded
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
        # Track per-process initial state leaves when with_grad — exposed via the result
        # dict so the caller can `.backward()` from any function of the inner outputs and
        # read `.grad` per process/per outer path.
        state_t_leaves = {} if with_grad else None
        # What the fork BORROWS, to give back exactly as found (see the finally below): a fork
        # over an outer-path SLICE would otherwise leave the state slice-sized, which is invisible
        # while forks only follow forks but corrupts the next outer batch under streaming.
        borrowed_batch, borrowed_fill = shared_mem.simulation_batch, shared_mem.fillvalue
        with grad_ctx:
            try:
                shared_mem.simulation_batch = B_outer
                shared_mem.reset_inner(self.num_factors, inner_time_grid,
                                       use_antithetic=self.params.get('Inner_Antithetic', 'No') == 'Yes',
                                       use_random=str(self.params.get('Inner_Draws', 'sobol')).lower() == 'random')

                market_t1_parts = []
                # The fork BORROWS this buffer, so an entry it does not rewrite is still the outer
                # run's - references only, read once by the publication below.
                outer_entries = dict(shared_mem.t_Scenario_Buffer)
                for key, proc_inner in self.stoch_factors_inner.items():
                    if key.type in utils.DimensionLessFactors:
                        continue
                    # Raw per-path init state for this factor's inner-MC precalculate fork. Never was
                    # type-specific: raw CommodityPrice `outer[t,:]`, raw ForwardRate/ForwardPrice/
                    # InterestRate `outer[t,:,:]` all equal `outer[key][t]`.
                    init_state = outer_scenario_buffer[key][t]
                    if with_grad:
                        # Leaf with grad: differentiates inner-sim + pricing back to state_t.
                        init_state = init_state.detach().clone().requires_grad_(True)
                        state_t_leaves[key] = init_state
                    proc_inner.precalculate(
                        inner_base_date, inner_time_grid,
                        init_state,
                        shared_mem, self.process_ofs[key],
                        implied_tensor=self._factor_precalc_args[key][1],
                    )
                    # per-outer-path t=0 privileged-state seed for this process's inner generate
                    for seed_key, seed_val in proc_inner.inner_fork_seed(key, outer_scenario_buffer, t).items():
                        shared_mem.t_Scenario_Buffer[seed_key] = seed_val
                    simulated = proc_inner.generate(shared_mem)
                    shared_mem.t_Scenario_Buffer[key] = simulated
                    # post-generate coherence: publish revealed state at t+1, return twin-loss leaves
                    inner_leaves = proc_inner.reseed_inner_state(
                        key, simulated, outer_scenario_buffer, t, shared_mem, self._inner_state_opts, with_grad)
                    if with_grad:
                        state_t_leaves.update(inner_leaves)
                    # market state at outer t+1 (inner-time index 1), in reveal order
                    for block, _kind in proc_inner.reveal_state_at(1, shared_mem.t_Scenario_Buffer):
                        market_t1_parts.append(block.reshape(-1, B_outer, B_inner))

                # publish past-then-forked rows; the past keeps ONE outer column per B_inner flat
                # columns, handed over as data rather than re-derived downstream
                past_columns = torch.arange(
                    B_flat, device=shared_mem.one.device) // B_inner
                # EVERY path series this fork WROTE and the outer path also carries - a factor's
                # grid and a process's own `(key, kind)` series alike, because a pricer reads both
                # through one seam and they cannot be on different logical grids. Each half of the
                # test excludes one thing: an outer entry the fork never rewrote (a dimensionless
                # factor, a burn-in seed), and this fork's own `<kind>_inner` seed.
                for key in [k for k, v in shared_mem.t_Scenario_Buffer.items()
                            if v is not outer_entries.get(k) and k in outer_scenario_buffer]:
                    inner_path = shared_mem.t_Scenario_Buffer[key]                  # (T_inner, ..., B, SB)
                    past = [utils.ScenarioBlock(outer_scenario_buffer[key][:cutoff_idx],
                                                batch_index=past_columns)] if cutoff_idx else []
                    shared_mem.t_Scenario_Buffer[key] = utils.ScenarioSource(
                        *past, utils.ScenarioBlock(
                            inner_path.reshape(*inner_path.shape[:-2], B_flat),
                            first_row=cutoff_idx))

                # Single-pass pricing — the chunk is sized so B_flat fits the memory budget.
                shared_mem.t_Buffer.clear()
                shared_mem.simulation_batch = B_flat
                # `fillvalue` is a batch-sized empty tensor frozen at State construction (the
                # energy-leg reset code uses it as the empty-cat fallback) — it must track the
                # current simulation_batch or cash_settle size-mismatches.
                shared_mem.fillvalue = shared_mem.one.new_zeros((0, 1, B_flat))
                # Per-chunk restricted DealStructures: same instruments + Factor_dep,
                # fresh Time_dep slicing off past events (windowed to [t, t+1] on the
                # one-step path), fresh Calc_res.
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
                    # The canonical deal guard swallows exceptions (e.g. CUDA OOM) into a scalar-0
                    # mark. Inside an inner fork that silently corrupts the solver's LABELS —
                    # fail loudly instead (the CRITICAL 'Deal skipped' log above names the cause).
                    raise RuntimeError(
                        f'inner-fork liability pricing degenerated (shape '
                        f'{tuple(mtm_flat.shape)}, expected (*, {B_outer * B_inner})) — a deal '
                        f'was skipped inside the fork; see the CRITICAL log above for the cause.')
                inner_mtm = mtm_flat.reshape(*mtm_flat.shape[:-1], B_outer, B_inner)

                def _fan(t):
                    # A STATIC tradable (a cash account off an unsimulated curve) marks with a
                    # 1-wide batch - path-independent is a legitimate mark, not a skip - so
                    # broadcast it over the fan-out rather than demanding the flat batch of it.
                    # Any OTHER width is a genuine shape error and expand fails loud.
                    if t.shape[-1] == B_outer * B_inner:
                        return t.reshape(*t.shape[:-1], B_outer, B_inner)
                    core = t.squeeze(-1) if t.shape[-1] == 1 else t
                    return core.reshape(*core.shape, 1, 1).expand(*core.shape, B_outer, B_inner)

                inner_trade_tensors = {
                    ref: _fan(t) for ref, t in inner_netting_sets.tensor_marks().items()}
                # The window prices {t, t+1}, so no terminal row exists to read: `L_T` is None
                # and the horizon stats are simply not produced.
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
                # Market state — every simulated factor's informative state (sufficient
                # statistic + price, carry curve, …) concatenated; factor order is the
                # `stoch_factors_inner` iteration order, identical for market_t/market_t1.
                market_t1 = torch.cat(market_t1_parts, dim=0).permute(1, 2, 0).contiguous()
                market_t_parts = []
                market_t_widths = []
                for key in self.stoch_factors_inner:
                    if key.type in utils.DimensionLessFactors:
                        continue
                    proc_inner = self.stoch_factors_inner[key]
                    width = 0
                    for block, _kind in proc_inner.reveal_state_at(t, outer_scenario_buffer):
                        b = block.reshape(-1, B_outer)
                        market_t_parts.append(b)
                        width += b.shape[0]
                    # Forward the process's differentiable-state-leaf suffixes to the solver's
                    # label projection, so it maps leaf grads → market columns with no model concept.
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
                    # pair each leaf with the market_t column width it occupies
                    result['state_t_leaves'] = state_t_leaves
                    result['state_t_leaf_widths'] = market_t_widths
            finally:
                # restore on ANY exit - a mid-fork raise must not leave the state flat-sized
                shared_mem.simulation_batch = borrowed_batch
                shared_mem.fillvalue = borrowed_fill
                shared_mem.t_Buffer.clear()
                shared_mem.t_Scenario_Buffer.clear()
                # drop the Sobol sample cache (keyed by sample_size; unbounded across t-steps)
                shared_mem.t_quasi_rng.clear()

        return result


def construct_calculation(calc_type, config, **kwargs):
    return globals().get(calc_type)(config, **kwargs)


if __name__ == '__main__':
    pass
