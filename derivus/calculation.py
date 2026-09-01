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


import time
import logging
import itertools
import pandas as pd
import numpy as np
import torch
from functools import reduce

from collections import namedtuple, defaultdict
from .riskfactors import construct_factor
from .stochasticprocess import REVEAL_CONTINUOUS, construct_process
from .instruments import get_fxrate_factor, get_survival_component, get_interest_factor, get_survival_factor
from .pricing import SensitivitiesEstimator
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
        self.dependencies = []
        self.sub_structures = []
        self.store_results = store_results

    @staticmethod
    def calc_time_dependency(base_date, deal, time_grid):
        """Return the deal's time dependency on `time_grid`, or None if it has expired."""
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
        """Compile `deal` into this structure. A structure's deals are netted off before
        the structure's own rules are applied.

        A compile failure is logged and the deal is SKIPPED, which is what lets a portfolio of
        thousands survive one deal it cannot bind. `utils.is_fatal_pricing_error` is the exception
        to that, and it is the same predicate `Deal.calculate` reads one layer down: a framework
        fault, or a schedule refused by name, must not become a deal that marks at nothing on a job
        that then reports success.
        """
        deal_time_dep = self.calc_time_dependency(base_date, deal, time_grid)
        if deal_time_dep is not None:
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
                if utils.is_fatal_pricing_error(e):
                    raise
                stats['Deals Skipped'] = stats.setdefault('Deals Skipped', 0) + 1

    def finalize_struct(self, base_date, time_grid):
        all_report_dates = [set(
            x.obj.Instrument.get_report_dates(time_grid, base_date)) for x in self.sub_structures]
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
                    utils.stamp_boundary_sets(shared, deal_mark, logging.root.name)
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
            finally:
                if mark is not None:
                    utils.claim_boundary_sets(shared, mark)
                    # whatever post_process itself registered (a margin call) has no deal to
                    # name it, so the structure does
                    utils.stamp_boundary_sets(shared, mark, logging.root.name)

        return accum

    def resolve_hedge_structure(self, shared, time_grid):
        """Accumulate the liability MTM across the structure and return `{'mtm': tensor}`.

        Unlike `resolve_structure` this skips `post_process`/`save_results` - and so the
        per-batch GPU->CPU copy of the mark - which is why the hedge sim loop and the inner
        MC use it for the liability."""
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
        liability-terminal horizon that caps the sim time grid. Each instrument is reset from
        its `field` first (idempotent), since the structure is not built yet when the horizon
        is needed."""
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

        self.calc_stats = {}
        self.params = {}
        self.gradient_index = None
        self.output = {}

    def execute(self, params):
        pass

    def factor_leaf(self, factor, current_val, requires_grad, offset=0.0):
        """Return the AAD leaf for `factor`, connected to a calibration where one exists.

        Every leaf is minted from a numpy array, which severs whatever produced those
        numbers: it does not raise, it reports a zero gradient. A curve - or one named
        parameter of a calibrated model - the library bootstrapped and kept the graph of is
        offered as `leaf + (theta - theta.detach())` instead: worth zero in the forward pass,
        derivative one, so the factor greek reported is the number it always was and dV/dq
        arrives in the same reverse sweep. A non-zero `offset` (Tenor_Offset) declines the
        attachment - a shifted curve is a different curve, and quote sensitivities are t0
        risk.

        Under `Quote_Propagation` the same seam changes the VALUE: the leaf is minted from
        the last artifact ridden to the quotes standing now, `theta* + dtheta/dq (q_now -
        q0)`, so a tick reaches a valuation without a re-solve. The ride is derived per call,
        never stored, and leaves the artifact id in `calc_stats['Calibrations']` - nothing
        else in the replay tuple distinguishes a ride from a refit. A `Tenor_Offset` REFUSES
        that ride rather than declining it: the shifted curve interpolates off coefficients
        fitted before the ride, so declining would silently price the stale curve.
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

    def set_deal_structures(self, deals, output, unit, deal_level_mtm=False):
        """Compile the deal tree. `unit` is the calculation's dtype/device anchor: a deal's
        schedules are BOUND to it as they compile, so the tensor half's birthday is this walk."""
        for node in deals:
            instrument = node['Instrument']
            if node.get('Ignore') == 'True':
                self.calc_stats['Ignored'] = self.calc_stats.setdefault('Ignored', 0) + 1
                continue

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


#: The quasi-random stream's fixed identity: the scramble seed of every Sobol engine and the
#: offset each one starts at. Historical values, and deliberately NOT derived from the job's
#: `Random_Seed` - which is why sharding raises a question about POSITION in this stream and not
#: about randomness, and why `batch_seed` below has no counterpart here.
QUASI_SEED = 1234
QUASI_ANCHOR = 1024


def batch_seed(random_seed, batch_index):
    """The seed a batch runs under: a 64-bit mix of (seed, global batch), not `seed + batch`.

    Deterministic sharding reseeds ONCE PER BATCH, so a 1024-batch job asks for 1024 seeds and
    consecutive integers are the one input a generator's initialization is least defensible on.
    CUDA's Philox is counter-based and key-independent by construction, so it does not care. CPU
    MT19937 is the gray case: its 624-word state is expanded from the seed by a linear recurrence,
    and near seeds producing near initial states is exactly the correlation the literature declines
    to rule out rather than one it endorses. A single SplitMix64 round costs three multiplies and
    closes it for both, so the question does not have to be answered per backend.

    ONE spelling, here, because a sharded run and an unsharded-but-batch-seeded run deriving the
    same batch's seed two ways would be a determinism bug that no gate comparing shard counts could
    see. Masked to 63 bits: `torch.manual_seed` takes a signed 64-bit value.
    """
    z = (random_seed + (batch_index + 1) * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (z ^ (z >> 31)) & 0x7FFFFFFFFFFFFFFF


class CMC_State(utils.Calculation_State):
    def __init__(self, cholesky, static_buffer, batch_size, one, mcmc_sims, report_currency,
                 seed, job_id, num_jobs, scale_survival=False, nomodel='Constant', keep_tensor=False):
        """Per-calculation Monte Carlo state: correlated random numbers, scenario buffers and
        the caches a batched exposure run needs on top of `Calculation_State`.

        `t_PreCalc` is the per-CALCULATION memo (`t_Buffer` is the per-batch one `reset`
        clears), so its presence is the marker pricers read for "exposure-based".
        `t_Bridge_Variance_Rate` holds each factor's annualized log-variance RATE, published
        once the processes are precalculated: a continuously monitored barrier crossing
        between two deal-grid dates is a conditional probability needing the SIMULATION
        variance of that interval, not a pricing implied vol.
        """
        super(CMC_State, self).__init__(
            static_buffer, one, mcmc_sims, report_currency, nomodel, batch_size, keep_tensor=keep_tensor)
        self.t_PreCalc = {}
        # decisions taken on simulated state, recorded forward so their derivative can be
        # restored before the reverse sweep. Per BATCH like t_Buffer - backward() runs once
        # per batch, so a correction assembled from another batch's graph is stale
        self.boundary_aad = False
        self.boundary_sets = []
        self.t_Bridge_Variance_Rate = {}
        self.t_cholesky = cholesky
        self.t_random_numbers = None
        self.t_Scenario_Buffer = {}
        self.t_quasi_rng = {}
        self.t_quasi_rng_batch = {}
        # Each dimension's Sobol engine and the absolute position it stands at. Tracking the
        # position is what lets one `quasi_rng` serve both arms: the historical one asks for where
        # it already is, an anchored one asks for its global batch's own place in the sequence.
        self.sobol_position = {}
        # The GLOBAL index of the batch being run, set by `set_quasi_batch` under deterministic
        # sharding only. None leaves `quasi_rng` indexing by this process's own draw count, which
        # is every unsharded caller.
        self.quasi_batch = None
        # seed each job by its offset
        torch.manual_seed(seed + job_id)
        self.job_id = job_id
        self.num_jobs = num_jobs
        self.scale_survival = scale_survival

    def quasi_rng(self, dimension, sample_size):
        """One quasi-random draw, memoized per `(dimension, sample_size, index)`.

        THE ONLY THING SHARDING CHANGES IS WHERE THE DRAW IS TAKEN FROM. The stream is a fixed
        scrambled Sobol sequence starting at a fixed offset, neither derived from the job's
        `Random_Seed`, so it is perfectly reproducible; what is not reproducible is POSITION. The
        historical arm indexes by THIS PROCESS's own draw count for the shape and reads wherever
        its engine has reached - a function of everything drawn before it here - so a worker
        holding only part of a job reads the points an earlier batch should have had. The anchored
        arm (`set_quasi_batch`, deterministic sharding only) indexes by the GLOBAL batch and reads
        `QUASI_ANCHOR + index * sample_size`, which depends on nothing local.

        The two agree on an unsharded run of one draw per batch per shape: the engine advances by
        `sample_size` a batch, landing exactly on the anchored position, which is why anchoring is
        a repositioning rather than a second model - `tests/test_multi_gpu.py` checks that
        arithmetic against `SobolEngine` directly. The historical arm nonetheless reads its
        engine's STANDING position rather than recomputing it from the index, because a dimension
        drawn at two different sample sizes shares one engine and interleaves: there the index
        arithmetic and the running engine part company, and the historical interleaving is what
        must survive byte for byte.

        The draw, the memo, the clamp and the icdf happen once, below, for both arms.
        """
        batch_key = (dimension, sample_size)
        call = self.t_quasi_rng_batch.setdefault(batch_key, 0)

        if self.quasi_batch is None:
            index = call
            position = self.sobol_position.get(dimension, (None, QUASI_ANCHOR))[1]
        else:
            if call:
                # a genuine second draw of one shape inside one batch, which has no distinct batch
                # position to take. The inner-MC replay idiom zeroes this counter through
                # `reset_qrg` first, so a re-read never lands here and still returns by identity.
                raise RuntimeError(
                    'deterministic sharding cannot cover a second draw of one quasi-random stream '
                    'inside a batch: (dimension={}, sample_size={}) was asked for a {} time while '
                    'running global batch {}, and anchoring gives a draw the position of its '
                    'BATCH, so two of them in one batch have no distinct position to take. Two '
                    'factors sharing a shape, or an inner Monte Carlo pricer drawing beside the '
                    'outer path, is the usual source - run such a job unsharded, or give the '
                    'second consumer its own dimension.'.format(
                        dimension, sample_size, call + 1, self.quasi_batch))
            index = self.quasi_batch
            position = QUASI_ANCHOR + index * sample_size

        sample_key = (dimension, sample_size, index)

        if sample_key not in self.t_quasi_rng:
            engine = self._sobol_at(dimension, position)
            sample_sobol = engine.draw(sample_size, dtype=self.one.dtype)
            self.sobol_position[dimension] = (engine, position + sample_size)
            margin = 1.0e-6
            u = sample_sobol.clamp(min=margin, max=1.0 - margin).to(self.one.device)
            self.t_quasi_rng[sample_key] = (utils.norm_icdf(u), u)

        self.t_quasi_rng_batch[batch_key] += 1
        return self.t_quasi_rng[sample_key]

    def _sobol_at(self, dimension, position):
        """This dimension's engine, standing at `position`.

        Seeks FORWARD and rebuilds to go backwards. The historical arm only ever asks for the
        position it already stands at, so it never seeks and never rebuilds - it is the same
        engine advancing draw after draw that it always was. A sharded worker pays the jump to the
        start of its own slice once and then advances the same way.
        """
        engine, standing = self.sobol_position.get(dimension, (None, QUASI_ANCHOR))
        if engine is None or position < standing:
            engine = torch.quasirandom.SobolEngine(
                dimension=dimension, scramble=True, seed=QUASI_SEED)
            engine.fast_forward(position)
        elif position > standing:
            engine.fast_forward(position - standing)
        return engine

    def reset_qrg(self):
        self.t_quasi_rng_batch = {}

    def set_quasi_batch(self, batch_num):
        """Anchor the quasi stream to a GLOBAL batch index (deterministic sharding only).

        Resets the per-`(dimension, sample_size)` counter along with it, because under anchoring
        that counter means "draws so far WITHIN this batch" rather than "draws so far in this
        process" - which is what makes a repeated draw detectable instead of silently aliasing
        onto the position belonging to the next batch.

        AND DROPS THE PREVIOUS BATCH'S MEMO, which anchoring is what makes safe. The memo exists so
        a re-read returns the drawn tensor rather than an equal one, and on the historical arm it
        has to live for the whole run: that arm's position is wherever its engine has crawled to,
        so an entry dropped there could never be recovered. Anchored, a draw's position is derived
        entirely from its key, so re-drawing is bit-identical and the memo is a WITHIN-batch
        convenience with nothing to say across batches. Run-long it grew without bound - measured
        at 20,480 bytes a batch on a two-state HMM - holding entries nothing would read again.

        Cleared at the START of a batch, so the within-batch replay idiom is untouched: the entries
        a batch accumulates stand until that batch is done with them.
        """
        self.quasi_batch = batch_num
        self.t_quasi_rng_batch = {}
        self.t_quasi_rng = {}

    def reset_cashflows(self, time_grid):
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
    """Inner-MC variant of `CMC_State` for nested simulation: each of `simulation_batch`
    outer-path states fans out into `simulation_sub_batch` (B2) independent forward paths.

    `reset()` is inherited unchanged, so outer-mode use of this state is transparent;
    `reset_inner()` swaps in random numbers shaped
    `(num_factors, T, simulation_batch, simulation_sub_batch)` in place of the base
    `(num_factors, T, simulation_batch)`. Stochastic processes dispatch on `Z.ndim` to pick
    the outer or the inner path."""

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
        `(num_factors, T, simulation_batch, simulation_sub_batch)`, and clear the per-batch
        caches.

        `use_random` (`Inner_Draws='random'`) swaps the shared Sobol tensor for iid
        Gaussians: one low-discrepancy stream strided across (T,B,B2) loses its uniformity
        on the per-(t,b) B2-slices as B grows (measured label/argmax degradation at B=512),
        while iid draws keep per-fork label noise B-independent.

        `use_antithetic` (`Inner_Antithetic='Yes'`) draws B2/2 quasi-normals per (t, outer
        path) and mirrors them (z, -z), halving the label/argmax variance of the inner-MC
        E[C] estimate. It stays unbiased because the folded emissions are symmetric in z;
        auxiliary streams (e.g. a discrete-state transition) draw separately and stay iid.
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
        self.jacobians = {}
        self.implied_factors = {}
        self.implied_var = {}

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
        """Construct the factor objects, tensors and shared memory, and precalculate the
        stochastic processes.

        Called by `update_factors` once the time grid and dependency sets are known;
        subclasses that build their own dependency sets can call it directly.

        IMPLIED-LEAF INVARIANT: a factor can be BOTH a static dependent factor and a spot
        process's implied factor. Minting a second leaf under the scope name the implied
        leaf already owns would split the gradient - the pricer (`t_Static_Buffer`) and the
        scenario path (`implied_tensor`) would read different tensors - so the single implied
        leaf is reused and `backward()` sums both consumers into it.

        `_factor_precalc_args` caches each factor's `(ScenarioTimeGrid, implied_tensor)` so a
        consumer needing a per-path initial state can precalculate again without re-deriving
        the dependency and time-grid plumbing.
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
                    factor_tenor_offset = utils.get_day_count_accrual(
                        base_date, tenor_offset, value.factor.get_day_count() if hasattr(
                            value.factor, 'get_day_count') else utils.DAYCOUNT_ACT365)
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
                    factor_tenor_offset = utils.get_day_count_accrual(
                        base_date, tenor_offset, value.get_day_count() if hasattr(
                            value, 'get_day_count') else utils.DAYCOUNT_ACT365)
                else:
                    factor_tenor_offset = 0.0
                current_val = value.current_value(offset=factor_tenor_offset)
                if isinstance(current_val, dict):
                    for k, v in current_val.items():
                        fkey = utils.Factor(key.type, key.name + (k,))
                        self.static_var[fkey] = implied_leaves[fkey] if fkey in implied_leaves else \
                            self.factor_leaf(fkey, v, calc_grad, factor_tenor_offset)
                else:
                    self.static_var[key] = implied_leaves[key] if key in implied_leaves else \
                        self.factor_leaf(key, current_val, calc_grad, factor_tenor_offset)

        shared_mem = self._init_shared_mem(
            int(params['Random_Seed']), params['NoModel'],
            params['Currency'], params['MCMC_Simulations'],
            job_id, num_jobs, calc_greeks=sensitivities if greeks else None)

        self.all_tenors = utils.update_tenors(self.base_date, self.all_factors)

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

        for key, value in self.stoch_factors.items():
            if key.type not in utils.DimensionLessFactors:
                value.calc_references(key, self.static_factors, self.stoch_factors, self.all_tenors, self.all_factors)

        return shared_mem

    def update_time_grid(self, base_date, reset_dates, settlement_currencies, dynamic_scenario_dates=False):
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

    def _init_shared_mem(self, seed, nomodel, reporting_currency, mcmc_sim, job_id, num_jobs, calc_greeks=None):
        """Allocate the `CMC_State` for this run (correlation cholesky, static buffer,
        reporting FX) and, when greeks are requested, build the flat AAD variable index over
        `calc_greeks`.

        `boundary_aad` has no JSON switch: wanting sensitivities IS the switch. The
        correction is worth exactly zero in the forward pass, so it can only ever change a
        derivative, and recording events nobody differentiates would just hold memory across
        a batch.
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
            seed, job_id, num_jobs, scale_by_survival, nomodel=self.params.get('NoModel', 'Constant'),
            keep_tensor=self.params.get('Keep_Tensor', 'No') == 'Yes')
        shared_mem.boundary_aad = calc_greeks is not None
        shared_mem.recompute_inner_mc = self.params.get('Recompute_Inner_MC', 'No') == 'Yes'
        return shared_mem

    def report(self, output):
        for result, data in output.items():
            if result == 'scenarios':
                scen = {}
                scenario_date_index = pd.DatetimeIndex(sorted(self.time_grid.scenario_dates))
                if self.params['Calc_Scenarios'] == 'At_Percentile':
                    dates = np.array(sorted(self.time_grid.mtm_dates))[self.time_grid.report_index]
                    mtms = pd.DataFrame(np.concatenate(output['mtm'], axis=-1).astype(np.float64), index=dates)
                    percentiles = self.params.get('Percentile', '95').replace(' ', '').split(',')
                    profiles = {x: np.percentile(mtms.values, float(x), axis=1) for x in percentiles}
                    index = {x: np.argmin(np.abs(mtms.values - profiles[x][:, np.newaxis]), axis=1) for x in percentiles}

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

    def execute(self, params, job_id=0, num_jobs=1, deterministic_batches=False):
        """Run the batched exposure simulation plus whichever sub-calculations `params`
        enables (collateral, initial margin, CVA, FVA, scenarios, cashflows) and return the
        netting sets, stats and reports.

        DETERMINISTIC BATCHES (`deterministic_batches`, set only by the `runparallel` dispatch).
        Off - the default, and every unsharded caller - the stream is seeded ONCE per worker in
        `CMC_State.__init__` as `manual_seed(seed + job_id)` and the batch loop then consumes it
        sequentially, so batch b's draws depend on how many batches ran before it in this process.
        That is fine for one worker and fatal for n of them: the same job sharded two ways covers
        different paths, and the answer moves with the device count.

        On, each batch seeds its OWN stream from its GLOBAL index b - `batch_seed(Random_Seed, b)`,
        a SplitMix64 mix rather than `Random_Seed + b`, because reseeding per batch asks for as
        many seeds as there are batches and consecutive ones are the weakest thing to hand a
        generator's initialization - where b counts from the start of the WHOLE job rather than
        from the start of this worker's slice. Workers take contiguous ranges - worker j owns `[j*k, (j+1)*k)` for k batches each -
        so batch b is the same batch, drawing the same numbers, whichever worker runs it and
        whichever device that worker sits on. The job is then deterministic in n: sharding it
        across two devices and running it on one produce the same batches in the same order, and
        the pooled result is bit-identical rather than merely close.

        BOTH STREAMS ARE ANCHORED, not just the generator. A world whose outer path draws quasi-
        random numbers (`MarkovHMMSpotModel`, `GARCHSpotModel`, `BasisLinkedSpotModel`) reads a
        Sobol sequence that is perfectly reproducible but POSITION-dependent: the historical path
        advances one engine through every draw before it in this process, so a worker starting
        part-way through the job reads the points an earlier batch should have had.
        `set_quasi_batch` gives the quasi stream the same global index the generator gets, and the
        anchored arm of `CMC_State.quasi_rng` then takes batch b's draw from absolute position
        `1024 + b * sample_size` however many draws the asking worker has already made. On an
        unsharded run that is the position the historical engine already stands at, which is why
        the default path is untouched. That position derives from the BATCH INDEX and the historical
        scramble seed 1234, never from `Random_Seed`, so the consecutive-seed question `batch_seed`
        answers for the generator does not arise for the quasi stream at all. The one shape anchoring cannot carry - two draws of one
        `(dimension, sample_size)` inside a single batch, which have no distinct batch position
        between them - refuses by name there.

        The CVA reduction keeps its grouping deliberately: a per-path vector cannot be
        reduced back to `mean over paths of a sum over time` in the same float order, and the
        reported number must not move by an ULP. `pricing.cva_per_scenario` is the same
        quantity for the counterfactuals, where only internal consistency matters.

        BOUNDARY AAD (CVA gradient): a hard transfer decision contributes a derivative the
        frozen-decision graph does not carry. The correction is worth exactly zero in the
        forward pass, so the reported `cva` is untouched and only the differentiated scalar
        gains a term. `Boundary_AAD_Bandwidth` defaults to 0.01, which needs roughly 32768
        paths to populate the near-boundary band; a thinner run should widen it and expect
        bias rather than noise.

        THE CVA HESSIAN (`Hessian: 'Yes'`) rides the reported trapezoid through
        `pricing.exposure_kink_term`, so the cva, the profile and grad_cva are untouched by
        construction. What it restores is the `delta(V) V_theta V_theta^T` term the double
        backward drops at the exposure relu, without which a LINEAR book reports a spot gamma
        of exactly zero. Two things refuse rather than report a plausible wrong matrix: a
        book that registered a boundary correction (a first-order estimator differentiated
        twice loses its flux block silently) and a reporting row whose bandwidth LADDER
        diverges. A row merely pinned at zero is neither, and keeps its second-order block.
        """
        # the declaration is the single source of an omitted field's default
        params = declared_defaults(type(self), params)
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
                # batch b owns its streams, keyed by its GLOBAL index - so neither which worker
                # runs it nor how many batches preceded it in this process can reach the numbers.
                # BOTH streams: the torch generator is reseeded, and the quasi stream, which is
                # reproducible but position-dependent, is repositioned onto the same batch.
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
                    # FVA carries its own objective - the CVA section is deleted by the
                    # shipped batch job, so the correction assembled there cannot fire here
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
                            'second-order block on a book without decision products. The estimator '
                            'that will answer this is the conditional-p mixture on the roadmap '
                            '(Second-order flux at a JUMP), pinned to the stride.'.format(
                                ', '.join(sorted({str(b.deal) for b in shared_mem.boundary_sets}))))
                    cva_for_aad = tensors['cva']
                    if shared_mem.boundary_sets:
                        objective = lambda mtm: pricing.cva_per_scenario(
                            torch.relu(mtm * fx_report * Dt_T) / fx_report[0], prob, recovery)
                        correction = pricing.boundary_correction(
                            shared_mem, objective, tensors['mtm'],
                            float(params.get('Boundary_AAD_Bandwidth', 0.01)))
                        if correction is not None:
                            cva_for_aad = cva_for_aad + correction
                    if hessian:
                        # the relu's argument above, SIGNED - nothing here is built unless
                        # second order is asked for
                        kink = pricing.exposure_kink_term(
                            tensors['mtm'] * fx_report * Dt_T / fx_report[0])
                        # mirrors the reported reduction exactly: same trapezoid, same prob, same
                        # recovery, so the (exact-zero) term rides the objective's own weights
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
                dates = np.array(sorted(self.time_grid.mtm_dates))
                for currency, values in shared_mem.t_Cashflows.items():
                    cash_index = dates[sorted(values.keys())]
                    output.setdefault('cashflows', {}).setdefault(currency, []).append(
                        pd.DataFrame(
                            [v.cpu().detach().numpy() for _, v in sorted(values.items())], index=cash_index))

            if self.params.get('Calc_Scenarios', 'No') != 'No':
                for key, value in self.stoch_factors.items():
                    output.setdefault('scenarios', {}).setdefault(key, []).append(
                        shared_mem.t_Scenario_Buffer[key].cpu().detach().numpy())

        self.calc_stats[execution_label] = time.monotonic() - self.calc_stats[execution_label]

        results = {'Netting': self.netting_sets, 'Stats': self.calc_stats, 'Jacobians': self.jacobians}
        results['Results'] = self.report(output)

        return results


class Base_Reval_State(utils.Calculation_State):
    def __init__(self, static_buffer, one, mcmc_sims, report_currency, calc_greeks, gamma, nomodel='Constant'):
        """Single-date, single-scenario valuation state (base MtM and its greeks).

        `boundary_aad` follows the same contract as `CMC_State`: a decision taken on
        simulated state is recorded during the forward pass so its derivative can be restored
        before the reverse sweep. Base valuation has one date and one scenario, but a Monte
        Carlo pricer still runs a full inner simulation underneath it - which is where a
        TARF's knock-in is decided - so the defect and the estimator are the same ones and
        only the objective is simpler.
        """
        super(Base_Reval_State, self).__init__(
            static_buffer, one, mcmc_sims, report_currency, nomodel, 1, False)
        self.calc_greeks = calc_greeks
        self.gamma = gamma
        self.boundary_aad = calc_greeks is not None
        self.boundary_sets = []

    @staticmethod
    def save_results(output, tensors):
        for k, v in tensors.items():
            output[k] = np.float64(v) if isinstance(v, float) else v.detach().cpu().numpy().astype(np.float64)


class Base_Revaluation(Calculation):
    """Simple deal revaluation - use this to reconcile with the source system.

    SECOND DERIVATIVES LIVE HERE AND NOWHERE ELSE. `Greeks: 'All'` asks the reverse sweep
    for `create_graph` and reports the result as `Greeks_Second` beside the first-order
    `Greeks_First`, which always accompanies it because the row labels are built off it.

    The shape is the FULL Hessian rather than a Hessian-vector product, because the number
    is a cross-gamma REPORT (spot-vol, spot-curve) that no caller arrives at with a
    direction to contract along. One date, one scenario and P = the factor knots the
    portfolio depends on keep that affordable - `report_hessian` runs P double-backward
    passes, measured at 0.9x to 10x the first-order pass. An exposure-sized P would flip
    that argument, and exposure does not come here: `Credit_Monte_Carlo` has its own
    CVA-Hessian route.

    The frame is the Hessian's SUPPORT: identically-zero rows and columns are dropped, so
    the matrix is square and symmetric but smaller than P, indexed on both axes by
    (Rate, Tenor, Tenor2, Tenor3) with the reporting reference as the columns' outer level.

    Two things refuse rather than report a plausible wrong number: a deal that registered a
    `BoundarySet` (`execute` below) and `Recompute_Inner_MC`
    (`pricing.InnerMCRecompute.backward`). `Branch_And_Weight` is the answer to the first
    and is declared here and nowhere else - the smooth estimator removes the decision
    instead of correcting it, so a deal priced under the switch registers nothing and the
    full Hessian flows. `Credit_Monte_Carlo` does not declare the field, so its exposure,
    cashflow and collateral semantics are structurally untouched.
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
        F('MCMC_Simulations', 'Integer', default=4096 * 8),
        F('Random_Seed', 'Integer', default=5120),
        F('Greeks', 'Text', default='No', values=['All', 'First', 'No'],
          description='First order factor sensitivities, or `All` for the second order block '
                      '(`Greeks_Second`) as well - see the class docstring for its shape'),
        F('Boundary_AAD_Bandwidth', 'Float', default=0.01,
          description='Kernel bandwidth of the boundary correction assembled into backward()'),
        F('Recompute_Inner_MC', 'Text', default='No', values=['Yes', 'No'],
          description='Re-simulate a Monte Carlo pricer\'s inner paths in backward() rather than '
                      'taping them; trades a second forward pass for the graph of every pricing'),
        F('Branch_And_Weight', 'Text', default='No', values=['Yes', 'No'],
          description='Price fixing-observed knockouts (TARF, accumulator, discrete barrier, '
                      'autocall) with the SMOOTH estimator: the fired branch of each fixing '
                      'integrated analytically against that interval\'s own lognormal law and the '
                      'continuing branch drawn from the truncated one. Same expectation, lower '
                      'variance, and no indicator on the tape - so second-order greeks flow where '
                      'the crisp estimator has to refuse them. GBM only; a non-GBM spot model '
                      'refuses by name (`pricing.branch_and_weight`), as does an AVERAGING '
                      'autocall, whose conditioning law is the distribution of a mean of spots '
                      'rather than one fixing interval\'s. Off is the crisp path bit for bit, and '
                      'on it is a RE-ESTIMATION of the same deal - it changes which estimator '
                      'prices a settlement convention, never which convention the deal settles on')
    ]

    def __init__(self, config, **kwargs):
        super(Base_Revaluation, self).__init__(config, **kwargs)
        self.base_date = None

        self.shared_memClass = namedtuple('shared_mem',
                                          't_Buffer t_Static_Buffer t_Feed_dict t_Cashflows calc_greeks \
                                          gpus riskneutral precision simulation_batch Report_Currency')

        self.static_var = {}

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
                current_val = value.current_value()
                if isinstance(current_val, dict):
                    for k, v in current_val.items():
                        fkey = utils.Factor(key.type, key.name + (k,))
                        self.static_var[fkey] = self.factor_leaf(fkey, v, calc_grad)
                else:
                    self.static_var[key] = self.factor_leaf(key, current_val, calc_grad)

        shared_mem = self.__init_shared_mem(
            params['Currency'], params['MCMC_Simulations'], calc_grad, params['Random_Seed'])

        self.all_tenors = utils.update_tenors(self.base_date, self.all_factors)

        return shared_mem

    def update_time_grid(self, base_date):
        self.time_grid = utils.TimeGrid({base_date}, {base_date}, {base_date})
        self.base_date = base_date
        self.time_grid.set_base_date(base_date)
        # the one date IS the reporting date - a boundary registration reads report_index to
        # know the grid its counterfactual lands on, and reads its absence as "not reportable"
        self.time_grid.set_report_dates(base_date, {base_date})

    def __init_shared_mem(self, reporting_currency, mcmc_sim, calc_greeks, random_seed):
        # fix the seed if we need to price mc instruments
        torch.manual_seed(random_seed)

        base_currency = utils.Factor(
            'FxRate', (self.config.params['System Parameters']['Base_Currency'],))

        all_vars_concat = None
        if calc_greeks:
            all_vars_concat = [x for x in self.static_var.items() if x[0] != base_currency]
            self.make_factor_index(list(self.static_var.items()))

        shared_mem = Base_Reval_State(
            self.static_var, torch.ones([1, 1], dtype=self.dtype, device=self.device),
            mcmc_sim, get_fxrate_factor(utils.check_rate_name(reporting_currency), self.static_factors, {}),
            all_vars_concat, self.params['Greeks'] == 'All')
        shared_mem.recompute_inner_mc = self.params.get('Recompute_Inner_MC', 'No') == 'Yes'
        # the SMOOTH estimator (`pricing.branch_and_weight`), declared on this calculation
        # alone; `execute` has completed the block, so the key is present and the read direct
        shared_mem.branch_and_weight = self.params['Branch_And_Weight'] == 'Yes'
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
                        data[k] = v.item()
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

        self.output = {}
        tag_titles = self.config.deals['Attributes'].get('Tag_Titles', '').split(',')
        mtm, greeks = check_prices(
            self.netting_sets, [('Parent', self.netting_sets.obj.Instrument.field.get('Reference'))])

        data = dict(
            [(field, self.netting_sets.obj.Instrument.field.get(field, 'Root')) for field in ['Reference', 'Object']])
        data['Value'] = sum([x.obj.Calc_res['Value'].item() for x in self.netting_sets.sub_structures])
        mtm.insert(0, data)

        self.output['mtm'] = pd.DataFrame(mtm)
        for greek_name, greek_val in greeks.items():
            # the transposed spelling is pandas' own migration off groupby(axis=1) (removed
            # in pandas 3) and runs identically back to 1.x
            if greek_name == 'Greeks_Second':
                summary = pd.concat(greek_val, axis=1).T.groupby(level=[0, 1, 2, 3, 4]).sum().T
            elif greek_name == 'Greeks_First':
                summary = pd.concat(greek_val, axis=1).T.groupby(level=0).sum().T
            else:
                raise Exception('Unknown Greek requested', greek_name)
            self.output.setdefault(greek_name, summary)

        return self.output

    def execute(self, params):
        # the declaration is the single source of an omitted field's default
        params = declared_defaults(type(self), params)
        base_date = pd.Timestamp(params['Run_Date'])
        self.params = params
        shared_mem = self.update_factors(params, base_date)
        logging.root.name = self.config.deals['Attributes'].get('Reference', self.config.file_ref)
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
                # the portfolio value IS the objective - one scenario, whose mean is the
                # reported number. Worth zero forward, so only the tape gains a term
                correction = pricing.boundary_correction(
                    shared_mem, lambda value: value.sum(axis=0), mtm,
                    float(params.get('Boundary_AAD_Bandwidth', 0.01)))
                if correction is not None:
                    mtm = mtm + correction
            pricing.greeks(shared_mem, ns_obj, mtm)
            self.calc_stats['Greek_Execution_Time'] = time.monotonic() - self.calc_stats['Greek_Execution_Time']

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
                      description='Half-spread bps of the calendar roll leg. Setting it arms the '
                                  'matched-leg pricing everywhere - the argmax charge, the fitted '
                                  'target and the realized accounting'),
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
                    F('Decision_Deadband_Sigma', 'Float', default=0.0,
                      description='No-trade band: the argmax must beat HOLDING the standing '
                                  'book by this many standard errors of the paired inner-draw '
                                  'difference before it trades; 0 = trade on any improvement. '
                                  'Execution policy only, like the cap above'),
                    F('Total_Position_Schedule', 'Table', default=None,
                      row=Row([F('Step', 'Integer', default=0),
                               F('Min_Total', 'Float', default=0.0),
                               F('Max_Total', 'Float', default=0.0)]),
                      description='Piecewise-constant corridor on the signed book total, by '
                                  'decision step'),
                    F('Allocation_Mode', 'Text', default='Exposure',
                      values=['Exposure', 'Carry_Variance'],
                      description='How the net cover splits across the hedge legs: Exposure = '
                                  'the declared Allocation_Weights table; Carry_Variance = the '
                                  'solver DERIVES per-step weights from the warmup sims (carry '
                                  'vs tracking vs the capital line), stamps them into the '
                                  'checkpoint, and a load restores the stamped table'),
                    F('Allocation_Weights', 'Table', default=None,
                      row=Row([F('Step', 'Integer', default=0),
                               F('Instrument', 'Text', default=''),
                               F('Weight', 'Float', default=0.0)]),
                      description='Piecewise-constant split of the NET cover across the hedge '
                                  'legs, by decision step. Present, the argmax searches one '
                                  'ladder over the total instead of the product of per-leg '
                                  'levels, and this table decides the composition')]),
              F('Solver', 'Container', default={},
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
                      description='Levels per hedge axis in the greedy action grid - or, under '
                                  'Evaluator.Allocation_Weights, rungs on the NET cover, where '
                                  'it must be at least as large as the net range in contracts'),
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
                      description='The backward DP aversion — the causal proxy for the '
                                  'clairvoyant seed floor no causal pass can enforce: divides '
                                  'the capital line in the LogWealth reward (1.0 neutral; '
                                  'higher = less capital at risk = more averse). The forward '
                                  'pass takes no dial.'),
                    F('DiffV2_Drift_Threshold_Sigmas', 'Float', default=3.0,
                      description='The drift tripwire tail: how many validation-measured null '
                                  'sigmas the inference CUSUM must exceed to trip'),
                    F('DiffV2_Drift_Beta', 'Float', default=0.0,
                      description='On-trip correction strength: the forecast used by the '
                                  'ranking is biased toward the REALIZED drift by beta times '
                                  'the observed average residual. 0 = report-only; tuned '
                                  'post-training by re-rolling saved checkpoints'),
                    F('DiffV2_Load_Horizon_Pad', 'Text', default='No', values=['Yes', 'No'],
                      description='Yes loads a value-function checkpoint fitted on a different '
                                  'decision horizon: per-step nets, trust bounds and scale '
                                  'schedules clamp to the saved range, the tail repeating the '
                                  'last fitted step. No refuses any t_min/T_dec mismatch'),
                    F('DiffV2_Returns_State', 'Text', default='No', values=['Yes', 'No'],
                      description='Yes makes the value state dimensionless: price columns as '
                                  'log-returns vs the calibrated t0 spot, basis columns as '
                                  'fractions of it, wealth as a fraction of the t0 book '
                                  'notional. Checkpoints stamp the coordinate system and refuse '
                                  'a mismatched load'),
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
                    F('DiffV2_Fit_Tol', 'Float', default=0.001,
                      description='Relative loss-plateau tolerance at which an INHERITED '
                                  'net\'s fit stops early; the terminal anchor always runs '
                                  'the full DiffV2_Fit_Iters budget. 0 = never stop early'),
                    F('DiffV2_Temporal_Proximity', 'Float', default=0.0,
                      description='Weight pulling each net\'s parameters toward its fitted '
                                  'successor\'s during the fit; 0 = off'),
                    F('DiffV2_Churn_Lambda', 'Float', default=0.0,
                      description='Quadratic repositioning charge in currency per contract^2, '
                                  'subtracted from the wealth entering the continuation at the '
                                  'argmax and at the training-label argmax; 0 = off'),
                    F('DiffV2_Position_State', 'Text', default='No', values=['Yes', 'No'],
                      description='Frictional Bellman: the signed net book fraction becomes a '
                                  'state coordinate of the fitted value and the repositioning '
                                  'charge enters the regressed target, so turnover compounds '
                                  'down the recursion instead of being a one-day toll'),
                    F('DiffV2_Wealth_Free_Value', 'Text', default='No', values=['Yes', 'No'],
                      description='Drop the wealth column from the value net\'s inputs, so the '
                                  'fitted residual reads market state (and the position, under '
                                  'DiffV2_Position_State) alone and the continuation bends in '
                                  'wealth exactly as the utility anchor does'),
                    F('DiffV2_Stepper_Rollout', 'Text', default='No', values=['Yes', 'No'],
                      description='Roll a frozen policy day-by-day through the real accounting'),
                    F('DiffV2_Decision_Curve_Dump', 'Text', default='',
                      description='Path a per-decision CSV of the stepper rollout\'s FULL '
                                  'ranking curve is written to (empty = off). Pure diagnostic: '
                                  'it changes no decision and no reported number'),
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
        """Build the `CMC_State_Inner` this calculation runs on, so one shared_mem hosts both
        the outer mode (inherited `reset()`) and the inner one (`reset_inner()`).
        HedgeMonteCarlo computes neither greeks nor FVA, so the parent's factor-index and
        survival-scaling setup is skipped."""
        return CMC_State_Inner(
            self.get_cholesky_decomp(), self.static_var, self.batch_size,
            torch.ones([1, 1], dtype=self.dtype, device=self.device), mcmc_sim, get_fxrate_factor(
                utils.check_rate_name(reporting_currency), self.static_factors, self.stoch_factors),
            seed, job_id, num_jobs,
            simulation_sub_batch=int(self.params.get('Inner_Sub_Batch', 0)),
            keep_tensor=True)  # hedge stepper replays need the kept tensor, unconditionally

    @staticmethod
    def _require_all_compiled(declared, structure, role):
        """Raise unless every declared deal compiled into `structure`.

        The pricing walk's skip-and-continue is the right contract for a reporting book -
        one broken deal should not lose the run - but here a skipped tradable silently
        shrinks the solver's menu and a skipped liability shrinks the target it is hedging,
        so the solve reports a confident answer to a different problem."""
        loaded = ({d.Instrument.field.get('Reference') for d in structure.dependencies} |
                  {s.obj.Instrument.field.get('Reference') for s in structure.sub_structures})
        missing = [n['Instrument'].field.get('Reference') for n in declared
                   if n['Instrument'].field.get('Reference') not in loaded]
        if missing:
            raise Exception(f'HedgeMonteCarlo: {role} legs failed to compile and were skipped: '
                            f'{missing} — a hedge book prices whole or not at all')

    def update_factors(self, params, base_date, job_id, num_jobs, end_date):
        """Build the factor state from the deal-driven dependencies plus the calculation's
        explicit `Scenario_Factors` - factors no deal reaches through a schema edge (e.g. a
        basis consumed only by a composed spot).

        The horizon is the latest tradable reset date capped at `end_date` (the liability
        terminal): hedge maturities past liability end leave the simulation horizon, while
        the hedges themselves price through liability end and any residual position closes
        out at fair value there."""
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
        """Return the static liability descriptors the symlog utility-scale needs, read
        straight off the cashflow schedules: `(total_leg_volume, last_payment_day)` - the
        summed |notional| across all liability legs and the latest payment day as an offset
        in days from base_date. `Bundle.from_batch` maps that day onto the (history-prefixed)
        bundle time grid to recover `last_settlement_index`."""
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
        """Simulate the scenario engine over batches, building the tensor bundle (tradable
        prices, liability MtM, factor paths), and return a `HedgeRuntimeExecutionResult`.

        A SOLVE IS A STREAM: `solve_hedge` builds one Bundle per batch inside the batch loop
        and hands it to a persistent solver as it is built - warmup on batch 1, step on each
        later batch, finish on a held-out final batch - so the inner-MC forks are only ever
        `Batch_Size` wide and every fit step sees fresh paths. `simulate_only` instead
        accumulates every batch into one bundle and exposes it for stepping, for which
        `Simulation_Batches` is a path multiplier rather than a stream length.

        LIABILITY-DRIVEN TIME-GRID CAP: the global grid stops at the liability's last
        cashflow / reval date, so outer and inner sim both stop there. Past it there is
        nothing to hedge, and any residual hedge position closes out at fair value.

        INNER-MC SETUP: the process copies are forked only after outer setup precalc has
        populated `factor_key` / `spot0` / etc., so inner-mode precalc on the copies cannot
        clobber outer attrs that outer generate reads each batch.

        `Randomize_Initial_State='Yes'` gives the diff-ML boundary label the variance in z_0
        it needs, via a per-batch burn-in: run each process once from the calibrated t=0,
        snapshot the terminal state per path, re-precalculate from that snapshot. The
        designer distribution is the process's own T-step pushforward, so there is no
        separate sampler. Batch k+1 REWINDS to the calibrated t=0 first, or the batch
        sequence random-walks away from it instead of drawing N independent worlds (measured
        over 5 streaming batches: the symlog scale drifted +94%, to the point of NaN sweeps
        on some months). Single-batch runs never rewind, so every `Simulation_Batches=1` job
        is bit-identical.

        `Observed_Scenario` (walk-forward backtest) replaces the simulated draw with
        grid-aligned realized paths from an .npz keyed by factor name; all preparation lives
        in the driver, and each process reseeds its own state from the replayed path.

        The declared underlying(s) are leafed so the base-delta / conditional-feature pass
        can read d(value)/d(spot) via AAD. The diff-ML solver differentiates the continuation
        off its own fresh state-at-t leaves (`Bundle.inner_mc_grad`), so it needs no outer
        leaf.
        """
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
        # canonical underlying (commodity-spot) factor-name set, derived once by the runtime
        # from the live CommodityPrice factors; the spot-leaf pass and `_find_spot_key` read
        # it rather than re-deriving it
        self._underlying_names = set(normalized_runtime['referenced_commodities'])

        # Inner-MC setup; the copies below are forked only after outer setup precalc has run
        inner_mc_enabled = params.get('Inner_MC_Enabled', 'No') == 'Yes'
        tradable_refs = sorted(normalized_runtime['names']['hedges']) if inner_mc_enabled else ()

        # solve_hedge: inner MC runs in the backward DP/MPC sweep, not the outer loop.
        # Cache the per-batch outer scenario buffer so inner MC can fork on demand later.
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

            # session-print conditioning from STATE: the calibrated t0 values ARE the state of
            # this batch's first step, so a process that informs another off a print in that
            # state publishes its first-step shift here (see StochasticProcess.print_seed)
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
                    # Each process owns its t=0 seed for the next run (regime / variance / none);
                    # captured now (detached) so it survives the buffer-clearing reset below.
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
                # The restart's state carries each path's OWN prints — condition on them
                # (per path), exactly as a fork conditions on its day's prints.
                restart_state = {k: v.reshape(1, 1, -1) for k, v in initial_t0.items()}
                print_keys |= self._publish_print_seeds(
                    self.stoch_factors.items(), restart_state, 0, shared_mem)

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

            # print seeds are CONSUMED state, not path series: every generate/replay above has
            # read them, and leaving them in the buffer would make the fork's republication
            # filter (fork-rewrote ∧ outer-carries) mistake a per-path scalar for a path.
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
                # this batch IS a bundle: build it, attach its forks, hand it to the solver.
                # The last batch is never fitted - it is the held-out world `finish` measures
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
                # fresh accumulators - this batch is consumed. The forks the solver ran
                # borrowed `shared_mem` and handed it back as found, so outer generation
                # resumes unaffected
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
            # world-before-solver tripwire: each hedge leg's expected drift over the live
            # window, in the very sim the policy trains on. A conditional-mean seam prints
            # HERE rather than surfacing months later as a policy that refuses to hedge
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

    # Inner-MC subsystem: forks the simulator from each outer-path state at an outer
    # timestep. Outer process instances are never touched - inner runs on shallow copies, so
    # per-instance precalc state cannot bleed across the outer/inner boundary.

    def _find_spot_key(self):
        """Return the unique underlying (commodity-spot) factor key, mapping the
        runtime-owned underlying name set back to the live key object via `stoch_factors`.
        Where more than one CommodityPrice is simulated, prefer the martingale primary - the
        spot exposing a revealed sufficient statistic (non-empty `privileged_layout`). Raises
        unless exactly one remains."""
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
        """Mirror `outer_struct` with each deal's `Time_dep` restricted to events at mtm
        positions >= `cutoff_mtm_idx` (`DealTimeDependencies.copy_restricted`), or windowed
        to [`cutoff_mtm_idx`, `window_end_idx`] when that is given (`copy_window` - the
        one-step fork prices exactly {t, t+1}). `Factor_dep` is shared by reference and
        `Calc_res` is fresh, so inner pricing cannot clobber outer storage; deals entirely in
        the past are dropped. Does not recurse into sub_structures - the inner-MC dependency
        list is flat.

        Aggregation storage is off while the per-deal `Calc_res` stays: the fork harvests on
        the DEVICE via `tensor_marks()` and `resolve_hedge_structure()`, and nothing reads
        the aggregate's stored 'Value', whose D2H copy was 93% of the fork's host egress."""
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
        """Attach the on-demand inner-MC forks to `bundle` as closures, so the solver can
        fork without a calc handle. Every fork is windowed to {t, t+1} - 2-row generation and
        a real 2-row pricing pass - giving the exact per-tradable F_t1 and L_t / L_t1 the
        diff-ML bootstrap reads. `outer_rows` lets the solver run the GRAD fork in
        outer-path sub-slices at large B_outer.

        The outer buffer is passed in rather than read off the calc: under streaming there is
        one bundle per batch, and each forks from ITS OWN batch."""
        bundle.inner_mc = lambda t: self._run_inner_mc_at_t(
            t, outer_buffer, shared_mem, base_date, tradable_refs)
        bundle.inner_mc_grad = lambda t, outer_rows=None: self._run_inner_mc_at_t(
            t, outer_buffer, shared_mem, base_date, tradable_refs,
            with_grad=True, outer_rows=outer_rows)

    def _publish_print_seeds(self, procs, state, t, shared_mem):
        """Publish every process's `print_seed` off `state` (factor key -> row-indexable
        snapshot) into the buffer, and return the published keys. Those keys are CONSUMED
        state, not path series - the caller drops them once the run has generated - and every
        seed is published before any generate."""
        keys = set()
        for key, proc in procs:
            for seed_key, seed_val in proc.print_seed(key, state, t).items():
                shared_mem.t_Scenario_Buffer[seed_key] = seed_val
                keys.add(seed_key)
        return keys

    def _run_inner_mc_at_t(self, t, outer_scenario_buffer, shared_mem, base_date,
                           tradable_refs, with_grad=False, outer_rows=None):
        """Run inner MC at a single outer timestep `t`, forking from `outer_scenario_buffer`
        - a snapshot of the outer `t_Scenario_Buffer` (factor keys plus every per-process aux
        key, batch dim B_outer). `outer_rows=(lo, hi)` forks only that contiguous outer-path
        range; labels are per-outer-path, so row slices are independent. The DP/MPC backward
        sweep calls this on demand outside the outer loop, via `Bundle.inner_mc`.

        Returns the inner samples the solver bootstraps from:
            F_t1          {ref: (B_outer, B_inner)}       futures price at outer t+1
            L_t, L_t1     (B_outer, B_inner)              liability MTM at outer t and t+1
            market_t1     (B_outer, B_inner, market_dim)  inner market state at outer t+1
            market_t      (B_outer, market_dim)           outer-realised market state at t
            t, cutoff_idx
        `market_t` / `market_t1` are every simulated factor's revealed segments concatenated
        in reveal order; each process owns its own packing via `reveal_state_at`, so the
        solvers consume the column block without knowing what the factors are.

        SINGLE PASS: generation, stuffing, pricing and extraction all run at
        `B_outer x Inner_Sub_Batch` flat, so peak memory is a function of two JSON fields
        (`Batch_Size`, `Inner_Sub_Batch`) and nothing else. A config too wide for the card
        raises CUDA OOM naming this fork; that is the contract, not a knob.

        THE WINDOW: the inner grid and every deal's `Time_dep` are truncated to {t, t+1}, so
        the pricing chain runs for real on 2 rows - exact per-tradable F_t1 and liability
        L_t / L_t1, and correct on mixed strips, where a market-only short-circuit would
        broadcast spot as every F_t1. Restricting the AAD tape to one forward step is what
        bounds its memory.

        TWO COORDINATE SYSTEMS: processes generate against the shifted-base
        `inner_time_grid`, while pricers run against the full outer `self.time_grid` with
        each deal's `Time_dep` restricted. Buffer stuffing prepends the outer-realized past
        so path-dependent payoffs see the realized fixings; that past is published as its own
        `ScenarioBlock` carrying a `past_columns` index rather than materialized B_inner
        times. Every path series goes through that publication - a factor's grid and a
        process's own `(key, kind)` series alike, since a pricer cannot tell them apart.

        PER-PROCESS HOOKS, no isinstance branch anywhere: `inner_fork_seed` supplies the
        per-outer-path t=0 privileged state (regime for the HMM, conditional variance h0 for
        GARCH), `reseed_inner_state` republishes whatever path-dependent revealed state
        `reveal_state_at` needs at t+1 and returns differentiable leaves for the twin loss
        (`self._inner_state_opts` is forwarded to it opaquely), and `reveal_state_at` yields
        each factor's informative segments from the live buffer. A factor without a revealed
        sufficient statistic returns an empty dict, so one uniform loop covers every model
        world. Under `with_grad`, `state_t_leaf_widths` pairs each leaf with the market_t
        column width it occupies, so the label projection never re-derives factor widths.

        FAIL LOUDLY on both halves: a tradable live in this fork but missing from
        `tensor_marks()` reads downstream as an expired contract and the solver retires it,
        and a liability swallowed by the canonical deal guard silently corrupts the solver's
        LABELS.

        The fork BORROWS `shared_mem` and the `finally` restores it on ANY exit - without
        that a mid-fork raise leaves the state flat-sized and the next t-step fails on shapes
        instead of the real cause. The Sobol sample cache is dropped there too: it is keyed
        by sample_size and would otherwise grow unbounded across t-steps.
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
        # closed-form V_T there; a caller querying `inner_mc` past terminal reaches this
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
        # what the fork BORROWS, to give back exactly as found (see the finally below): a
        # fork over an outer-path SLICE would otherwise leave the state slice-sized, which
        # corrupts the next outer batch under streaming
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
                live_procs = []
                fork_state = {}
                for key, proc_inner in self.stoch_factors_inner.items():
                    if key.type in utils.DimensionLessFactors:
                        continue
                    # raw per-path init state for this factor's inner-MC precalculate fork
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
                    # the day-t state snapshot the print seeds read; under `with_grad` this IS
                    # the leaf, so conditioning derived from a print stays on the tape
                    fork_state[key] = init_state.unsqueeze(0)
                    live_procs.append((key, proc_inner))
                # EVERY fork seed before ANY generate: a seed reads only the OUTER state, and
                # a process may seed a factor that generates before it in topological order.
                # `inner_fork_seed` keeps the detached buffer - its seeds are detached by design
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
                # every path series this fork WROTE that the outer path also carries - a
                # factor's grid and a process's own `(key, kind)` series alike. Each half of
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
                shared_mem.simulation_batch = B_flat
                # `fillvalue` is a batch-sized empty tensor frozen at State construction; it
                # must track the current simulation_batch or cash_settle mismatches on size
                shared_mem.fillvalue = shared_mem.one.new_zeros((0, 1, B_flat))
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
                    # 1-wide batch - path-independent is a legitimate mark, not a skip - so
                    # broadcast it. Any other width is a shape error and expand fails loud.
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
