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
import copy
import time
import logging
import threading
from collections import namedtuple, OrderedDict
from functools import partial

# third party stuff
import numpy as np
import pandas as pd
import torch

# Internal modules
from . import utils, pricing, instruments, riskfactors, stochasticprocess
from .schema import (F, OPTION_QUOTE, QUOTE_TWO_WAY, REQUIRED, Row, declared_defaults,
                     partition_market_price)
from ._version import __version__

import scipy.optimize
import scipy.stats


def resolve_factor(name, price_factors, candidates):
    """The factor `name` refers to, typed by the first candidate the price factors hold a block for.

    A block names its inputs by name only, so the bootstrapper declares the candidate types
    (`utils.TwoDimensionalFactors` is the candidate list for a vol surface)."""
    rate = utils.check_rate_name(name)
    return utils.Factor(next(x for x in candidates if utils.check_tuple_name(
        utils.Factor(x, rate)) in price_factors), rate)


def reference_fields(factor_types, required_by_quote_type, notes):
    """One `F` per named factor reference, then its optional `_Type` sibling.

    The `_Type` values are the candidate list, so a new candidate cannot miss the schema. REQUIRED
    only where every quote type requires it; a reference required under one and inert under another
    states that half through `notes` and is enforced where it is read.

    A module-level function because a class-scope name is not visible inside a class-body
    comprehension, so neither the quote-type map nor `notes` could be read there.
    """
    always = {field for field in factor_types
              if all(field in required for required in required_by_quote_type.values())}
    return [F(field, 'Text', default=REQUIRED if field in always else '',
              description='The {} factor - one of {}{}'.format(
                  field.replace('_', ' ').lower(), ', '.join(types), notes.get(field, '')))
            for field, types in factor_types.items()] + [
        F(field + '_Type', 'Text', default='', values=[''] + list(types),
          description='Names the factor type explicitly, where the name exists under more than one')
        for field, types in factor_types.items()]


def implied_vol(premium, forward, strike, rate, steps, T, units, parity):
    """One quoted premium as a Black vol: the units and the yield rescale stripped, the put put
    back on parity, and `bs_implied_total_var` off the same forward. `rate` is per step over
    `steps` of them, which is the clock a GARCH family counts in; `T` is the year fraction the
    total variance is annualised by."""
    call = float(premium) / float(units) + parity
    return np.sqrt(max(utils.bs_implied_total_var(
        call, float(forward), float(strike), float(rate), int(steps)), 0.0) / T)


def column_scale(jacobian):
    """`(J/||J_:,j||, ||J_:,j||)` - the `x_scale='jac'` matrix `least_squares` itself steps on,
    which is what both the identification table and the quote contraction read. An all-zero column
    keeps unit scale rather than dividing by nothing."""
    norms = jacobian.norm(dim=0)
    norms = torch.where(norms > 0.0, norms, torch.ones_like(norms))
    return jacobian / norms, norms


def active_set(x, lower, upper, g, tol=1e-4):
    """The KKT active set at `x` as a boolean mask - the coordinates the BOX holds: within `tol` of
    a bound relative to the box's width AND the objective's gradient `g = J^T r` pointing into it.
    An infinite half-width takes the finite side's width, or `max(1, |x|)` where both are infinite.

    Both conditions, because either alone is wrong: a solver stops a hair short of a floor it is
    jammed against, and a coordinate resting against a bound the data pulls away from is free.
    """
    x, lower, upper, g = (np.asarray(v, dtype=float) for v in (x, lower, upper, g))
    span = np.where(np.isfinite(upper), upper, 0.0) - np.where(np.isfinite(lower), lower, 0.0)
    edge = tol * np.where(np.isfinite(upper) | np.isfinite(lower),
                          np.abs(span), np.maximum(1.0, np.abs(x)))
    return ((x - lower <= edge) & (g > 0.0)) | ((upper - x <= edge) & (g < 0.0))


def active_bounds(labels, x, lower, upper, g):
    """One line per coordinate `active_set` holds - a CALIBRATION statement, reported whether or
    not quote sensitivities were asked for: the data is pushing that parameter through a bound, so
    the surface wants something outside the box and the number written there is not a fitted one.
    """
    held = active_set(x, lower, upper, g)
    return ['{} HELD at {:.6g} on its {} bound {:.6g}, gradient {:+.3e} pushing into it'.format(
        labels[i], x[i], 'lower' if g[i] > 0.0 else 'upper',
        lower[i] if g[i] > 0.0 else upper[i], g[i]) for i in np.flatnonzero(held)]


def null_basis(scaled, norms, rcond):
    """An orthonormal basis of the null space of the UNSCALED Jacobian at the declared cutoff - the
    right singular vectors the cutoff DISCARDS, mapped back by `D^-1` and re-orthonormalised.

    The pseudo-inverse is minimum-norm in the metric the solve steps in, so the part of a quote
    delta lying here is that CONVENTION and not anything the quotes identify.
    """
    _, values, right = torch.linalg.svd(scaled, full_matrices=True)
    dropped = torch.ones(right.shape[0], dtype=torch.bool, device=right.device)
    dropped[:values.numel()] = values <= rcond * values.max()
    basis = (right[dropped] / norms).t()
    return torch.linalg.qr(basis)[0] if basis.shape[1] else basis


class swaption_schedule_class(namedtuple('swaption_schedule', 'expiry pay_times accruals')):
    """One benchmark swaption's FIXED leg, in the curve's own year fractions.

    The clock is the interest rate factor's `get_day_count_accrual` and not `utils.DAYS_IN_YEAR`:
    that is what `read_cache` builds `time_grid_years` with, hence the grid every `J` integral is
    taken on. The two are 7e-4 years apart at a 1Y expiry, enough to miss a grid node.

    `expiry` is the same float the premium was struck on, so a normal vol round-trips exactly.

    `accruals` and `pay_times` are the fixed leg's, because the annuity is. Where both legs share a
    frequency `set_fixed_amount` writes the coupon into the float leg, which is then the leg read.
    """


class market_swap_class(namedtuple('market_swap', 'deal_data price weight schedule quote premium',
                                   defaults=(None, None))):
    """One benchmark swaption of a risk-neutral IR calibration: the compiled par swap, the market
    premium the model has to reproduce, the weight it carries in the objective, and the fixed leg
    the analytic objective reads.

    `quote` and `premium` are the quote side and absent by default - the float64 leaf the market
    number arrived on, and the map from that leaf to this swaption's premium (`create_market_swaps`).
    Both objectives splice the same pair onto their own residual, so `quote_leaves` is one shape.

    `premium` is a CALLABLE so the twin is rebuilt inside every evaluation: `make_basin_hopping_loss`
    calls `backward()` with no `retain_graph`, and a compile-time subgraph hanging off the residual
    would be freed with the first evaluation. It costs one scalar Black per benchmark per call.
    """

    def error(self, model, resid):
        """This swaption's weighted relative pricing error against its `model` price.

        The quote rides in as the splice `base + (carried - detach(carried))`: exactly zero in the
        forward pass, derivative one, so enabling the quote side cannot move a mark.

        `model` is detached in the carried half and only there. Left attached it would reach the
        model parameters as well as the quote and double the calibration Jacobian.

        The splice sits at the error and not at the price because `price` is a numpy scalar and
        torch divides a tensor by a scalar at the scalar's precision - a float64 tensor there rounds
        twice where the engine rounds once, moving the residual by an ulp.
        """
        base = self.weight * resid(100.0 * (self.price / model - 1.0))
        if self.premium is None:
            return base
        carried = self.weight * resid(100.0 * (self.premium(self.quote) / model.detach() - 1.0))
        return base + (carried - carried.detach()).to(base.dtype)

    def market_normal_vol(self, annuity):
        """This swaption's market premium as an ATM normal (Bachelier) vol, in closed form.

        At the money the Bachelier premium is $A\\sigma_N\\sqrt{T_0/2\\pi}$, so the inversion is a
        division and not a root find:

        $$\\sigma_N = \\frac{P}{A}\\sqrt{\\frac{2\\pi}{T_0}}$$

        Every quoting convention rides in through the premium `create_market_swaps` already built,
        struck on `schedule.expiry` itself - so under `'Normal'` the round trip is exact.

        `annuity` is the analytic price's own annuity off the t=0 curve, built in numpy, so it
        carries no derivative in theta and the residual is the premium residual over a constant.

        The quote side is the splice `base + (carried - detach(carried))`. Nothing is detached here,
        unlike `error`: the carried half divides by that severed annuity, so the market side is a
        function of the quote alone and $\\partial^2 r/\\partial\\theta\\partial q$ is structurally
        zero - the cross term Gauss-Newton drops is absent rather than small.
        """
        base = self.price * np.sqrt(2.0 * np.pi / self.schedule.expiry) / annuity
        if self.premium is None:
            return base
        carried = self.premium(self.quote) * np.sqrt(
            2.0 * np.pi / self.schedule.expiry) / annuity.double()
        return base + (carried - carried.detach()).to(base.dtype)

    def normal_vol_error(self, swaption):
        """This swaption's weighted normal-vol residual against the market, plain.

        Vols against vols and not squared. `error` returns a residual that is already a square, so
        `least_squares` minimises a quartic and $J = \\partial r/\\partial\\theta$ carries a factor
        of the pricing error in every row
        ([Quote Sensitivities](quote_sensitivities.md#the-stationarity-contract)). Here the residual
        is the difference itself, in absolute normal vol, and `least_squares` does the squaring.

        This chain reaches $\\|J^Tr\\|$ 8.63e-7 on the identified block against the squared
        residual's 3.16e2 - either side of `Stationarity_Tol`'s 1e-3 default.

        The residual is separable, a theta-function minus a q-function, so $\\partial r/\\partial q$
        is diagonal and the mixed second derivative is exactly zero. `market_normal_vol` carries the
        splice that puts the market half on the tape.
        """
        return self.weight * (swaption.normal_vol - self.market_normal_vol(swaption.annuity))


date_desc = {'years': 'Y', 'months': 'M', 'days': 'D'}
# date formatter
date_fmt = lambda x: ''.join(['{0}{1}'.format(v, date_desc[k]) for k, v in x.kwds.items()])


class RiskNeutralInterestRate_State(utils.Calculation_State):
    def __init__(self, scenario_keys, batch_size, device, dtype, nomodel='Constant'):
        super(RiskNeutralInterestRate_State, self).__init__(
            None, torch.ones([1, 1], dtype=dtype, device=device), 2048, None, nomodel, batch_size, False)
        # these are tensors
        self.t_PreCalc = {}
        self.scenario_keys = scenario_keys
        self.t_random_batch = None
        self.batch_index = 0
        self.t_Scenario_Buffer = {}

    @property
    def t_random_numbers(self):
        return self.t_random_batch[self.batch_index]

    def clear(self):
        """Empties the memo buffers, which every evaluation must do before it starts.

        `t_Buffer` is keyed by factor and time and `t_PreCalc` by factor and integrand, neither by
        the tensor's identity, so a state carried across two parameter sets would answer the second
        call with the first's curves. Only the Monte Carlo objective goes on to need a sample, which
        is why `reset` is a second call.
        """
        self.t_Buffer.clear()
        self.t_PreCalc.clear()

    def reset(self, num_batches, numfactors, time_grid):
        # clear the buffers
        self.clear()

        if self.t_random_batch is None:
            # the sobol engine in torch > 1.8 goes up to dimension 21201 - so this should be fine
            self.sobol = torch.quasirandom.SobolEngine(
                dimension=time_grid.time_grid_years.size * numfactors, scramble=True, seed=1234)
            # skip this many samples
            self.sobol.fast_forward(2048)
            # make sure we don't include 1 or 0
            sample_sobol = self.sobol.draw(self.simulation_batch * num_batches).reshape(
                num_batches, self.simulation_batch, -1)
            sample = torch.erfinv(2 * (0.5 + (1 - torch.finfo(sample_sobol.dtype).eps) * (
                    sample_sobol - 0.5)) - 1).reshape(
                num_batches, self.simulation_batch, -1) * 1.4142135623730951
            self.t_random_batch = sample.transpose(1, 2).reshape(
                num_batches, numfactors, -1, self.simulation_batch).to(self.one.device)


def create_float_cashflows(base_date, cashflow_obj, frequency):
    cashflows = []
    for cashflow, reset in zip(cashflow_obj.schedule, cashflow_obj.Resets.schedule):
        cashflows.append({
            'Payment_Date': base_date + pd.offsets.Day(cashflow[utils.CASHFLOW_INDEX_Pay_Day]),
            'Notional': 1.0,
            'Accrual_Start_Date': base_date + pd.offsets.Day(cashflow[utils.CASHFLOW_INDEX_Start_Day]),
            'Accrual_End_Date': base_date + pd.offsets.Day(cashflow[utils.CASHFLOW_INDEX_End_Day]),
            'Accrual_Year_Fraction': cashflow[utils.CASHFLOW_INDEX_Year_Frac],
            'Fixed_Amount': cashflow[utils.CASHFLOW_INDEX_FixedAmt],
            'Resets': [[base_date + pd.offsets.Day(reset[utils.RESET_INDEX_Reset_Day]),
                        base_date + pd.offsets.Day(reset[utils.RESET_INDEX_Start_Day]),
                        base_date + pd.offsets.Day(reset[utils.RESET_INDEX_End_Day]),
                        reset[utils.RESET_INDEX_Accrual],
                        frequency, 'ACT_365', '0D', 0.0, 'No', utils.Percent(0.0)]],
            'Margin': utils.Basis(0.0)
        })
    return cashflows


#: The two quoting conventions this family prices, each as the matched pair `create_market_swaps`
#: needs: the numpy pricer that builds the market premium, and the tensor twin of that same formula
#: which the quote side differentiates. Keyed by `InterestYieldVol`'s declared `Distribution_Type`,
#: whose declared default is `'Lognormal'`.
PREMIUM_CONVENTIONS = {
    'Lognormal': (utils.black_european_option_price, utils.black_european_option),
    'Normal': (utils.bachelier_european_option_price, utils.bachelier_european_option)}

#: The HW2F reversion-speed seed, deliberately asymmetric. Equal alphas beside equal sigma curves
#: make the objective exactly exchange-symmetric in `(alpha_i, sigma_i)`, so the first local
#: minimisation is confined to the symmetric hyperplane: on the identified 25-quote block basin
#: hopping's iteration-0 L-BFGS-B reaches 8.95e-6 from `0.1, 0.1` and 5.33e-6 from this seed.
#:
#: The ratio is 10x - a fast factor and a slow one, half-lives `ln2/alpha` of 1.39y and 13.9y,
#: bracketing the expiry ladders this family is quoted on. Both sit above every small-alpha series
#: threshold (`hw_alpha_series_B` 1e-3, `_H` 1e-2, `_IJK` 3e-2), strictly inside `alpha_bounds`,
#: and positive; their sum 0.55 is far from the singular `alpha_1 + alpha_2 -> 0` hyperplane.
#:
#: The sigma seeds stay identical: separating the reversion speeds already breaks the exchange.
ALPHA_SEED = (0.5, 0.05)

#: The bracket the `Volatility_Delta` implied-vol re-solve runs in, as a function of the row's own
#: quoted vol. Co-keyed with `PREMIUM_CONVENTIONS` off the same declared `Distribution_Type`, so the
#: convention that picks the pricer picks the scale its bracket is in.
#:
#: A lognormal vol is a fraction of the rate and a 1% floor sits under every quoted surface. A normal
#: vol is an absolute rate move, where 0.01 is 100 basis points - above ordinary EUR and JPY levels,
#: and a quote below it left both bracket ends the same sign - so that bracket is multiplicative
#: around the quote instead. Two orders either side suffices because the ATM Bachelier premium is
#: exactly linear in the vol, so the bracket has only to contain a division; further out is a broken
#: premiums file and still refuses. The quote is floored at 1e-6 so the band cannot collapse.
IMPLIED_VOL_BRACKETS = {
    'Lognormal': lambda vol: (0.01, vol + .5),
    'Normal': lambda vol: (max(vol, 1e-6) * 0.01, max(vol, 1e-6) * 100.0)}


def market_premium(pvbp, strike, expiry, delta, option, quote):
    """One ATM swaption's premium as a differentiable function of its vol quote.

    `option` is the tensor half of this surface's `PREMIUM_CONVENTIONS` pair, so this is the twin of
    the numpy premium `create_market_swaps` builds beside it; the two share a signature, so the
    convention arrives bound rather than branched on. At the money, the only place this is called,
    the pairs agree to 1e-12 (Black) and to the hex digit (Bachelier).
    """
    return pvbp * option(
        quote.new_tensor(strike), quote.new_tensor(strike), quote + delta, expiry, 1.0, 1.0, None)


def create_market_swaps(base_date, time_grid, curve_index, vol_surface, curve_factor,
                        instrument_definitions, rate=None, unit=None):
    """The benchmark swaptions of one risk-neutral IR calibration: a compiled par swap, the market
    premium the model has to reproduce, and the objective weight.

    THE QUOTE SIDE. `unit` is the residual's unit tensor when the block asks for `Quote_Sensitivity`
    and `None` otherwise. The market premium is numpy, so each swaption carries a pair - the quote as
    a float64 leaf and the map back to its premium - which `market_swap_class.error` splices on. A
    vol-quoted row carries the vol and maps through `market_premium`; a premium-quoted one carries
    the premium and the map is the identity.

    The premium is priced in the surface's declared convention, read through `get_subtype` as the
    deal path reads it: see `PREMIUM_CONVENTIONS`. The `Volatility_Delta` re-solve brackets in that
    same declared scale, `IMPLIED_VOL_BRACKETS` being co-keyed with it. The displacement is
    `vol_surface.displacement`, where the declared `Shift` outranks the `Property_Aliases` legacy
    (see `riskfactors.InterestYieldVol.displacement`). An absent or zero `Market_Volatility` refuses.

    THE SCHEDULE the analytic objective reads is extracted here for every benchmark whatever the
    block's `Objective` - see `swaption_schedule_class` for why the curve's own clock.

    ONE EXPIRY YEAR FRACTION, and it is `curve_factor.get_day_count_accrual`: it prices the numpy
    premium, strikes the float64 twin, brackets the brentq re-solve and is `schedule.expiry`. The
    DATES are untouched - `exp_days`, the `mtm_time_grid` search and both leg generators read days
    and the instrument's own day counts, and 365.25 still converts vol tenors to grid days.
    """
    # a brentq implied-vol solve carries no derivative, so the quote side declines that combination
    if unit is not None and vol_surface.premiums is not None and vol_surface.delta:
        raise Exception('Quote_Sensitivity: a premium re-struck at Volatility_Delta reaches the '
                        'residual through a brentq implied-vol solve, which carries no derivative')
    # store these benchmark swap definitions if necessary
    benchmarks = []
    # store the benchmark instruments
    all_deals = {}
    # the surface's declared convention, read once - `get_subtype` is the deal path's own read
    distribution = vol_surface.get_subtype()[0]
    if distribution not in PREMIUM_CONVENTIONS:
        raise Exception(
            "InterestYieldVol declares Distribution_Type '{}', which is not a convention this "
            'calibration prices a benchmark premium in - they are {}. Correct the surface\'s '
            'Distribution_Type to one of those'.format(
                distribution, ' and '.join(sorted(PREMIUM_CONVENTIONS))))
    price_option, tensor_option = PREMIUM_CONVENTIONS[distribution]
    # the re-solve's bracket off that same read - the quote's scale is the convention's
    vol_bracket = IMPLIED_VOL_BRACKETS[distribution]
    # cater for shifted lognormal vols - declared `Shift` first, `Property_Aliases` behind it
    shift_parameter = vol_surface.displacement
    for instrument in instrument_definitions:
        # set up the instrument
        effective = base_date + instrument['Start']
        maturity = effective + instrument['Tenor']
        exp_days = (effective - base_date).days
        # one clock, the curve's: this prices the premium, strikes the twin, brackets the re-solve
        # and is `schedule.expiry` below, so the Bachelier inversion reads back what it struck on
        expiry = float(curve_factor.get_day_count_accrual(base_date, exp_days))
        time_index = np.searchsorted(time_grid.mtm_time_grid, [exp_days], side='right') - 1
        swaption_name = 'Swaption_{}_{}'.format(
            date_fmt(instrument['Start']), date_fmt(instrument['Tenor']))

        float_pay_dates = instruments.generate_dates_backward(
            maturity, effective, instrument['Floating_Frequency'])

        float_cash = utils.generate_float_cashflows(
            base_date, time_grid, float_pay_dates, 1.0, None, None,
            instrument['Floating_Frequency'], pd.DateOffset(month=0),
            utils.get_day_count(instrument['Floating_Day_Count']), 0.0)

        K, pvbp = float_cash.get_par_swap_rate(base_date, curve_factor)

        if instrument['Fixed_Frequency'] != instrument['Floating_Frequency']:
            fixed_pay_dates = instruments.generate_dates_backward(
                maturity, effective, instrument['Fixed_Frequency'])
            fixed_cash = utils.generate_fixed_cashflows(
                base_date, fixed_pay_dates, 1.0, None, utils.get_day_count(instrument['Fixed_Day_Count']), 0.0)
            pv_float = K * pvbp
            pvbp = fixed_cash.get_par_swap_rate(base_date, curve_factor)
            K = pv_float / pvbp
            fixed_cash.set_fixed_amount(K)
            fixed_indices = float_cash[:, utils.CASHFLOW_INDEX_Pay_Day].searchsorted(
                fixed_cash[:, utils.CASHFLOW_INDEX_Pay_Day])

            if not (float_cash[fixed_indices, utils.CASHFLOW_INDEX_Pay_Day] ==
                    fixed_cash[:, utils.CASHFLOW_INDEX_Pay_Day]).all():
                logging.error('Float leg and Fixed legs do not coincide')
                raise Exception('Float leg and Fixed legs do not coincide')

            # set the float leg fixed amount
            float_cash.schedule[fixed_indices, utils.CASHFLOW_INDEX_FixedAmt] = \
                -fixed_cash[:, utils.CASHFLOW_INDEX_FixedAmt]
            fixed_schedule = fixed_cash.schedule
        else:
            float_cash.set_fixed_amount(-K)
            fixed_schedule = float_cash.schedule

        # the annuity's own leg, in the CURVE's year fractions - see `swaption_schedule_class`
        schedule = swaption_schedule_class(
            expiry=expiry,
            pay_times=curve_factor.get_day_count_accrual(
                base_date, fixed_schedule[:, utils.CASHFLOW_INDEX_Pay_Day]),
            accruals=fixed_schedule[:, utils.CASHFLOW_INDEX_Year_Frac].copy())

        # a benchmark has to carry a quote: neither an absent nor a zero vol is a price
        if 'Market_Volatility' not in instrument:
            raise Exception(
                '{}: the benchmark carries no Market_Volatility, and a swaption with no quote is '
                'not a benchmark. Author the vol on the row, or drop the row'.format(swaption_name))
        vol = instrument['Market_Volatility'].amount
        if not vol:
            raise Exception(
                '{}: Market_Volatility is quoted ZERO, and a zero vol is not a price - it used to '
                "read the surface's own ATM instead, which calibrates against a quote nobody gave. "
                'Author the vol on the row, or drop the row'.format(swaption_name))

        deal_data = utils.DealDataType(
            Instrument=None, Factor_dep={'Cashflows': float_cash, 'Forward': curve_index,
                                         'Discount': curve_index, 'CompoundingMethod': 'None'},
            Time_dep=utils.DealTimeDependencies(time_grid.mtm_time_grid, time_index), Calc_res=None)

        shifted_strike = K + shift_parameter
        # first check if we have the actual premium (not implied)
        if vol_surface.premiums is not None:
            swaption_price = vol_surface.get_premium(date_fmt(instrument['Start']), date_fmt(instrument['Tenor']))
            if vol_surface.delta:
                # one bracket for both solves, in the scale this surface quotes its vols in
                bracket = vol_bracket(vol)
                try:
                    implied_vol = scipy.optimize.brentq(lambda v: pvbp * price_option(
                        shifted_strike, shifted_strike, 0.0, v, expiry, 1.0, 1.0) - swaption_price,
                        *bracket)
                except:
                    modified_k = vol_surface.get_strike_from_premiums(date_fmt(instrument['Start']),
                                                                      date_fmt(instrument['Tenor']))
                    logging.warning(
                        'Implied vol calc during delta bump failed - calculated strike is {} - using strike from premium file {}'.format(
                            K, modified_k))
                    shifted_strike = modified_k + shift_parameter
                    implied_vol = scipy.optimize.brentq(lambda v: pvbp * price_option(
                        shifted_strike, shifted_strike, 0.0, v, expiry, 1.0, 1.0) - swaption_price,
                        *bracket)

                swaption_price = pvbp * price_option(
                    shifted_strike, shifted_strike, 0.0, implied_vol + vol_surface.delta, expiry, 1.0, 1.0)
        else:
            swaption_price = pvbp * price_option(
                shifted_strike, shifted_strike, 0.0, vol + vol_surface.delta, expiry, 1.0, 1.0)

        # the quote side - a float64 leaf and the map back to this swaption's premium, see docstring
        quote, premium = None, None
        if unit is not None:
            premium_quoted = vol_surface.premiums is not None
            quote = unit.new_tensor(
                swaption_price if premium_quoted else vol, dtype=torch.float64).requires_grad_(True)
            premium = (lambda q: q) if premium_quoted else partial(
                market_premium, pvbp, shifted_strike, expiry, vol_surface.delta, tensor_option)

        all_deals[swaption_name] = market_swap_class(
            deal_data=deal_data, price=swaption_price, weight=instrument['Weight'],
            schedule=schedule, quote=quote, premium=premium)

        if rate is not None:
            benchmarks.append(
                instruments.construct_instrument(
                    {'Object': 'CFFloatingInterestListDeal',
                     'Reference': swaption_name,
                     'Currency': curve_factor.param['Currency'],
                     'Discount_Rate': '.'.join(rate),
                     'Forecast_Rate': '.'.join(rate),
                     'Buy_Sell': 'Buy',
                     'Cashflows': {'Items': create_float_cashflows(
                         base_date, float_cash, instrument['Floating_Frequency'])}},
                    {})
            )

    return all_deals, benchmarks


class CSForwardPriceModelParameters(object):
    documentation = (
        'Energy',
        ['For Risk Neutral simulation, the Clewlow Strickland Model is calibrated to a set of European Energy',
         'futures options $J$.',
         'an integrated curve $\\bar{\\sigma}(t)$ needs to be specified and is',
         'interpreted as the average volatility at time $t$. This is typically obtained from the corresponding',
         'ATM volatility. This is then used to construct a new variance curve $V(t)$ which is defined as',
         '$V(0)=0, V(t_i)=\\bar{\\sigma}(t_i)^2 t_i$ and $V(t)=\\bar{\\sigma}(t_n)^2 t$ for $t>t_n$ where',
         '$t_1,...,t_n$ are discrete points on the ATM volatility curve.',
         '',
         'Points on the curve that imply a decrease in variance (i.e. $V(t_i)<V(t_{i-1})$) are adjusted to',
         '$V(t_i)=\\bar\\sigma(t_i)^2t_i=V(t_{i-1})$. This curve is then used to construct *instantaneous* curves',
         'that are then input to the corresponding stochastic process.',
         '',
         'The relationship between integrated $F(t)=\\int_0^t f_1(s)f_2(s)ds$ and instantaneous curves $f_1, f_2$',
         'where the instantaneous curves are defined on discrete points $P={t_0,t_1,..,t_n}$ with $t_0=0$ is defined',
         'on $P$ by Simpson\'s rule:',
         '',
         '$$F(t_i)=F(t_{i-1})+\\frac{t_i-t_{i-1}}{6}\\Big(f(t_i)+4f(\\frac{t_i+t_{i-1}}{2})+f(t_i)\\Big)$$',
         '',
         'and $f(t)=f_1(t)f_2(t)$. Integrated curves are flat extrapolated and linearly interpolated.'
         ]
    )

    market_factor_type = 'CSForwardPriceModelPrices'
    fields = [
        F('Energy', 'Text', default=REQUIRED, description='The ForwardPrice factor to calibrate'),
        F('Forward_Volatility', 'Text', default=REQUIRED,
          description='The ForwardPriceVol surface the quoted vols are read off'),
        F('Discount_Rate', 'Text', default=REQUIRED,
          description='The InterestRate curve the premiums discount on'),
        F('Quote_Type', 'Text', default='Implied_Volatility', values=['Implied_Volatility'],
          description='How Quoted_Market_Value reads - this family takes vols only'),
        F('Energy_Futures_Options', 'Table', default='null',
          row=Row(OPTION_QUOTE[:1] + [F('Settlement_Date', 'Date',
                                        description='Futures settlement, which sets the '
                                                    'Clewlow-Strickland decay term')] +
                  OPTION_QUOTE[1:]),
          description='The option quotes sigma and alpha are fitted to')
    ]

    def __init__(self, param, device, dtype):
        self.device = device
        self.prec = dtype
        self.param = param

    def bootstrap(self, sys_params, price_models, price_factors, factor_interp, market_prices, calendars, debug=None):
        '''
        Checks for Declining variance in the ATM vols of the relevant price factor and corrects accordingly.
        '''

        def B(a, t):
            return (1.0 - np.exp(-a * t)) / a if a != 0 else t

        def V(sigma, alpha, T, S):
            return sigma * sigma * np.exp(-2.0 * alpha * S) * B(-2.0 * alpha, T)

        def calc_error(x, options):
            sigma, alpha = x
            error = 0.0

            for option in options:
                discount = np.exp(-option['r'] * option['T'])
                error += option['Weight'] * (option['Premium'] - utils.black_european_option_price(
                    option['Forward'], option['Strike'], 0.0, np.sqrt(V(sigma, alpha, option['T'], option['S'])),
                    1.0, option['Units'], 1.0 if option['Option_Type'] == 'Call' else -1.0) * discount) ** 2
            return error

        for market_price, implied_params in market_prices.items():
            rate = utils.check_rate_name(market_price)
            market_factor = utils.Factor(rate[0], rate[1:])

            if market_factor.type == self.market_factor_type:
                # get the vol surface
                if 'ForwardPriceVol.' + implied_params['instrument']['Forward_Volatility'] in price_factors:
                    vol_factor = utils.Factor('ForwardPriceVol', utils.check_rate_name(
                        implied_params['instrument']['Forward_Volatility']))
                if 'ForwardPrice.' + implied_params['instrument']['Energy'] in price_factors:
                    energy_factor = utils.Factor('ForwardPrice', utils.check_rate_name(
                        implied_params['instrument']['Energy']))
                if 'InterestRate.' + implied_params['instrument']['Discount_Rate'] in price_factors:
                    discount_factor = utils.Factor('InterestRate', utils.check_rate_name(
                        implied_params['instrument']['Discount_Rate']))

                # this shouldn't fail - if it does, need to log it and move on
                try:
                    vol_surface = riskfactors.construct_factor(vol_factor, price_factors, factor_interp)
                    vol_surface.delta = sys_params.get('Volatility_Delta', 0.0)
                    forward = riskfactors.construct_factor(energy_factor, price_factors, factor_interp)
                    discount = riskfactors.construct_factor(discount_factor, price_factors, factor_interp)
                except Exception:
                    logging.error('Unable to bootstrap {0} - skipping'.format(market_price), exc_info=True)
                    continue

                # need to loop over this and create some market prices.
                quote_type = implied_params['instrument']['Quote_Type']
                for option in implied_params['instrument']['Energy_Futures_Options']:
                    t = discount.get_day_count_accrual(
                        sys_params['Base_Date'], (option['Expiry_Date'] - sys_params['Base_Date']).days)
                    d = discount.get_day_count_accrual(
                        sys_params['Base_Date'], (option['Settlement_Date'] - sys_params['Base_Date']).days)
                    expiry_excel = (option['Expiry_Date'] - utils.excel_offset).days
                    settlement_excel = (option['Settlement_Date'] - utils.excel_offset).days
                    forward_at_exp = forward.current_value(expiry_excel)
                    forward_at_settle = forward.current_value(settlement_excel)
                    r = discount.current_value(t)
                    if quote_type == 'Implied_Volatility':
                        sigma = vol_surface.current_value([[t, d, 1.0]])[0] if not option['Quoted_Market_Value'] else \
                            option['Quoted_Market_Value']
                        sigma += vol_surface.delta
                    else:
                        logging.error('quote_type {} not supported yet'.format(quote_type))
                        continue

                    option['Strike'] = forward_at_exp if not option['Strike'] else option['Strike']
                    option['Forward'] = forward_at_settle
                    option['r'] = r
                    option['S'] = d
                    option['T'] = t
                    option['sigma'] = sigma
                    option['Premium'] = utils.black_european_option_price(
                        option['Forward'], option['Strike'], r, sigma, t,
                        option['Units'], 1.0 if option['Option_Type'] == 'Call' else -1.0)

                result = scipy.optimize.minimize(
                    calc_error, (0.5, 0.1),
                    args=(implied_params['instrument']['Energy_Futures_Options'],),
                    bounds=[(0.001, 2.5), (-1, 2.0)])

                # log the results
                for option in implied_params['instrument']['Energy_Futures_Options']:
                    vol = np.sqrt(V(result.x[0], result.x[1], option['T'], option['S']) / option['T'])
                    discount = np.exp(-option['r'] * option['T'])
                    fitted_premium = utils.black_european_option_price(
                        option['Forward'], option['Strike'], 0.0,
                        np.sqrt(V(result.x[0], result.x[1], option['T'], option['S'])),
                        1.0, option['Units'], 1.0 if option['Option_Type'] == 'Call' else -1.0) * discount
                    err = (fitted_premium - option['Premium']) ** 2
                    logging.info(
                        'Commodity {} strike {}, expiry {}, vol {}, c_vol {}, premium {}, c_premium {}, err {}'.format(
                            implied_params['instrument']['Energy'], option['Strike'], option['Expiry_Date'],
                            option['sigma'], vol, option['Premium'], fitted_premium, err))

                price_param = utils.Factor(self.__class__.__name__, market_factor.name)

                price_factors[utils.check_tuple_name(price_param)] = {
                    'Property_Aliases': None,
                    'Sigma': result.x[0],
                    'Alpha': result.x[1]}


class OptionQuoteFamily(object):
    documentation = (
        'Fx And Equity',
        ['The quote preparation every European option family in this module shares - what a block',
         'names, what its quote type reads, the forward its premia are priced off and where the',
         'vol surface is looked up. The MODEL is the subclass\'s; this is the ladder.',
         '',
         'A family here is ASSET CLASS AGNOSTIC - the *Underlying* may be any spot (0D) price',
         'factor (**FxRate**, **EquityPrice**, **CommodityPrice**, **FuturesPrice**) and the',
         '*Volatility* any (moneyness, expiry) vol surface (**FXVol**, **EquityPriceVol**,',
         '**CommodityPriceVol**); the type of each is looked up from the price factors, or named',
         'explicitly with *Underlying_Type* / *Volatility_Type*. The optional *Yield* (a dividend,',
         'repo, convenience or carry curve) enters as $q$ - the drift is $r-q$ and the value',
         'carries the extra $e^{-qt}$ factor - so equity, FX and commodity underlyings are all',
         'handled by the same objective.',
         '',
         'THE FORWARD IS THE PRICER\'S. $r$ is the *Discount_Rate* curve and is what the premium',
         'discounts on; the forward GROWS at the optional *Funding_Rate* curve instead where one is',
         'named, which is the curve `utils.calc_eq_forward` integrates - an equity\'s own repo curve',
         '(**EquityPrice.Interest_Rate**), not the curve its deals discount on. Left blank the two are',
         'one curve, which is the one-curve world and what an FX pair always was; named, an index',
         'carrying a repo/borrow spread calibrates at the forward it is priced at rather than one a',
         'spread away from it.',
         '',
         'Target premia are the Black prices at the corresponding vol surface point (as per the Clewlow',
         'Strickland bootstrapper) unless *Quote_Type* is **Premium**, in which case the quoted values are',
         'used directly. A previously bootstrapped price factor (if present) is used to warm start the fit.',
         '',
         'WHICH REFERENCES ARE REQUIRED IS THE QUOTE TYPE\'S. **Implied_Volatility** reads *Underlying*,',
         '*Volatility* and *Discount_Rate*; **Premium** reads *Underlying* and *Discount_Rate* and NO',
         'surface at all - a listed chain is calibrated to its own prints rather than to somebody\'s fit',
         'to them. *Yield* and *Funding_Rate* are optional under both. A reference a block does not name',
         'and its quote type reads REFUSES by name; it does not skip.',
         '',
         'MONEYNESS CONVENTION. Unlike the other bootstrappers this one queries the surface AWAY FROM',
         'THE MONEY, where the five moneyness conventions in this framework no longer coincide, so the',
         'lookup point is produced by `pricing.calc_moneyness` - the same dispatch every option deal',
         'uses - off the surface *SubType*, with *Use_Forward* and *Invert_Moneyness* (Yes/No, both',
         'defaulting to **No**, i.e. $\\frac{S}{K}$, as they do in the pricing path). Supported',
         '*Surface_Types* are **Explicit**, **Relative_Forward** and **Malz** - the ones whose vol at a',
         'strike is a table lookup. **SVI** and **Skew** surfaces are parametric (the vol needs the',
         'ATM_Ref/wing machinery of the pricing path) and are REFUSED with an error rather than',
         'mis-looked-up: quote those premiums directly with *Quote_Type* **Premium**.'
         ]
    )

    # Every premium here is read in double: the framework default (float32) destroys the
    # cancellation, so the dtype this is constructed with is deliberately ignored.
    prec = torch.float64
    # candidate types per input: any spot (0D) factor, any (moneyness, expiry) surface - so one
    # instrument definition serves FX, equity and commodity underlyings
    factor_types = {'Underlying': ['FxRate', 'EquityPrice', 'CommodityPrice', 'FuturesPrice'],
                    'Volatility': utils.TwoDimensionalFactors,
                    'Discount_Rate': ['InterestRate'],
                    'Yield': ['DividendRate', 'InterestRate'],
                    'Funding_Rate': ['InterestRate']}

    #: What each quote type requires. `Implied_Volatility` prices its target premium off the
    #: surface; `Premium` is handed the number, so a `Volatility` name on such a block is inert and
    #: a chain-sourced ladder fits with no surface in the book at all. Declared here rather than as
    #: a `default=REQUIRED`, which one static default cannot express.
    quote_type_references = {'Implied_Volatility': ('Underlying', 'Volatility', 'Discount_Rate'),
                             'Premium': ('Underlying', 'Discount_Rate')}

    #: The carry references, read whenever named and required by no quote type: no `Yield` is no
    #: carry, no `Funding_Rate` is a forward funded by `Discount_Rate`. A fit reads these plus what
    #: its quote type requires; a reference in neither list is not looked at.
    optional_references = ('Yield', 'Funding_Rate')

    #: What an optional reference's absence means, appended to its declared description.
    reference_notes = {
        'Volatility': '. REQUIRED under Quote_Type Implied_Volatility, which prices its target '
                      'premium off it; INERT under Premium, where the quote IS the premium and no '
                      'surface is read at all',
        'Yield': '. Blank is no carry, q = 0',
        'Funding_Rate': '. The curve the FORWARD grows at - an equity\'s own repo curve '
                        '(EquityPrice.Interest_Rate), which is what utils.calc_eq_forward '
                        'integrates, rather than the curve the premium discounts on. Blank funds '
                        'the forward off Discount_Rate, which is the one-curve world and what an '
                        'FX pair always was'}
    # Surface_Types whose vol at a strike is a table lookup, hence usable here. SVI/Skew are
    # parametric - Factor2D returns the parameters, not a vol - so a synthesised premium would be
    # silently wrong.
    tabular_surfaces = ('Explicit', 'Relative_Forward', 'Malz')

    # the five factor references, each with the optional `_Type` `resolve` reads; what is REQUIRED
    # is derived from `quote_type_references` - see `reference_fields`
    fields = reference_fields(factor_types, quote_type_references, reference_notes) + [
        F('Quote_Type', 'Text', default='Implied_Volatility',
          values=['Implied_Volatility', 'Premium'],
          description='Whether Quoted_Market_Value is a vol to price at or a premium to fit'),
        F('Use_Forward', 'Text', default='No', values=['Yes', 'No'],
          description='Moneyness against the forward rather than the spot'),
        F('Invert_Moneyness', 'Text', default='No', values=['Yes', 'No'],
          description='Moneyness as K/S rather than S/K'),
        F('Steps_Per_Year', 'Float', default=252.0,
          description='GARCH steps an expiry is spread over'),
        F('Quote_Timestamp', 'Date', default='',
          description='When the quotes were seen - the vol surface\'s own as-of where this block '
                      'was authored off one (fx_surface_block). Stored, logged and reported; '
                      'nothing in the fit reads it, because what counts as too old is the '
                      'consumer\'s policy and not the parameters\''),
        F('Quote_Source', 'Text', default='',
          description='How this block was authored, in one line: what the vols were read off and '
                      '- where the surface does not carry an expiry the ladder asks for - the '
                      'nearest quoted one used instead. Logged beside the fitted parameters, so a '
                      'substituted pillar is in the record rather than interpolated silently'),
        F('European_Options', 'Table', default='null', row=Row(OPTION_QUOTE + QUOTE_TWO_WAY),
          description='The option quotes the five parameters are fitted to, each with the two-way '
                      'it was dealt on and the print\'s own clock where the source printed them')]

    def __init__(self, param, device, dtype):
        self.device = device
        self.param = param

    @classmethod
    def resolve(cls, instrument, field, price_factors):
        """The factor named by instrument[field], typed by the first candidate that exists in the
        price factors, or by an explicit instrument[field + '_Type']. None if the field is unset."""
        if not instrument.get(field):
            return None
        return resolve_factor(instrument[field], price_factors, [instrument[field + '_Type']]
                              if instrument.get(field + '_Type') else cls.factor_types[field])

    @classmethod
    def resolve_references(cls, market_price, instrument, price_factors, factor_interp):
        """`{field: constructed factor}` for every reference this block names and its quote type
        reads - and a named refusal for every one it needs and cannot get.

        A missing reference refuses; it does not skip. What is required is the quote type's
        (`quote_type_references`), so a `Volatility` name on a `Premium` block is not resolved at
        all; `optional_references` are read whenever named. Everything is resolved before an option
        is looked at, so a book carrying two ladders fails on the one that is wrong.
        """
        quote_type = instrument.get('Quote_Type')
        if quote_type not in cls.quote_type_references:
            raise ValueError(
                '{}: Quote_Type {!r} is not one this family fits - it takes {}. Implied_Volatility '
                'prices each quote at the Volatility surface and fits that premium; Premium fits '
                'the number in Quoted_Market_Value directly'.format(
                    market_price, quote_type, ' or '.join(sorted(cls.quote_type_references))))
        required = cls.quote_type_references[quote_type]
        resolved = {}
        for field in tuple(required) + cls.optional_references:
            if not instrument.get(field):
                if field in required:
                    raise ValueError(
                        '{}: Quote_Type {} requires {}, and {} is blank. Name the factor the fit '
                        'should read; a reference the block does not name is a calibration that '
                        'writes no price factor{}'.format(
                            market_price, quote_type, '/'.join(required), field,
                            '. Quote the premiums directly (Quote_Type Premium), which requires '
                            'only {}, if the book carries no surface to price them off'.format(
                                '/'.join(cls.quote_type_references['Premium']))
                            if field == 'Volatility' else ''))
                continue
            try:
                factor = cls.resolve(instrument, field, price_factors)
            except StopIteration:
                raise ValueError(
                    '{}: {} names {!r} and the book\'s Price Factors carry no {} block for it. Add '
                    'the factor, or point the field at one the book carries'.format(
                        market_price, field, instrument[field],
                        '/'.join('{}.{}'.format(candidate, instrument[field])
                                 for candidate in ([instrument[field + '_Type']]
                                                   if instrument.get(field + '_Type')
                                                   else cls.factor_types[field]))))
            try:
                resolved[field] = riskfactors.construct_factor(
                    factor, price_factors, factor_interp)
            except Exception as failure:
                raise ValueError(
                    '{}: {} names {!r}, which resolved to {} and would not construct: {}'.format(
                        market_price, field, instrument[field],
                        utils.check_tuple_name(factor), failure))
        return resolved

    def tensor(self, x):
        return torch.tensor(float(x), device=self.device, dtype=self.prec)

    def vector(self, xs):
        return torch.tensor([float(x) for x in xs], device=self.device, dtype=self.prec)

    @classmethod
    def moneyness(cls, strike, spot, forward, vol_surface, use_forward, invert_moneyness):
        """The moneyness coordinate to look the vol surface up at.

        Five conventions, dispatched off the surface's SubType, so this delegates to
        `pricing.calc_moneyness` - the same function every option deal uses. That reads only the
        SubType out of deal_data, so a minimal `DealDataType` carrying it is all it needs.
        """
        deal_data = utils.DealDataType(
            Instrument=None, Time_dep=None, Calc_res=None,
            Factor_dep={'Volatility': [(None, None, vol_surface.get_subtype())]})
        return float(pricing.calc_moneyness(
            *[torch.tensor(float(x), dtype=cls.prec) for x in (strike, spot, forward)],
            deal_data, use_forward, invert_moneyness))

    #: The FX ladder the desk deals: the ATM term structure identifies H0/Beta/Omega, the 25
    #: delta wings identify Gamma_Star (the skew) and Alpha (the wings' width). Nothing past 1Y -
    #: TARFs and accumulators are sub-year products.
    fx_atm_expiries = (1.0 / 12.0, 2.0 / 12.0, 0.25, 0.5, 0.75, 1.0)
    fx_wing_expiries = (0.25, 0.5)
    #: The delta pillars each wing expiry is quoted at. One here; a family whose buckets are the
    #: wing expiries wants two, a bucket freeing one parameter per wing quote (spec 5.4.5).
    fx_wing_pillars = (0.25,)
    #: Days a surface expiry in years is emitted as: a quote block carries DATES, so `Expiry_Date`
    #: is the nearest whole day and the residual is the rounding alone (a 1M pillar emits as 30).
    fx_days_per_year = 365.0
    #: How far past the ladder's longest rung a surface pillar may still be snapped to. Snapping is
    #: an argmin and has no ceiling, so without this a 2Y/5Y-only surface answers every rung with
    #: 2Y. A week is the width of the same pillar quoted from a different date.
    fx_expiry_tolerance = 7.0 / 365.0
    #: Distinct (expiry, strike) contracts the ladder must survive snapping with. Ten rungs are not
    #: ten quotes: a two-pillar surface collapses them onto four, and four do not identify five
    #: parameters. A floor rather than a guarantee - the fit still reports parameters on a bound.
    fx_minimum_contracts = 6

    @classmethod
    def fx_surface_expiry(cls, surface, expiry, cap):
        """The surface's own expiry nearest `expiry` at or under `cap`, and whether it had to
        substitute. `(None, True)` where the surface carries no admissible pillar at all.

        The quote moves to the nearest pillar the surface was BUILT from and the block records it in
        `Quote_Source`; interpolating between two would put a number nobody quoted into the
        objective. `cap` is the ladder's longest rung widened by `fx_expiry_tolerance`, and a rung
        with nothing admissible under it is dropped and recorded.
        """
        admissible = surface.expiry[surface.expiry <= cap + cls.fx_expiry_tolerance]
        if not admissible.size:
            return None, True
        nearest = float(admissible[np.argmin(np.abs(admissible - expiry))])
        return nearest, not np.isclose(nearest, expiry)

    @staticmethod
    def fx_atm_coordinate(vol_at, T, iterations=64):
        """`(x, vol)` of the delta-neutral straddle on a Malz surface at expiry `T`.

        The ATM convention `FXVolSurfaceParameters` writes and `Factor2D.malz_skew` places the +-0.5
        label's vol at: `K = F exp(-sigma^2 T/2)`, so `x = sigma^2 T / 2` with `sigma` the surface's
        own vol at that x. Reading at `x = 0` would be the ATMF vol, a different number on a skewed
        smile. Iterated as a fixed point - a contraction, slope of order 1e-2 here.
        """
        vol = vol_at(0.0)
        for _ in range(iterations):
            moved = vol_at(0.5 * vol * vol * T)
            if abs(moved - vol) < 1e-14:
                vol = moved
                break
            vol = moved
        return 0.5 * vol * vol * T, vol

    @staticmethod
    def fx_pillar_delta(vol_at, T, x, side):
        """The premium-adjusted forward delta magnitude at log-moneyness `x = log(F/K)`, on the call
        wing (`side` +1) or the put wing (-1) - `(K/F)N(d2)` and `(K/F)N(-d2)`.

        The one delta convention the Malz solve inverts, so inverting this finds the strike that
        solve placed the pillar's vol at. `d2` is built off the surface's own vol at `x`, which
        makes it a smile delta rather than a flat-vol one.
        """
        vol = vol_at(x)
        d2 = (x - 0.5 * vol * vol * T) / (vol * np.sqrt(T))
        return float(np.exp(-x) * scipy.stats.norm.cdf(side * d2))

    @classmethod
    def fx_pillar_coordinate(cls, vol_at, T, pillar, side, x_atm, iterations=100):
        """The log-moneyness whose premium-adjusted forward delta is `pillar` on one wing.

        Bisection between the delta-neutral straddle and a strike three log-units out: the delta is
        monotone in `x` along each wing. An unbracketed pillar refuses by name rather than clamping,
        a clamped strike being one that enters the objective as if it were the wing.
        """
        far = 3.0
        low, high = (-far, x_atm) if side > 0 else (x_atm, far)
        error = lambda x: cls.fx_pillar_delta(vol_at, T, x, side) - pillar
        if error(low) * error(high) > 0.0:
            raise ValueError(
                'the {:g} delta {} at expiry {:.4f} is not reachable on this surface - its delta '
                'runs from {:.4f} to {:.4f} over log-moneyness [{:g}, {:g}]. Quote the wing the '
                'surface carries, or widen it'.format(
                    pillar, 'call' if side > 0 else 'put', T,
                    cls.fx_pillar_delta(vol_at, T, low, side),
                    cls.fx_pillar_delta(vol_at, T, high, side), low, high))
        for _ in range(iterations):
            middle = 0.5 * (low + high)
            if error(low) * error(middle) <= 0.0:
                high = middle
            else:
                low = middle
        return 0.5 * (low + high)

    @staticmethod
    def fx_black_vega(forward, strike, rate, vol, T):
        """Black vega of one unit of the option - `exp(-rT) F n(d1) sqrt(T)`.

        The objective weight before normalisation: vega makes it scale-free across a term structure,
        where an unweighted least squares would fit the back end alone. Puts and calls share it.
        """
        stddev = vol * np.sqrt(T)
        d1 = (np.log(forward / strike) + 0.5 * stddev * stddev) / stddev
        return float(np.exp(-rate * T) * forward * scipy.stats.norm.pdf(d1) * np.sqrt(T))

    @classmethod
    def fx_surface_block(cls, pair, price_factors, sys_params, factor_interp):
        """`(Market Prices name, block)` - this family's quote block, authored off a pair's built
        `FXVol` surface.

        THE LADDER: vega-weighted implied vols read off the surface - ATM at `fx_atm_expiries`,
        plus each of `fx_wing_pillars` on both wings at `fx_wing_expiries` - normalised by Black
        vega off the same surface.
        An expiry the surface does not carry moves to the nearest quoted one at or under 1Y, or is
        dropped where it carries none; `Quote_Source` records either.

        Ten rungs are not ten quotes: a substituted rung lands on a contract another rung already
        named, and a repeat is a weight rather than an observation. So DISTINCT `(expiry, strike)`
        contracts are counted after snapping and a ladder below `fx_minimum_contracts` refuses.

        The vols are the surface's, UNSHIFTED - `Volatility_Delta` is a shift the fit applies to
        every quoted vol it prices a premium off, and applying it here too would bump twice.

        TWO CLOCKS, deliberately. `T` is the surface's own expiry axis and is what the surface is
        read at; `t` is what the emitted `Expiry_Date` resolves to through the discount curve's day
        count and is what the FORWARD hangs off. They agree only under ACT_365 - reading the surface
        at `t` under ACT_360 puts the 1Y rung past the last expiry the surface carries.

        The strikes are the surface's own coordinates: the ATM one is the delta-neutral straddle
        `K = F exp(-sigma^2 T/2)`, each wing the strike whose premium-adjusted forward delta is one
        of `fx_wing_pillars`, found by inverting the delta the Malz solve inverted off the same
        vols.

        No `Funding_Rate` is declared, and an FX pair needs none: `Discount_Rate` and `Yield` are
        exactly the pair `utils.calc_fx_forward` builds the priced forward from, so the calibrated
        forward already grows at the curve the pricer grows it on.

        ORIENTATION. An `FXVol.A.B` x-axis is `log(F/K)` for `A` priced in `B`, while the `FxRate`
        fitted is priced in the DOMESTIC currency - so the underlying is whichever token is not
        domestic, and the block declares `Use_Forward` Yes with `Invert_Moneyness` as the deal sets
        it. Inverting flips the sign of Gamma_Star's skew, so what is written describes the rate the
        pricer simulates, orientation included.

        Refuses by name, with the remedy, on: no built surface, a surface type no strike can be
        looked up on, a ladder below `fx_minimum_contracts`, a cross against the reporting currency,
        and a missing spot or discount curve.
        """
        name = utils.check_rate_name(pair)
        vol_name = utils.check_tuple_name(utils.Factor('FXVol', name))
        if vol_name not in price_factors:
            raise ValueError(
                'no {} in the book\'s Price Factors - there is no built surface to read {} off. '
                'Tick the pair\'s FXVolPrices block first (/book/market or /book/bloomberg), '
                'which bootstraps it'.format(vol_name, pair))

        surface = riskfactors.construct_factor(
            utils.Factor('FXVol', name), price_factors, factor_interp)
        subtype = surface.get_subtype()
        if subtype[0] not in cls.tabular_surfaces:
            raise ValueError(
                '{} has Surface_Type {} - only {} surfaces carry a vol AT A STRIKE, which is what '
                'a quote is. Author the quotes as premiums (Quote_Type Premium) instead'.format(
                    vol_name, subtype[0], '/'.join(cls.tabular_surfaces)))

        # `Factor2D.current_value` only interpolates a grid with two of each coordinate; handed a
        # degenerate one it answers the whole flat vol vector
        if surface.expiry.size < 2 or surface.moneyness.size < 2:
            raise ValueError(
                '{} carries {} expiries x {} moneyness nodes - a surface has to be a grid before a '
                'vol can be read off it at a strike. Quote the pair at more than one '
                'expiry'.format(vol_name, surface.expiry.size, surface.moneyness.size))

        base_date = sys_params['Base_Date']
        domestic = sys_params.get('Base_Currency', 'USD')
        # the underlying is whichever token is not domestic; the moneyness inverts exactly where
        # FXOptionDeal inverts it, on the surface's first token being the domestic
        if domestic not in name:
            raise ValueError(
                '{} is a cross against the reporting currency {} - neither leg is an FxRate this '
                'family can fit, because an FxRate is priced in the domestic currency. Author '
                'the {} block by hand, naming the Underlying and its Discount_Rate/Yield '
                'explicitly'.format(pair, domestic, cls.market_factor_type))
        underlying = name[1] if name[0] == domestic else name[0]
        invert = name[0] == domestic

        spot_name = utils.check_tuple_name(utils.Factor('FxRate', (underlying,)))
        if spot_name not in price_factors:
            raise ValueError('no {} in the book\'s Price Factors - a smile is quoted around a '
                             'spot, and the parameters this writes describe that rate\'s own '
                             'dynamics. Add the FxRate block for {}'.format(
                                 spot_name, underlying))
        spot_block = price_factors[spot_name]
        # the carry legs: the FxRate's own foreign curve and the one it is priced in - the pair the
        # FX forward is built from
        carry_name = spot_block.get('Interest_Rate') or underlying
        discount_name = spot_block.get('Domestic_Currency') or domestic
        for curve in (discount_name, carry_name):
            if utils.check_tuple_name(
                    utils.Factor('InterestRate', utils.check_rate_name(curve))) not in price_factors:
                raise ValueError(
                    'no InterestRate.{0} in the book\'s Price Factors - the strikes hang off the '
                    'forward, and the forward is this pair\'s two curves. Add the {0} curve, or '
                    'point {1}\'s Interest_Rate / Domestic_Currency at curves the book '
                    'carries'.format(curve, spot_name))

        spot = float(riskfactors.construct_factor(
            utils.Factor('FxRate', (underlying,)), price_factors, factor_interp).current_value()[0])
        discount = riskfactors.construct_factor(
            utils.Factor('InterestRate', utils.check_rate_name(discount_name)),
            price_factors, factor_interp)
        carry = riskfactors.construct_factor(
            utils.Factor('InterestRate', utils.check_rate_name(carry_name)),
            price_factors, factor_interp)
        cap = max(cls.fx_atm_expiries)

        def pillar(expiry):
            """One admissible expiry's `(T, moved, days, t, F, r, vol_at)`, or `None` where the
            surface carries no pillar the ladder may snap to. `T` is the surface's coordinate and
            `t` the emitted date's accrual - see the two-clock note in `fx_surface_block`."""
            T, moved = cls.fx_surface_expiry(surface, expiry, cap)
            if T is None:
                return None
            days = int(round(T * cls.fx_days_per_year))
            t = discount.get_day_count_accrual(base_date, days)
            rate = float(discount.current_value(t))
            forward = spot * np.exp((rate - float(carry.current_value(t))) * t)
            # the surface unshifted - the fit applies `Volatility_Delta` to every vol it reads
            return T, moved, days, t, forward, rate, (
                lambda x: float(surface.current_value([[x, T]])[0]))

        quotes, substituted = [], []

        def quote(days, forward, rate, t, x, vol):
            """One `OPTION_QUOTE` row: the strike this coordinate names in the underlying's own
            units (inverting `calc_moneyness`, as `Invert_Moneyness` declares), the surface's vol
            there, and the Black vega that becomes its weight."""
            strike = forward * np.exp(x if invert else -x)
            quotes.append({
                'Expiry_Date': base_date + pd.DateOffset(days=days), 'Strike': strike,
                # the OTM leg, which is the one a desk deals; the fit is blind to the choice, puts
                # being priced by parity off the call
                'Option_Type': 'Call' if strike >= forward else 'Put', 'Units': 1.0,
                'Weight': cls.fx_black_vega(forward, strike, rate, vol, t),
                'Quoted_Market_Value': vol})

        for expiry in cls.fx_atm_expiries:
            found = pillar(expiry)
            if found is None:
                substituted.append('ATM {:g} DROPPED - no pillar at or under {:g}'.format(
                    expiry, cap))
                continue
            T, moved, days, t, forward, rate, vol_at = found
            x, vol = cls.fx_atm_coordinate(vol_at, T)
            quote(days, forward, rate, t, x, vol)
            if moved:
                substituted.append('ATM {:g} -> {:g}'.format(expiry, T))

        wings = '/'.join('{:g}'.format(x) for x in cls.fx_wing_pillars)
        for expiry in cls.fx_wing_expiries:
            found = pillar(expiry)
            if found is None:
                substituted.append('{}d {:g} DROPPED - no pillar at or under {:g}'.format(
                    wings, expiry, cap))
                continue
            T, moved, days, t, forward, rate, vol_at = found
            x_atm, _ = cls.fx_atm_coordinate(vol_at, T)
            for delta in cls.fx_wing_pillars:
                for side in (1.0, -1.0):
                    x = cls.fx_pillar_coordinate(vol_at, T, delta, side, x_atm)
                    quote(days, forward, rate, t, x, vol_at(x))
            if moved:
                substituted.append('{}d {:g} -> {:g}'.format(wings, expiry, T))

        # a repeated contract is a weight rather than an observation, so what is counted is the
        # number of DISTINCT (expiry, strike) contracts
        contracts = {(point['Expiry_Date'], point['Strike']) for point in quotes}
        if len(contracts) < cls.fx_minimum_contracts:
            raise ValueError(
                '{} carries pillars {} - the ladder (ATM {}, {}d wings {}) collapses onto {} '
                'distinct contract{} on it, and {} do not identify {}, and a collapsed ladder has '
                'no term structure in it. Quote the pair at more expiries (at least {} distinct '
                'contracts, so at least three pillars at or under {:g}), or author the '
                '{} block by hand. What each rung did: {}'.format(
                    vol_name, '/'.join('{:g}'.format(x) for x in surface.expiry),
                    '/'.join('{:g}'.format(x) for x in cls.fx_atm_expiries), wings,
                    '/'.join('{:g}'.format(x) for x in cls.fx_wing_expiries), len(contracts),
                    '' if len(contracts) == 1 else 's', len(contracts),
                    cls.identification_note, cls.fx_minimum_contracts, cap,
                    cls.market_factor_type,
                    ', '.join(substituted) or 'every rung landed on a pillar it was asked for'))

        # normalised over the rungs as emitted - the weights are relative in the objective
        total = sum(point['Weight'] for point in quotes)
        if not total > 0.0:
            raise ValueError('{} priced every quote at zero vega, so there is no weight to '
                             'normalise and nothing the fit would be sensitive to. Quote the '
                             'surface at a positive vol'.format(vol_name))
        for point in quotes:
            point['Weight'] /= total

        source = '{} ATM {} + {}d wings {}, off {} as at {}'.format(
            len(quotes), '/'.join('{:g}'.format(x) for x in cls.fx_atm_expiries),
            wings, '/'.join('{:g}'.format(x) for x in cls.fx_wing_expiries), vol_name,
            price_factors[vol_name].get('Quote_Timestamp') or 'no stated time')
        if substituted:
            source += ('; rungs the surface does not carry, moved to the nearest quoted at or '
                       'under {:g} or dropped where it carries none: {}'.format(
                           cap, ', '.join(substituted)))

        declared = {field.name: field.default for field in cls.fields}
        return utils.check_tuple_name(utils.Factor(cls.market_factor_type, (underlying,))), {
            'instrument': {
                'Underlying': underlying, 'Underlying_Type': 'FxRate',
                'Volatility': '.'.join(name), 'Volatility_Type': 'FXVol',
                'Discount_Rate': discount_name, 'Discount_Rate_Type': 'InterestRate',
                'Yield': carry_name, 'Yield_Type': 'InterestRate',
                'Quote_Type': 'Implied_Volatility',
                # the surface's x-axis is log(F/K) on the pair, so the lookup is against the
                # forward and inverts where the pair's own deals invert it
                'Use_Forward': 'Yes', 'Invert_Moneyness': 'Yes' if invert else 'No',
                # the step clock is what the fitted parameters mean - a deal's `Steps_Per_Year`
                # must be this number - so it is stated, at the field's own declared default
                'Steps_Per_Year': declared['Steps_Per_Year'],
                'Quote_Timestamp': price_factors[vol_name].get('Quote_Timestamp') or '',
                'Quote_Source': source,
                'European_Options': quotes}}

    @staticmethod
    def effective_yield(discount_rate, funding, carry, t):
        """The `q` the objective runs on at accrual `t`: the dividend (or foreign) carry, plus the
        basis between the curve the premium discounts on and the curve the forward grows at.

        The whole arithmetic hangs off `r` and `q`: the forward is `spot exp((r-q)t)`, the per-step
        carry `(r-q)t/n`, and the value carries `exp(-qt)` so it discounts at `r`. Folding the
        funding basis `r - f` into `q` grows the forward at `f - carry` - what
        `utils.calc_eq_forward` integrates - while the premium still discounts at `r`.

        With no `Funding_Rate` the basis term is not evaluated, so `q` is the plain carry. Every leg
        is read at the accrual the `Discount_Rate` curve's own day count gives.
        """
        q = 0.0 if carry is None else float(carry.current_value(t))
        return q if funding is None else q + (discount_rate - float(funding.current_value(t)))

    @classmethod
    def resolve_block(cls, market_price, instrument, price_factors, factor_interp, sys_params):
        """`({field: constructed factor}, spot)` - everything a block names, resolved before an
        option is looked at, so a book carrying two ladders fails on the one that is wrong.

        A missing reference refuses by name (`resolve_references`), and so does a surface whose vol
        is not a table lookup at a strike: a mis-looked-up vol converges to the wrong answer.
        """
        factors = cls.resolve_references(market_price, instrument, price_factors, factor_interp)
        surface = factors.get('Volatility')
        if surface is not None:
            surface.delta = sys_params.get('Volatility_Delta', 0.0)
        if instrument['Quote_Type'] == 'Implied_Volatility':
            subtype = surface.get_subtype()
            if subtype[0] not in cls.tabular_surfaces:
                raise ValueError(
                    '{0}: volatility {1} has Surface_Type {2} (Moneyness_Rule {3}); only {4} '
                    'surfaces carry a vol AT A STRIKE, which is what Quote_Type '
                    'Implied_Volatility prices its target premium at. Quote the premiums directly '
                    '(Quote_Type Premium) instead'.format(
                        market_price, instrument['Volatility'], subtype[0], subtype[1],
                        '/'.join(cls.tabular_surfaces)))
        return factors, float(factors['Underlying'].current_value()[0])

    def prepare_quotes(self, sys_params, instrument, factors, spot):
        """Each `European_Options` row as `(option, t, r, q, forward, sign, strike, sigma,
        premium)` - the numbers every family in this hierarchy builds its objective from.

        `t` is the `Discount_Rate` curve's own day count accrual, the forward grows at
        `effective_yield`'s `r - q`, and a blank `Strike` is the forward. Under
        `Implied_Volatility` the premium is Black at the surface's vol AT THE STRIKE; under
        `Premium` the quote IS the premium and `sigma` is inverted back out of it, which is what
        seeds a fit and what its diagnostics are read in.
        """
        discount, surface = factors['Discount_Rate'], factors.get('Volatility')
        carry, funding = factors.get('Yield'), factors.get('Funding_Rate')
        quote_type = instrument['Quote_Type']
        use_forward = instrument.get('Use_Forward') == 'Yes'
        invert_moneyness = instrument.get('Invert_Moneyness') == 'Yes'
        for option in instrument['European_Options']:
            t = discount.get_day_count_accrual(
                sys_params['Base_Date'], (option['Expiry_Date'] - sys_params['Base_Date']).days)
            r = float(discount.current_value(t))
            q = self.effective_yield(r, funding, carry, t)
            forward = spot * np.exp((r - q) * t)
            sign = 1.0 if option['Option_Type'] == 'Call' else -1.0
            strike = forward if not option['Strike'] else option['Strike']
            if quote_type == 'Implied_Volatility':
                moneyness = self.moneyness(
                    strike, spot, forward, surface, use_forward, invert_moneyness)
                sigma = surface.current_value([[moneyness, t]])[0] if not option[
                    'Quoted_Market_Value'] else option['Quoted_Market_Value']
                sigma += surface.delta
                premium = utils.black_european_option_price(
                    forward, strike, r, sigma, t, option['Units'], sign)
            else:
                premium = option['Units'] * option['Quoted_Market_Value']
                # back out the Black vol of the quote (seeds the fit and the diagnostics)
                call = option['Quoted_Market_Value'] + (0.0 if sign > 0 else
                                                        forward - strike) * np.exp(-r * t)
                sigma = np.sqrt(utils.bs_implied_total_var(
                    call, spot * np.exp(-q * t), strike, r * t, 1) / t)
            yield option, t, r, q, forward, sign, strike, sigma, premium

    @staticmethod
    def quote_trailer(instrument):
        """Where the quotes came from, in the record beside the parameters they produced."""
        if instrument.get('Quote_Source') or instrument.get('Quote_Timestamp'):
            logging.info('  quotes: {} (as at {})'.format(
                instrument.get('Quote_Source') or 'authored by hand',
                instrument.get('Quote_Timestamp') or 'no stated time'))


#: One quote inside the fit: its block index `j`, the contract (`ratio` being the strike over spot,
#: built once because the price reads it every evaluation), and the two market numbers the
#: vol-space residual is built from - the target premium and the Black vega at the quoted vol.
LVQuote = namedtuple(
    'lv_quote',
    'row j spot strike ratio is_call units T rate carry forward premium vega weight '
    'sigma quoted')

#: One forward-start target: a window `[j1, j2]` of the same walk, a strike as a fraction of
#: S_T1, and the vol it is aimed at - spec 5.3's objective is in vol points, so this block carries
#: no premium. Under `Prior` the target is the model's own spot smile and `target` is unread.
LVForward = namedtuple('lv_forward', 'j1 j2 T1 tenor strike ratio carry weight target')

#: The three structural numbers the PRICE FACTOR declares, read here rather than re-spelt so the
#: calibrator's bound and the factor's own load assertion cannot disagree.
LV_FACTOR_DEFAULTS = {field.name: field.default
                      for field in riskfactors.LogVar2FJModelParameters.fields}

#: Which lever an expiry's wing quotes free first in `Bootstrap` mode (spec 5.4.1): two 25 delta
#: wings free the first two, four wings free all four, and what is left is TIED.
LV_FREE_ORDER = ('Mu_J', 'Sigma_S', 'Rho_S', 'Sigma_J')

#: The tenor grammar `Forward_Tenors` is written in.
LV_TENOR_UNITS = {'d': 1.0 / 365.0, 'w': 7.0 / 365.0, 'm': 1.0 / 12.0, 'y': 1.0}


def lv_tenor(text):
    """`'6m'` as years. The one grammar `Forward_Tenors` is written in."""
    text = text.strip().lower()
    if text[-1:] not in LV_TENOR_UNITS:
        raise ValueError(
            'Forward_Tenors: {!r} is not a tenor - write a number and one of {} (6m, 1y, 2w), '
            'each pair as T1:Delta and the list comma separated'.format(
                text, '/'.join(sorted(LV_TENOR_UNITS))))
    return float(text[:-1]) * LV_TENOR_UNITS[text[-1]]


class LVFit(object):
    """ONE LogVar2FJ calibration: the prepared quotes, the walk they are priced on, the fitted
    state, and every verb that moves it.

    The state is one dict of plain numbers (`self.state`) and one list of L levels; a fitted vector
    `x` is spliced into it by `build`, which is what makes a stage a two-line call. `bootstrap`
    prepares the quotes and orders the stages; everything else is here.
    """

    #: What the walk is handed per STEP: the four fitted levers and the structural intensity.
    step_names = utils.LV_BUCKET_NAMES + utils.LV_STRUCTURAL_CURVES
    #: The fitted box, per coordinate (spec 5.2). `Rho_S`'s own lower edge is derived from `Rho_L`
    #: and `C_Min` at every stage and so cannot live here.
    #: `Sigma_L`'s floor is for EXPOSURES (spec 5.2 stage 4): a two-year surface cannot claim
    #: five-year vol is certain, and a zero slow factor collapses a CVA profile's vol distribution
    #: onto the fast factor's spread, which reverts within months.
    box = {'Sigma_L': (0.3, 2.0), 'Rho_L': (-0.6, 0.0), 'Sigma_S': (0.5, 5.0),
           'Nu': (0.0, 0.6), 'Sigma_J': (0.02, 0.25), 'Mu_J': (-0.40, -0.03)}
    #: Stage 4's prior per ASSET CLASS, keyed by the factor type `Underlying` resolves to. The
    #: pair is a MAGNITUDE - its sign is the fitted fast leverage's, so the slow skew is never
    #: pinned against the fast one, which on an FX pair quoted the other way up is the whole of
    #: the rule. `Sigma_L`'s floor is the box beneath all four and never their default: a
    #: floor-by-default understates every pinned name's 2-5 year vol in one direction.
    slow_priors = {'FxRate': (0.2, 0.5), 'EquityPrice': (0.4, 1.0),
                   'CommodityPrice': (0.4, 1.0), 'FuturesPrice': (0.4, 1.0)}
    #: The horizon the three-way prior report reads its log-vol sd at, in years - the exposure
    #: tail the slow factor's prior is for, and the number a phase-3 exposure row will read.
    exposure_horizon = 5.0
    #: The spot slope or butterfly under which a stickiness RATIO divides by nothing, in vol
    #: points: the report prints the DIFFERENCES the fit targets and no ratio (spec 5.3).
    psi_floor = 0.5
    #: How far inside `C_Min` and inside `share_box` the soft penalties start, and what a full
    #: margin's violation costs: 1.25 vol points of residual, which the data rows at a one-point
    #: RMSE just outweigh - so a soft edge bites in the last percent and the BOX stops the fit.
    c_margin, soft_penalty = 0.05, 0.25
    #: What a degraded vanilla RMSE costs stage 5, per spec 5.3's failure mode: 0.1 vol points of
    #: degradation at the target maturities buys 10 vol points of residual, so forward skew is
    #: bought only where the spot smile can pay for it. Read in the FIRST-ORDER metric.
    vanilla_guard, vanilla_band = 100.0, 0.001
    #: The jump share's own box (5.1 step 2), applied to the REALISED share
    #: lambda(mu_J^2 + sigma_J^2)/xi_0 - below it the one-month wing is unreachable, above it the
    #: 6-12 month smiles carry more jump convexity than the market has.
    share_box = (0.05, 0.35)
    #: Chord steps one L pillar gets per round, how many rounds (each ending in one refreshed
    #: slope) it gets, and the largest move in log-variance one may take: the price is monotone in
    #: the level, so the damping only tames a first step off a bad seed.
    l_iterations, l_rounds, l_damping = 12, 3, 0.5
    #: The mass of path-days within 5 Cap_Beta of the cap that REFUSES the fit (spec 2.7), and the
    #: stationary log-vol sd VIX options imply, outside which the guard of 5.4.6 fires.
    cap_headroom_max, log_vol_sd_band = 1e-5, (0.4, 0.9)
    #: The relative ATM miss a pillar may still carry when the fit is done. The bootstrap solves to
    #: `Pillar_Tolerance` at every iterate; anything left here is a pillar the OTHER parameters
    #: have put out of reach.
    atm_miss_max = 1e-4
    #: Where the spec's stages cut the ladder, in years: stage 2 the wing's own horizon, stage 3
    #: the sub-year smile, stage 4 what is left beyond it.
    stage_horizons = (0.25, 1.0)
    #: Spec 5.3's forward-start strikes, as fractions of S_T1 - what `Forward_Smile_Source` Prior
    #: quotes its own rows at - and the three both stickiness ratios are read at: the 90-110 slope
    #: over its log-strike gap, and the 90/110 average less the ATM.
    prior_strikes = (0.90, 0.95, 1.00, 1.05, 1.10)
    psi_strikes = (0.90, 1.00, 1.10)
    #: The shortest WING expiry that identifies the slow pair (spec 5.2 stage 4), in years.
    slow_horizon = 1.5
    #: What one point of `Diffusive_Share` miss costs, on the same scale the other soft terms use.
    split_penalty = 0.25

    def __init__(self, family, market_price, instrument, factors, previous):
        self.family, self.market_price, self.instrument = family, market_price, instrument
        self.prec, self.device = family.prec, family.device
        self.tensor, self.vector = family.tensor, family.vector
        self.factors = factors
        #: the block COMPLETED by its own declarations, so every read is an index and the block
        #: and the declaration cannot disagree
        self.instrument = read = declared_defaults(type(family), instrument)
        self.mode = read['Fit_Mode']
        self.delta = 1.0 / float(read['Steps_Per_Year'])
        self.c_min, self.rcond = float(read['C_Min']), float(read['Jacobian_Rcond'])
        self.stationarity = float(read['Stationarity_Tol'])
        self.tolerance, self.max_iter = float(read['Tolerance']), int(read['Max_Iterations'])
        self.pillar_tol, self.smoothness = float(read['Pillar_Tolerance']), float(
            read['Bucket_Smoothness'])
        self.is_vol = read['Quote_Type'] == 'Implied_Volatility'
        self.source = read['Forward_Smile_Source']
        self.previous, self.tables, self.calls = previous, [], {'n': 0, 'j': 0, 'l': 0, 'b': 0}
        self.targets, self.guarded, self.base_rmse, self.ties = [], [], 0.0, {}
        #: the target DIFFERENCE pair per forward tenor, `Diffusive_Share`'s per-segment target
        #: where one is declared, the spot rung each forward tenor's own smile is read at, the
        #: history's slow pair where the job carries one, and the report lines the slow pair, its
        #: three-way table and an identified event day each earn
        self.tilt, self.diffusive, self.spot_rung = {}, None, {}
        self.pinned_slow, self.identified_days, self.history, self.slow_rows = '', [], None, []
        self.notes, self.final, self.leaf, self.theta = [], {}, None, None
        #: the last stage's box as `(lower, upper)` and the gradient `J^T r` it stopped on, which
        #: `interior` and `report` read the KKT active set off; and the stage that hit its cap
        self.edges, self.slope, self.capped = None, None, None

    def table(self, name):
        """One optional Table as rows: a completed block carries the declared blank `'null'` where
        a document carries a list, which is `schema.quote_rows`' own reading."""
        rows = self.instrument[name]
        return rows if isinstance(rows, list) else []

    def draw(self, paths, steps, seed):
        """The walk's two normals and its jump UNIFORMS - fixed for the whole fit, and antithetic.

        The uniforms rather than the counts, because a count is `lv_counts(u, lambda, .)` and stage
        5 moves lambda. Pseudo-random off `Random_Seed`: a Sobol block 3n dimensions wide buys
        nothing where every evaluation reads the same draws.
        """
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        half = max(int(paths) // 2, 1)
        kw = {'generator': gen, 'dtype': self.prec, 'device': self.device}
        z_l, z_s = torch.randn(half, steps, **kw), torch.randn(half, steps, **kw)
        u = torch.rand(half, steps, **kw)
        return torch.cat([z_l, -z_l]), torch.cat([z_s, -z_s]), torch.cat([u, 1.0 - u])

    def counted(self):
        """The integer counts at the CURRENT lambda strip, redrawn only where stage 5's bounded
        search for the jump share moves it - nothing else in the fit can."""
        return self.uniforms[0], self.uniforms[1], utils.lv_counts(
            self.uniforms[2], self.lam_at(), self.deltas)

    def lam_at(self):
        """`lambda(t)` at the walk's step STARTS: the segment strip of spec 5.1, read as `L` is."""
        return utils.bucket_at(self.knots[:-1], self.vector(self.state['Lambda']),
                               self.times[:-1])

    def lam_strip(self, w=None):
        """Spec 5.1's rule per segment, `lambda = w_J xi_mkt / (mu_J^2 + sigma_J^2)`, off the
        market's own forward-variance increments - so the jump share is the same number on every
        segment and the diffusive strip follows the market's shape. STRUCTURAL: never a fitted
        coordinate, re-derived from the strip whenever `w_J` moves."""
        share = self.share if w is None else w
        return [share * x / self.jump_size for x in self.xi]

    def walk(self, scalars, levers, curve, n):
        """`(M, Sigma^2)` CUMULATED to every grid point, off ONE internal-step walk.

        A maturity is then a prefix and a forward window the difference of two prefixes, so the
        whole quote set and the whole forward block cost one pass. The levers and `lambda(t)` are
        read at the ABSOLUTE step-START times, as `L` is: buckets and segments are calendar time
        from the base date, which is how `pricing.LogVar2FJKit` reads them. The fit is on the
        FxRate's own axis, never the reciprocal - one law per pair, carried at the deal.
        """
        params = dict(scalars, Lambda=self.lam_at()[:n],
                      **{name: utils.bucket_at(self.buckets, levers[name], self.times[:n])
                         for name in utils.LV_BUCKET_NAMES})
        eta_l, eta_s, counts = self.draws
        s = eta_l.new_zeros(eta_l.shape[0])
        l, M, var, a = s + curve[0], [], [], 0
        for b in [int(x) for x in self.upto if x <= n]:
            m, v, l, s = utils.lv_walk(
                dict(params, **{x: params[x][a:b] for x in self.step_names}),
                curve[a:b + 1], self.deltas[a:b], eta_l[:, a:b], eta_s[:, a:b], counts[:, a:b],
                (l, s), False)
            M.append(m)
            var.append(v)
            a = b
        return torch.stack(M, -1).cumsum(-1), torch.stack(var, -1).cumsum(-1)

    @staticmethod
    def conditional_black(M, var, carry, strike):
        """`(S_T/S - k)^+` PER PATH - spec 2.4's vanilla, which IS `pricing.lognormal_fired_gain`
        at this block's own Gaussian law; the caller averages, or weighs by the share measure.

        Priced off the SAMPLE forward: at the default paths the fixed draws leave
        `E[exp(M + var/2)]` up to 15 basis points off the analytic `exp(carry)`, which the L
        bootstrap would otherwise absorb into `L` as a fifth of a vol point of calibration noise.
        Dividing it out is a martingale control variate - one `logsumexp`, and it takes the level
        bias out of every strike of the block at once.
        """
        sigma = utils.sqrt_or_zero(var)
        drift = M + carry - torch.logsumexp(M + 0.5 * var, 0) + np.log(M.shape[0])
        return pricing.lognormal_fired_gain(
            1.0, drift, sigma, (torch.log(strike) - drift) / sigma, strike, True)

    def implied(self, price, forward, strike, T):
        """A conditional-Black price as a Black vol, DIFFERENTIABLE: the numpy inversion off the
        tape and then one Newton splice at its own vega, so the value is the true inversion and
        `dsigma/dtheta` is `dP/dtheta` over that vega - the splice the L pillar already returns."""
        sd = np.sqrt(max(utils.bs_implied_total_var(
            float(price.detach()), forward, strike, 0.0, 1), 0.0))
        d1 = (np.log(forward / strike) + 0.5 * sd * sd) / sd
        vega = max(forward * np.exp(-0.5 * d1 * d1) / np.sqrt(2.0 * np.pi), 1.0e-12)
        return (sd + (price - price.detach()) / vega) / np.sqrt(T)

    def smile(self, cum, j, carry, T, strikes):
        """The model's own smile at block `j`, as vols at `strikes` given as fractions of that
        block's forward - the reading both the forward rows and the stickiness ratios are taken
        on, so neither can drift from the other."""
        forward = np.exp(carry)
        return [self.implied(self.conditional_black(
            cum[0][..., j], cum[1][..., j], carry, self.tensor(k * forward)).mean(),
            forward, k * forward, T) for k in strikes]

    @classmethod
    def shape(cls, vols):
        """`(atm, slope, butterfly)` of a `psi_strikes` smile in DECIMAL vol: the 90-110 difference
        over its log-strike gap, and the 90/110 average less the ATM."""
        low, atm, high = vols
        return atm, (low - high) / np.log(cls.psi_strikes[2] / cls.psi_strikes[0]), \
            0.5 * (low + high) - atm

    def spot_shape(self, cum, tenor):
        """The model's OWN spot smile at the quoted maturity nearest `tenor`, as
        `(atm, slope, butterfly)` - what `Stickiness_Prior` tilts and what psi divides by."""
        j, carry, T = self.spot_rung[tenor]
        return self.shape(self.smile(cum, j, carry, T, self.psi_strikes))

    def market_shape(self, quotes):
        """The MARKET's own `(atm, slope, butterfly)` at one expiry, its quoted vols read in
        log-moneyness at `psi_strikes` - a quote itself wherever the rung carries one there. The
        side of the difference a source's own forward smile is measured against (spec 5.3).

        A rung that does not REACH 90 or 110 reads its nearest quote there, which understates the
        slope and the butterfly the difference is taken against; that is a target being measured
        off strikes nobody quoted, so it is named rather than left to the residual.
        """
        rows = sorted((np.log(quote.strike / quote.forward), quote.sigma) for quote in quotes)
        wants = [np.log(k) for k in self.psi_strikes]
        if min(wants) < rows[0][0] or max(wants) > rows[-1][0]:
            logging.warning(
                '{}: the {:g}y rung is quoted over {:.0%}-{:.0%} of its own forward and spec 5.3 '
                'reads the spot slope and butterfly at {}, so an end outside that is the NEAREST '
                'quote rather than the strike asked for, and the target DIFFERENCE this tenor '
                'is measured against understates both. Quote the wings at that expiry'.format(
                    self.market_price, quotes[0].T, np.exp(rows[0][0]), np.exp(rows[-1][0]),
                    '/'.join('{:g}'.format(k) for k in self.psi_strikes)))
        return self.shape(np.interp(wants, *zip(*rows)))

    def target_vol(self, target, shape):
        """The vol one forward row is aimed at: the model's OWN spot smile at that tenor, tilted
        by the target DIFFERENCES in slope and butterfly (spec 5.3).

        The quadratic in log-strike through the ATM whose slope and butterfly are the spot
        smile's plus `(Delta_skew, Delta_bfly)` - declared by `Stickiness_Prior` under `Prior`,
        measured off the source's own smiles under `Quotes` and `Reference`, both zero being
        sticky-delta. The forward ATM LEVEL is targeted by neither: the ATM ladder pins it.
        """
        d_skew, d_bfly = self.tilt[(target.T1, target.tenor)]
        atm, slope, bfly = shape[0], shape[1] + d_skew, shape[2] + d_bfly
        a, b = np.log(self.psi_strikes[0]), np.log(self.psi_strikes[2])
        q = -(bfly + 0.5 * slope * (a + b)) / (a * b)
        u = np.log(target.strike)
        return atm + (-slope - q * (a + b)) * u + q * u * u

    def value(self, cum, quote):
        """One quote's model premium: the conditional Black over the paths, the discount and the
        yield rescale as the component family applies them, and a put by parity off the ANALYTIC
        forward - so a put and a call at one strike cannot disagree by the sample."""
        gain = quote.spot * self.conditional_black(
            cum[0][..., quote.j], cum[1][..., quote.j], quote.carry, quote.ratio).mean()
        return quote.units * np.exp(-quote.rate * quote.T) * (
            gain if quote.is_call else gain - (quote.forward - quote.strike))

    def forward_vol(self, cum, target):
        """One forward-start row's model vol - `forward_value` inverted by the same splice the
        spot smile uses, spec 5.3's objective being in vol points."""
        return self.implied(self.forward_value(cum, target), np.exp(target.carry), target.strike,
                            target.tenor)

    def forward_value(self, cum, target):
        """One forward-start target's model premium in units of `S_T1`: the block over `[T1, T2]`
        ALONE, whose law given the draws is Gaussian, so `(S_T2/S_T1 - k)^+` is the same
        conditional Black over that window and nothing is drawn at `T1` (spec 5.3).

        A TRADED forward-start pays `S_T1 (R - k)^+` and is quoted as
        `E[S_T1 (R - k)^+]/E[S_T1]`, so where the source is market `Quotes` the same per-path gain
        is averaged under the SHARE MEASURE - one extra factor `exp(M_1 + V_1/2)` on the tape,
        normalised, which is the only difference between the two instruments. A reference model's
        slopes are ratio expectations and want the plain average.
        """
        gain = self.conditional_black(
            cum[0][..., target.j2] - cum[0][..., target.j1],
            cum[1][..., target.j2] - cum[1][..., target.j1], target.carry, target.ratio)
        if self.source != 'Quotes':
            return gain.mean()
        share = torch.softmax(cum[0][..., target.j1] + 0.5 * cum[1][..., target.j1], 0)
        return (share * gain).sum()

    def market(self, quotes):
        """Every quote's market premium with its QUOTED number on the tape.

        The number itself is what the fit reads; the splice `b - detach(b)` is worth zero forward
        and carries `dPremium/dq`, so turning `Quote_Sensitivity` on cannot move a fitted digit and
        `dr/dq` is still the derivative of the premium the residual measures. Under
        `Implied_Volatility` that derivative is Black's own vega at the quoted vol; under `Premium`
        the quote IS the premium and the splice is linear.
        """
        premia = self.vector([quote.premium for quote in quotes])
        if self.leaf is None:
            return premia
        rows = [self.at_quote(quote, self.leaf[quote.row]) for quote in quotes]
        spliced = torch.stack(rows)
        return premia + (spliced - spliced.detach())

    def at_quote(self, quote, quoted):
        """One quote's premium as a function of the number quoted - Black in TENSORS at a vol, the
        units times it at a premium."""
        if not self.is_vol:
            return quote.units * quoted
        sd = quoted * np.sqrt(quote.T)
        d1 = (np.log(quote.forward / quote.strike) + 0.5 * sd * sd) / sd
        sign = 1.0 if quote.is_call else -1.0
        return quote.units * sign * np.exp(-quote.rate * quote.T) * (
            quote.forward * utils.norm_cdf(sign * d1)
            - quote.strike * utils.norm_cdf(sign * (d1 - sd)))

    def quote_vol(self, cum, quote):
        """The model-minus-market difference in Black vol at one quote, both premia inverted off
        the same forward with the units and the yield rescale stripped.

        The objective is a vol residual only to first order; this is the inversion itself, taken
        OFF THE TAPE, so what the report prints is what a desk would read. Returned as a decimal.
        """
        both = [implied_vol(
            premium, quote.spot * np.exp(quote.carry - quote.rate * quote.T), quote.strike,
            quote.rate * quote.T, 1, quote.T, quote.units,
            0.0 if quote.is_call else (quote.forward - quote.strike) * np.exp(
                -quote.rate * quote.T))
            for premium in (float(self.value(cum, quote).detach()), quote.premium)]
        return both[0] - both[1]

    def rmse(self, cum, quotes):
        """The vol-space RMS residual over `quotes` to first order, off the tape - the reading
        stage 5's guard hinges on and the one its own rows carry."""
        return float(np.sqrt(np.mean([
            ((float(self.value(cum, quote).detach()) - quote.premium) / quote.vega) ** 2
            for quote in quotes])))

    def cap_headroom(self, scalars, levers, curve):
        """The mass of path-days with headroom `(a - (l+s))/beta` under 5 (spec 2.7).

        The state's own recursion, which `lv_walk` consumes and does not publish, run ONCE at the
        parameters actually written: the cap is a guard, and a calibration that reaches it is a
        failure this report has to be able to state.
        """
        eta_l, eta_s, counts = self.draws
        a, beta, nu = (float(scalars[x]) for x in ('Cap_A', 'Cap_Beta', 'Nu'))
        sigma_s = utils.bucket_at(self.buckets, levers['Sigma_S'], self.times[:-1])
        phi_s, w_s = utils.lv_ou_step_weights(scalars['Kappa_S'], sigma_s, self.deltas)
        phi_l, w_l = utils.lv_ou_step_weights(scalars['Kappa_L'], scalars['Sigma_L'], self.deltas)
        counts = counts.to(eta_l.dtype)
        s = torch.zeros_like(eta_l[:, 0])
        l, near = s + curve[0], 0.0
        for k in range(self.deltas.shape[0]):
            near += float(((a - (l + s)) / beta < 5.0).sum())
            s = phi_s[..., k] * s + w_s[..., k] * eta_s[:, k] + nu * counts[:, k]
            l = curve[k + 1] + phi_l[k] * (l - curve[k]) + w_l[k] * eta_l[:, k]
        return near / (eta_l.shape[0] * self.deltas.shape[0])

    @classmethod
    def jensen(cls, scalars, lam, sigma_s, delta, ends):
        """Spec 2.6's Jensen factor per SEGMENT: `E[exp(l+s)]/exp(L)` averaged over that segment's
        own steps, which is what lifts the model's diffusive variance above `exp(L)`.

        The OU deviation from `L` is independent of `L`, so with `l_0 = L(0)`, `s_0 = 0`,
        `E[exp(l+s)] = exp(L(t) + (Var(l) + Var(s))/2 + sum_j lam*d*(exp(nu*phi_s^m) - 1))` whatever
        shape `L` has - the Gaussian half of it is the Jensen share the split report prints, and
        the running sum is the co-jump's. `ends` is the step each ATM expiry lands on.
        """
        nu = float(scalars['Nu'])
        sigma = {'S': sigma_s, 'L': float(scalars['Sigma_L'])}
        phi = {x: np.exp(-float(scalars['Kappa_' + x]) * delta) for x in ('S', 'L')}
        weight = {x: sigma[x] ** 2 * (1.0 - phi[x] ** 2) / (2.0 * float(scalars['Kappa_' + x]))
                  for x in ('S', 'L')}
        var, compound, factors, rows, k = {'S': 0.0, 'L': 0.0}, 0.0, [], [], 0
        for step in range(ends[-1]):
            rows.append(0.5 * (var['S'] + var['L']) + compound)
            compound += lam[k] * delta * (np.exp(nu * phi['S'] ** step) - 1.0)
            var = {x: phi[x] ** 2 * var[x] + weight[x] for x in ('S', 'L')}
            if step + 1 == ends[k]:
                factors.append(np.mean(np.exp(rows)))
                rows, k = [], k + 1
        return np.array(factors)

    @classmethod
    def seed_curve(cls, xi, scalars, lam, mu_j, sigma_j, sigma_s, delta, ends):
        """Spec 2.6's mapping, one level per SEGMENT: the level whose UNCONDITIONAL diffusive
        variance AVERAGES the market's own forward variance over that segment's own steps - the
        market's total less that segment's own jump variance, over its Jensen factor. A SEED, the
        bootstrap then matching the model's own ATM price per segment."""
        jump = np.asarray(lam, dtype=float) * (mu_j ** 2 + sigma_j ** 2)
        diffusive = np.asarray(xi, dtype=float) - jump
        if (diffusive <= 0.0).any():
            worst = int(np.argmin(diffusive))
            raise ValueError(
                'the seeded jump variance lambda(t)*(mu_J^2 + sigma_J^2) is at or above this '
                'surface\'s own forward variance on {} of {} segments (worst {:.6g} against '
                '{:.6g}), so xi_diff <= 0 and spec 2.6\'s mapping has no answer. Lower Jump_Share '
                'or the wing the jump is sized off (Wing_Strike), or pin a smaller Lambda'.format(
                    int((diffusive <= 0.0).sum()), diffusive.size, float(jump[worst]),
                    float(xi[worst])))
        return np.log(diffusive) - np.log(cls.jensen(scalars, lam, sigma_s, delta, ends))

    def l_knots(self, levels):
        """The L curve as it is WRITTEN, off the segment levels solved so far: one level per
        SEGMENT on knots that START it - tenor 0, then every ATM expiry but the last - and flat
        beyond (spec 5.4.3). `L(0)` is the first segment's level.

        An `Event_Days` date carries its own KNOT PAIR, a one-day segment at
        `Event_Variance_Prior` times the enclosing segment's level and that level again after it,
        so the variance on the event day is one number.
        """
        pillars = torch.stack(list(levels))
        knots = self.knots[:pillars.numel()]
        if not self.event_times.size:
            return knots, pillars
        extra = utils.bucket_at(knots, pillars, self.vector(self.event_times)) + self.event_log
        merged = np.concatenate([knots, self.event_times])
        order = np.argsort(merged, kind='stable')
        return merged[order], torch.cat([pillars, extra])[order.tolist()]

    def l_at(self, levels):
        """`L` at the walk's grid times."""
        return utils.bucket_at(*self.l_knots(levels), t=self.times)

    def solve_l(self, scalars, levers):
        """The inner triangular bootstrap, RE-RUN AT EVERY OUTER ITERATE: the L SEGMENTS solved one
        at a time against their own ATM premium, so every candidate reprices the ATM term structure
        exactly and is judged on the smile alone (spec 5.4.2).

        Triangular because the model is - an option to `T_k` reads `L` only on `[0, T_k]`, and
        segment k's level nowhere before `T_{k-1}` - and the premium is monotone in that level, so
        the root is unique and the search is a CHORD off the previous sweep's slope, warm started
        at the previous sweep's answer and run to `Pillar_Tolerance` OFF THE TAPE. A stale chord
        that misses gets one refreshed slope, which is the whole of the fallback.

        The level RETURNED is one NEWTON STEP at that root, `L_k* - F_k/detach(dF_k/dL_k)`, off the
        one graph pass a pillar costs - so `dL_k/dtheta`, `dL_k/dL_j` down the triangle and
        `dL_k/dq` at the ATM quote are all the implicit function theorem written as an EXPRESSION,
        which is the component family's own spelling of it. Spending the backward on the root alone
        is what makes the per-iterate bootstrap affordable: measured on the walk at 8192 paths over
        504 daily steps, a forward pass is 0.280 s and a pass with its backward 0.756 s.

        A sweep is deterministic in `x` to `Pillar_Tolerance` - a relative premium miss two orders
        under the outer `Tolerance` - which is the same argument the component family's brentq
        makes for its own bracket; the warm start moves the answer no further than that.
        """
        targets = self.market(self.atm)
        levels, misses = [], []
        for k, quote in enumerate(self.atm):

            def priced(at):
                self.calls['l'] += 1
                return self.value(self.walk(scalars, levers, self.l_at(levels + [at]),
                                            int(self.upto[quote.j])), quote) - targets[k]

            at = self.warm[k]
            for _ in range(self.l_rounds):
                if self.slopes[k] is not None:
                    with torch.no_grad():
                        for _ in range(self.l_iterations):
                            miss = float(priced(at)) / quote.premium
                            if abs(miss) < self.pillar_tol:
                                break
                            at = at + max(-self.l_damping, min(
                                self.l_damping, -miss * quote.premium / self.slopes[k]))
                pillar = at.detach().requires_grad_(True)
                shift = priced(pillar)
                slope = torch.autograd.grad(shift, pillar, retain_graph=True)[0].detach()
                self.calls['b'] += 1
                self.slopes[k] = float(slope)
                if abs(float(shift.detach()) / quote.premium) < self.pillar_tol:
                    break
            levels.append(pillar.detach() - shift / slope)
            self.warm[k] = pillar.detach()
            misses.append(float(shift.detach()) / quote.premium)
        self.atm_misses = misses
        return levels

    def build(self, x, coords):
        """The full parameter set as tensors: the fitted coordinates taken off `x` - the leaf a
        Jacobian hangs on - everything else off `self.state`, and then the TIES of `Bootstrap` mode
        applied in bucket order, so a tie carried from the previous bucket chains."""
        scalars = {name: self.tensor(value) for name, value in self.state.items()
                   if name not in self.step_names}
        levers = {name: [self.tensor(v) for v in self.state[name]]
                  for name in utils.LV_BUCKET_NAMES}
        for i, (name, bucket) in enumerate(coords):
            if bucket is None:
                scalars[name] = x[i]
            else:
                levers[name][bucket] = x[i]
        for (name, bucket), rule in sorted(self.ties.items(), key=lambda item: item[0][1]):
            levers[name][bucket] = (0.5 * torch.abs(levers['Mu_J'][bucket]) if rule == 'half'
                                    else levers[name][bucket - 1])
        return scalars, {name: torch.stack(v) for name, v in levers.items()}

    def evaluate(self, x=None, coords=()):
        """One outer iterate: the parameters, the L strip re-bootstrapped at them, and the walk.
        The strip is BANKED, so `written` and `connect` read the one solve `finish` took."""
        scalars, levers = self.build(x, coords)
        self.levels = self.solve_l(scalars, levers)
        return scalars, levers, self.walk(scalars, levers, self.l_at(self.levels), self.n)

    def rows(self, cum, judged, forwards):
        """Every row's residual, weighted: a vanilla's premium miss over its own market vega, and
        a forward-start's vol miss itself (spec 5.3's own objective term)."""
        market, shapes = self.market(judged), self.spot_shapes(cum, forwards)
        return ([quote.weight * (self.value(cum, quote) - market[i]) / quote.vega
                 for i, quote in enumerate(judged)] +
                [target.weight * (self.forward_vol(cum, target)
                                  - self.target_vol(target, shapes[target.tenor]))
                 for target in forwards])

    def residual(self, x, coords, judged, forwards=(), guard=None, smooth=None):
        """The stage's residual VECTOR: its own rows FIRST, which is what the identification table
        reads, then the penalties the spec states - the idiosyncratic share and the REALISED jump
        share approaching their boxes, and where the mode or stage 5 asks, the vanilla degradation
        at the target maturities and the levers' smoothness across adjacent buckets.

        `smooth` is the LAST bucket the stage has fitted: a difference row against a bucket still
        at its seed measures the seed, so `Bootstrap` sees only the buckets behind it."""
        scalars, levers, cum = self.evaluate(x, coords)
        terms = self.rows(cum, judged, forwards)
        terms.append(self.soft_penalty * torch.relu(
            self.c_min + self.c_margin
            - (1.0 - levers['Rho_S'] ** 2 - scalars['Rho_L'] ** 2)))
        realised = self.state['Lambda'][0] * (
            levers['Mu_J'][0] ** 2 + levers['Sigma_J'][0] ** 2) / self.xi_0
        terms.append(self.soft_penalty * (torch.relu(realised - self.share_box[1])
                                          + torch.relu(self.share_box[0] - realised)))
        if self.diffusive is not None:
            terms.append(self.split_penalty
                         * (self.split(levers)[1] - self.vector(self.diffusive)))
        if guard is not None:
            held, base = guard
            kept = torch.stack([(self.value(cum, quote) - quote.premium) / quote.vega
                                for quote in held])
            terms.append(self.vanilla_guard * torch.relu(
                (kept * kept).mean().sqrt() - base - self.vanilla_band))
        if smooth:
            terms += [self.smoothness * levers[name][:smooth + 1].diff()
                      for name in utils.LV_BUCKET_NAMES]
        return torch.cat([term.reshape(-1) for term in terms])

    def split(self, levers):
        """The split per SEGMENT (spec 5.4.8): `(jump variance, diffusive share)` at the jump
        sizes in force there, both taken of the MARKET's own forward variance - the denominator a
        desk quotes, and the one `Diffusive_Share` is declared against."""
        starts = self.vector(self.knots[:-1])
        jump = self.vector(self.state['Lambda']) * (
            utils.bucket_at(self.buckets, levers['Mu_J'], starts) ** 2
            + utils.bucket_at(self.buckets, levers['Sigma_J'], starts) ** 2)
        return jump, 1.0 - jump / self.vector(self.xi)

    def jacobian(self, x, coords, judged, forwards, **kw):
        """dr/dx by autograd at a fitted leaf - ONE vmapped backward over the residual rows, which
        reads the per-row loop's answer to the bit at a fifth of its cost on this graph."""
        self.calls['j'] += 1
        leaf = torch.tensor(np.asarray(x, dtype=float), device=self.device,
                            dtype=self.prec, requires_grad=True)
        terms = self.residual(leaf, coords, judged, forwards, **kw)
        return torch.autograd.grad(
            terms, leaf, torch.eye(terms.numel(), dtype=self.prec, device=self.device),
            is_grads_batched=True)[0].numpy()

    def label(self, coord):
        return coord[0] if coord[1] is None else '{}[{:g}y]'.format(
            coord[0], self.buckets[coord[1]])

    def value_of(self, coord):
        return self.state[coord[0]] if coord[1] is None else self.state[coord[0]][coord[1]]

    def bounds_of(self, coord, coords):
        """`Rho_S`'s box is `C_Min`'s own - the SAME declaration the factor asserts at load -
        re-derived off `Rho_L` unless this stage is moving it too, where the widest box and the
        soft penalty do the work the spec asks of them."""
        if coord[0] != 'Rho_S':
            return self.box[coord[0]]
        rho_l = 0.0 if ('Rho_L', None) in coords else self.state['Rho_L']
        return -np.sqrt(max(1.0 - rho_l * rho_l - self.c_min, 0.0)), 0.0

    def stage(self, tag, coords, judged, forwards=(), **kw):
        """One stage: `least_squares` over `coords` against its own rows, the fitted values written
        back into `self.state`, its identification table kept over the DATA rows alone, and the L
        strip left at what it landed on - which is also the warm start every iterate of the NEXT
        stage bootstraps from, so a stage's residual is a function of its own `x` alone.

        The stage's own BOX is kept with them, for `interior` to read its KKT active set off at
        theta*: a coordinate the box holds is held there and not by the data, which is what the
        quote contraction is taken over (`LeastSquaresSolve`)."""
        edges = [self.bounds_of(coord, coords) for coord in coords]
        x0 = np.clip([self.value_of(coord) for coord in coords], *map(np.array, zip(*edges)))

        def residual(x):
            self.calls['n'] += 1
            return self.residual(self.vector(x), coords, judged, forwards, **kw).detach().numpy()

        started = time.time()
        result = scipy.optimize.least_squares(
            residual, x0, bounds=tuple(zip(*edges)), method='trf', x_scale='jac',
            jac=lambda x, *rest: self.jacobian(x, coords, judged, forwards, **kw),
            ftol=self.tolerance, xtol=1e-12, max_nfev=self.max_iter)
        for coord, value in zip(coords, result.x):
            if coord[1] is None:
                self.state[coord[0]] = float(value)
            else:
                self.state[coord[0]][coord[1]] = float(value)
        self.tables.append((tag, [self.label(coord) for coord in coords],
                            result.jac[:len(judged) + len(forwards)]))
        self.edges, self.slope = np.array(edges, dtype=float).T, result.grad
        self.capped = tag if result.nfev >= self.max_iter else None
        logging.info('  stage {}: {} rows, {} evaluations, residual {:.4e}{}, {:.1f}s'.format(
            tag, result.fun.size, result.nfev, float(np.sqrt((result.fun ** 2).sum())),
            '' if result.nfev < self.max_iter else ' CAPPED at Max_Iterations',
            time.time() - started))
        self.settle(coords)
        self.fitted, self.judged, self.final = coords, judged, kw

    def settle(self, coords=()):
        """The L strip at what the stage landed on, kept as the next stage's warm start; the ties
        of `Bootstrap` mode written back into `self.state` so the report reads what was fitted."""
        scalars, levers = self.build(None, ())
        self.levels = [x.detach() for x in self.solve_l(scalars, levers)]
        self.warm = list(self.levels)
        for name in utils.LV_BUCKET_NAMES:
            self.state[name] = [float(v) for v in levers[name]]

    def solve_share(self):
        """Stage 5's third variable, by a BOUNDED SCALAR search rather than a coordinate.

        `w_J` sets the whole lambda STRIP by 5.1, lambda sets the law of the integer counts, and
        a count drawn by inverse CDF off a fixed uniform is a STEP function of it - so no Jacobian
        carries this direction and least_squares would read the flat one it has. The same
        weighted sum of squares, at counts redrawn per candidate; the box IS spec 5.1's.
        """
        def objective(w):
            self.state['Lambda'] = self.lam_strip(w)
            self.draws = self.counted()
            self.calls['n'] += 1
            _, levers, cum = self.evaluate()
            split = ((self.split_penalty * (self.split(levers)[1] - self.vector(self.diffusive)))
                     ** 2).sum() if self.diffusive is not None else 0.0
            return sum(float(row.detach()) ** 2
                       for row in self.rows(cum, self.guarded, self.targets)) + float(split) + (
                self.vanilla_guard * max(
                    self.rmse(cum, self.guarded) - self.base_rmse - self.vanilla_band, 0.0)) ** 2

        found = scipy.optimize.minimize_scalar(
            objective, bounds=self.share_box, method='bounded',
            options={'xatol': 1e-3, 'maxiter': 20})
        objective(found.x)
        self.settle()
        return float(found.x)

    def solve(self):
        """The staged fit. Returns theta* over the last stage's coordinates - the flat vector
        `LeastSquaresSolve` hangs the quote derivative on."""
        started = time.time()
        self.settle()
        # spec 2.7's LEVEL rule, on the seeded curve and again once the spread is fitted
        self.state['Cap_A'] = self.cap_level()
        (self.bootstrap_stages if self.mode == 'Bootstrap' else self.global_stages)()
        self.elapsed = time.time() - started
        self.theta = self.vector([self.value_of(coord) for coord in self.fitted])
        return self.theta

    def __call__(self, x):
        """The last stage's residual at fitted coordinates `x` - the rows the solver stepped on, so
        the quote derivative cannot drift from the answer it is taken at."""
        return self.residual(x, self.fitted, self.judged, self.targets, **self.final)

    @property
    def labels(self):
        """One name per fitted coordinate, so a coordinate the box holds is named."""
        return [self.label(coord) for coord in self.fitted]

    @property
    def descriptors(self):
        """One name per quote, in `leaf`'s own order - what the quote deltas are reported against."""
        return ['{:g}y {:g}'.format(quote.T, quote.strike) for quote in self.quotes]

    def interior(self, x, g):
        """The last stage's free coordinates at `(x, g)` - what the KKT active set does not hold."""
        return np.flatnonzero(~active_set(x, self.edges[0], self.edges[1], g)).tolist()

    def cap_level(self):
        """Spec 2.7's level rule `a = max(Cap_A, L(0) + 6 s_inf)`, at the widest bucket's spread -
        raised where the fit asks for it and never lowered."""
        spread = max(self.spreads())
        return max(float(self.instrument['Cap_A']), float(self.levels[0]) + 12.0 * spread)

    def spreads(self, horizon=None, sigma_l=None):
        """The log-VOL sd per bucket (spec 2.2.1), half the log-variance one - the quantity VIX
        options price and the `Stationary_Spread` guard reads.

        STATIONARY where no horizon is given. At `horizon` years it is the spread an exposure row
        at that node carries, which the three-way prior report reads at each prior's `sigma_l`.
        """
        grown = lambda k: 1.0 if horizon is None else -np.expm1(-2.0 * k * horizon)
        sigma_l = self.state['Sigma_L'] if sigma_l is None else sigma_l
        return [0.5 * np.sqrt(sigma_s ** 2 * grown(self.state['Kappa_S'])
                              / (2.0 * self.state['Kappa_S'])
                              + sigma_l ** 2 * grown(self.state['Kappa_L'])
                              / (2.0 * self.state['Kappa_L']))
                for sigma_s in self.state['Sigma_S']]

    def global_stages(self):
        """Spec 5.2's order: the jump off the short end, the fast pair off the sub-year smile, the
        slow pair off what is beyond it, the forward block of 5.3, then a joint polish."""
        short = [q for q in self.quotes if q.T <= self.stage_horizons[0]]
        middle = [q for q in self.quotes if q.T <= self.stage_horizons[1]]
        long = [q for q in self.quotes if q.T > self.stage_horizons[1]]
        self.stage('2 (mu_J, sigma_J)', [('Mu_J', 0), ('Sigma_J', 0)], short or middle
                   or self.quotes)
        self.stage('3 (rho_s, sigma_s, nu)', [('Rho_S', 0), ('Sigma_S', 0), ('Nu', None)],
                   middle or self.quotes)
        if self.identified_slow():
            self.stage('4 (rho_l, sigma_l)', [('Rho_L', None), ('Sigma_L', None)], long)
        else:
            self.pin_slow()
        self.state['Cap_A'] = self.cap_level()

        self.guarded = [q for q in self.quotes if any(
            min(abs(q.T - t.T1 - t.tenor), abs(q.T - t.T1)) < self.delta for t in self.targets)]
        self.base_rmse = self.rmse(self.evaluate()[2], self.guarded) if self.guarded else 0.0
        last = self.buckets.size - 1
        if self.targets and last:
            later = [b for b in range(1, self.buckets.size)]
            guard = (self.guarded, self.base_rmse)
            self.stage('5a mu_J(t)', [('Mu_J', b) for b in later], self.guarded, self.targets,
                       guard=guard, smooth=last)
            self.stage('5b rho_s(t)', [('Rho_S', b) for b in later], self.guarded, self.targets,
                       guard=guard, smooth=last)
        if self.targets and not self.pinned:
            self.share = self.solve_share()

        polish = [(name, None) for name in
                  (('Sigma_L', 'Rho_L', 'Nu') if self.identified_slow() else ('Nu',))]
        polish += [(name, b) for name in utils.LV_BUCKET_NAMES
                   for b in range(self.buckets.size)]
        self.stage('6 joint polish', polish, self.quotes, self.targets, smooth=last)

    def identified_slow(self):
        """Does this ladder identify `(rho_l, sigma_l)` - WING quotes at `slow_horizon` or longer
        (spec 5.2 stage 4)? Otherwise both are held at a prior and the report says so, exactly as
        the kappas are: a fit to nothing writes a number a desk would read as one."""
        return any(T >= self.slow_horizon for T in self.wings)

    @property
    def asset_class(self):
        """The factor TYPE the block's `Underlying` resolves to - `slow_priors`' own key."""
        return type(self.factors['Underlying']).__name__

    def slow_prior(self):
        """`(rho_l, sigma_l, source)` for the stage-4 pin, in the order spec 5.2 and 5.5.3 give:
        `Slow_Factor_Prior` where the block declares one, else the historical estimate where the
        job's `Price Models` carries one for this underlying, else the asset class's own default.

        The class default's MAGNITUDE is `slow_priors`', its SIGN that of the `Rho_S` in force at
        the pin - stage 3 precedes stage 4 - so the slow skew is never set against the fast one.
        The floor `Sigma_L >= 0.3` is the box beneath all three: a DECLARED prior under it refuses
        by name, an ESTIMATE is taken to the floor and the report says so.
        """
        rows = [float(x) for x in
                str(self.instrument['Slow_Factor_Prior']).split(',') if x.strip()]
        floor, keys = self.box['Sigma_L'][0], utils.LV_SLOW_HISTORY[1]
        if len(rows) == 2 and rows[1] >= floor:
            return rows[0], rows[1], 'the declared Slow_Factor_Prior'
        if rows:
            raise ValueError(
                "{}: Slow_Factor_Prior reads {!r}. It is the PAIR rho_l,sigma_l held where the "
                "ladder carries no wing at {:g}y or longer, with sigma_l AT OR ABOVE the {:g} "
                "floor spec 5.2 stage 4 keeps beneath any prior - a two-year surface cannot claim "
                "five-year vol is certain, and a zero slow factor collapses a CVA profile's vol "
                "distribution onto the fast factor's spread, which reverts within months. Write "
                "both at or above the floor, or leave the field blank for the {} default".format(
                    self.market_price, self.instrument['Slow_Factor_Prior'], self.slow_horizon,
                    floor, self.asset_class))
        if self.history is not None:
            rho_l, sigma_l = (float(self.history[name]) for name in keys[:2])
            return rho_l, max(sigma_l, floor), (
                "the history's estimate, Rho_L SE {:.4f} and Sigma_L SE {:.4f} (spec 5.5.3){}"
                .format(*[float(self.history[name]) for name in keys[2:]] +
                        ['' if sigma_l >= floor else
                         ', its Sigma_L {:.4f} TAKEN TO the {:g} floor'.format(sigma_l, floor)]))
        rho_l, sigma_l = self.slow_priors[self.asset_class]
        fast = self.state['Rho_S'][0]
        return float(np.copysign(rho_l, fast or -1.0)), sigma_l, (
            'the {} class default, signed by the Rho_S {:+.4f} in force at the pin'.format(
                self.asset_class, fast))

    def pin_slow(self):
        """Stage 4 where the ladder does not identify the slow pair: the prior into the state, and
        the line the report earns for it."""
        rho_l, sigma_l, source = self.slow_prior()
        self.state['Rho_L'], self.state['Sigma_L'] = rho_l, sigma_l
        self.pinned_slow = (
            'pinned: not identified by this ladder - Rho_L {:+.4f} and Sigma_L {:.4f} are held at '
            '{}, the longest wing expiry being {} against the {:g}y spec 5.2 stage 4 asks '
            'for'.format(rho_l, sigma_l, source,
                         '{:g}y'.format(max(self.wings)) if self.wings else 'none at all',
                         self.slow_horizon))

    def prior_report(self):
        """The ladder read under each of spec 5.2 stage 4's three slow priors - the FLOOR beneath
        any of them, the class default and the index seed - as `(name, rho_l, sigma_l, wing RMSE,
        log-vol sd at exposure_horizon)`. Empty where the ladder identified the pair.

        ONE forward pass a prior: everything but the slow pair stays at theta*, the L strip is
        re-bootstrapped so every ATM still reprices, and the wings are re-priced, with no outer
        search. It APPROXIMATES the re-fit it is not - a re-fit would let the fast pair take back
        some of what the slow one gives - and the sd is the closed form, the outer generator
        carrying `(S, l, s)` being phase 3. Two priors that land on the same pair - an index, whose
        class default IS the seed - are ONE pass named for both.
        """
        if not self.pinned_slow:
            return []
        sign, floor = self.state['Rho_S'][0] or -1.0, self.box['Sigma_L'][0]
        mine, seed = self.slow_priors[self.asset_class], self.slow_priors['EquityPrice']
        priors = {}
        for name, rho_l, sigma_l in (('the floor', mine[0], floor),
                                     ('the class default', mine[0], mine[1]),
                                     ('the index seed', seed[0], seed[1])):
            priors.setdefault((float(np.copysign(rho_l, sign)), sigma_l), []).append(name)
        rows = []
        for (rho_l, sigma_l), names in priors.items():
            scalars, levers = self.build(None, ())
            scalars['Rho_L'], scalars['Sigma_L'] = self.tensor(rho_l), self.tensor(sigma_l)
            cum = self.walk(scalars, levers, self.l_at(self.solve_l(scalars, levers)), self.n)
            rows.append((' = '.join(names), rho_l, sigma_l, self.wing_rmse(cum),
                         max(self.spreads(self.exposure_horizon, sigma_l))))
        return rows

    def wing_rmse(self, cum):
        """The vol-point RMS miss over the ladder's WING quotes - the headline the slow pair
        moves, the ATM rungs being a constraint the L strip solves to zero."""
        rows = [self.quote_vol(cum, quote) for rung in self.wings.values() for quote in rung]
        return 100.0 * np.sqrt(np.mean(np.square(rows))) if rows else float('nan')

    def bootstrap_stages(self):
        """Spec 5.4.1's `Bootstrap`: the wing expiries ARE the buckets, and bucket `k` is fitted to
        expiry `k`'s wing quotes GIVEN buckets `< k`, sequentially - the same triangular discipline
        the L strip already runs on, applied to the smile.

        How many of `LV_FREE_ORDER` a bucket frees is how many wing quotes its expiry carries; what
        is left is TIED, `Sigma_J` to half `|Mu_J|` and the other two carried forward, so a bucket
        never has more parameters than quotes. The slow pair stays global and the smoothness
        penalty is not optional here.
        """
        long = [q for q in self.quotes if q.T > self.stage_horizons[1]]
        if self.identified_slow():
            self.stage('4 (rho_l, sigma_l)', [('Rho_L', None), ('Sigma_L', None)], long)
        else:
            self.pin_slow()
        self.state['Cap_A'] = self.cap_level()
        for k, expiry in enumerate(self.wings):
            rung = self.wings[expiry]
            free = LV_FREE_ORDER[:min(len(rung), len(LV_FREE_ORDER))]
            self.ties.update({(name, k): ('half' if name == 'Sigma_J' else 'carry')
                              for name in LV_FREE_ORDER[len(free):] if k or name == 'Sigma_J'})
            self.free[k] = free
            self.stage('bootstrap bucket {:g}y ({} wing quote{}, free {})'.format(
                expiry, len(rung), '' if len(rung) == 1 else 's', '/'.join(free)),
                [(name, k) for name in free], rung, smooth=k)

    def own_segment(self, t):
        """Is the ATM segment holding `t` ONE internal step - the event day its own pillar, with
        expiries on the business days either side (spec 5.4.3)?"""
        k = int(np.searchsorted(self.knots, t + utils.BUCKET_TOL)) - 1
        return 0 <= k < self.knots.size - 1 and round(
            (self.knots[k + 1] - self.knots[k]) / self.delta) <= 1

    def finish(self, theta):
        """The fit as it stands: the walk at theta*, every quote's TRUE vol-point miss, and - where
        a forward source is set - the polish's identification table taken a SECOND time without the
        forward rows, which is what says whether a later bucket is pinned by them or by nothing.

        The walk is taken at THETA, not at the state it was written back into, so the L strip this
        banks carries the graph `connect` publishes - one solve, read by both."""
        if self.targets and self.tables:
            self.tables.append((self.tables[-1][0] + ', vanillas only',
                                self.labels,
                                self.jacobian([self.value_of(x) for x in self.fitted], self.fitted,
                                              self.quotes, (), **self.final)[:len(self.quotes)]))
        self.slow_rows = self.prior_report()
        self.scalars, self.levers, self.cum = self.evaluate(theta, self.fitted)
        self.misses = [100.0 * self.quote_vol(self.cum, quote) for quote in self.quotes]
        self.realised = self.state['Lambda'][0] * (
            self.state['Mu_J'][0] ** 2 + self.state['Sigma_J'][0] ** 2) / self.xi_0

    def verify(self):
        """The failures a calibrated surface may not carry, raised BEFORE the report so a refusal
        is a message rather than half a log, and the two the block declares `Refuse | Floor` on -
        never a silent third option (spec 5.4.6). Leaves the cap headroom the report prints."""
        stuck = [i for i, x in enumerate(self.atm_misses) if abs(x) > self.atm_miss_max]
        if stuck:
            raise ValueError(
                '{}: no L level reprices the {} ATM pillar{} - {}. The triangular bootstrap is '
                'monotone in a pillar\'s own level, so a pillar that walks its Newton steps out '
                'and still misses is one the OTHER parameters have put out of reach: at these '
                'globals the jumps alone carry more variance than the pillar has. Lower '
                'Jump_Share, pin a smaller Lambda, or quote a surface this model can reach'.format(
                    self.market_price, len(stuck), '' if len(stuck) == 1 else 's',
                    ', '.join('{:g}y {:+.2%}'.format(self.knots[i + 1], self.atm_misses[i])
                              for i in stuck)))

        # the BOX is the event, not the soft margin above it: the fit is bounded at exactly C_Min,
        # so a bucket landing there is one the box stopped rather than an interior optimum
        c = 1.0 - np.array(self.state['Rho_S']) ** 2 - self.state['Rho_L'] ** 2
        edge = np.flatnonzero(c <= self.c_min + 1e-9)
        if edge.size:
            convexity = self.band(1.05, 1.25)
            message = (
                '{}: the idiosyncratic share c = 1 - Rho_S^2 - Rho_L^2 is ON its floor C_Min={:g} '
                'in {} of {} bucket{} ({}), so the box - not the data - is what stopped Rho_S, and '
                'this surface wants MORE leverage than one shock plus a co-jump can carry: it '
                'wants a ONE-SHOCK model (spec 2.2.2). The 110-120% convexity residual that goes '
                'with the floor is {}'.format(
                    self.market_price, self.c_min, edge.size, c.size, '' if c.size == 1 else 's',
                    ', '.join('{:g}y c {:.3f} at Rho_S {:+.4f}'.format(
                        self.buckets[i], c[i], self.state['Rho_S'][i]) for i in edge),
                    'nothing quoted there' if convexity is None else
                    '{:+.3f} vol points RMS, worst {:+.3f}'.format(*convexity)))
            self.guard('Idiosyncratic_Share', message,
                       'Floor to TAKE C_Min and have the fit say so, lower C_Min (a book may run '
                       'it near 0.06, at the cost of the second-order noise the share buys), or '
                       'fit a surface whose skew this structure can reach')

        # nothing is scaled and nothing re-solved: theta* is where the fit left it, and the guard
        # is a READING of it - which is what keeps the quote contraction taken at the same point
        spreads, (lo, hi) = self.spreads(), self.log_vol_sd_band
        stuck = [i for i, sd in enumerate(spreads) if not lo <= sd <= hi]
        if stuck:
            message = (
                '{}: the stationary log-vol sd 0.5*sqrt(Sigma_S^2/2Kappa_S + Sigma_L^2/2Kappa_L) '
                'sits outside the {:g}-{:g} VIX options imply in {} of {} bucket{} ({}), so this '
                'surface wants vol dynamics the VIX market does not price (spec 2.2.1)'.format(
                    self.market_price, lo, hi, len(stuck), len(spreads),
                    '' if len(spreads) == 1 else 's',
                    ', '.join('{:g}y sd {:.3f} at Sigma_S {:.4f}, Sigma_L {:.4f}'.format(
                        self.buckets[i], spreads[i], self.state['Sigma_S'][i],
                        self.state['Sigma_L']) for i in stuck)))
            self.guard('Stationary_Spread', message,
                       'Floor to TAKE the fit and have it say so, widen Kappa_S or Kappa_L (the '
                       'sd is theirs as much as the vol-of-vol pair), or quote a surface whose '
                       'vol-of-vol sits in the band')

        scalars, levers = self.build(None, ())
        self.headroom = self.cap_headroom(scalars, levers, self.l_at(self.levels))
        if self.headroom > self.cap_headroom_max:
            raise ValueError(
                '{}: {:.3e} of path-days sit within 5*Cap_Beta of the cap at Cap_A={:.4g}, above '
                'the {:g} a calibrated surface may carry. The cap exists to make E[S^p] finite and '
                'to stop an exp overflowing, NOT to shape a smile, so a fit that reaches it is a '
                'failure rather than a warning (spec 2.7). Raise Cap_A, or fit a surface whose '
                'vol-of-vol this model can carry'.format(
                    self.market_price, self.headroom, self.state['Cap_A'],
                    self.cap_headroom_max))

    def guard(self, field, message, remedy):
        """One `Refuse | Floor` guard (5.4.6): Refuse raises the diagnostic with the remedy, Floor
        takes what the fit reached and says so. Never a silent third option."""
        if self.instrument[field] == 'Refuse':
            raise ValueError('{}. Set {} to {}'.format(message, field, remedy))
        self.notes.append('FLOORED - ' + message.split(': ', 1)[1])

    def written(self):
        """The `LogVar2FJModelParameters` price factor: the scalars, the structural ones the kit
        reads, and the six curves - `L` and `lambda(t)` on the segments, the four levers on their
        buckets. The L strip is the ONE `finish` banked, so the factor and `connect`'s tensors
        are one solve."""
        knots, values = self.l_knots(self.levels)
        values = values.detach()
        return {'Property_Aliases': None,
                **{name: float(self.state[name])
                   for name in utils.LV_PARAM_NAMES + utils.LV_STRUCTURAL_NAMES},
                'C_Min': self.c_min,
                'Lambda': utils.Curve([], [[float(t), float(v)] for t, v
                                           in zip(self.knots[:-1], self.state['Lambda'])]),
                'L_Curve': utils.Curve([], [[float(t), float(v)]
                                            for t, v in zip(knots, values)]),
                **{name: utils.Curve([], [[float(t), float(v)] for t, v
                                          in zip(self.buckets, self.state[name])])
                   for name in utils.LV_BUCKET_NAMES}}

    def connect(self):
        """What `Calculation.factor_leaf` is offered when `Quote_Sensitivity` is on: every fitted
        parameter still connected to the quotes, keyed as `_build_factor_state` mints its leaf.

        The L curve is the strip `finish` bootstrapped at theta*, so `dL/dq` is
        `dL/dtheta . dtheta/dq + dL/dq` - the chain the Newton splice and the Gauss-Newton
        contraction each hold one half of - and the numbers written out are the numbers connected.
        """
        _, values = self.l_knots(self.levels)
        return dict({name: self.scalars[name].reshape(1) for name in utils.LV_PARAM_NAMES},
                    L_Curve=values, **{name: self.levers[name] for name in utils.LV_BUCKET_NAMES})

    def identification(self, tag, labels, jacobian):
        """The SVD of the DATA rows of `least_squares`' own Jacobian at the fitted point, its
        columns scaled by `1/||J_:,j||` - the `x_scale='jac'` scaling the solve itself runs on and
        returns the matrix without.

        The penalty and smoothness rows are LEFT OUT: their largest singular value is an algebraic
        constant of `C_Min` and `c_margin`, so a table taken over them reports a threshold rather
        than what the quotes identify. The singular values in order, and for every direction below
        `Jacobian_Rcond` times the largest, the PARAMETER LOADINGS of its right singular vector -
        so a flat or collinear direction is NAMED rather than averaged over. A stage with fewer
        rows than parameters has exact null directions and says so. The unscaled COLUMN NORMS go
        beside them, because scaling makes the table read conditioning: a well-conditioned
        direction whose column norm is 1e-6 moves nothing, and only the norm says so.
        """
        matrix = np.atleast_2d(np.asarray(jacobian, dtype=float))
        norms = np.linalg.norm(matrix, axis=0)
        values, vectors = np.linalg.svd(matrix / np.where(norms > 0.0, norms, 1.0))[1:]
        values = np.concatenate([values, np.zeros(len(labels) - values.size)])
        logging.info('  identification, {}: singular values {}; column norms {}'.format(
            tag, '  '.join('{:.3e}'.format(x) for x in values),
            ', '.join('{} {:.2e}'.format(name, x) for name, x in zip(labels, norms))))
        for i in np.flatnonzero(values <= self.rcond * max(values[0], 1e-300)):
            logging.info('    FLAT at {:.3e} ({:.1e} of the largest): {}'.format(
                values[i], values[i] / values[0], ', '.join(
                    '{} {:+.3f}'.format(labels[k], vectors[i][k])
                    for k in np.argsort(-np.abs(vectors[i])))))

    def forward_smile(self, cum):
        """`{(T1, Delta): {strike: (model vol, target vol)}}` - every forward-start target's own
        implied vol beside the one it was aimed at, both by the splice the rows are built on."""
        smiles, shapes = {}, self.spot_shapes(cum)
        for target in self.targets:
            smiles.setdefault((target.T1, target.tenor), {})[target.strike] = (
                float(self.forward_vol(cum, target).detach()),
                float(self.target_vol(target, shapes[target.tenor])))
        return smiles

    def spot_shapes(self, cum, forwards=None):
        """The model's own spot `(atm, slope, butterfly)` at each forward row's own `Delta` - the
        smile every source's target DIFFERENCE is applied to (spec 5.3)."""
        rows = self.targets if forwards is None else forwards
        return {target.tenor: self.spot_shape(cum, target.tenor) for target in rows}

    def stickiness(self, cum):
        """Per forward tenor the forward slope and butterfly, the target's and the model's own
        SPOT counterparts at maturity `Delta`, six numbers in vol points (spec 5.3).

        BOTH quantities, because the two ingredients of the split have opposite signatures - jump
        skew is sticky while its convexity dilutes at the forward date, leverage skew dilutes
        while vol-of-vol convexity amplifies - so the pair separates what the slope alone cannot.
        What the fit targets is each DIFFERENCE from the spot side; `ratio` is what the report
        divides them by where the spot side is big enough to divide by. A tenor whose rows do not
        carry `psi_strikes` has no shape and is left out.
        """
        rows = {}
        for (T1, tenor), smile in self.forward_smile(cum).items():
            if not set(self.psi_strikes) <= set(smile):
                continue
            model = self.shape([smile[k][0] for k in self.psi_strikes])
            target = self.shape([smile[k][1] for k in self.psi_strikes])
            spot = [float(x) for x in self.spot_shape(cum, tenor)]
            rows[(T1, tenor)] = tuple(100 * x for x in (model[1], target[1], spot[1],
                                                        model[2], target[2], spot[2]))
        return rows

    def ratio(self, model, target, spot):
        """One stickiness ratio, model against target, or why there is none: a spot quantity
        within `psi_floor` vol points of zero divides the forward one by nothing (spec 5.3)."""
        return ('{:.3f} against {:.3f}'.format(model / spot, target / spot)
                if abs(spot) > self.psi_floor else
                'no ratio, spot {:+.2f} inside {:g} vol points'.format(spot, self.psi_floor))

    def band(self, lo, hi):
        """`(RMS, worst)` vol-point miss over the moneyness band `[lo, hi]`, or `None` where the
        surface quotes nothing in it."""
        rows = [x for x, quote in zip(self.misses, self.quotes)
                if lo <= quote.strike / quote.forward <= hi]
        return (np.sqrt(np.mean([x * x for x in rows])), max(rows, key=abs)) if rows else None

    def report(self):
        """What the fit MEASURED, logged beside what it wrote (spec 5.2's diagnostics, 5.3's
        stickiness ratios and failure mode, 5.4.8's per-bucket table and the identification one).

        The vol-point readings go through `utils.bs_implied_total_var` OFF THE TAPE: the objective
        is a vol residual only to first order, and this is the true inversion of both premia, so
        what a desk reads here is what a desk would read. Both the UNWEIGHTED RMSE and the
        vega-weighted one the objective actually minimises are printed, because they are different
        functionals and a fit is quoted in whichever the reader had in mind.
        """
        quotes, buckets, misses = self.quotes, self.buckets, self.misses
        # keyed by POSITION: two quotes at one strike and expiry compare equal as tuples
        rung_of = lambda T: [i for i, quote in enumerate(quotes) if quote.T == T]
        logging.info('{} LogVar2FJ ({} mode): {}'.format(
            self.market_price, self.mode, ', '.join(
                '{} {:.6g}'.format(name, self.state[name])
                for name in utils.LV_PARAM_NAMES + utils.LV_STRUCTURAL_NAMES)))
        jump = self.split(self.levers)[0].detach().numpy()
        factors = self.jensen(self.state, self.state['Lambda'], self.state['Sigma_S'][0],
                              self.delta, [int(self.upto[q.j]) for q in self.atm])
        logging.info('  THE SPLIT per segment (spec 5.4.8), each share taken of the MARKET\'s '
                     'own forward variance there. The model\'s total is its MEAN diffusive '
                     'variance plus the jump, and sits ABOVE the market by what matching the '
                     'ATM PRICE rather than the variance costs - the vol-of-vol\'s convexity:')
        for lo, hi, level, x, factor, j in zip(self.knots[:-1], self.knots[1:], self.levels,
                                               self.xi, factors, jump):
            flat = np.exp(float(level.detach()))
            model = flat * factor + j
            logging.info(
                '    {:5.3f}-{:5.3f}y  market {:.2%} vol, model {:.2%}, annualised DIFFUSIVE vol '
                '{:.2%}; jump share {:.1%}, Jensen share {:.1%}'.format(
                    lo, hi, np.sqrt(max(x, 0.0)), np.sqrt(model), np.sqrt(flat),
                    j / x, flat * (factor - 1.0) / x))
        for name in utils.LV_BUCKET_NAMES:
            logging.info('  {}: {}'.format(name, ', '.join(
                '{:g}y {:+.4f}{}'.format(t, v, '' if not self.free else ' ({})'.format(
                    'free' if name in self.free.get(b, ()) else
                    'tied' if (name, b) in self.ties else 'held'))
                for b, (t, v) in enumerate(zip(buckets, self.state[name])))))
        for T in sorted({quote.T for quote in quotes}):
            rung = rung_of(T)
            worst = max(rung, key=lambda i: abs(misses[i]))
            logging.info('    {:5.3f}y  RMSE {:5.3f} vol points over {} quotes, worst {:+.3f} at '
                         '{:.0%} of forward'.format(
                             T, np.sqrt(np.mean([misses[i] ** 2 for i in rung])), len(rung),
                             misses[worst], quotes[worst].strike / quotes[worst].forward))
        weights = np.array([quote.weight for quote in quotes]) ** 2
        logging.info('  RMSE {:.3f} vol points unweighted over {} quotes, {:.3f} vega-weighted '
                     '(the objective\'s own); the bootstrap\'s ATM misses {}'.format(
                         np.sqrt(np.mean([x ** 2 for x in misses])), len(quotes),
                         np.sqrt(np.dot(weights, np.square(misses)) / weights.sum()),
                         ', '.join('{:+.1e}'.format(x) for x in self.atm_misses)))
        for tag, lo, hi in (('wing 70-80%', 0.65, 0.85), ('convexity 110-120%', 1.05, 1.25)):
            found = self.band(lo, hi)
            logging.info('  {} residual: {}'.format(
                tag, 'nothing quoted there' if found is None else
                '{:+.3f} vol points RMS, worst {:+.3f}'.format(*found)))

        c = 1.0 - np.array(self.state['Rho_S']) ** 2 - self.state['Rho_L'] ** 2
        logging.info(
            '  jump share {:.1%} asked of the {:g}y variance and {:.1%} REALISED as '
            'lambda(mu_J^2 + sigma_J^2)/xi_0 (the lambda strip {}); leverage products '
            'rho_s*sigma_s {}, rho_l*sigma_l {:+.3f}; c {} with efficiency {}'.format(
                self.share, self.knots[1], self.realised,
                '/'.join('{:.4g}'.format(x) for x in self.state['Lambda']),
                '/'.join('{:+.3f}'.format(x * y) for x, y
                         in zip(self.state['Rho_S'], self.state['Sigma_S'])),
                self.state['Rho_L'] * self.state['Sigma_L'],
                '/'.join('{:.3f}'.format(x) for x in c),
                '/'.join('{:.1f}x'.format((np.sqrt(x) / 0.22) ** 3) for x in c)))
        for line in ([] if self.slope is None else active_bounds(
                self.labels, np.array([self.value_of(x) for x in self.fitted]),
                self.edges[0], self.edges[1], self.slope)):
            logging.info('  {}'.format(line))
        logging.info('  stationary log-vol sd {}; cap headroom {:.2e} of path-days within '
                     '5*Cap_Beta of Cap_A={:.4g}'.format(
                         '/'.join('{:.3f}'.format(x) for x in self.spreads()),
                         self.headroom, self.state['Cap_A']))
        for note in self.notes:
            logging.warning('  {}: {}'.format(self.market_price, note))
        for line in ([self.pinned_slow] if self.pinned_slow else []) + self.identified_days:
            logging.info('  {}'.format(line))
        for name, rho_l, sigma_l, wing, spread in self.slow_rows:
            logging.info(
                '    under {} ({:+.4f}, {:.4f}): wing RMSE {:.3f} vol points, {:g}-year log-vol '
                'sd {:.3f}{}'.format(
                    name, rho_l, sigma_l, wing, self.exposure_horizon, spread,
                    ' - THE PIN IN FORCE' if (rho_l, sigma_l) == (self.state['Rho_L'],
                                                                  self.state['Sigma_L']) else ''))
        if self.slow_rows:
            logging.info('    each row is ONE forward pass with everything but the slow pair at '
                         'theta* - the L strip re-bootstrapped, the wings re-priced, no outer '
                         'search - so it APPROXIMATES the re-fit it is not')
        if self.diffusive is not None:
            logging.info('  Diffusive_Share is ON: the split is targeted at {} per segment as a '
                         'soft term of weight {:g}, which is an explicit override of it and '
                         'nothing else'.format(
                             '/'.join('{:.1%}'.format(x) for x in self.diffusive),
                             self.split_penalty))

        for (T1, tenor), row in self.stickiness(self.cum).items():
            slope, bfly = row[:3], row[3:]
            logging.info(
                '  psi({:g}y into {:g}y): forward slope model {:.2f} target {:.2f} over spot '
                '{:.2f} - Delta_skew {:+.2f} against {:+.2f}, psi_skew {}; butterfly {:.2f} / '
                '{:.2f} over {:.2f} - Delta_bfly {:+.2f} against {:+.2f}, psi_bfly {}'.format(
                    T1, tenor, *slope + (slope[0] - slope[2], slope[1] - slope[2],
                                         self.ratio(*slope)) + bfly
                    + (bfly[0] - bfly[2], bfly[1] - bfly[2], self.ratio(*bfly))))
        # spec 5.3's composition check is the smile BEYOND the last bucket boundary: that rung is
        # the composition of the conditional laws either side of it, and the rungs inside the last
        # bucket are ordinary vanillas the earlier buckets already fitted
        beyond = [T for T in sorted({q.T for q in quotes})
                  if self.targets and buckets.size > 1 and T > buckets[-1]]
        if beyond:
            worst = np.sqrt(np.mean([misses[i] ** 2 for i in rung_of(beyond[0])]))
            logging.info('  composition residual {:.3f} vol points at {:g}y, the first rung beyond '
                         'the {:g}y bucket boundary{}'.format(
                             worst, beyond[0], buckets[-1], '' if worst <= 0.3 else
                             ' - ABOVE 0.3, spec 5.3\'s failure mode'))
        if self.guarded:
            now = self.rmse(self.cum, self.guarded)
            logging.info('  vanilla RMSE at the forward targets\' maturities moved {:+.3f} vol '
                         'points over stage 5, in the FIRST-ORDER metric the guard is in{}'.format(
                             100.0 * (now - self.base_rmse),
                             '' if now - self.base_rmse <= self.vanilla_band else
                             ' - DEGRADED past 0.1, spec 5.3\'s failure mode'))
        for tag, labels, jacobian in self.tables:
            self.identification(tag, labels, jacobian)
        sweeps = max(self.calls['n'] + self.calls['j'], 1)
        logging.info('  {} evaluations and {} Jacobians in {:.1f}s; the inner bootstrap cost {} '
                     'pillar passes of which {} carried a backward - {:.1f} and {:.1f} a sweep '
                     'over {} pillars'.format(
                         self.calls['n'], self.calls['j'], self.elapsed, self.calls['l'],
                         self.calls['b'], self.calls['l'] / sweeps, self.calls['b'] / sweeps,
                         len(self.atm)))
        self.family.quote_trailer(self.instrument)


class LogVar2FJModelParameters(OptionQuoteFamily):
    documentation = (
        'Fx And Equity',
        ['The LogVar2FJ model (`logvar2fj_spec.md`) fitted to European options and, where the desk',
         'has them, to FORWARD-START smiles. Two mean-reverting log-variance factors carry a',
         'compound-Poisson co-jump:',
         '',
         '$$h_t=\\exp\\big(\\mathrm{cap}(\\ell_t+s_t)\\big),\\qquad',
         '\\ell\\to L(t)\\ \\text{at}\\ \\kappa_\\ell,\\qquad s\\to0\\ \\text{at}\\ \\kappa_s$$',
         '',
         'and an event moves the return by $N(\\mu_J(t),\\sigma_J(t)^2)$ while lifting $s$ by',
         '$\\nu$. GIVEN the two shocks and the counts a block return is EXACTLY Gaussian, so a',
         'vanilla is that block\'s conditional Black averaged over paths -',
         '`pricing.lognormal_fired_gain` per path, no new closed form - and the whole objective',
         'stays on one AAD tape. Each block is priced off its own SAMPLE forward, a martingale',
         'control variate that costs one `logsumexp` and takes the fixed draws\' level bias out of',
         'every strike at once.',
         '',
         'THE OBJECTIVE IS IN VOL SPACE TO FIRST ORDER. Each quote contributes',
         '$(V_{model}-V_{market})/\\mathcal{V}_{market}$ with $\\mathcal{V}$ the Black vega at the',
         'quoted vol, so autograd carries the residual with no root find on the tape. The draws are',
         'FIXED and antithetic, which is what makes the objective deterministic and its gradient',
         'the derivative of the number `least_squares` is minimising. The report prints the TRUE',
         'inversion of both premia, unweighted and vega-weighted: three functionals, named, because',
         'a fit is quoted in whichever the reader had in mind.',
         '',
         'ONE WALK PER EVALUATION, ON THE QUOTES OWN CLOCK. The internal grid steps one trading',
         'day on the *Steps_Per_Year* clock - the day is the model - and a block ends at',
         'every quoted maturity and at every forward target $T_1$ and $T_2$ - its last step a STUB',
         'landing it exactly on $T$, so the variance the fit reads at a maturity is the variance a',
         'pricer reads at that tenor of the curve written out. The block sums are cumulated, so a',
         'maturity is a prefix and a forward window the difference of two prefixes.',
         '',
         'THE FIT IS TWO NESTED SOLVES, and the inner one runs at EVERY outer iterate.',
         '',
         '1. THE INNER TRIANGULAR BOOTSTRAP. $L$ is PIECEWISE CONSTANT on the segments between ATM',
         'expiries - flat forward variance, the shape var-swap strips are quoted in (5.4.3) - and',
         'given candidate shape parameters its levels are solved SEQUENTIALLY, one per segment',
         'against that segment own ATM expiry premium. A segment integral depends on nothing but',
         'its own level, so no recurrence runs along the strip; an option to $T$ never reads $L$',
         'beyond $T$, so the system is exactly triangular; and the premium is monotone in the',
         'level, so a damped Newton off the previous sweep converges in a step or two and is',
         'iterated to *Pillar_Tolerance*. $L(0)$ is the first segment level.',
         '2. THE OUTER FIT concentrates $L$ out - the ATM ladder is a CONSTRAINT and not a term, so',
         'no smile improvement may pay for an ATM miss - plus the soft constraints of 5.2 and, in',
         '`Global` mode with a forward source, the forward-smile block of 5.3.',
         '',
         'The gradient is what makes that affordable. The level each pillar RETURNS is one NEWTON',
         'STEP at its own root, $L_k=L_k^*-F_k/\\mathrm{detach}(\\partial F_k/\\partial L_k)$, taken',
         'off the same graph the last iteration built - so the implicit function theorem across the',
         'triangle is an EXPRESSION autograd differentiates rather than a rule, and the outer',
         'solver is `least_squares` with an exact vmapped Jacobian rather than a simplex.',
         '',
         'BOOTSTRAP MODE IS THAT TRIANGULAR DISCIPLINE APPLIED TO THE SMILE (5.4.1): the ladder',
         'WING EXPIRIES are the calendar buckets, bucket $k$ is fitted to expiry $k$ wing quotes',
         'given buckets $<k$, and how many parameters it frees is how many wing quotes that expiry',
         'carries, in the order $\\mu_J,\\sigma_s,\\rho_s,\\sigma_J$; the rest are TIED,',
         '$\\sigma_J=|\\mu_J|/2$ and the other two carried from the previous bucket. The slow pair',
         'stays global and $\\psi$ is REPORTED rather than targeted.',
         '',
         'RISK IN QUOTE SPACE. *Quote_Sensitivity* **Yes** keeps the written parameters connected',
         'to the numbers quoted, so one backward pass reports $dV/dq$ beside $dV/d\\theta$. The',
         'outer fit is a least-squares minimum, so its half is the Gauss-Newton contraction at the',
         'stationarity point $(J^TJ)\\,d\\theta/dq=-J^T dr/dq$ - `LeastSquaresSolve`, the node the',
         'swaption family also solves through - taken over the coordinates the KKT active set',
         'leaves FREE, since one the box holds is held by the box and its own derivative is zero;',
         'the $L$ strip half is the same Newton splice the inner solve already carries, so',
         '`dL/dq` needs no rule of its own. The market premium each row measures against carries',
         'its quote as a splice worth zero forward, so turning this on cannot move a fitted digit.',
         'REFUSED in `Bootstrap` mode: bucket $k$ is fitted given buckets $<k$ and $\\theta^*$ is',
         'then a stationary point of no single objective, so the contraction would report the last',
         'bucket derivative as the whole.',
         '',
         'THE STAGES (spec 5.2), warm-started, each `least_squares` over its own subset:',
         '',
         '0. $\\kappa_\\ell$, $\\kappa_s$, $\\lambda(t)$ and the cap are STRUCTURAL and',
         'never enter a fitted vector. $\\mu_J$ seeds at $\\log$ *Wing_Strike* and $\\sigma_J$ at',
         'half that magnitude, then $\\lambda(t)=w_J\\xi_{mkt}(t)/(\\mu_J^2+\\sigma_J^2)$ PER',
         'SEGMENT: the wing a desk hedges is ONE jump, vanillas do not identify $\\lambda$ apart',
         'from the jump sizes, and a CONSTANT intensity would force the diffusive share to move',
         'inversely to the market own strip. The one free number in the split is its level $w_J$.',
         '1. The curve $L$ from the ATM term structure through spec 2.6 mapping, then the',
         'triangular bootstrap - at every iterate thereafter.',
         '2. $(\\mu_J,\\sigma_J)$ on the 1-3 month rows. 3. $(\\rho_s,\\sigma_s,\\nu)$ on the 1-12',
         'month rows. 4. $(\\rho_\\ell,\\sigma_\\ell)$ beyond one year, and ONLY where',
         'the ladder carries wing quotes at 18 months or longer; otherwise both are PINNED with',
         'the report line *pinned: not identified by this ladder*, as the $\\kappa$ pair is. WHAT',
         'THEY ARE PINNED AT is read in one order: *Slow_Factor_Prior* where the block declares',
         'one, else a LogVar2FJ history for the underlying in *Price Models*, reported with its',
         'standard errors, else the ASSET CLASS default off the factor type *Underlying* resolves',
         'to - FX $(0.2,0.5)$, an index $(0.4,1.0)$ - whose SIGN is that of the $\\rho_s$ in force',
         'at the pin, so the slow skew is never set against the fast one stage 3 has just fitted.',
         'A pin is applied to every name without an 18-month wing, so a floor-by-default would',
         'understate a whole book long-horizon vol in ONE direction; the floor is the BOX beneath',
         'all three instead - $\\sigma_\\ell\\ge0.3$, there for exposures, a zero slow factor',
         'collapsing a CVA profile vol distribution onto the fast factor spread, which reverts',
         'within months. A DECLARED prior under it refuses by name; a fit sitting on it with long',
         'expiries present is a RESULT, reported and never refused. Where the pair is pinned the',
         'report reads the ladder under all THREE priors - the floor, the class default and the',
         'index seed - at ONE extra forward pass each, everything but the slow pair held at',
         '$\\theta^*$, the $L$ strip re-bootstrapped and the wings re-priced with no outer search,',
         'so every row but the one in force APPROXIMATES the re-fit it is not.',
         '5. THE FORWARD BLOCK, in the order spec 5.3 states and only where *Forward_Smile_Source*',
         'names one: the later buckets of $\\mu_J(t)$, then of $\\rho_s(t)$, then the jump share',
         '$w_J$ by a BOUNDED SCALAR SEARCH - $w_J$ moves the whole $\\lambda$ strip, which moves',
         'integer counts, so it is not a coordinate any Jacobian can carry. THE TARGETS ARE',
         'DIFFERENCES, in vol points and BOTH of them: $\\Delta_{skew}$ is the forward 90-110',
         'slope less the spot one at maturity $\\Delta$, $\\Delta_{bfly}$ the same for the 90/110',
         'butterfly. *Prior* DECLARES the pair (*Stickiness_Prior*, `0.0,0.0` being sticky-delta);',
         '*Quotes* and *Reference* MEASURE it, the source own forward smile less the MARKET own',
         'spot one. So the residual is ONE term whatever the source - the model difference less',
         'the target - taken on the model OWN spot smile per evaluation, in VOL POINTS. A RATIO',
         'would divide by a spot quantity within a few tenths of zero (the butterfly at 3-6',
         'months on an index, the skew on any symmetric FX smile) and ask the fit to match noise.',
         'Neither targets the forward ATM LEVEL, which the ATM ladder already pins.',
         'The spot smile at $T$ never sees a bucket',
         'later than $T$ and a forward smile prices on nothing else, which is why the lever is',
         'CALENDAR TIME and not the vol state (2.3.1, measured three ways).',
         '6. A joint polish over everything but $\\lambda$, the $\\kappa$ pair and the cap.',
         '',
         'BOUNDS LIVE HERE AND NOWHERE ELSE. The engine puts no transform on $\\rho_s$ - one cost a',
         'tenth of the spot skew before it was found - so the box is the calibrator own:',
         '$\\rho_s\\in[-\\sqrt{1-\\rho_\\ell^2-c_{min}},0]$, re-derived as $\\rho_\\ell$ moves, with a',
         'soft penalty sized to bite in the last percent of it. A bucket landing ON that box is the',
         'box stopping $\\rho_s$ rather than the data, which is a surface asking for a ONE-SHOCK',
         'model. Neither guard Floor moves a fitted number: both TAKE what the fit reached and say',
         'so, which is what leaves the quote contraction at the point it was taken.',
         '',
         'WHAT THE FIT REPORTS. The vol-point miss per maturity and over the surface, unweighted',
         'and vega-weighted; the wing and convexity residuals; per bucket the parameters and',
         'which were free; the jump share asked and the one REALISED; the leverage products and',
         '$c$; the stationary log-vol sd; the cap headroom; THE SPLIT per segment - the market',
         'own forward variance, the model total as MEAN diffusive plus jump, the jump share and',
         'the Jensen share the vol-of-vol implies, so a gap between the curve level and the',
         'market reads as the split it is and not as a miss (the total sits ABOVE the market by',
         'what matching the ATM PRICE rather than the variance costs); per $(T_1,\\Delta)$ the',
         'forward slope and butterfly, the spot ones they are differenced against, and BOTH',
         'DIFFERENCES model beside target - with $\\psi_{skew}$ and $\\psi_{bfly}$ beside them',
         'only where the spot side exceeds 0.5 vol points, and named as the division by nothing',
         'it is where it does not; the THREE-WAY prior table wherever the slow pair is pinned;',
         'and THE',
         'IDENTIFICATION TABLE, the SVD of the DATA rows of the Jacobian at each fitted point, its',
         'columns scaled by $1/\\|J_{:,j}\\|$ and the unscaled norms beside it - scaling reads',
         'conditioning, and only the norm says that a well-conditioned direction moves nothing. The',
         'penalty rows are left out because their singular value is an algebraic constant of',
         '*C_Min*. With a forward source the polish table is taken TWICE, with the forward rows and',
         'without, which is what says whether the later bucket is pinned by them or by nothing.',
         '',
         'THE FAILURE MODE IS REPORTED BY NAME (spec 5.3): a vanilla RMSE degraded by more than 0.1',
         'vol points at the target maturities, or a composition residual above 0.3 at the first',
         'rung BEYOND the last bucket boundary - the rung that is the composition of the two',
         'conditional laws - means the forward target is asking for a conditional law this model',
         'does not have with these buckets. The structure is not refined further without a',
         'decision.'
         ]
    )

    market_factor_type = 'LogVar2FJModelPrices'
    #: The fit runs on the CPU whatever device the job was constructed with. One evaluation is a
    #: Python loop over internal steps on `[paths]` tensors and so is dispatch-bound rather than
    #: bandwidth-bound: 504 daily steps at 8192 paths measure 0.36 s here against 0.57 s on an
    #: RTX 3090, and the vmapped backward 1.9 s either way.
    device = torch.device('cpu')

    identification_note = ('the forward-smile targets: the later buckets of Rho_S and Mu_J are '
                           'identified by them and by nothing else')

    #: Four wing expiries at TWO delta pillars: a `Bootstrap` bucket with two wing quotes frees
    #: two parameters and one with four frees four (5.4.5), and an FX smile is quoted at both
    #: wings anyway.
    fx_wing_expiries = (1.0 / 12.0, 0.25, 0.5, 1.0)
    fx_wing_pillars = (0.25, 0.10)

    #: The shared block plus this family's own. `European_Options` stays LAST, as it is there.
    fields = OptionQuoteFamily.fields[:-1] + [
        F('Fit_Mode', 'Text', default='Global', values=['Global', 'Bootstrap'],
          description='Global fits one bucket of shape parameters to every wing quote jointly and '
                      'is what an autocall wants; Bootstrap makes the ladder\'s WING EXPIRIES the '
                      'calendar buckets and fits each given the ones before it, which is what a '
                      'TARF read at many fixings wants (spec 5.4.1)'),
        F('Paths', 'Integer', default=8192,
          description='Paths the fixed antithetic draws carry. The objective is deterministic in '
                      'them, so this sets the noise floor under every fitted number rather than a '
                      'confidence interval around it: at the default, re-running a 34-quote fit at '
                      'three seeds moves the ATM term structure by 0.1 to 0.6 vol points and an L '
                      'level by up to 2.6'),
        F('Random_Seed', 'Integer', default=1,
          description='Seeds the fixed draws. Re-running at another seed is the honest way to '
                      'read how much of a parameter is the surface and how much is the sample'),
        F('Kappa_L', 'Float', default=0.5,
          description='STRUCTURAL slow reversion speed, per year - a prior, never fitted: a '
                      'sub-year ladder does not identify a reversion speed apart from the '
                      'vol-of-vol it multiplies'),
        F('Kappa_S', 'Float', default=6.0,
          description='STRUCTURAL fast reversion speed, per year, on the same terms as Kappa_L'),
        F('Cap_A', 'Float', default=LV_FACTOR_DEFAULTS['Cap_A'],
          description='STRUCTURAL log-variance cap level, raised to L(0) + 6*s_inf where the '
                      'fitted spread asks for it (spec 2.7 level rule) and never lowered. The '
                      'default is 1000% vol; a fit reaching within 5 Cap_Beta of it is refused'),
        F('Cap_Beta', 'Float', default=LV_FACTOR_DEFAULTS['Cap_Beta'],
          description='STRUCTURAL log-variance cap width'),
        F('C_Min', 'Float', default=LV_FACTOR_DEFAULTS['C_Min'],
          description='Floor on the idiosyncratic share c = 1 - Rho_S^2 - Rho_L^2, written onto '
                      'the factor and asserted there at load. THE SAME NUMBER bounds Rho_S here, '
                      'so the fit cannot land past what the model will read. A dial between '
                      'second-order noise and the 105-120% residual; a book may run it near 0.06'),
        F('Idiosyncratic_Share', 'Text', default='Refuse', values=['Refuse', 'Floor'],
          description='What a bucket landing ON the C_Min box does. Refuse names the bucket, the '
                      'c it wanted and the 110-120% convexity residual that goes with the floor; '
                      'Floor takes C_Min and says so (spec 5.4.6)'),
        F('Stationary_Spread', 'Text', default='Refuse', values=['Refuse', 'Floor'],
          description='The same semantics for a stationary log-vol sd outside the 0.4-0.9 VIX '
                      'options imply: Refuse names the bucket, its sd and the pair that produced '
                      'it; Floor takes the fit as it stands and says so. Nothing is scaled - the '
                      'guard is a reading of theta*, which is what leaves the quote contraction '
                      'taken at the point the fit reached'),
        F('Jump_Share', 'Float', default=0.25,
          description='w_J of spec 5.1 - the jump share of total variance at the SHORTEST '
                      'calibrated maturity, which is what fixes Lambda. Seeds stage 5\'s bounded '
                      'search for it, and is the answer where no forward source is named'),
        F('Wing_Strike', 'Float', default=0.85,
          description='The wing the desk hedges, as a ratio of spot at the shortest maturity: '
                      'Mu_J seeds at its log and Sigma_J at half that magnitude. A wing is ONE '
                      'jump, not many small ones (spec 5.1)'),
        F('Wing_Side', 'Text', default='Put', values=['Put', 'Both'],
          description='Which wing Wing_Weight lifts: Put, which is what an index desk hedges, or '
                      'Both, which is how an FX smile is dealt'),
        F('Wing_Weight', 'Float', default=1.0,
          description='Multiplier on the vega weight of the wing quotes Wing_Side names. 1.0 is '
                      'OFF: a weight is a CHOICE about what the fit is for, and a default that '
                      're-weights every surface silently is not one'),
        F('Expiry_Weights', 'Text', default='',
          description='Optional tenor:weight list (3m:2,1y:0.5) moving the fit\'s emphasis along '
                      'the term structure - each rung matched to the quoted maturity nearest the '
                      'tenor. With the mode and the curve\'s knots this is where a fit is '
                      'strongest (spec 5.4.1), so it is a field rather than a rule'),
        F('Lambda', 'Text', default='',
          description='Blank DERIVES the jump intensity PER SEGMENT by spec 5.1, '
                      'w_J*xi_mkt(t)/(mu_J^2 + sigma_J^2) off Jump_Share, the seeded jump sizes '
                      'and the market\'s own forward-variance strip - so the jump share is the '
                      'same number on every segment and the diffusive strip follows the market\'s '
                      'shape rather than its inverse. A number PINS it flat, which is what a '
                      'structural review writes back. Never a fitted coordinate - its whole '
                      'content is the law of integer counts, which no Jacobian carries'),
        F('Param_Buckets', 'Table', default='null', row=Row([
            F('Tenor', 'Float', description='Years from the base date the bucket STARTS at')]),
          description='Calendar-time buckets the four levers are piecewise constant on. Empty is '
                      'ONE bucket - the constant-parameter model. Align them to the forward-smile '
                      'horizons: a spot smile at T never sees a bucket later than T. IGNORED under '
                      'Fit_Mode Bootstrap, where the ladder\'s wing expiries are the buckets'),
        F('Event_Days', 'Text', default='',
          description='Optional comma-separated dates each carrying a ONE-DAY L segment, so the '
                      'variance on the event day is one number and the ATM pillar straddling it '
                      're-solves around it (spec 5.4.3). For short-dated FX this single lever '
                      'outweighs any smile parameter'),
        F('Event_Variance_Prior', 'Float', default=1.0,
          description='The multiplier an event day\'s own diffusive variance carries over the '
                      'segment enclosing it. 1.0 is OFF; 3.0 says the day carries three ordinary '
                      'days of variance and the segment around it gives that back'),
        F('Forward_Smile_Source', 'Text', default='None',
          values=['None', 'Quotes', 'Reference', 'Prior'],
          description='Where spec 5.3\'s forward-skew target comes from, and therefore which '
                      'instrument is priced. Quotes are TRADED forward-starts, E[S_T1 (R-k)^+] / '
                      'E[S_T1], read off Forward_Smiles; Reference is a reference model\'s own '
                      'forward smiles, the ratio expectation E[(R-k)^+], off the same table; Prior '
                      'needs no table and tilts the market\'s spot smile at each Delta by '
                      'Stickiness_Prior over the Forward_Tenors pairs. None skips stage 5, and a '
                      'vanilla-only calibration is then what it is - not an autocall calibration. '
                      'Global mode only'),
        F('Forward_Smiles', 'Table', default='null', row=Row([
            F('T1', 'Float', description='Forward start, in years'),
            F('Delta', 'Float', description='Tenor of the forward-start option, in years'),
            F('Strike', 'Float', description='Strike as a fraction of S_T1'),
            F('Target_Vol', 'Float', description='The forward-start implied vol to hit'),
            F('Weight', 'Float', default=1.0,
              description='Relative weight within the forward block')]),
          description='The rows Forward_Smile_Source Quotes or Reference reads its targets from'),
        F('Forward_Tenors', 'Text', default='6m:6m,1y:1y,1y:3m',
          description='The (T1:Delta) pairs Forward_Smile_Source Prior builds its rows on'),
        F('Stickiness_Prior', 'Text', default='0.0,0.0',
          description='The PAIR Delta_skew,Delta_bfly in VOL POINTS that Forward_Smile_Source '
                      'Prior targets on the model\'s OWN spot smiles: 0.0,0.0 is sticky-delta, '
                      'the FX convention and the default; a few tenths negative on the skew is an '
                      'LSV-like view. DIFFERENCES rather than ratios, because a ratio divides by a '
                      'spot quantity within 0.3 vol points of zero - the butterfly at 3-6m on an '
                      'index, the skew on any symmetric FX smile - and asks the fit to match '
                      'noise. Both, because jump skew is sticky while its convexity dilutes at the '
                      'forward date and leverage skew dilutes while vol-of-vol convexity '
                      'amplifies: the slope alone cannot separate the jump share from the '
                      'vol-of-vol'),
        F('Slow_Factor_Prior', 'Text', default='',
          description='The PAIR rho_l,sigma_l held where the ladder carries no wing quote at 18 '
                      'months or longer and spec 5.2 stage 4 fits neither. Blank takes the ASSET '
                      'CLASS default off the factor type Underlying resolves to - FxRate 0.2/0.5, '
                      'EquityPrice 0.4/1.0 - signed by the Rho_S in force at the pin, so the slow '
                      'skew is never set against the fast one; a LogVar2FJ history for the '
                      'underlying in Price Models replaces it and is reported with its standard '
                      'errors. Sigma_L >= 0.3 is the box beneath all three, there for the 2-5 year '
                      'exposure tail rather than for the smile - a zero slow factor collapses a '
                      'CVA profile\'s vol distribution onto the fast factor\'s spread - and a '
                      'declared prior under it REFUSES rather than being floored silently'),
        F('Diffusive_Share', 'Text', default='',
          description='Target share of each L segment\'s total variance carried by the DIFFUSIVE '
                      'part - one number for the strip or one per segment - as a soft term on the '
                      'split and NOTHING ELSE. Blank is off and there is no default: it is a prior '
                      'on a number the desk has no market for. Tightening the jump-share box is '
                      'not a substitute; that box touches one side of the gap'),
        F('Forward_Weight', 'Float', default=1.0 / 3.0,
          description='The forward block\'s share of the total objective weight, the vanillas '
                      'carrying the rest'),
        F('Bucket_Smoothness', 'Float', default=0.02,
          description='Weight on the difference between adjacent buckets of any lever, over the '
                      'buckets FITTED so far - a difference against a bucket still at its seed '
                      'measures the seed. At the default a 0.1 step costs 0.2 vol points of '
                      'residual, which a forward target outweighs and sampling noise does not. '
                      'Not optional in Bootstrap mode'),
        F('Stationarity_Tol', 'Float', default=1e-4,
          description='The Gauss-Newton contraction is taken at a stationary point, so a fit that '
                      'stopped somewhere else - a stage CAPPED at Max_Iterations - has no quote '
                      'derivative to report. ||J^T r|| over the free coordinates above this '
                      'REFUSES Quote_Sensitivity by name. The norm is absolute and the scale is '
                      'the weighted vol-space residual, so it is declared per block'),
        F('Jacobian_Rcond', 'Float', default=1e-3,
          description='A singular value below this times the largest names a FLAT direction, '
                      'whose right singular vector the identification table prints as parameter '
                      'loadings - on the COLUMN-SCALED Jacobian, and the quote contraction '
                      'pseudo-inverts that same matrix at that same cutoff. It is therefore the '
                      'dial: a direction just inside it amplifies dtheta/dq by one over its '
                      'singular value, so a quote delta riding a direction the quotes barely '
                      'identify is raised out of the answer here'),
        F('Max_Iterations', 'Integer', default=150,
          description='EVALUATIONS least_squares may spend per stage (scipy\'s max_nfev). One '
                      'evaluation is one L bootstrap plus one walk; each accepted point costs a '
                      'Jacobian on top (one vmapped backward). A stage stopping here reports '
                      'itself CAPPED with the residual it reached, which is the tolerance it '
                      'actually got to'),
        F('Tolerance', 'Float', default=1e-8,
          description='Convergence tolerance (scipy\'s ftol) on each stage\'s weighted vol-space '
                      'residual'),
        F('Pillar_Tolerance', 'Float', default=1e-10,
          description='The relative ATM miss each L segment\'s own Newton solve stops at, at EVERY '
                      'outer iterate. It is what makes the objective a function of x alone rather '
                      'than of the sweep it warm-started from, so it wants to sit well under '
                      'Tolerance; every step below that costs one prefix walk'),
        F('Quote_Sensitivity', 'Text', default='No', values=['Yes', 'No'],
          description='Keep the written parameters connected to the numbers quoted, so a '
                      'calculation\'s backward pass reports dV/dq beside dV/dtheta. The fitted '
                      'parameters are identical either way. REFUSED under Fit_Mode Bootstrap')
    ] + OptionQuoteFamily.fields[-1:]

    def __init__(self, param, device, dtype):
        # the constructed device and dtype are ignored - see the `device` note and `prec`
        self.param = param
        #: What `Quote_Sensitivity` leaves behind: every fitted parameter still connected to its
        #: quotes, keyed as `_build_factor_state` mints its leaf, plus the quote leaf per block.
        #: `Config.bootstrap` harvests both - tensors cannot live in `Price Factors`.
        self.calibrated = {}
        self.quote_leaves = {}

    def bootstrap(self, sys_params, price_models, price_factors, factor_interp, market_prices,
                  calendars, debug=None):
        """Calibrates the LogVar2FJ parameters and writes a `LogVar2FJModelParameters` price
        factor.

        The quote preparation is the plain family's, quote for quote. The quotes then SPLIT into an
        ATM ladder the inner bootstrap consumes one pillar at a time - the ATM quote at an expiry
        being the one nearest its own forward, so a hand-authored block reads as an emitted one
        does - and the whole set, which every fitted stage is judged on. A previously written
        factor warm starts every scalar, every lever and the curve. `LVFit` owns the state and the
        verbs; what is here is the quotes, the grid they are priced on, and the order the stages
        run in.
        """
        for market_price, implied_params in market_prices.items():
            rate = utils.check_rate_name(market_price)
            market_factor = utils.Factor(rate[0], rate[1:])
            if market_factor.type != self.market_factor_type:
                continue
            instrument = implied_params['instrument']
            factors, spot = self.resolve_block(
                market_price, instrument, price_factors, factor_interp, sys_params)
            param_name = utils.check_tuple_name(
                utils.Factor(self.__class__.__name__, market_factor.name))
            fit = LVFit(self, market_price, instrument, factors,
                        price_factors.get(param_name))
            fit.history = self.slow_history(market_price, instrument, price_models)
            if not self.prepare(fit, sys_params, factors, spot):
                continue

            connect = fit.instrument['Quote_Sensitivity'] == 'Yes'
            if connect and fit.mode == 'Bootstrap':
                raise ValueError(
                    '{}: Quote_Sensitivity is not available under Fit_Mode Bootstrap. Bucket k is '
                    'fitted to expiry k\'s wings GIVEN the buckets before it, so theta* is a '
                    'stationary point of no single objective and the Gauss-Newton contraction '
                    'would report the LAST bucket\'s quote derivative as the whole surface\'s. '
                    'Fit Global, which carries it, or read the risk on the parameters'.format(
                        market_price))
            if connect:
                fit.leaf = torch.tensor(
                    [quote.quoted for quote in fit.quotes], device=self.device, dtype=self.prec,
                    requires_grad=True)

            theta = LeastSquaresSolve.apply(fit, fit.rcond, fit.stationarity, fit.leaf)
            if connect and fit.capped:
                raise ValueError(
                    '{}: stage {} stopped CAPPED at Max_Iterations={}, so theta* is where the '
                    'evaluation budget ran out and not where the objective is stationary - the '
                    'Gauss-Newton contraction has no fixed point to be taken at. Raise '
                    'Max_Iterations, loosen Tolerance, or read the risk on the parameters'.format(
                        market_price, fit.capped, fit.max_iter))
            fit.finish(theta)
            fit.verify()
            fit.report()
            price_factors[param_name] = fit.written()

            if connect:
                factor = utils.Factor(self.__class__.__name__, market_factor.name)
                self.calibrated.update({
                    utils.Factor(factor.type, factor.name + (name,)): value
                    for name, value in fit.connect().items()})
                self.quote_leaves[market_price] = (fit.descriptors, fit.leaf)

    def prepare(self, fit, sys_params, factors, spot):
        """The quotes, the buckets and the grid the whole fit is priced on - everything `LVFit`
        needs before a stage runs. False where the block carries nothing to fit.

        THE GRID IS THE QUOTES' OWN: a block ends at every quoted maturity and at every forward
        window's two ends, its steps one trading day except the last, a STUB landing the
        block exactly on T. So the variance the fit reads at a maturity is the variance a pricer
        reads at that tenor of the curve written out, to the digit.
        """
        quotes = [LVQuote(
            row=None, j=None, spot=spot, strike=strike, ratio=self.tensor(strike / spot),
            is_call=sign > 0, units=option['Units'], T=t, rate=r, carry=(r - q) * t,
            forward=forward, premium=premium, weight=float(option['Weight']), sigma=sigma,
            quoted=sigma if fit.is_vol else option['Quoted_Market_Value'],
            vega=option['Units'] * self.fx_black_vega(forward, strike, r, sigma, t))
            for option, t, r, q, forward, sign, strike, sigma, premium
            in self.prepare_quotes(sys_params, fit.instrument, factors, spot)]
        dead = [quote for quote in quotes if not quote.vega > 0.0]
        if dead:
            logging.warning(
                '{}: {} of {} quotes carry a Black vega of zero at their own quoted vol and are '
                'DROPPED - the residual is a premium miss over that vega, so a contract nothing '
                'prices has no vol-space reading. Strikes: {}'.format(
                    fit.market_price, len(dead), len(quotes),
                    ', '.join('{:.4g} at {:.3f}y'.format(x.strike, x.T) for x in dead)))
            quotes = [quote for quote in quotes if quote.vega > 0.0]
        if not quotes:
            logging.error('{} carries no quotes - nothing to bootstrap'.format(fit.market_price))
            return False

        # the split, by POSITION - two quotes at one strike and expiry compare equal as tuples:
        # one ATM per distinct expiry (nearest its own forward), the rest that expiry's wings
        by_expiry = {}
        for i, quote in enumerate(quotes):
            by_expiry.setdefault(quote.T, []).append(i)
        atm = [min(by_expiry[T], key=lambda i: abs(quotes[i].strike / quotes[i].forward - 1.0))
               for T in sorted(by_expiry)]
        wings = OrderedDict((T, [i for i in by_expiry[T] if i not in set(atm)])
                            for T in sorted(by_expiry) if len(by_expiry[T]) > 1)
        fit.buckets = self.param_buckets(fit, wings)
        fit.free = {}

        quotes = self.emphasis(fit, quotes, set(atm))
        rungs = {T: [quotes[i] for i in rows] for T, rows in by_expiry.items()}
        targets = self.forward_rows(fit, factors)
        ends = sorted(set(by_expiry) | {t for target in targets
                                        for t in (target.T1, target.T1 + target.tenor)})
        at = {T: j for j, T in enumerate(ends)}
        # every forward tenor's own spot rung: the quoted maturity nearest it, which is where the
        # model's spot smile is read for the two ratios and for what `Prior` tilts
        near = {target.tenor: min(rungs, key=lambda T: abs(T - target.tenor))
                for target in targets}
        fit.spot_rung = {d: (at[T], rungs[T][0].carry, T) for d, T in near.items()}
        fit.tilt = self.tilts(fit, targets, rungs)
        spans = np.diff([0.0] + ends)
        counts = np.maximum(np.round(spans / fit.delta), 1.0).astype(int)
        fit.upto = np.cumsum(counts)
        fit.n = int(fit.upto[-1])
        share = float(fit.instrument['Forward_Weight']) if targets else 0.0
        total = sum(quote.weight for quote in quotes)
        fit.quotes = [quote._replace(row=i, j=at[quote.T],
                                     weight=np.sqrt(quote.weight * (1.0 - share) / total))
                      for i, quote in enumerate(quotes)]
        fit.atm = [fit.quotes[i] for i in atm]
        fit.wings = OrderedDict((T, [fit.quotes[i] for i in rows])
                                for T, rows in wings.items())
        if targets:
            total = sum(target.weight for target in targets)
            targets = [target._replace(
                j1=at[target.T1], j2=at[target.T1 + target.tenor],
                weight=np.sqrt(target.weight * share / total)) for target in targets]
        fit.targets = targets

        fit.deltas = self.vector(np.concatenate(
            [np.append(np.full(n - 1, fit.delta), span - (n - 1) * fit.delta)
             for n, span in zip(counts, spans)]))
        fit.times = torch.cat([fit.deltas.new_zeros(1), fit.deltas.cumsum(0)])
        fit.uniforms = fit.draw(int(fit.instrument['Paths']), fit.n, int(fit.instrument['Random_Seed']))
        fit.knots = np.array([0.0] + [quote.T for quote in fit.atm])
        variance = np.array([quote.sigma ** 2 * quote.T for quote in fit.atm])
        fit.xi = np.diff(np.concatenate([[0.0], variance])) / np.diff(fit.knots)
        self.event_knots(fit, sys_params)
        self.seed(fit)
        return True

    def emphasis(self, fit, quotes, atm):
        """The desk's own emphasis on the vega weights (5.4.4): `Wing_Weight` on the wings
        `Wing_Side` names, and `Expiry_Weights`' tenor:weight list along the term structure, each
        rung matched to the quoted maturity nearest the tenor.

        Both are RELATIVE - the normalisation that follows divides them out - so the defaults
        (1.0 and blank) leave every weight the vega it was.
        """
        wing, side = float(fit.instrument['Wing_Weight']), fit.instrument['Wing_Side']
        expiries = sorted({quote.T for quote in quotes})
        by_expiry = {}
        for pair in str(fit.instrument['Expiry_Weights']).split(','):
            if pair.strip():
                tenor, value = pair.split(':')
                by_expiry[min(expiries, key=lambda T: abs(T - lv_tenor(tenor)))] = float(value)
        if wing == 1.0 and not by_expiry:
            return quotes
        logging.info('  {}: the fit is weighted {} on the {} wing{} and {} along the term '
                     'structure'.format(
                         fit.market_price, wing, side.lower(), '' if side == 'Put' else 's',
                         ', '.join('{:g}y {:g}'.format(T, x) for T, x in sorted(by_expiry.items()))
                         or 'evenly'))
        lifted = lambda i, quote: i not in atm and (
            side == 'Both' or quote.strike < quote.forward)
        return [quote._replace(weight=quote.weight * by_expiry.get(quote.T, 1.0)
                               * (wing if lifted(i, quote) else 1.0))
                for i, quote in enumerate(quotes)]

    def param_buckets(self, fit, wings):
        """The calendar buckets the four levers are piecewise constant on: `Param_Buckets` in
        `Global` mode, and in `Bootstrap` mode the ladder's own WING EXPIRIES, a bucket starting
        where the previous expiry ended so an option to `E_k` reads buckets 0..k (5.4.1).

        A ladder whose wings all land on one expiry has no smile term structure in it and is
        REFUSED in that mode rather than fitted as one bucket under another name.
        """
        if fit.mode != 'Bootstrap':
            return np.array([0.0] + sorted(
                float(row['Tenor']) for row in fit.table('Param_Buckets')
                if float(row['Tenor']) > 0.0))
        if len(wings) < 2:
            raise ValueError(
                '{}: Fit_Mode Bootstrap makes the ladder\'s WING EXPIRIES the calendar buckets, '
                'and this ladder carries wing quotes at {} - a ladder whose wings collapse onto '
                'one expiry has no smile term structure to bootstrap through. Quote wings at more '
                'expiries (the emitter asks for {} at {} delta), or fit Global, where one bucket '
                'is the model'.format(
                    fit.market_price,
                    ', '.join('{:g}y'.format(T) for T in wings) or 'no expiry at all',
                    '/'.join('{:g}y'.format(T) for T in self.fx_wing_expiries),
                    '/'.join('{:g}'.format(p) for p in self.fx_wing_pillars)))
        if fit.table('Param_Buckets'):
            logging.warning('{}: Param_Buckets is ignored under Fit_Mode Bootstrap - the buckets '
                            'are the ladder\'s wing expiries {}'.format(
                                fit.market_price,
                                ', '.join('{:g}y'.format(T) for T in wings)))
        return np.array([0.0] + list(wings)[:-1])

    def forward_rows(self, fit, factors):
        """Spec 5.3's forward-start rows, in the instrument `Forward_Smile_Source` names.

        `Quotes` and `Reference` read `Forward_Smiles`; `Prior` builds its own rows off
        `Forward_Tenors` and aims them at the MODEL's own spot smile, tilted by both
        `Stickiness_Prior` ratios per evaluation. A table with no source, or a source in
        `Bootstrap` mode, refuses by name.
        """
        rows, source = fit.table('Forward_Smiles'), fit.source
        if source != 'None' and fit.mode == 'Bootstrap':
            raise ValueError(
                '{}: Forward_Smile_Source {} is Global mode only (spec 5.4.1). In Bootstrap mode '
                'forward smiles are CONSEQUENCES of the bucket term structure and psi is reported '
                'rather than targeted - the report prints it. Set Forward_Smile_Source to None, or '
                'fit Global'.format(fit.market_price, source))
        if rows and source == 'None':
            raise ValueError(
                '{}: Forward_Smiles carries {} rows and Forward_Smile_Source is None, so nothing '
                'would be measured against them. Say what they are: Quotes (traded forward-starts, '
                'priced under the share measure), or Reference (a reference model\'s forward '
                'smiles, priced as the ratio expectation)'.format(fit.market_price, len(rows)))
        if not rows and source in ('Quotes', 'Reference'):
            raise ValueError(
                '{}: Forward_Smile_Source {} reads its targets from Forward_Smiles and the table '
                'is empty, so stage 5 would be skipped and a vanilla-only fit written under a '
                'block that asked for a forward target. Author the rows, or name Prior, which '
                'needs none, or None'.format(fit.market_price, source))
        if source == 'Prior':
            rows = self.prior_rows(fit)
        elif source == 'None':
            return []

        discount, carry = factors['Discount_Rate'], factors.get('Yield')
        funding = factors.get('Funding_Rate')
        targets = []
        for row in rows:
            t1, t2 = float(row['T1']), float(row['T1']) + float(row['Delta'])
            r1, r2 = float(discount.current_value(t1)), float(discount.current_value(t2))
            # the window's own carry, differenced off the two maturities the curves read
            window = ((r2 - self.effective_yield(r2, funding, carry, t2)) * t2
                      - (r1 - self.effective_yield(r1, funding, carry, t1)) * t1)
            targets.append(LVForward(
                j1=None, j2=None, T1=t1, tenor=t2 - t1, strike=float(row['Strike']),
                ratio=self.tensor(float(row['Strike'])), carry=window,
                weight=float(row.get('Weight', 1.0)), target=float(row['Target_Vol'])))
        return targets

    def prior_rows(self, fit):
        """The rows a prior on the two ratios builds: spec 5.3's strikes at each `(T1, Delta)`,
        aimed at the MODEL's own spot smile at the quoted maturity nearest `Delta` with its slope
        and butterfly scaled by `Stickiness_Prior` (5.3 source 3).

        The target therefore moves with the fit and what is pinned is the RATIO pair, not a vol -
        which is what makes `(1.0, 1.0)` sticky-delta rather than a smile from yesterday. The row
        carries no target vol of its own; `LVFit.target_vol` builds it per evaluation.
        """
        return [{'T1': lv_tenor(pair.split(':')[0]), 'Delta': lv_tenor(pair.split(':')[1]),
                 'Strike': k, 'Target_Vol': 0.0}
                for pair in fit.instrument['Forward_Tenors'].split(',')
                for k in fit.prior_strikes]

    def tilts(self, fit, targets, rungs):
        """`{(T1, Delta): (Delta_skew, Delta_bfly)}` in DECIMAL vol - the differences spec 5.3
        targets, ONE pair per forward tenor whatever the source, so the residual is one term: the
        model's difference less the target's.

        `Prior` declares the pair in vol points. `Quotes` and `Reference` MEASURE it - the
        source's own forward slope and butterfly at `psi_strikes` less the MARKET's own spot ones
        at the rung nearest Delta, read off the quoted vols in log-moneyness. Neither targets the
        forward ATM LEVEL: the ATM ladder pins it and the L strip reprices it exactly.
        """
        if fit.source == 'Prior':
            pair = self.stickiness_prior(fit)
            return {(target.T1, target.tenor): pair for target in targets}
        smiles = {}
        for target in targets:
            smiles.setdefault((target.T1, target.tenor), {})[
                round(target.strike, 10)] = target.target
        tilt = {}
        for (T1, tenor), smile in smiles.items():
            missing = [k for k in fit.psi_strikes if round(k, 10) not in smile]
            if missing:
                raise ValueError(
                    '{}: Forward_Smiles quotes {:g}y into {:g}y at strikes {} and spec 5.3 targets '
                    'the DIFFERENCES - the forward 90-110 slope and 90/110 butterfly less the spot '
                    'ones at that maturity - so the tenor needs {}. Quote {}'.format(
                        fit.market_price, T1, tenor,
                        '/'.join('{:g}'.format(k) for k in sorted(smile)),
                        '/'.join('{:g}'.format(k) for k in fit.psi_strikes),
                        ', '.join('{:g}'.format(k) for k in missing)))
            forward = fit.shape([smile[round(k, 10)] for k in fit.psi_strikes])
            spot = fit.market_shape(rungs[fit.spot_rung[tenor][2]])
            tilt[(T1, tenor)] = (forward[1] - spot[1], forward[2] - spot[2])
        return tilt

    def stickiness_prior(self, fit):
        """`Stickiness_Prior` as `(Delta_skew, Delta_bfly)` in DECIMAL vol - the two DIFFERENCES
        spec 5.3 targets, the FX default `0.0,0.0` being sticky-delta."""
        rows = [x for x in str(fit.instrument['Stickiness_Prior']).split(',') if x.strip()]
        if len(rows) != 2:
            raise ValueError(
                '{}: Stickiness_Prior is the PAIR Delta_skew,Delta_bfly in VOL POINTS (spec 5.3) '
                'and reads {!r}. The forward smile\'s slope and its convexity dilute differently '
                '- jump skew is sticky while jump convexity dilutes, leverage skew dilutes while '
                'vol-of-vol convexity amplifies - so one number cannot say what the desk assumes. '
                'Write both, 0.0,0.0 for sticky-delta'.format(
                    fit.market_price, fit.instrument['Stickiness_Prior']))
        return tuple(0.01 * float(x) for x in rows)

    def slow_history(self, market_price, instrument, price_models):
        """The historical slow pair where the job's `Price Models` carries a `LogVar2FJCalibration`
        block for this underlying (spec 5.5.3), read by the shape `utils.LV_SLOW_HISTORY` declares
        for both lanes. The estimator is the calibration's; this is its reader."""
        model, keys = utils.LV_SLOW_HISTORY
        block = price_models.get(utils.check_tuple_name(utils.Factor(
            model, utils.check_rate_name(instrument['Underlying']))))
        missing = [] if block is None else [name for name in keys if name not in block]
        if missing:
            raise ValueError(
                '{}: Price Models carries {}.{}, which spec 5.2 stage 4 reads the slow pair off '
                'where the ladder does not identify it, and it is missing {}. A historical prior '
                'is the pair AND its standard errors ({}) - re-run the calibration, or drop the '
                'block and the asset class default stands'.format(
                    market_price, model, instrument['Underlying'], '/'.join(missing),
                    '/'.join(keys)))
        return block

    def diffusive_share(self, fit):
        """`Diffusive_Share` as one target per L SEGMENT - one number for all of them, or one
        each. An explicit override of the split and nothing else (spec 5.3), never a default."""
        rows = [float(x) for x in str(fit.instrument['Diffusive_Share']).split(',') if x.strip()]
        if not rows:
            return None
        if len(rows) not in (1, fit.xi.size):
            raise ValueError(
                '{}: Diffusive_Share carries {} numbers against {} L segments. It is a target '
                'share of each segment\'s total variance carried by the DIFFUSIVE part, so write '
                'one number for the whole strip or one per segment'.format(
                    fit.market_price, len(rows), fit.xi.size))
        return rows * fit.xi.size if len(rows) == 1 else rows

    def event_knots(self, fit, sys_params):
        """`Event_Days` as a one-INTERNAL-STEP SEGMENT each - a day at the default delta (spec
        5.4.3) - carrying `Event_Variance_Prior` times the enclosing segment's level, the knot
        after it restoring that level. A day whose ATM segment is ALREADY one step is IDENTIFIED
        by its own pillar, so the prior is ignored there and the report says so. The variance on
        the event day is then one number and the ATM pillar straddling it gives that variance back
        over the days around it; a date with no straddling expiry carries the prior alone."""
        days = [x.strip() for x in str(fit.instrument['Event_Days']).split(',') if x.strip()]
        prior = float(fit.instrument['Event_Variance_Prior'])
        base, discount = sys_params['Base_Date'], fit.factors['Discount_Rate']
        marks = {}
        for day in days:
            t = discount.get_day_count_accrual(base, (pd.Timestamp(day) - base).days)
            if fit.own_segment(t):
                fit.identified_days.append(
                    'the event day {} is its OWN L segment - expiries on the business days either '
                    'side - so the ordinary pillar bootstrap sets its variance and '
                    'Event_Variance_Prior is IGNORED (spec 5.4.3)'.format(day))
                continue
            marks.setdefault(t + fit.delta, 0.0)                  # the restore, unless a day is
            marks[t] = np.log(prior)                              # already bumped there
        keep = sorted(t for t in marks
                      if t > 0.0 and np.min(np.abs(fit.knots - t)) > utils.BUCKET_TOL)
        fit.event_times = np.array(keep)
        fit.event_log = self.vector([marks[t] for t in keep])
        priced = len(days) - len(fit.identified_days)
        if priced:
            logging.info('  {} event day{} carrying {:g}x the diffusive variance of the day '
                         'around them, as {} extra L knots'.format(
                             priced, '' if priced == 1 else 's', prior, fit.event_times.size))

    def seed(self, fit):
        """STAGE 0, the structural half: the jump sized off the wing the desk hedges, the lambda
        STRIP off the share it carries along the market's own forward variance (5.1), and the L
        strip off spec 2.6's mapping. The kappas and the cap width are read and never moved; a
        previous factor then replaces every seed but those.
        """
        read, wing = fit.instrument, float(fit.instrument['Wing_Strike'])
        n, fit.xi_0 = fit.buckets.size, fit.atm[0].sigma ** 2
        fit.state = {'Kappa_L': float(read['Kappa_L']), 'Kappa_S': float(read['Kappa_S']),
                     'Sigma_L': 1.0, 'Rho_L': -0.4, 'Nu': 0.3,
                     'Cap_A': float(read['Cap_A']), 'Cap_Beta': float(read['Cap_Beta']),
                     'Rho_S': [-0.75] * n, 'Mu_J': [np.log(wing)] * n,
                     'Sigma_S': [2.4] * n, 'Sigma_J': [0.5 * abs(np.log(wing))] * n}
        # 5.1's OWN jump sizes, off the wing the desk hedges - never off what stage 2 later fits. A
        # lambda chasing mu_J puts the day-to-day stability back on the parameter 5.1 exists to
        # take it off, and a mu_J on its upper bound then explodes it.
        fit.jump_size = 1.25 * np.log(wing) ** 2
        fit.pinned = str(read['Lambda']).strip()
        fit.share = float(read['Jump_Share'])
        fit.diffusive = self.diffusive_share(fit)

        previous, levels = fit.previous, None
        if previous:
            fit.state.update({name: float(previous[name]) for name in utils.LV_PARAM_NAMES
                              if not name.startswith('Kappa')})
            fit.state.update({name: utils.bucket_at(
                previous[name].array[:, 0], self.vector(previous[name].array[:, 1]),
                self.vector(fit.buckets)).tolist() for name in utils.LV_BUCKET_NAMES})
            levels = list(utils.bucket_at(previous['L_Curve'].array[:, 0],
                                          self.vector(previous['L_Curve'].array[:, 1]),
                                          self.vector(fit.knots[:-1])))
        # spec 5.1: lambda(t) is RE-DERIVED from today's strip and never warm started - what
        # carries across days is w_J, read back off a previous factor's own first segment
        if previous and not fit.pinned:
            fit.share = float(previous['Lambda'].array[0, 1]) * fit.jump_size / fit.xi_0
        fit.state['Lambda'] = ([float(fit.pinned)] * fit.xi.size if fit.pinned
                               else fit.lam_strip())
        if levels is None:
            levels = [self.tensor(x) for x in LVFit.seed_curve(
                fit.xi, fit.state, fit.state['Lambda'], fit.state['Mu_J'][0],
                fit.state['Sigma_J'][0], fit.state['Sigma_S'][0], fit.delta,
                [int(fit.upto[quote.j]) for quote in fit.atm])]
        fit.levels, fit.warm = list(levels), list(levels)
        fit.slopes = [None] * len(levels)
        fit.draws = fit.counted()


class GBMAssetPriceTSModelParameters(object):
    documentation = (
        'Fx And Equity',
        ['For Risk Neutral simulation, an integrated curve $\\bar{\\sigma}(t)$ needs to be specified and is',
         'interpreted as the average volatility at time $t$. This is typically obtained from the corresponding',
         'ATM volatility. This is then used to construct a new variance curve $V(t)$ which is defined as',
         '$V(0)=0, V(t_i)=\\bar{\\sigma}(t_i)^2 t_i$ and $V(t)=\\bar{\\sigma}(t_n)^2 t$ for $t>t_n$ where',
         '$t_1,...,t_n$ are discrete points on the ATM volatility curve.',
         '',
         'Points on the curve that imply a DECREASE in forward variance are adjusted up to the least',
         'variance that step can reach, which is the one a zero instantaneous vol over it leaves:',
         '$V(t_i)=V(t_{i-1})+\\frac{t_i-t_{i-1}}{3}\\sigma(t_{i-1})^2$. This curve is then used to construct',
         '*instantaneous* curves that are then input to the corresponding stochastic process.',
         '',
         'The relationship between integrated $F(t)=\\int_0^t f_1(s)f_2(s)ds$ and instantaneous curves $f_1, f_2$',
         'where the instantaneous curves are defined on discrete points $P={t_0,t_1,..,t_n}$ with $t_0=0$ is defined',
         'on $P$ by Simpson\'s rule:',
         '',
         '$$F(t_i)=F(t_{i-1})+\\frac{t_i-t_{i-1}}{6}\\Big(f(t_i)+4f(\\frac{t_i+t_{i-1}}{2})+f(t_i)\\Big)$$',
         '',
         'and $f(t)=f_1(t)f_2(t)$. Integrated curves are flat extrapolated and linearly interpolated.'
         ]
    )

    market_factor_type = 'GBMAssetPriceTSModelPrices'
    factor_types = {'Asset_Price_Volatility': utils.TwoDimensionalFactors}
    #: The precision the TAPE runs in - the value path is numpy and has no dtype to pick. Float64 on
    #: the CPU whatever the job asked for; `construct_bootstrapper`'s dtype does not reach it.
    dtype = torch.float64
    fields = [
        F('Asset_Price_Volatility', 'Text', default=REQUIRED,
          description='The vol surface whose ATM column becomes the integrated vol curve'),
        F('Quote_Sensitivity', 'Text', default='No', values=['Yes', 'No'],
          description='Keep the integrated vol curve connected to the ATM vols it was built from, '
                      'so a calculation\'s backward pass reports dV/dq beside dV/dtheta. The '
                      'written curve is identical either way')
    ]

    def __init__(self, param, device, dtype):
        self.device = device
        self.prec = dtype
        self.param = param
        #: What `Quote_Sensitivity` leaves behind: the integrated vol curve still connected to its
        #: ATM quotes, keyed as `_build_factor_state` mints its `Vol` leaf, plus the quote leaf per
        #: block. `Config.bootstrap` harvests both - tensors cannot live in `Price Factors`.
        self.calibrated = {}
        self.quote_leaves = {}

    @staticmethod
    def atm_column(vol_factor, vol_surface, market_prices, price_factors):
        """The ATM vol per surface expiry, and where each number came from.

        Two sources, chosen by the surface's PROVENANCE. Where this market data carries an
        `FXVolPrices` block for the surface being integrated and that surface is the one the block
        wrote, its ATM rows ARE its ATM vols (`Factor2D.malz_skew` puts the +-0.5 label's vol at the
        delta-neutral straddle strike), so they are taken straight off it.

        Provenance is evidence, not a name: a hand-authored surface can sit under a name a quote
        block also uses. What is checked is the fingerprint `FXVolSurfaceParameters` leaves and
        `pinned_grid` reads back - the `Malz` subtype beside its `Grid_Tolerance`.

        Anything else is authored data, and the ATM column is `np.interp` at moneyness 1. Where the
        surface carries a node there, that read is the node itself.

        KNOWN LIMITATION: moneyness 1 is the ATM coordinate of a RATIO surface, while a `Malz`
        axis is log(F/K), whose ATM is at 0 - so a hand-authored Malz surface reads a wing.
        """
        family = FXVolSurfaceParameters
        quoted = market_prices.get(utils.check_tuple_name(utils.Factor(
            family.market_factor_type, vol_factor.name)))
        written = price_factors.get(utils.check_tuple_name(vol_factor), {})
        if quoted is not None and written.get('Surface_Type') == family.surface_type and \
                'Grid_Tolerance' in written:
            atm = family.atm_quotes(family.used(quoted['instrument']))
            if set(atm) != set(vol_surface.expiry):
                raise ValueError(
                    '{} is quoted at expiries {} and carries a surface over {} - the quotes moved '
                    'since it was built, so re-bootstrap the surface before integrating it'.format(
                        utils.check_tuple_name(vol_factor),
                        ', '.join('{:g}'.format(T) for T in sorted(atm)),
                        ', '.join('{:g}'.format(T) for T in sorted(vol_surface.expiry))))
            return [atm[expiry] for expiry in vol_surface.expiry], 'its own ATM quotes'

        mn_ix = np.searchsorted(vol_surface.moneyness, 1.0)
        return [np.interp(1, vol_surface.moneyness[mn_ix - 1:mn_ix + 1], y) for y in
                vol_surface.get_vols()[:, mn_ix - 1:mn_ix + 1]], 'the surface at moneyness 1'

    @staticmethod
    def integrated_vol(atm_vol, expiry):
        """The ATM column as the integrated vol curve the process reads - the value path.

        `V(t_i) = sigma_bar(t_i)^2 t_i` is the total variance the column implies, and the walk is
        Simpson's rule inverted for the instantaneous vol over each step,

            V(t_i) - V(t_{i-1}) = (dt/3)(sigma_{i-1}^2 + sigma_{i-1} sigma_i + sigma_i^2)

        a quadratic in `sigma_i` whose positive root is taken. Returns the curve and the expiries
        the repair fired at. The numpy walk is the only thing a mark is built from; `carried_vol`
        is the derivative twin.

        The map is piecewise and the switch is a KINK. A column implying a declining forward
        variance has no root, so `V(t_i)` is floored at what `sigma_i = 0` leaves and the written
        vol is that floor rather than the quote - `d/dq` is 1 on the smooth side and 0 on the
        floored one, in that column and every later one.

        Only `sigma_bar` is written; `sigma` is the walk's own state, sizing the next step's floor.
        """
        if expiry.size == 1:
            return list(atm_vol), []

        dt = np.diff(np.append(0, expiry))
        var = expiry * np.array(atm_vol) ** 2
        sig, vol, var_tm1, floored = atm_vol[:1], atm_vol[:1], var[0], []

        for var_t, delta_t, t_i in zip(var[1:], dt[1:] / 3.0, expiry[1:]):
            M = var_tm1 + delta_t * (sig[-1] ** 2)
            if var_t < M:
                floored.append(t_i)
                var_t = M

            a, b, c = delta_t, sig[-1] * delta_t, M - var_t
            sig.append((-b + np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a))
            vol.append(np.sqrt(var_t / t_i))
            var_tm1 = var_t

        return vol, floored

    @staticmethod
    def carried_vol(atm_vol, expiry):
        """The same walk on a tape - a derivative carrier, never a value.

        It rides in as the splice `integrated_vol + (carried - carried.detach())`: exactly zero in
        the forward pass, derivative one, so the shipped curve is the numpy walk's bit for bit.

        The two walks stay separate because they do not agree to the bit: `torch.sqrt` is one ulp
        below `np.sqrt` on better than one float64 in a hundred here and re-associates the
        expression tree, which moved the shipped vols on 24% of 4000 random ATM columns.

        The discriminant is guarded, and only here. One repair leaves `sigma` exactly zero, so a
        second reaches a discriminant of exactly zero, where `sqrt` has an infinite derivative: the
        backward pass multiplies it by the zero `d(b^2)/db` and NaNs the whole Jacobian. The root
        there is zero, so it is written as zero and `sqrt` never sees the point.
        """
        third = np.diff(np.append(0.0, expiry)) / 3.0
        variance = atm_vol * atm_vol * atm_vol.new_tensor(expiry)
        sigma, curve, previous = atm_vol[0], [atm_vol[0]], variance[0]

        for i in range(1, expiry.size):
            floor = previous + third[i] * sigma * sigma
            previous = floor if variance[i] < floor else variance[i]
            b = sigma * third[i]
            disc = b * b - 4.0 * third[i] * (floor - previous)
            real = disc > 0
            root = torch.where(real, torch.sqrt(torch.where(real, disc, torch.ones_like(disc))),
                               torch.zeros_like(disc))
            sigma = (-b + root) / (2.0 * third[i])
            curve.append(torch.sqrt(previous / expiry[i]))

        return torch.stack(curve)

    def bootstrap(self, sys_params, price_models, price_factors, factor_interp, market_prices, calendars, debug=None):
        '''
        Turns the ATM column of the named vol surface into the integrated vol curve the risk neutral
        process reads, repairing any declining variance on the way - see `integrated_vol`.

        `Quote_Sensitivity` leaves that curve behind still connected to its ATM vols, so
        `Calculation.factor_leaf` can offer the connected tensor rather than minting a `Vol` leaf out
        of numpy. The map is explicit - no solve, no implicit function theorem - and the tape is a
        SPLICE over the shipped walk, so every number written comes out of `integrated_vol`.
        '''
        for market_price, implied_params in market_prices.items():
            rate = utils.check_rate_name(market_price)
            market_factor = utils.Factor(rate[0], rate[1:])

            if market_factor.type == self.market_factor_type:
                # get the vol surface
                vol_factor = resolve_factor(implied_params['instrument']['Asset_Price_Volatility'],
                                            price_factors, self.factor_types['Asset_Price_Volatility'])
                implied_param = vol_factor.name
                # whether the thing being MODELLED is an fx rate - the model is named after the
                # underlying, while the surface's tag names its own asset class
                is_fx = utils.check_tuple_name(utils.Factor('FxRate', rate[1:])) in price_factors

                # this shouldn't fail - if it does, need to log it and move on
                try:
                    vol_surface = riskfactors.construct_factor(vol_factor, price_factors, factor_interp)
                except Exception:
                    logging.error('Unable to bootstrap {0} - skipping'.format(market_price), exc_info=True)
                    continue

                connect = implied_params['instrument'].get('Quote_Sensitivity', 'No') == 'Yes'
                atm_vol, source = self.atm_column(
                    vol_factor, vol_surface, market_prices, price_factors)
                curve, floored = self.integrated_vol(atm_vol, vol_surface.expiry)

                # store the output
                price_param = utils.Factor(self.__class__.__name__, market_factor.name)
                model_param = utils.Factor('GBMAssetPriceTSModelImplied', market_factor.name)
                vol = utils.Curve(['Integrated'], list(zip(vol_surface.expiry, curve)))

                if is_fx:
                    quanto_fx_corr = 0.0
                else:
                    quanto_fx_corr = price_factors.get(
                        'Correlation.EquityPrice.{}.{}/FxRate.{}.{}'.format(
                            rate[-1], implied_param[-1], *sorted([sys_params['Base_Currency'], implied_param[-1]])),
                        {'Value': 0.0})['Value']

                price_factors[utils.check_tuple_name(price_param)] = {
                    'Property_Aliases': None,
                    'Vol': vol,
                    'Quanto_FX_Volatility': None,
                    'Quanto_FX_Correlation': quanto_fx_corr}
                price_models[utils.check_tuple_name(model_param)] = {'Risk_Premium': None}

                if connect:
                    quotes = torch.tensor(atm_vol, dtype=self.dtype, requires_grad=True)
                    carried = self.carried_vol(quotes, vol_surface.expiry)
                    self.calibrated[utils.Factor(
                        price_param.type, price_param.name + ('Vol',))] = torch.tensor(
                        curve, dtype=self.dtype) + (carried - carried.detach())
                    self.quote_leaves[market_price] = (
                        ['ATM {:g}'.format(expiry) for expiry in vol_surface.expiry], quotes)

                logging.info('{} built from {} ATM vols off {}'.format(
                    utils.check_tuple_name(price_param), len(atm_vol), source))
                if floored:
                    logging.warning('Fixed declining variance for {} at {}'.format(
                        market_price, ', '.join('{:g}'.format(expiry) for expiry in floored)))


class swaption_objective_class(namedtuple('swaption_objective', 'loss reduce reprice')):
    """What a risk-neutral swaption calibration minimises, as one record: the residual closure, the
    scalar the optimizer chain compares two candidates with, and the estimator that audits it.

    `loss(implied_var)` is `(model value per benchmark, residual per benchmark)`. The model value is
    a PREMIUM under either objective, so `SwaptionCalibration.solve`'s log is one thing; the residual
    is the objective's own and the two are not the same shape.

    `reduce(residuals)` exists because the two stages do not minimise the same function on the Monte
    Carlo path: `market_swap_class.error` returns a residual that is already a square, so basin
    hopping's `sum(r)` is the sum of squared pricing errors while `least_squares` sees a quartic. On
    the analytic path the residual is plain and `sum(r^2)` is both stages' objective. It takes a
    numpy array or a torch tensor.

    `reprice` is the Monte Carlo closure on a block that solved ANALYTICALLY, `None` otherwise - see
    `SwaptionCalibration.honesty_reprice`.

    Both objectives carry a quote side, the same leaf spliced onto two different residuals: the
    Monte Carlo one is already squared, so the dropped Gauss-Newton terms are each half what they
    correct and cancel ([Quote Sensitivities](quote_sensitivities.md#the-dropped-term)); the
    analytic one is separable, so its cross term is structurally zero.
    """


class SwaptionCalibration(object):
    """One risk-neutral swaption calibration as an operand: the residual, and the solve over it.

    The residual is what `calc_loss_on_ir_curve` builds - one weighted error per
    `Instrument_Definitions` row, per the block's `Objective` - and the solve is the optimizer chain
    `calc_loss` hands over. Holding both beside the parameter dict they share lets
    `LeastSquaresSolve` run the ordinary solve forward and differentiate the same residual backward.

    The parameter vector is FLAT here and a dict everywhere else: scipy takes a vector, the process
    takes `{name: tensor}`, and the two scipy adapters own that boundary with `tn_var.data = ...`.
    `__call__` is the third crossing and the only differentiable one - it builds VIEWS of a flat
    tensor rather than writing `.data`, which would sever the edge the theorem needs.
    """

    def __init__(self, name, objective, implied_var, optimizers, process, market_swaps):
        self.name = name
        self.objective = objective
        self.implied_var = implied_var
        self.optimizers = optimizers
        self.process = process
        self.market_swaps = market_swaps
        self.keys = list(implied_var)
        self.sizes = [implied_var[key].numel() for key in self.keys]
        #: the fitted box, which `interior` reads its KKT active set off. A chain with no
        #: least-squares stage declares none, and an unbounded box holds nothing.
        box = next((optim[4] for optim in optimizers or () if optim[0] == 'leastsq'), None)
        self.edges = (np.array(box, dtype=float) if box is not None
                      else np.tile([[-np.inf], [np.inf]], sum(self.sizes)))

    @property
    def quotes(self):
        """The quote leaf per benchmark, or `()` where the block asked for no `Quote_Sensitivity` -
        which is what makes the wrapper a pass-through with no edge recorded."""
        return tuple(swap.quote for swap in self.market_swaps.values() if swap.quote is not None)

    @property
    def descriptors(self):
        """The benchmark names of `quotes`, in its order - what `quote_leaves` pairs them with."""
        return [name for name, swap in self.market_swaps.items() if swap.quote is not None]

    @property
    def labels(self):
        """One name per coordinate of the flat vector, so a coordinate the box holds is named."""
        return ['{}[{}]'.format(key, i) if size > 1 else key
                for key, size in zip(self.keys, self.sizes) for i in range(size)]

    def interior(self, x, g):
        """The free coordinates at `(x, g)` - what the KKT active set does not hold on a bound."""
        return np.flatnonzero(~active_set(x, self.edges[0], self.edges[1], g)).tolist()

    def split(self, theta):
        """`{name: tensor}` in the closure's own parameter order, sharing theta's graph.

        The one place the flat vector is taken apart, so a factor leaf cannot be handed the wrong
        slice of the vector the Jacobian was read off.
        """
        return dict(zip(self.keys, theta.split(self.sizes)))

    def unflatten(self, theta):
        """`{name: numpy}` in the closure's own parameter order - the shape `save_params` takes."""
        return {name: value.detach().cpu().numpy() for name, value in self.split(theta).items()}

    def __call__(self, x):
        """The residual vector at flat parameters `x`, differentiable in `x` and in the quotes.

        A fresh dict of views rather than the standing `implied_var`, whose `.data` the scipy
        adapters overwrite. `x` carries the closure's own precision, so the residual is the number
        the solve stopped on - the float64 promotion belongs to the linear algebra downstream.
        """
        return torch.stack(list(self.objective.loss(self.split(x))[1].values()))

    def honesty_reprice(self, theta):
        """What the engine's own estimator makes of an analytically-solved theta*, or `None`.

        The analytic objective fits Schrager-Pelsser normal vols, which freeze the annuity's
        weights, so such a block has never been asked what the Monte Carlo the rest of the library
        prices with makes of the answer. One pass at theta*, at the block's own path count, reports
        the worst benchmark's relative PREMIUM residual by name - the premium gap being what a mark
        moves by, and what carries the simulation's own numeraire error. No tolerance, no move.
        """
        if self.objective.reprice is None:
            return None
        # `.data` on a CLONE, not on the view: a leaf aliasing theta's storage would move with
        # anything that later wrote through theta
        for name, value in self.split(theta).items():
            self.implied_var[name].data = value.detach().clone()
        prices, _ = self.objective.reprice(self.implied_var)
        errors = {name: float(value.detach()) / self.market_swaps[name].price - 1.0
                  for name, value in prices.items()}
        worst = max(errors, key=lambda name: abs(errors[name]))
        return worst, errors[worst]

    def solve(self):
        """theta* as a flat tensor: the optimizer chain, run exactly as a bootstrap runs it.

        Basin hopping then least squares, `x0` chained from one to the next, and a candidate is
        accepted only if it beats the running best and the process it implies is well posed - so the
        answer can be the seed, which is what `LeastSquaresSolve` checks stationarity for.

        The acceptance test compares one scalar across the seed and both stages, so that scalar is
        `objective.reduce` rather than a `sum` spelled three times. Which coordinates the box holds
        is `interior`'s reading at theta* and not a stage's report.
        """
        calibrated_swaptions, errors = self.objective.loss(self.implied_var)
        batch_loss = self.objective.reduce(
            torch.stack(list(errors.values()))).cpu().detach().numpy()
        vars = {k: v.cpu().detach().numpy() for k, v in self.implied_var.items()}
        # initialize the soln with the current values
        soln = (batch_loss, vars)
        logging.info('{} - Batch loss {}'.format(self.name, batch_loss))
        for k, v in sorted(vars.items()):
            logging.info('{} - {}'.format(k, v))

        for k, v in sorted(calibrated_swaptions.items()):
            value = v.cpu().detach().numpy()
            price = self.market_swaps[k].price
            logging.debug('{},market_value,{:f},sim_model_value,{:f},error,{:.0f}%'.format(
                k, price, value, 100.0 * (price - value) / price))

        # minimize
        result = None
        num_optimizers = len(self.optimizers)
        for op_loop in range(num_optimizers):
            optim = self.optimizers[op_loop % num_optimizers]
            x0 = result['x'] if result is not None else optim[1]
            if optim[0] == 'basin':
                result = scipy.optimize.basinhopping(
                    optim[2], x0=x0, take_step=optim[3], accept_test=optim[4], T=5.0, niter=50,
                    minimizer_kwargs={"method": "L-BFGS-B", "jac": True, "bounds": optim[5]},
                    rng=optim[6])
                batch_loss = float(optim[2](result['x'])[0])
            elif optim[0] == 'leastsq':
                result = scipy.optimize.least_squares(
                    optim[2], x0=x0, jac=optim[3], bounds=optim[4])
                batch_loss = self.objective.reduce(optim[2](result['x']))

            if batch_loss < soln[0] and self.process.params_ok:
                sim_swaptions, errors = self.objective.loss(self.implied_var)
                vars = {k: v.cpu().detach().numpy() for k, v in self.implied_var.items()}
                soln = (batch_loss, vars)
                logging.info('{} - run {} - Batch loss {}'.format(self.name, op_loop, batch_loss))
                for k, v in sorted(vars.items()):
                    logging.info('{} - {}'.format(k, v))
                for k, v in sim_swaptions.items():
                    value = v.cpu().detach().numpy()
                    price = self.market_swaps[k].price
                    logging.info('{},market_value,{:f},sim_model_value,{:f},error,{:.0f}%'.format(
                        k, price, value, 100.0 * (price - value) / price))

        theta = torch.tensor(np.concatenate([soln[1][key] for key in self.keys]),
                             dtype=self.implied_var[self.keys[0]].dtype,
                             device=self.implied_var[self.keys[0]].device)
        # ONE evaluation, to say which bounds bind - a calibration statement, not a sensitivity
        x = theta.detach().requires_grad_(True)
        residual = self(x)
        jacobian = torch.autograd.grad(
            residual, x, torch.eye(residual.numel(), dtype=x.dtype, device=x.device),
            is_grads_batched=True)[0].double()
        for line in active_bounds(self.labels, theta.detach().cpu().numpy(), self.edges[0],
                                  self.edges[1],
                                  (jacobian.t() @ residual.detach().double()).cpu().numpy()):
            logging.info('{} - {}'.format(self.name, line))
        return theta


class LeastSquaresSolve(torch.autograd.Function):
    """A least-squares calibration as one differentiable node: quotes in, calibrated parameters out.

    The operand is the calibration - `solve()` for theta*, `__call__(x)` for the residual at it,
    `interior(x, g)` for the coordinates the box leaves free, `labels` to name them - so the two
    families differ in the operand and in nothing here.

    FORWARD IS THE ORDINARY SOLVE, re-enabling the grad mode autograd turns off around it, so
    asking for quote gradients cannot move theta*.

    BACKWARD IS THE IMPLICIT FUNCTION THEOREM at a stationarity point rather than a root: what is
    held fixed is `g = J^T r = 0`, and dropping the term in `d(J^T)/dtheta . r` - Gauss-Newton,
    second order on these residuals - leaves `(J^T J) dtheta/dq = -J^T dr/dq`, a cotangent `v`
    contracting as `dL/dq = -(dr/dq)^T J (J^T J)^+ v`. THE BOX IS PART OF THAT FIXED POINT: the
    condition at theta* is KKT, so the contraction is taken over `interior` alone and a coordinate
    the box holds has `dtheta/dq` zero. It runs on the COLUMN-SCALED Jacobian `J/||J_:,j||` - the
    `x_scale='jac'` matrix the solve steps on - at `Jacobian_Rcond` on ITS singular values, so one
    number cuts one matrix here and in the identification table, and `J (J^T J)^+` being `(J^+)^T`
    one `pinv` says both.

    Every quote delta is logged beside the share of it lying in the null space `Jacobian_Rcond`
    discards - minimum-norm being a CONVENTION there, and one in the scaled metric.

    Both Jacobians come from ONE fresh evaluation at `(theta*, q)` through `autograd.grad` and not
    off `.grad`, which accumulates across the optimizer's evaluations and would be a path sum; they
    are MEMOED, theta* not moving between cotangents. Every `grad` retains the graph, for the
    reason `CalibrationSolve` gives.

    REFUSED: `create_graph`, the backward carrying no second derivative; and `||J^T r||` over the
    free coordinates above `Stationarity_Tol`, `solve` being free to return the seed.
    """

    @staticmethod
    def forward(ctx, calibration, rcond, stationarity, *quotes):
        with torch.enable_grad():
            theta = calibration.solve()
        ctx.calibration, ctx.theta, ctx.quotes = calibration, theta, quotes
        ctx.rcond, ctx.stationarity, ctx.memo = rcond, stationarity, None
        return theta

    @staticmethod
    def backward(ctx, cotangent):
        # grad mode here means `create_graph` - a second differentiation Gauss-Newton cannot give
        if torch.is_grad_enabled():
            raise Exception('Calibration: create_graph is not supported - the backward is a '
                            'Gauss-Newton contraction and carries no second derivative')
        calibration = ctx.calibration
        with torch.enable_grad():
            if ctx.memo is None:
                x = ctx.theta.detach().requires_grad_(True)
                residual = calibration(x)
                eye = torch.eye(residual.numel(), dtype=x.dtype, device=x.device)
                jacobian = torch.autograd.grad(residual, x, eye, is_grads_batched=True,
                                               retain_graph=True)[0].double()
                slope = jacobian.t() @ residual.detach().double()
                free = calibration.interior(x.detach().cpu().numpy(), slope.cpu().numpy())
                inner = jacobian[:, free]
                gradient = float(slope[free].norm())
                held = [name for i, name in enumerate(calibration.labels) if i not in set(free)]
                logging.info(
                    '  quote sensitivity: ||J^T r|| {:.3e} against ||r|| {:.3e} over {} rows and '
                    '{} fitted coordinates{} - the Gauss-Newton contraction is exact where the '
                    'first is zero'.format(
                        gradient, float(residual.detach().norm()), residual.numel(), x.numel(),
                        '' if not held else ', {} of them HELD by the box ({}), whose quote '
                        'derivative is zero'.format(len(held), ', '.join(held))))
                if gradient > ctx.stationarity:
                    raise Exception(
                        'Calibration: theta* is not stationary - ||J^T r|| is {:.6g} against a '
                        'Stationarity_Tol of {:.6g}, so the implicit function theorem does not '
                        'hold there'.format(gradient, ctx.stationarity))
                scaled, norms = column_scale(inner)
                columns = torch.cat([column.reshape(residual.numel(), -1)
                                     for column in torch.autograd.grad(
                                         residual, ctx.quotes, eye, is_grads_batched=True,
                                         retain_graph=True)], 1).double()
                pseudo = torch.linalg.pinv(scaled, rtol=ctx.rcond)
                ctx.memo = (residual, free, norms, pseudo.t(),
                            -(pseudo @ columns) / norms[:, None],
                            null_basis(scaled, norms, ctx.rcond))
            residual, free, norms, contraction, delta, null = ctx.memo
            v = cotangent.double()[free]
            projected = null @ (null.t() @ delta)
            logging.info(
                '  quote deltas, minimum-norm in the COLUMN-SCALED metric over a {}-dimensional '
                'null space - the share lying in it is that convention and not identified:'.format(
                    null.shape[1]))
            for j, descriptor in enumerate(calibration.descriptors):
                value = float(v @ delta[:, j])
                logging.info(
                    '    {}: dV/dq {:+.6g}, direction share {:.3f}, value share {}'.format(
                        descriptor, value,
                        float(projected[:, j].norm() / delta[:, j].norm()),
                        'n/a' if value == 0.0 else
                        '{:+.3f}'.format(float(v @ projected[:, j]) / value)))
            grads = torch.autograd.grad(
                residual, ctx.quotes, retain_graph=True,
                grad_outputs=-(contraction @ (v / norms)).to(residual.dtype))
        return (None, None, None) + grads


class RiskNeutralInterestRateModel(object):
    def __init__(self, param, device, dtype):
        self.param = param
        self.device = device
        self.prec = dtype
        #: The Monte Carlo sample shape of the last block built - a REPORT. Nothing prices off
        #: these: the residual closure captures its own shape as locals, because one bootstrapper
        #: runs every curve and a closure reaching through `self` would take the next block's count.
        self.batch_size = None
        self.num_batches = None
        #: What `Quote_Sensitivity` leaves behind: theta* still connected to its quotes, one entry
        #: per named model parameter, plus the quote leaf per block. `Config.bootstrap` harvests
        #: both - tensors cannot live in `Price Factors`.
        self.calibrated = {}
        self.quote_leaves = {}

    def calc_loss_on_ir_curve(self, implied_params, base_date, time_grid, process,
                              implied_obj, ir_factor, vol_surface, resid=lambda x: x * x, jac=False):
        """The swaption calibration's residual closure: implied parameters in, one weighted error
        per benchmark out.

        TWO OBJECTIVES, ONE DECLARED SWITCH, `Analytic` the default. It prices each benchmark with
        `stochasticprocess.HullWhite2FactorImpliedInterestRateModel.schrager_pelsser_swaption` and
        differences NORMAL VOLS, plain (`market_swap_class.normal_vol_error`). `Monte_Carlo` prices
        each through the engine's own `pv_float_cashflow_list` and differences the weighted relative
        pricing error, ALREADY SQUARED.

        Why the closed form is the default: it sits inside one Monte Carlo evaluation's own noise at
        22 of 25 benchmarks (the simulation's numeraire bias -0.35% to -1.61% exceeds the freezing
        bias -0.13 to +2.17bp), its residual is quadratic rather than quartic so `||J'r||` at theta*
        is 8.63e-7 against 3.16e2, it is deterministic in the sample, and it is 4.6x faster on the
        four-quote block. `Monte_Carlo` keeps being the engine's OWN estimator.

        The Monte Carlo closure is built either way: on an analytic block it is the auditor, and
        `SwaptionCalibration.honesty_reprice` runs it once at theta*.

        Both objectives price under the DOMESTIC measure, arranged upstream: `process` arrives built
        on a quanto-suppressed implied object, so `precalculate` assembles `K = 0`. See
        `implied_process` for the Girsanov argument.

        COMMON RANDOM NUMBERS ARE FROZEN PER SOLVE - the Sobol engine is built once and `reset`
        re-seeds nothing once `t_random_batch` exists, so the optimizer differences the parameters
        rather than the sample. `clear` is the memo half alone, all the analytic path needs. The
        sample shape is frozen as LOCALS: this residual outlives its block, `LeastSquaresSolve`
        holding it for a backward that runs after the loop.

        The batch loop clears `t_Buffer` and not `t_PreCalc`. `calc_time_grid_curve_rate` keys on
        the curve code and time grid rather than the batch, so clearing only outside the loop makes
        every later batch re-read batch zero's curve. `t_PreCalc` holds integrals in theta rather
        than in the sample, so clearing it per batch would re-integrate the same numbers.

        THE QUOTE SIDE severs at the market price and nowhere else, on either objective: `swap.price`
        is a numpy scalar, and the splice that closes it sits on `market_swap_class.error` or on
        `market_swap_class.market_normal_vol`. One quote leaf per benchmark serves both. Two
        severances stay open deliberately, their upstream being the calibrated curve rather than a
        quote of THIS calibration: `get_par_swap_rate` and `set_fixed_amount`.
        """
        block = implied_params['instrument']
        objective = block.get('Objective', 'Analytic')
        quote_sensitivity = block.get('Quote_Sensitivity', 'No')
        if objective not in ('Monte_Carlo', 'Analytic'):
            raise Exception(
                "Swaption calibration: Objective '{}' is not one this family prices - it is "
                "'Analytic' (the default: the Schrager-Pelsser normal vols) or 'Monte_Carlo' "
                "(every benchmark through the engine's own Monte Carlo). Correct the block's "
                "Objective to one of those two".format(objective))
        # the closures below capture THESE locals, not the attributes they are mirrored onto
        batch_size = int(block.get('Simulations', 8192))
        num_batches = int(block.get('Batches', 1))
        self.batch_size, self.num_batches = batch_size, num_batches

        def loss(implied_var):
            # first, reset the shared_mem
            shared_mem.reset(num_batches, numfactors, time_grid)
            # now set up the calc
            process.precalculate(base_date, time_grid, stoch_var, shared_mem, 0, implied_tensor=implied_var)
            tensor_swaptions = {}
            # needed to interpolate the zero curve
            delta_scen_t = np.diff(time_grid.scen_time_grid).reshape(-1, 1)

            for batch_index in range(num_batches):
                # the curve memo is keyed by curve and time and not by batch; `t_PreCalc` stays
                shared_mem.t_Buffer.clear()
                # load up the batch
                shared_mem.batch_index = batch_index
                # simulate the price factor - only need the full curve at the mtm time points
                shared_mem.t_Scenario_Buffer = process.generate(shared_mem)
                # get the discount factors
                Dfs = utils.calc_time_grid_curve_rate(
                    curve_index_reduced, time_grid.calc_time_grid(time_grid.scen_time_grid[:-1]),
                    shared_mem)
                # get the index in the deflation factor just prior to the given grid
                deflation = Dfs.reduce_deflate(delta_scen_t, time_grid.mtm_time_grid, shared_mem)
                # go over the instrument definitions and build the calibration
                for swaption_name, market_data in market_swaps.items():
                    expiry = market_data.deal_data.Time_dep.mtm_time_grid[
                        market_data.deal_data.Time_dep.deal_time_grid[0]]
                    DtT = deflation[expiry]
                    par_swap = pricing.pv_float_cashflow_list(
                        shared_mem, time_grid, market_data.deal_data,
                        pricing.pricer_float_cashflows, settle_cash=False)
                    sum_swaption = torch.sum(torch.relu(DtT * par_swap))
                    if swaption_name in tensor_swaptions:
                        tensor_swaptions[swaption_name] += sum_swaption
                    else:
                        tensor_swaptions[swaption_name] = sum_swaption

            calibrated_swaptions = {k: v / (batch_size * num_batches) for k, v in tensor_swaptions.items()}
            errors = {k: swap.error(calibrated_swaptions[k], resid)
                      for k, swap in market_swaps.items()}
            return calibrated_swaptions, errors

        def analytic_loss(implied_var):
            """The same benchmarks, priced by Schrager-Pelsser, differenced as normal vols.

            `precalculate` builds H, I and J and is the only thing this runs - no sample, no
            simulated curve - so the two objectives share their whole front half: the same
            reversion-speed floors, series branches, `params_ok` and `Correlation` leaves.

            `clear` and not `reset`: the memo tables go per evaluation, the Sobol draw is not paid.
            """
            shared_mem.clear()
            process.precalculate(
                base_date, time_grid, stoch_var, shared_mem, 0, implied_tensor=implied_var)
            swaptions = {name: process.schrager_pelsser_swaption(
                market_data.schedule.expiry, market_data.schedule.pay_times,
                market_data.schedule.accruals)
                for name, market_data in market_swaps.items()}
            return ({name: swaption.premium for name, swaption in swaptions.items()},
                    {name: market_swaps[name].normal_vol_error(swaption)
                     for name, swaption in swaptions.items()})

        # set up the stochastic factors
        stochastic_factors = {ir_factor: process}
        # calculate a reverse lookup for the tenors and store the daycount code
        all_tenors = utils.update_tenors(base_date, stochastic_factors)
        # calculate the curve indices
        index_keys = {'full': utils.Factor(ir_factor.type, ir_factor.name + ('full',)),
                      'reduced': utils.Factor(ir_factor.type, ir_factor.name + ('reduced',))}
        # calculate the tenor curve index
        c_index = instruments.calc_factor_index(ir_factor, {}, stochastic_factors, all_tenors)
        # now edit the curve indices with the correct names - one reduced, one full
        curve_index = [(c_index[utils.FACTOR_INDEX_Stoch], index_keys['full']) + c_index[2:]]
        curve_index_reduced = [(c_index[utils.FACTOR_INDEX_Stoch], index_keys['reduced']) + c_index[2:]]
        # set up a common context - we leave out the random numbers and pass it in explicitly below
        shared_mem = RiskNeutralInterestRate_State(index_keys, batch_size, self.device, self.prec)
        # the unit tensor switches the quote side on and puts its leaves on the right device
        market_swaps, benchmarks = create_market_swaps(
            base_date, time_grid, curve_index, vol_surface, process.factor,
            block['Instrument_Definitions'], ir_factor.name,
            shared_mem.one if quote_sensitivity == 'Yes' else None)
        # number of random factors to use
        numfactors = process.num_factors()
        # compiled here rather than by a DealStructure, so they bind here
        for market_data in market_swaps.values():
            utils.bind_schedules(market_data.deal_data.Factor_dep, shared_mem.one)
        # set up the variables
        implied_var = {}
        stoch_var = torch.tensor(
            process.factor.current_value(), device=self.device, dtype=self.prec, requires_grad=jac)

        for param_name, param_value in implied_obj.current_value(include_quanto=jac).items():
            implied_var[param_name] = torch.tensor(
                param_value, dtype=self.prec, device=self.device, requires_grad=True)

        # `reduce` squares on the analytic path because the residual does not; `reprice` is the
        # Monte Carlo standing by as auditor. The switch is read once, here.
        chosen = swaption_objective_class(
            loss=analytic_loss, reduce=lambda r: (r * r).sum(), reprice=loss
        ) if objective == 'Analytic' else swaption_objective_class(
            loss=loss, reduce=lambda r: r.sum(), reprice=None)

        if jac:
            return stoch_var, implied_var, chosen.loss
        else:
            return implied_var, chosen, market_swaps, benchmarks

    def bootstrap(self, sys_params, price_models, price_factors, factor_interp, market_prices, calendars, debug=None):
        base_date = sys_params['Base_Date']
        base_currency = sys_params['Base_Currency']
        master_curve_list = sys_params.get('Master_Curves')

        if sys_params.get('Swaption_Premiums') is not None:
            swaption_premiums = pd.read_csv(sys_params['Swaption_Premiums'], index_col=0)
            ATM_Premiums = swaption_premiums[swaption_premiums['Strike'] == 'ATM']
        else:
            ATM_Premiums = None

        for market_price, implied_params in market_prices.items():
            rate = utils.check_rate_name(market_price)
            market_factor = utils.Factor(rate[0], rate[1:])
            if market_factor.type == self.market_factor_type:
                # fetch the factors
                ir_factor = utils.Factor('InterestRate', rate[1:])
                vol_factor = utils.Factor('InterestYieldVol', utils.check_rate_name(
                    implied_params['instrument']['Swaption_Volatility']))

                # this shouldn't fail - if it does, need to log it and move on
                try:
                    swaptionvol = riskfactors.construct_factor(vol_factor, price_factors, factor_interp)
                    swaptionvol.delta = sys_params.get('Volatility_Delta', 0.0)
                    ir_curve = riskfactors.construct_factor(ir_factor, price_factors, factor_interp)
                    swaptionvol.set_premiums(ATM_Premiums, ir_curve.get_currency())
                except KeyError as k:
                    logging.warning('Missing price factor {} - Unable to bootstrap {}'.format(k.args, market_price))
                    continue
                except Exception:
                    logging.error('Unable to bootstrap {0} - skipping'.format(market_price), exc_info=True)
                    continue

                if master_curve_list and master_curve_list.get(ir_curve.get_currency()[0]) != rate[1]:
                    logging.warning('curve is not Risk Free {} - skipping and will reassign later'.format(market_price))
                    continue

                # set of dates for the calibration
                mtm_dates = set(
                    [base_date + x['Start'] for x in implied_params['instrument']['Instrument_Definitions']])

                # grab the implied process
                implied_obj, process, vol_tenors = self.implied_process(
                    base_currency, price_factors, price_models, ir_curve, rate,
                    vol_tenors=self.sigma_knots(implied_params['instrument'].get('Sigma_Knots')))

                # set up the time grid
                time_grid = utils.TimeGrid(mtm_dates, mtm_dates, mtm_dates)
                # add a delta of 10 days to the time_grid_years (without changing the scenario grid
                # this is needed for stochastically deflating the exposure later on
                time_grid.set_base_date(base_date, delta=(10, vol_tenors * utils.DAYS_IN_YEAR))

                # calculate the error
                objective, optimizers, implied_var, market_swaptions, benchmarks = self.calc_loss(
                    implied_params, base_date, time_grid, process, implied_obj, ir_factor, swaptionvol)

                if debug is not None:
                    debug.deals['Deals']['Children'] = [{'instrument': x} for x in benchmarks]
                    try:
                        debug.write_trade_file(market_factor.name[0] + '.aap')
                    except Exception:
                        logging.error('Could not write output file {}'.format(market_factor.name[0] + '.aap'))

                # check the time
                time_now = time.monotonic()
                calibration = SwaptionCalibration(
                    market_factor.name[0], objective, implied_var, optimizers, process,
                    market_swaptions)
                # through the implicit-function wrapper either way: with no quotes on the tape no
                # edge is recorded and the wrapper is a pass-through
                theta = LeastSquaresSolve.apply(
                    calibration,
                    float(declared_defaults(type(self),
                                            implied_params['instrument'])['Jacobian_Rcond']),
                    float(declared_defaults(type(self),
                                            implied_params['instrument'])['Stationarity_Tol']),
                    *calibration.quotes)

                # reported by name rather than checked against a tolerance - see `honesty_reprice`
                reprice = calibration.honesty_reprice(theta)
                if reprice is not None:
                    logging.info(
                        '{} - Analytic objective - at theta* the engine\'s own Monte Carlo prices '
                        'its worst benchmark, {}, {:+.2f}% away from market'.format(
                            market_factor.name[0], reprice[0], 100.0 * reprice[1]))

                # save this - `unflatten` detaches, so `Price Factors` gets plain numpy
                self.save_params(calibration.unflatten(theta), price_factors, implied_obj, rate)

                # the connected half: one entry per named parameter, under the key
                # `_build_factor_state` mints its leaf with
                if calibration.quotes:
                    params_factor = utils.Factor(self.__class__.__name__, rate[1:])
                    self.calibrated.update({
                        utils.Factor(params_factor.type, params_factor.name + (name,)): value
                        for name, value in calibration.split(theta).items()})
                    # the chain called backward() per evaluation, so a `.grad` standing here is the
                    # sum over its whole path - the leaf is handed over clean
                    for quote in calibration.quotes:
                        quote.grad = None
                    self.quote_leaves[market_price] = (calibration.descriptors, calibration.quotes)

                # record the time
                logging.info('This took {} seconds.'.format(time.monotonic() - time_now))


class HullWhite2FactorModelParameters(RiskNeutralInterestRateModel):
    documentation = (
        'Interest Rates',
        ['A set of parameters $\\sigma_1, \\sigma_2, \\alpha_1, \\alpha_2, \\rho$ are estimated from ATM',
         'swaption volatilities. Swaption volatilities are preferred to caplets to better estimate $\\rho$.'
         'Although assuming that $\\sigma_1, \\sigma_2$ are constant makes the calibration of this model',
         'considerably easier, in general, $\\sigma_1, \\sigma_2$ should be allowed a piecewise linear term',
         'structure dependent on the underlying swaptions.',
         '',
         'For a set of $J$ ATM swaptions, we need to minimize:',
         '',
         '$$E=\\sum_{j\\in J} \\omega_j (V_j(\\sigma_1, \\sigma_2, \\alpha_1, \\alpha_2, \\rho)-V_j)^2$$',
         '',
         'Where $V_j(\\sigma_1, \\sigma_2, \\alpha_1, \\alpha_2, \\rho)$ is the price of the $j^{th}$ swaption',
         'under the model, $V_j$ is the market value of the $j^{th}$ swaption and $ \\omega_j$ is the corresponding',
         'weight. The market value is calculated using the standard pricing functions',
         '',
         'To find a good minimum of the model value, basin hopping as implemented [here](https://docs.scipy.org/doc\
/scipy/reference/generated/scipy.optimize.basinhopping.html) as well as',
         'least squares [optimization](https://docs.scipy.org/doc/scipy/reference/generated/\
scipy.optimize.leastsq.html) are used.',
         '',
         'The error $E$ is algorithmically differentiated and then solved via brute-force monte carlo',
         'using tensorflow and scipy.',
         '',
         'If the currency of the interest rate is not the same as the base currency, then a quanto correction needs',
         'to be made. Assume $C$ is the value of the interest rate/FX correlation price factor (can be estimated from',
         'historical data), then the FX rate follows:',
         '',
         '$$d(log X)(t)=(r_0(t)-r(t)-\\frac{1}{2}v(t)^2)dt+v(t)dW(t)$$',
         '',
         'with $r(t)$ the short rate and $r_0(t)$ the short rate in base currency. The short rate with a quanto',
         'correction is:',
         '',
         '$$dr(t)=r_T(0,t)dt+\\sum_{i=1}^2 (\\theta_i(t)-\\alpha_i x_i(t)- \\bar\\rho_i\\sigma_i v(t))dt+\\sigma_i dW_i(t)$$',
         '',
         'where $W_1(t),W_2(t)$ and $W(t)$ are standard Wiener processes under the rate currency\'s risk neutral measure',
         'and $r_T(t,T)$ is the partial derivative of the instantaneous forward rate r(t,T) with respect to the maturity ',
         'date $T$.'
         '',
         'Define:',
         '',
         '$$F(u,v)=\\frac{\\sigma_1u+\\sigma_2v}{\\sqrt{\\sigma_1^2+\\sigma_2^2+2\\rho\\sigma_1\\sigma_2}}$$',
         '',
         'Then $\\bar\\rho_1, \\bar\\rho_2$ are assigned:',
         '',
         '$$\\bar\\rho_1=F(1,\\rho)C$$',
         '',
         '$$\\bar\\rho_2=F(\\rho,1)C$$',
         '',
         'That correction belongs to the SIMULATION and not to this fit. The market premium being',
         'repriced is $E^{Q_{dom}}[D_{dom}\\cdot\\text{payoff}]$ - struck, deflated and quoted in the',
         'rate currency - so the calibration prices it on domestic-measure paths whatever the base',
         'currency of the job, and $\\bar\\rho_1,\\bar\\rho_2$ are held at zero throughout the solve.',
         'Girsanov moves drifts and leaves quadratic variation alone, so the fitted',
         '$\\sigma_1,\\sigma_2,\\alpha_1,\\alpha_2,\\rho$ are the same numbers under either measure:',
         'they are calibrated domestically and $\\bar\\rho_1,\\bar\\rho_2$ are assembled from them',
         'above and written to the price factor, where a scenario run reads them.',
         ]
    )

    market_factor_type = 'HullWhite2FactorModelPrices'
    fields = [
        F('Swaption_Volatility', 'Text', default=REQUIRED,
          description='The InterestYieldVol surface the benchmark swaptions are priced off'),
        F('Instrument_Definitions', 'Table', default='null', row=Row([
            F('Start', 'Period', description='Forward start, from the base date'),
            F('Tenor', 'Period', description='Swap tenor, from the start'),
            F('Floating_Frequency', 'Period'), F('Fixed_Frequency', 'Period'),
            F('Floating_Day_Count', 'Text',
              values=['ACT_365', 'ACT_360', 'ACT_365_ISDA', '_30_360', '_30E_360', 'ACT_ACT_ICMA']),
            F('Fixed_Day_Count', 'Text',
              values=['ACT_365', 'ACT_360', 'ACT_365_ISDA', '_30_360', '_30E_360', 'ACT_ACT_ICMA']),
            F('Market_Volatility', 'Percent',
              description='The quoted ATM vol, in the convention the surface declares - a '
                          'lognormal Black vol, or an absolute normal one where Distribution_Type '
                          'is Normal. Required: a zero (which used to read the surface\'s own ATM '
                          'instead) and an absent one both refuse by name'),
            F('Weight', 'Float', description='Relative weight in the objective')]),
          description='The forward starting swaps the swaptions are struck on'),
        F('Sigma_Knots', 'Table', default='null', row=Row([F('Tenor', 'Period')]),
          description='The knots of both sigma term structures, as periods from the base date; absent, '
                      'the ten at 0, 1M, 3M, 6M, 1Y, 2Y, 4Y, 6Y, 8Y and 10Y'),
        F('Objective', 'Text', default='Analytic', values=['Monte_Carlo', 'Analytic'],
          description='What the solve minimises. Analytic is the default: it prices every benchmark '
                      'with the Schrager-Pelsser closed form and differences NORMAL VOLS, plain, so '
                      'it is exact in the sample rather than estimated, deterministic at a given '
                      'Random_Seed, quadratic in the pricing error where the other path is quartic '
                      '(||J\'r|| at theta* 8.63e-7 against 3.16e2, inside Stationarity_Tol\'s own '
                      '1e-3 default rather than five orders outside it), differentiable in the '
                      'quotes off a residual that is separable in (theta, q), and 4.6x faster on '
                      'the four-quote block. Monte_Carlo prices every benchmark through the '
                      'engine\'s own paths and differences the squared relative PREMIUM error, '
                      'which is what this family did before the measurement: it remains fully '
                      'supported as the engine\'s own estimator and is the oracle the closed form '
                      'was measured against - the analytic price sits inside one Monte Carlo '
                      'evaluation\'s own noise at 22 of the 25 benchmarks and is the MORE accurate '
                      'of the two over most of that grid, because the simulation\'s numeraire bias '
                      'exceeds Schrager-Pelsser\'s freezing bias almost everywhere. An analytic '
                      'solve ends by repricing theta* through that estimator and logging what it '
                      'makes of it. Quote_Sensitivity works on either, off the same quote leaf - '
                      'what differs is the residual it is spliced onto'),
        F('Simulations', 'Integer', default=8192,
          description='Paths per batch the Monte Carlo objective prices its benchmarks on, from a '
                      'Sobol sample frozen for the whole solve. Ignored by the Analytic objective, '
                      'which draws none - except by its honesty reprice, which prices at this count'),
        F('Batches', 'Integer', default=1,
          description='How many such batches. The sample is Simulations x Batches Sobol points '
                      'drawn once and walked a block at a time, so batches buy PATHS at the cost '
                      'of wall clock while leaving the memory one batch needs where it was: '
                      '(2048 x 4) is the same estimate as (8192 x 1) to one ulp'),
        F('Random_Seed', 'Integer', default=5120,
          description='Seeds the basin-hopping random search - the step taker and the Metropolis '
                      'accept test both draw from it. Without it the search draws from the process '
                      'global and the calibration is a function of whatever ran before it: on the '
                      'gate fixture theta* moves 0.93 absolute between ambient seeds. The Monte '
                      'Carlo paths are a separately frozen Sobol sample and do not move with it'),
        F('Quote_Sensitivity', 'Text', default='No', values=['Yes', 'No'],
          description='Keep each benchmark swaption connected to the quote it was priced off - the '
                      'row\'s Market_Volatility or the premium - so the '
                      'residual differentiates in the quote as well as in the model parameters. '
                      'The splice is worth exactly zero in the forward pass, so the calibrated '
                      'parameters are identical either way. Built on BOTH objectives, off the same '
                      'leaf: Monte_Carlo splices the twin premium onto the squared relative pricing '
                      'error, Analytic inverts it to a normal vol, and the analytic residual is '
                      'separable in (theta, q) so its Gauss-Newton cross term is structurally zero. '
                      'A premium re-struck by Volatility_Delta is refused on either'),
        F('Jacobian_Rcond', 'Float', default=1e-5,
          description='Relative cutoff on the singular values of the COLUMN-SCALED Jacobian '
                      'J/||J_:,j|| the backward pass pseudo-inverts - the x_scale=jac matrix the '
                      'solve itself steps on. J has one row per benchmark and 23 columns, so it is '
                      'rank deficient on every block quoting fewer swaptions than that: below the '
                      'cutoff a direction is one the quotes do not identify and its dtheta/dq is '
                      'the minimum-norm representative in that metric. The default sits in the one '
                      'gap the identified block measures - 2.97e-4 of the largest against 1.43e-6, '
                      'two hundred fold - and keeps 16 of the 23. Only used with Quote_Sensitivity '
                      'Yes'),
        F('Stationarity_Tol', 'Float', default=1e-3,
          description='How far off stationarity theta* may be before the quote Jacobian is refused, '
                      'as the 2-norm of J\'r. The optimizer chain accepts whatever it returned - '
                      'possibly the seed, if nothing beat it - and the implicit function theorem '
                      'holds only where that gradient vanishes, so above this the backward raises '
                      'and names the norm rather than reporting a quietly wrong number. The norm is '
                      'absolute and the objective sets its scale, so the default is the Analytic '
                      'path\'s: that chain reaches 8.63e-7 on the repository\'s identified block '
                      'while the Monte_Carlo one stops at 3.16e2 and has to declare a tolerance of '
                      'its own'),
        F('Generate_Instruments', 'Text', default='No', values=['Yes', 'No'],
          description='Unbuilt: generate the definitions from Generation_Parameters instead'),
        F('Generation_Parameters', 'Container', default={
            'First_Start': '1Y', 'Last_Start': '9Y', 'First_Tenor': '1Y', 'Last_Tenor': '9Y',
            'First_Maturity': '10Y', 'Last_Maturity': '10Y', 'Fixed_Frequency': '6M',
            'Floating_Frequency': '6M', 'Day_Count': 'ACT_365', 'Index_Offset': 0},
          sub_fields=[
            F('First_Start', 'Period', default='1Y'), F('Last_Start', 'Period', default='9Y'),
            F('First_Tenor', 'Period', default='1Y'), F('Last_Tenor', 'Period', default='9Y'),
            F('First_Maturity', 'Period', default='10Y'),
            F('Last_Maturity', 'Period', default='10Y'),
            F('Fixed_Frequency', 'Period', default='6M'),
            F('Floating_Frequency', 'Period', default='6M'),
            F('Day_Count', 'Text', default='ACT_365',
              values=['ACT_365', 'ACT_360', 'ACT_365_ISDA', '_30_360', '_30E_360', 'ACT_ACT_ICMA']),
            F('Index_Offset', 'Integer', default=0)],
          description='Unbuilt: the grid Generate_Instruments would sweep'),
        F('Quote_Timestamp', 'Date', default='',
          description='When the ladder was seen - the terminal\'s own as-of where this block was '
                      'authored off a screen (derivus_bloomberg.swaption_vol). Stored and '
                      'reported; nothing in the fit reads it, because what counts as too old is '
                      'the consumer\'s policy and not the parameters\''),
        F('Quote_Source', 'Text', default='',
          description='How this block was authored, in one line: what the vols were read off, the '
                      'convention they are quoted in, and the surface whose Distribution_Type the '
                      'calibration will actually price them under. Declared so a machine-fetched '
                      'ladder\'s provenance is data on the block rather than an undeclared key '
                      'bootstrap reads past')
    ]

    def __init__(self, param, device, dtype):
        super(HullWhite2FactorModelParameters, self).__init__(param, device, dtype)
        self.sigma_bounds = (1e-5, 0.09)
        self.alpha_bounds = (-0.5, 2.4)
        self.corr_bounds = (-.95, 0.95)

    def calc_loss(self, implied_params, base_date, time_grid, process, implied_obj, ir_factor, vol_surface):

        def split_param(x):
            corr = x[2:3]
            alpha = x[:2]
            sigmas = x[3:]
            return sigmas, alpha, corr

        def make_basin_callbacks(step, sigma_min_max, alpha_min_max, corr_min_max, rng):
            """The two callbacks basin hopping needs, drawing from `rng` and never from the process
            global - off `np.random` the calibration is a function of whatever ran before it in the
            same interpreter (theta* moves 0.93 absolute between ambient seeds on the gate fixture).
            The same generator serves the Metropolis test in `SwaptionCalibration.solve`."""

            def bounds_check(**kwargs):
                x = kwargs["x_new"]
                sigmas, alpha, corr = split_param(x)
                sigma_ok = (sigmas > sigma_min_max[0]).all() and (sigmas < sigma_min_max[1]).all()
                alpha_ok = (alpha > alpha_min_max[0]).all() and (alpha < alpha_min_max[1]).all()
                corre_ok = (corr > corr_min_max[0]).all() and (corr < corr_min_max[1]).all()
                return sigma_ok and alpha_ok and corre_ok and process.params_ok

            def basin_step(x):
                sigmas, alpha, corr = split_param(x)
                # update vars
                sigmas = (sigmas * np.exp(rng.uniform(-step, step, sigmas.size))).clip(*sigma_min_max)
                alpha = (alpha * np.exp(rng.uniform(-step, step, alpha.size))).clip(*alpha_min_max)
                corr = (corr + rng.uniform(-step, step, corr.size)).clip(*corr_min_max)

                return np.concatenate((alpha, corr, sigmas))

            return bounds_check, basin_step

        def make_basin_hopping_loss(objective, implied_vars, device, with_grad=False):
            """The scipy basinhopper's scalar-and-gradient adapter over the residual closure.

            The scalar is `objective.reduce` and not a `sum` written here, because `solve`'s
            acceptance test compares it against the least-squares stage's own number.
            """
            loss_fn = objective.loss

            def basin_hopper(x):
                for tn_var, np_var in zip(implied_vars.values(), np.split(x, split_param)):
                    tn_var.grad = None
                    tn_var.data = torch.from_numpy(np_var).to(device)

                try:
                    _, error = loss_fn(implied_vars)
                except Exception as e:
                    print("Warning x ({}) - {}".format(x, e.args))
                    return 100.0 * sum(len_vars), [100.0 * sum(len_vars)] * sum(len_vars)
                else:
                    total_loss = objective.reduce(torch.stack(list(error.values())))
                    if with_grad:
                        total_loss.backward()
                        grad = torch.cat([x.grad for x in implied_vars.values()]).cpu().detach().numpy()
                        return total_loss.cpu().detach().numpy(), grad
                    else:
                        return total_loss.cpu().detach().numpy()

            len_vars = [len(x) for x in implied_vars.values()]
            split_param = np.cumsum(len_vars[:-1])
            return basin_hopper

        def make_least_squares_loss(loss_fn, implied_vars, device):
            # makes it possible to call the scipy least squares algo
            def calc_loss(x):
                for tn_var, np_var in zip(implied_vars.values(), np.split(x, split_param)):
                    tn_var.grad = None
                    tn_var.data = torch.from_numpy(np_var).to(device)
                _, error = loss_fn(implied_vars)
                return torch.stack(list(error.values()))

            def jacobian(x):
                loss = calc_loss(x)
                # full jacobian - takes a second or so
                jac = torch.stack([torch.cat(torch.autograd.grad(
                    loss, list(implied_vars.values()), x, retain_graph=True))
                    for x in torch.eye(len(loss), device=device)])
                return jac.cpu().numpy()

            def least_squares(x):
                return calc_loss(x).cpu().detach().numpy()

            len_vars = [len(x) for x in implied_vars.values()]
            split_param = np.cumsum(len_vars[:-1])
            return least_squares, jacobian

        # get the swaption error and market values
        implied_var_dict, objective, market_swaptions, benchmarks = self.calc_loss_on_ir_curve(
            implied_params, base_date, time_grid, process, implied_obj, ir_factor, vol_surface)

        bounds = []
        for k, v in implied_var_dict.items():
            if k.startswith('Alpha'):
                bounds.append([self.alpha_bounds])
            elif k == 'Correlation':
                bounds.append([self.corr_bounds])
            else:
                bounds.append([self.sigma_bounds] * len(v))

        var_to_bounds = np.vstack(bounds)
        # one generator for the whole random search - the step taker here, the Metropolis test in
        # `solve` - so the search is a function of `Random_Seed` alone
        rng = np.random.RandomState(int(implied_params['instrument'].get('Random_Seed', 5120)))
        bounds_ok, make_step = make_basin_callbacks(
            0.125, self.sigma_bounds, self.alpha_bounds, self.corr_bounds, rng)

        # both adapters are the objective's, whichever the block declared - one `.data` boundary
        basin_hopper_fn_grad = make_basin_hopping_loss(objective, implied_var_dict, self.device, True)
        x0 = torch.cat(list(implied_var_dict.values())).cpu().detach().numpy()
        lsq_fn, jacobian = make_least_squares_loss(objective.loss, implied_var_dict, self.device)

        optimizers = [('basin', x0, basin_hopper_fn_grad, make_step, bounds_ok, var_to_bounds, rng),
                      ('leastsq', x0, lsq_fn, jacobian, list(zip(*var_to_bounds)))]

        return objective, optimizers, implied_var_dict, market_swaptions, benchmarks

    @staticmethod
    def sigma_knots(rows):
        """The declared `Sigma_Knots` as years (months / 12, the default grid's own arithmetic), or
        None for the default. A knot with a day part is refused: the grid is monthly."""
        if not rows:
            return None
        months = []
        for row in rows:
            offset = row[0] if isinstance(row, (list, tuple)) else row
            kwds = getattr(offset, 'kwds', {})
            if kwds.get('days') or kwds.get('weeks'):
                raise ValueError('Sigma_Knots takes whole months: {} has a day part'.format(offset))
            months.append(12 * kwds.get('years', 0) + kwds.get('months', 0))
        return np.array(sorted(set(months)), dtype=float) / 12.0

    def implied_process(self, base_currency, price_factors, price_models, ir_curve, rate,
                        vol_tenors=None):
        """The seed parameters and the process the objective prices through - two implied objects on
        a quanto'd curve, carrying the same numbers.

        CALIBRATE DOMESTICALLY, SIMULATE GLOBALLY. The market premium being repriced is
        $E^{Q_{dom}}[D_{dom}\\cdot(\\text{payoff})]$, struck and quoted in the RATE currency, so it
        is priced on domestic-measure paths whatever the job's base currency. Girsanov moves drifts
        and leaves quadratic variation alone, so $\\sigma_1,\\sigma_2,\\alpha_1,\\alpha_2,\\rho$ are
        the same numbers under either measure.

        So the two FX inputs `precalculate` builds $K$ out of are SUPPRESSED on the implied object
        the process is built on, taking that assembly down its base-currency branch to the bit, and
        left standing on the object returned, which is what `save_params` emits
        `Quanto_FX_Volatility` and `Quanto_FX_Correlation_1/2` off. All three consumers of the
        objective go through this one process. The simulator is untouched - `precalculate` still
        installs $K$ in a scenario run, being handed the emitted factor rather than this twin.

        The seed is asymmetric by ruling; `ALPHA_SEED` says why. A block whose parameter factor
        already exists warm-starts off it instead, clipped to the declared bounds.
        """
        if vol_tenors is None:
            vol_tenors = np.array([0, 1, 3, 6, 12, 24, 48, 72, 96, 120]) / 12.0
        # construct an initial guess - need to read from params
        param_name = utils.check_tuple_name(
            utils.Factor(type=self.__class__.__name__, name=rate[1:]))

        # check if we need a quanto fx vol
        fx_factor = utils.Factor('GBMAssetPriceTSModelParameters', ir_curve.get_currency())
        ir_factor = utils.Factor('InterestRate', ir_curve.get_currency())
        fx_factor_name = utils.check_tuple_name(fx_factor)
        ir_factor_name = utils.check_tuple_name(ir_factor)

        if fx_factor_name in price_factors:
            quanto_fx = price_factors[fx_factor_name]['Vol']
            curr_pair = sorted((base_currency,) + ir_curve.get_currency())
            correlation_name = 'Correlation.FxRate.{}/{}'.format('.'.join(curr_pair), ir_factor_name)
            # check if the quote is against the base currency
            sign = 1.0
            if curr_pair[0] == base_currency:
                sign = -1.0
                logging.info('Reversing Correlation as {} is quoted against the base currency'.format(correlation_name))
            # the correlation between fx and ir - needed to establish Quanto Correlation 1 and 2
            C = sign * price_factors.get(correlation_name, {'Value': 0.0})['Value']
        else:
            C = None
            quanto_fx = None

        if param_name in price_factors:
            param = price_factors[param_name]
            implied_obj = riskfactors.HullWhite2FactorModelParameters(
                {'Quanto_FX_Volatility': quanto_fx,
                 'short_rate_fx_correlation': C,
                 'Alpha_1': np.clip(param['Alpha_1'], *self.alpha_bounds),
                 'Alpha_2': np.clip(param['Alpha_2'], *self.alpha_bounds),
                 'Correlation': np.clip(param['Correlation'], *self.corr_bounds),
                 'Sigma_1': utils.Curve([], list(zip(
                     vol_tenors, np.interp(vol_tenors, *param['Sigma_1'].array.T).clip(*self.sigma_bounds)))),
                 'Sigma_2': utils.Curve([], list(zip(
                     vol_tenors, np.interp(vol_tenors, *param['Sigma_2'].array.T).clip(*self.sigma_bounds))))})
        else:
            implied_obj = riskfactors.HullWhite2FactorModelParameters(
                {'Quanto_FX_Volatility': quanto_fx,
                 'short_rate_fx_correlation': C,
                 'Alpha_1': ALPHA_SEED[0], 'Alpha_2': ALPHA_SEED[1], 'Correlation': 0.01,
                 'Sigma_1': utils.Curve([], list(zip(vol_tenors, [0.01] * vol_tenors.size))),
                 'Sigma_2': utils.Curve([], list(zip(vol_tenors, [0.01] * vol_tenors.size)))})

        # the domestic twin: every invariant, minus the two FX inputs, so `precalculate` assembles
        # K = 0. The None survives `read_cache` because Factor1D.get_tenor normalizes it to a Curve
        domestic_obj = riskfactors.HullWhite2FactorModelParameters(
            dict(implied_obj.param, Quanto_FX_Volatility=None, short_rate_fx_correlation=None))

        # need to create a process and params as variables to pass to tf
        process = stochasticprocess.HullWhite2FactorImpliedInterestRateModel(
            ir_curve, {'Lambda_1': 0.0, 'Lambda_2': 0.0}, domestic_obj)

        return implied_obj, process, vol_tenors

    def save_params(self, vars, price_factors, implied_obj, rate):
        param_name = utils.check_tuple_name(
            utils.Factor(type=self.__class__.__name__, name=rate[1:]))
        # grab the sigma tenors
        sig1_tenor, sig2_tenor = implied_obj.get_vol_tenors()
        # store the basic paramters
        param = {'Property_Aliases': None,
                 'Quanto_FX_Volatility': None,
                 'Alpha_1': float(vars['Alpha_1'][0]),
                 'Sigma_1': utils.Curve([], list(zip(sig1_tenor, vars['Sigma_1']))),
                 'Alpha_2': float(vars['Alpha_2'][0]),
                 'Sigma_2': utils.Curve([], list(zip(sig2_tenor, vars['Sigma_2']))),
                 'Correlation': float(vars['Correlation'][0])}

        # grab the quanto fx correlations
        quanto_fx1, quanto_fx2 = implied_obj.get_quanto_correlation(
            vars['Correlation'], [vars['Sigma_1'], vars['Sigma_2']])

        if quanto_fx1 is not None and quanto_fx2 is not None:
            param.update({
                'Quanto_FX_Volatility': implied_obj.param['Quanto_FX_Volatility'],
                'Quanto_FX_Correlation_1': quanto_fx1,
                'Quanto_FX_Correlation_2': quanto_fx2})

        price_factors[param_name] = param
        # return the final implied object
        return riskfactors.HullWhite2FactorModelParameters(param)


def leaf_deals(node):
    """The deals a deal-tree node prices - itself, or its children if it is a container."""
    if node.get('Children'):
        return [leaf for child in node['Children'] for leaf in leaf_deals(child)]
    return [node['Instrument']]


class Benchmark_State(utils.Calculation_State):
    """The pricing state a t0 benchmark valuation needs: one date, one path, float64.

    `t_Static_Buffer` is the point of it - every pricer reads a static curve from that dict, so a
    `requires_grad` tensor placed there is what puts the curve's nodes on the tape. It is built
    fresh per evaluation because `t_Buffer` is memoized by `(stoch, Factor)`, not by identity.

    Boundary registration is off: a deposit, an FRA and a swap leg take no decision on simulated
    state, so there is nothing for the correction to carry.
    """

    def __init__(self, static_buffer, one, report_currency):
        super(Benchmark_State, self).__init__(
            static_buffer, one, 1, report_currency, 'Constant', 1, False)
        self.boundary_aad = False
        self.boundary_sets = []


class BenchmarkInstruments(object):
    """The benchmark instruments of one curve solve, compiled once and priced at t0 off curve node
    TENSORS - so `torch.autograd.grad(pv, theta)` is the calibration Jacobian's row.

    A benchmark is a deal-tree NODE, `{'Instrument': deal, 'Children': [...]}`, as `Trade Data`
    authors it: a deposit or FRA is one deal, a par swap one `SwapInterestDeal`, an OIS swap a
    container over a compounded floating leg and a fixed one. Its PV is the sum of its leaves' PVs,
    each already in the reporting currency; no netting or collateral rule on top, which is what lets
    this stay out of `DealStructure`.

    **The graph audit.** The factor-construction path severs autograd in four places, every one on
    the way IN to `t_Static_Buffer`:

    - `Calculation._build_factor_state` and `Base_Revaluation.update_factors` mint every leaf off a
      numpy array. This class writes theta straight into the buffer and never calls `current_value`
      for a curve it is solving.
    - `riskfactors.Factor1D.current_value` is numpy end to end, and `Factor1D.get_tenor` REWRITES
      `param['Curve'].array` as a side effect of construction - so the node order theta is indexed
      by is the rewritten one, read back off the constructed factor.
    - `Factor1D.check_interpolation` precomputes the Hermite `(g, c)` pair from the numpy rate
      column. The pricing path does not use it: `utils.Interpolation.build` re-derives the pair from
      the buffer TENSOR, so a Hermite curve differentiates.
    - `utils.TensorSchedule.bind` mints the schedule's tensor half with `new_tensor`, which is where
      the QUOTE stops being differentiable. `_carry_quotes` builds the overlay that closes it.

    One trap that is not a severance: `utils.CurveTenor` caches its tenor grid as a tensor built
    from the first tensor that queries it. `all_tenors` is rebuilt per instance here, so a float64
    solve cannot inherit a float32 grid.

    `quotes` and `bumped_nodes` are the quote side: the quotes the set was authored at, in percent,
    and the same set authored one percent higher - the second says which schedule columns the quote
    writes (see `_carry_quotes`).
    """

    #: The solve is float64 whatever the simulation runs in: a bootstrap converging to 1e-10 cannot
    #: be done in float32, and the Jacobian is only as good as the residual it came from.
    dtype = torch.float64

    def __init__(self, nodes, price_factors, factor_interp, base_date, currency, calendars,
                 solve_for, device, quotes=None, bumped_nodes=None):
        # `config` imports from this module, so the package edge runs one way only
        from .config import Config

        cfg = Config(base_currency=currency)
        cfg.params['Price Factors'] = price_factors
        cfg.params['Price Factor Interpolation'] = factor_interp
        cfg.params['System Parameters']['Base_Date'] = base_date
        cfg.holidays = calendars
        cfg.set_calculation_children(nodes)
        # the engine's own discovery, so the set pulls exactly the factors a valuation would.
        # Single currency by construction, so every `calc_fx_cross` is the identity
        dependent_factors, _, _, _ = cfg.discover_factors(
            {'Currency': currency}, base_date, '0d', False)

        self.factors = {factor: riskfactors.construct_factor(
            factor, price_factors, factor_interp, base_date=base_date) for factor in dependent_factors}
        self.solve_for = tuple(solve_for)
        # the knot grid theta is indexed by, read off the factor AFTER `get_tenor` has rewritten it
        self.tenors = {factor: self.factors[factor].tenors for factor in self.solve_for}
        self.all_tenors = utils.update_tenors(base_date, self.factors)
        self.time_grid = utils.TimeGrid({base_date}, {base_date}, {base_date})
        self.time_grid.set_base_date(base_date)
        self.time_grid.set_report_dates(base_date, {base_date})
        self.one = torch.ones([1, 1], dtype=self.dtype, device=device)
        self.report_currency = instruments.get_fxrate_factor(
            utils.check_rate_name(currency), self.factors, {})
        # every factor the solve is NOT solving for is a constant of it
        self.constants = {factor: torch.tensor(
            obj.current_value(), dtype=self.dtype, device=device)
            for factor, obj in self.factors.items()
            if factor.type not in utils.DimensionLessFactors and factor not in self.solve_for}

        self.benchmarks = [[self._compile(leaf, base_date, calendars) for leaf in leaf_deals(node)]
                           for node in nodes]
        self.quotes = None if quotes is None else torch.tensor(
            quotes, dtype=self.dtype, device=device, requires_grad=True)
        if self.quotes is not None:
            self._carry_quotes(bumped_nodes, base_date, calendars)
        # compiled outside a calculation, so it binds its own schedules - and binds them LAST,
        # because the quote overlay is spliced into the copy `bind` makes
        for legs in self.benchmarks:
            for leg in legs:
                utils.bind_schedules(leg.Factor_dep, self.one)

    def _compile(self, deal, base_date, calendars):
        """One leaf deal's compiled form - the same `Factor_dep` / `Time_dep` pair a valuation
        builds, on a grid holding the base date alone."""
        return utils.DealDataType(
            Instrument=deal,
            Factor_dep=deal.calc_dependencies(
                base_date, self.factors, {}, self.factors, self.all_tenors, self.time_grid, calendars),
            Time_dep=self.time_grid.calc_deal_grid({base_date}),
            Calc_res=None)

    def _carry_quotes(self, bumped_nodes, base_date, calendars):
        """Put the quote leaf on every schedule column the quote WRITES.

        Which columns those are is MEASURED: the same set authored one percent higher is compiled,
        and the columns that moved are the value columns with the difference as their slope. The
        authoring map is affine in the quote, so one bumped compile IS the derivative - which keeps
        `QUOTE_WRITERS` the only place a quotable instrument is declared.

        The splice is `base + (q - q.detach()) * slope`, exactly zero forward with derivative one,
        so enabling quote gradients cannot move the solve. It is a derivative carrier and NOT a
        reparameterisation - the pricers memoize payment tensors off the schedule, so a different
        quote needs a fresh closure.

        Resets carry no overlay - a reset value also leaves through `known_resets`, which reads
        numpy - and a moved reset column raises here.

        A quote that moves NO column raises too: an `FXForwardDeal` writes its outright into
        `Buy_Amount`, which `generate` reads as a float off the deal, so nothing of its compiled
        form moves and `dF/dq` would be a silent zero row. Being measured, the refusal stops firing
        on its own the day such a type grows a schedule.
        """
        for index, (legs, node) in enumerate(zip(self.benchmarks, bumped_nodes)):
            delta = self.quotes[index] - self.quotes[index].detach()
            # the bumped set never went through discovery, and discovery is what resets a deal
            bumped_legs = leaf_deals(node)
            for leaf in bumped_legs:
                leaf.reset(calendars)
            carried = 0
            for leg, plus in zip(legs, [self._compile(leaf, base_date, calendars)
                                        for leaf in bumped_legs]):
                for name, schedule in leg.Factor_dep.items():
                    if not isinstance(schedule, utils.TensorCashFlows):
                        continue
                    bumped = plus.Factor_dep[name]
                    if schedule.Resets is not None and (
                            bumped.Resets.schedule != schedule.Resets.schedule).any():
                        raise Exception(
                            'Curve bootstrap: {} writes its quote into a RESET column, which the '
                            'schedule overlay does not reach'.format(name))
                    moved = bumped.schedule - schedule.schedule
                    columns = np.flatnonzero(np.abs(moved).max(axis=0))
                    if columns.size:
                        carried += columns.size
                        schedule.carry({int(column): self._column(schedule.schedule[:, column]) +
                                        delta * self._column(moved[:, column]) for column in columns})
            if not carried:
                raise Exception(
                    'Quote_Sensitivity: benchmark {} ({}) writes its quote into no cashflow '
                    'schedule column, so the increment-1 overlay reaches nothing and dV/dq for it '
                    'would be reported as a silent zero rather than refused. An FXForwardDeal '
                    'quote lands in Buy_Amount, which the pricer reads as a float off the deal and '
                    'not off a schedule - the only seam the overlay carries. Leave '
                    'Quote_Sensitivity at No and Quote_Propagation at No on a block carrying such '
                    'a quote; the solved curve is identical either way.'.format(
                        ' + '.join(str(leg.Instrument.field.get('Reference', '?')) for leg in legs),
                        ' + '.join(type(leg.Instrument).__name__ for leg in legs)))

    def _column(self, values):
        return torch.tensor(values, dtype=self.dtype, device=self.one.device)

    def reads(self, theta):
        """The factors outside `solve_for` this residual actually reads, measured rather than
        declared - the coupling detector a multi-curve set is grouped by.

        Every constant is made a leaf and the residual differentiated once; what a backward pass
        reaches is what it reads. One residual and one backward, a fraction of a Newton iteration,
        and it catches the coupling a `Discount_Rate` field cannot state - what a benchmark PROJECTS
        off is authored inside its own deal block.

        A residual with no graph reads nothing, which is the self-discounting single-curve case.
        """
        constants = list(self.constants)
        with torch.enable_grad():
            for factor in constants:
                self.constants[factor].requires_grad_(True)
            residual = self(theta)
            gradients = torch.autograd.grad(
                residual.sum(), [self.constants[factor] for factor in constants],
                allow_unused=True) if residual.requires_grad else [None] * len(constants)
            for factor in constants:
                self.constants[factor].requires_grad_(False)
        return {factor for factor, gradient in zip(constants, gradients)
                if gradient is not None and gradient.abs().max() > 0}

    def __call__(self, theta):
        """The benchmark PV vector at curve nodes `theta`, a `{Factor: tensor}` over `solve_for`."""
        shared = Benchmark_State({**self.constants, **theta}, self.one, self.report_currency)
        # one date and one path, so a leg's PV is a scalar - `reshape` says so and fails loud
        return torch.stack([
            sum(leg.Instrument.generate(shared, self.time_grid, leg).reshape(()) for leg in legs)
            for legs in self.benchmarks])


def damped_newton(residual, theta, n_iter, tol, halvings):
    """Solve `residual(theta) = 0` for a `{Factor: tensor}` of curve nodes, in float64.

    The curves are flattened into ONE system, so a projection curve solved against a discount curve
    in the same call is a single Jacobian. That Jacobian comes from autograd on the residual - one
    backward pass per benchmark gives a row - which is the same derivative the implicit function
    theorem needs, so the residual is written once and differentiated twice.

    Damping is a backtracking line search on the residual's max-norm: full step first, halved until
    it decreases. Near the root Newton takes the full step.

    `n_iter`, `tol` and `halvings` are declared fields of the block being solved.
    """
    keys = list(theta)
    sizes = [theta[key].numel() for key in keys]

    def unflatten(flat):
        return dict(zip(keys, flat.split(sizes)))

    x = torch.cat([theta[key].detach() for key in keys])
    for iteration in range(n_iter):
        x = x.detach().requires_grad_(True)
        f = residual(unflatten(x))
        jacobian = torch.stack([torch.autograd.grad(f[i], x, retain_graph=True)[0]
                                for i in range(f.numel())])
        step = torch.linalg.solve(jacobian, f.detach())

        # convergence is tested on the step BEFORE the line search: a step this small is inside the
        # linear solve's own rounding, and a residual at noise level cannot decrease again
        if step.abs().max() <= tol:
            return unflatten((x - step).detach())

        norm = f.detach().abs().max()
        damping = 1.0
        for _ in range(halvings + 1):
            trial = x.detach() - damping * step
            if residual(unflatten(trial)).abs().max() < norm:
                break
            damping *= 0.5
        else:
            raise Exception('Curve bootstrap: no damped Newton step reduces the residual '
                            '(iteration {}, residual {:.6g})'.format(iteration, float(norm)))
        x = trial

    raise Exception('Curve bootstrap: {} Newton iterations without converging'.format(n_iter))


def split_theta(benchmarks, theta):
    """The flat solved vector back as the `{Factor: nodes}` the residual takes, in `solve_for`
    order - which is the order `CalibrationSolve` concatenated it in."""
    sizes = [benchmarks.tenors[factor].size for factor in benchmarks.solve_for]
    return dict(zip(benchmarks.solve_for, theta.split(sizes)))


def residual_jacobians(benchmarks, theta):
    """The residual at `theta` and both its Jacobians: `dF/dtheta` (n x n) and `dF/dq` (n x m).

    One backward pass per benchmark gives both. Materialising the whole `dF/dq` costs nothing over
    contracting one cotangent through it, which is why `CalibrationSolve.backward`, the artifact's
    calibration Jacobian and its drift metric all read this one function.

    Every `grad` retains the graph: the residual's subgraph is shared with the forward pass
    (`pv_fixed_cashflows` memoizes its payment tensor in `Factor_dep`), so freeing it would take the
    forward pass's graph with it.
    """
    x = torch.cat([theta[factor] for factor in benchmarks.solve_for]).detach().requires_grad_(True)
    residual = benchmarks(split_theta(benchmarks, x))
    rows = [torch.autograd.grad(residual[i], [x, benchmarks.quotes], retain_graph=True)
            for i in range(residual.numel())]
    return (residual, torch.stack([row[0] for row in rows]),
            torch.stack([row[1] for row in rows]))


def calibration_jacobian(benchmarks, theta):
    """`dtheta/dq` at the fixed point.

    The implicit function theorem in matrix form, `dtheta/dq = -(dF/dtheta)^-1 (dF/dq)` - which is
    `CalibrationSolve.backward`'s arithmetic with every cotangent solved at once. No second solve:
    the fixed point is where the forward pass left it, so this costs one Newton iteration.

    `dF/dtheta` has to be invertible, which is a ROOT FIND's property; a least-squares fixed point
    would contract a pseudo-inverse instead. `J` is n x m and nothing assumes the two are equal.

    Over a COUPLED SET this is the whole block matrix, so `dtheta_2/dq_1` falls out of the one
    inverse - see `coupled_sets`.
    """
    with torch.enable_grad():
        _, d_theta, d_quote = residual_jacobians(benchmarks, theta)
    return -torch.linalg.solve(d_theta, d_quote)


class CalibrationSolve(torch.autograd.Function):
    """The bootstrap as one differentiable node: quotes in, calibrated nodes out.

    FORWARD IS THE ORDINARY SOLVE - `damped_newton` and nothing else - so enabling quote gradients
    cannot move a mark. Autograd runs `forward` with grad mode off, which the solve needs on for its
    own Jacobian, so it is re-enabled here and the iteration's graph discarded with it.

    BACKWARD IS THE IMPLICIT FUNCTION THEOREM, never an unrolled solver. At the fixed point
    `F(theta*, q) = 0`, so `dtheta/dq = -(dF/dtheta)^-1 (dF/dq)` and a cotangent `v = dL/dtheta*`
    contracts to

        w = (dF/dtheta)^-T v      then      dL/dq = -(dF/dq)^T w

    Both come from `residual_jacobians` at `(theta*, q)`. The residual is written once and
    differentiated twice, so the quote derivative cannot drift from the solve's own, nor from the
    `dtheta/dq` an artifact publishes. The Jacobian is recomputed at theta* rather than reused from
    the last Newton step, which was taken at the iterate before it.
    """

    @staticmethod
    def forward(ctx, benchmarks, seed, n_iter, tol, halvings, quotes):
        with torch.enable_grad():
            theta = damped_newton(benchmarks, seed, n_iter, tol, halvings)
        ctx.benchmarks, ctx.theta = benchmarks, theta
        return torch.cat([theta[factor] for factor in benchmarks.solve_for])

    @staticmethod
    def backward(ctx, cotangent):
        with torch.enable_grad():
            _, d_theta, d_quote = residual_jacobians(ctx.benchmarks, ctx.theta)
            w = torch.linalg.solve(d_theta.t(), cotangent)
        return None, None, None, None, None, -(d_quote.t() @ w)


class CalibrationArtifact(object):
    """One calibration of one coupled set, frozen as an operator - `(theta*, J, q0, timestamp)` and
    the compiled benchmark set the first two were read off.

    `theta*` is the solved node vector in `solve_for` order, `J` is `dtheta/dq` at that fixed point
    (exact by the implicit function theorem), `q0` the quote vector it was fitted at, in percent.
    Between two fits a small tick propagates linearly: `theta ~ theta* + J (q_now - q0)`, one matvec.

    IT COVERS THE SET, NOT THE BLOCK. `members` are the `Market Prices` blocks that solve as one
    system, in the order their quotes and nodes are concatenated in, and `J` is the whole block
    matrix. A partial ride is unrepresentable - one theta, one q0, one drift number for the set.

    `timestamp` is REPORTED rather than read: it reaches no number and no hash, so a wall clock
    cannot make two runs disagree.

    Plan-side and content-addressed. `key` is the SLOT (`plan_key`), so every tick of one strip
    lands on the same slot and a re-authored strip addresses a different one. `artifact_id` is the
    slot plus the quotes fitted at, so it MOVES with every refit and is a replay coordinate.

    Nothing here mutates. `ride` is a pure function of this artifact and the quotes it is handed,
    stored nowhere, so two EXECUTEs off one `(artifact, q_now)` are bit-identical; a refit publishes
    a NEW artifact into the same slot.

    It holds tensors and a compiled deal tree, so it cannot live in `Price Factors` and cannot be
    serialised: it lives in `ARTIFACTS`, in process. A cold start has none and the first tick
    REFUSES rather than pricing something else.
    """

    def __init__(self, key, members, theta, jacobian, quotes, benchmarks, drift=None):
        # `config` imports from this module, so the package edge runs one way only
        from . import content_hash

        self.key = key
        self.members = tuple(members)
        self.theta = theta
        self.jacobian = jacobian
        self.quotes = quotes
        self.benchmarks = benchmarks
        self.drift = drift
        self.timestamp = pd.Timestamp.utcnow()
        self.artifact_id = content_hash({'key': key, 'quotes': quotes.tolist()})

    @property
    def factors(self):
        """The curves this operator carries, in the order `theta` concatenates them."""
        return self.benchmarks.solve_for

    @property
    def jacobian_norm(self):
        """`||J||inf`, the induced max-row-sum norm - the conversion between the quote-space units
        `Drift_Tolerance` is declared in and the curve units a desk reads staleness in, since
        `||theta_ridden - theta_refit||inf <= ||J||inf ||r||inf` to first order."""
        return float(self.jacobian.abs().sum(dim=1).max())

    def ride(self, quotes):
        """`theta* + J (q_now - q0)` - the operator. Pure, and a matvec."""
        return self.theta + self.jacobian @ (quotes - self.quotes)

    def nodes(self, theta, factor):
        """One member curve's slice of a set-wide theta, as the numpy column a price factor is."""
        return split_theta(self.benchmarks, theta)[factor].detach().cpu().numpy()

    def mispricing(self, theta, quotes):
        """Every benchmark's residual at `(theta, quotes)`, in QUOTE SPACE: the move in that
        benchmark's own quote, in percent, that would close it. Exact at any theta and any quote.

        The set was compiled at `q0`, and needs no re-compile to be scored elsewhere: a benchmark's
        PV is affine in its own quote at fixed theta (measured in `_carry_quotes`, second difference
        exactly zero), so

            F(theta, q) = F(theta, q0) + (dF/dq)(q - q0)

        holds with no remainder - provided `dF/dq` is taken at the theta being scored.
        `residual_jacobians` re-differentiates HERE, at the ridden theta: 71.7ms against a 594ms
        refit on the ZAR strip. Reusing the `dF/dq` stored at `theta*` is cheaper by one backward
        and misses by `(d2F/dtheta dq) dtheta dq` - the same order as the residual it estimates -
        reading low (0.886 of the truth at worst) on the tick shapes the tolerance exists to refuse.

        Dividing each row by its own quote sensitivity is what makes `Drift_Tolerance` a number a
        desk can set. It is a row max rather than a diagonal, so a family whose benchmarks are not
        one-quote-each stays expressible.
        """
        with torch.enable_grad():
            residual, _, d_quote = residual_jacobians(
                self.benchmarks, split_theta(self.benchmarks, theta))
        d_quote = d_quote.detach()
        return ((residual.detach() + d_quote @ (quotes - self.quotes)) /
                d_quote.abs().amax(dim=1))


class ArtifactStore(object):
    """Calibration artifacts under their plan keys - the PlanCache's discipline for the other half
    of a prepared job.

    Bounded and least-recently-used, because an artifact is a refit and never the record of
    anything - the replay tuple is that. Locked because a slot is written by whatever thread ran the
    bootstrap and read by whatever runs the EXECUTE.

    Content-addressed, so an entry is immutable under its key: a refit REPLACES the artifact in a
    slot. A moved quote NUMBER keeps the slot, which is what makes a ride possible, while a
    re-authored quote SET addresses a different one and finds it empty.

    `covering` returns CANDIDATES - every artifact holding that curve, most-recently-used first -
    and never picks one: the caller recomputes each candidate's slot off the market data standing
    now. Two artifacts can cover one curve at once (a Hermite job and a linear one).

    Scanned rather than indexed by factor: an index can disagree with the store, and this holds 32.
    """

    def __init__(self, size=32):
        self.size = size
        self.artifacts = OrderedDict()
        self.lock = threading.Lock()

    def put(self, artifact):
        with self.lock:
            self.artifacts[artifact.key] = artifact
            self.artifacts.move_to_end(artifact.key)
            if len(self.artifacts) > self.size:
                self.artifacts.popitem(last=False)
            return artifact.key

    def get(self, key):
        with self.lock:
            if key not in self.artifacts:
                return None
            self.artifacts.move_to_end(key)
            return self.artifacts[key]

    def covering(self, factor):
        with self.lock:
            return [self.artifacts[key] for key in reversed(self.artifacts)
                    if factor in self.artifacts[key].factors]


#: Where a published calibration artifact lives - in process, beside the service's plan cache. It
#: holds tensors and a compiled benchmark set, so neither `Price Factors` nor a file is an option.
ARTIFACTS = ArtifactStore()


def quote_nodes(points, discount_rate, shift=0.0):
    """The used quotes as deal-tree nodes, each authored at its own quote plus `shift` percent.

    Deep-copied because authoring WRITES the quote and the discount curve into the block.
    """
    nodes = []
    for point in points:
        authored = dict(copy.deepcopy(point['Deal']), Object=point['DealType'])
        author_quote(authored, point['Quoted_Market_Value'] + shift, discount_rate)
        nodes.append(quote_node(authored, {}))
    return nodes


def _pin_deposit_schedule(deal, quote):
    """A deposit has no rate field of its own. Pinning every accrual start is what makes it price
    as a fixed leg, which is also what keeps it off the forecast curve the solve is building -
    `DepositDeal.reset` drops that dependency when the schedule covers every start."""
    starts = instruments.generate_dates_backward(
        deal['Maturity_Date'], deal['Effective_Date'], deal['Payment_Frequency'])[:-1]
    deal['Interest_Rate_Schedule'] = utils.DateList({date: quote for date in starts})


def _fixed_cashflow_rate(deal, quote):
    """The fixed leg of a two-leg benchmark carries the quote on every row of its schedule."""
    for item in deal['Cashflows']['Items']:
        item['Rate'] = utils.Percent(quote)


def _fx_forward_outright(deal, quote):
    """An FX forward's quote is the FORWARD OUTRIGHT - units of `Buy_Currency` per one unit of
    `Sell_Currency` - and the amount it buys is where that number lands.

    The authored benchmark fixes `Sell_Amount` and both discount-rate names, so the quote moves
    `Buy_Amount` alone and `FXForwardDeal.generate` is exactly affine in it at fixed curves - which
    is what `CalibrationArtifact.mispricing` reads as an exact quote-space residual.

    The outright is not a percent and nothing here converts it, because no writer converts anything:
    a percent-quoted type carries its scaling in its own field semantics (`DepositDeal` divides by
    100, `FRADeal` wraps in a `Basis`, `_fixed_cashflow_rate` writes a `utils.Percent`).
    """
    deal['Buy_Amount'] = quote * deal['Sell_Amount']


#: Where a quote's number goes, per instrument type, keyed by the `Object` string - the one thing
#: the family knows about a type beyond that type's own declarations. A registry, so a new quotable
#: instrument is a row. A container carries no rate; its fixed leg does.
QUOTE_WRITERS = {
    'DepositDeal': _pin_deposit_schedule,
    'FRADeal': lambda deal, quote: deal.update({'FRA_Rate': quote}),
    'SwapInterestDeal': lambda deal, quote: deal.update({'Swap_Rate': quote}),
    'CFFixedInterestListDeal': _fixed_cashflow_rate,
    'FXForwardDeal': _fx_forward_outright,
}


def author_quote(deal, quote, discount_rate):
    """Author an instrument block AT its quote, discounting on `discount_rate`.

    What an instrument PROJECTS off it names itself; what the quote set DISCOUNTS on is a property
    of the curve set, stated once on the block. Recurses into `Children`, so a two-leg benchmark
    gets the quote on the leg that holds a rate and the discount curve on both.
    """
    for child in deal.get('Children', ()):
        author_quote(child, quote, discount_rate)
    deal['Discount_Rate'] = discount_rate
    writer = QUOTE_WRITERS.get(deal['Object'])
    if writer:
        writer(deal, quote)


def quote_node(deal, valuation_options):
    """A deal-tree node from an authored instrument block - the shape `set_calculation_children`
    takes. `Config.parse_json` builds it from `.Deal` markers and a `Children` list; a quote carries
    the same block inline, so it is built here instead."""
    node = {'Instrument': instruments.construct_instrument(
        {key: value for key, value in deal.items() if key != 'Children'}, valuation_options)}
    if deal.get('Children'):
        node['Children'] = [quote_node(child, valuation_options) for child in deal['Children']]
    return node


def quote_knots(nodes, base_date, day_count, calendars):
    """The curve's knot grid: one knot per benchmark, at that benchmark's last cashflow date.

    The only placement that makes the system square - a knot with no instrument maturing at it is
    unidentified, and two instruments between one pair of knots leave the curve under-determined.
    Below the shortest knot the curve is flat by `CurveTenor`'s clipping, so the front stub costs no
    unknown. The output grid IS this grid: interpolating onto a second would stop the curve
    repricing its quotes.

    Returned in NODE order and in the curve's own day count, so a caller can pair each knot with the
    quote that identifies it; the curve itself is sorted.
    """
    code = utils.get_day_count(day_count)
    maturities = []
    for node in nodes:
        leaves = leaf_deals(node)
        for leaf in leaves:
            leaf.reset(calendars)
        maturities.append(max(max(leaf.get_reval_dates()) for leaf in leaves))
    return np.array([utils.get_day_count_accrual(
        base_date, (maturity - base_date).days, code) for maturity in maturities])


class InterestRateCurveParameters(object):
    """A zero curve solved from deposit, FRA, swap and FX forward quotes, priced by the engine's
    own pricers.

    A quote is an instrument, a `Quote_Type` and a number - see the developer note on
    [Market Prices](../developer/market_prices.md). Each `Points` entry names an instrument type in
    `DealType` and carries a block of it in `Deal`, so the `Instrument` store's declarations ARE
    this family's quote schema. The family authors that block at its `Quoted_Market_Value`, and a
    fair benchmark prices to zero, so the solve is a root find on the t0 PV vector.

    Two blocks make a multi-curve set - an OIS discount curve, then a projection curve discounting
    on it - and `Discount_Rate` is what orders them. A blank `Discount_Rate` discounts on the curve
    being built, the single-curve configuration and the harder solve.

    Unlike the other families this writes an `InterestRate` price factor rather than a
    `<ClassName>` parameter block, which is what `price_factor_type` declares.
    """
    market_factor_type = 'InterestRatePrices'
    #: The `Price Factors` type this family writes. The others write a block named for their own
    #: class, so the emitter recovers it; no rule recovers `InterestRate` from this class name.
    price_factor_type = 'InterestRate'
    #: The instrument types a quote may be, each a declared `Instrument` type - so the quote's
    #: schema IS that type's declarations. `StructuredDeal` is how a two-leg benchmark is authored;
    #: `FXForwardDeal` crosses currencies, its quote being a forward OUTRIGHT held at par.
    quote_instruments = ('DepositDeal', 'FRADeal', 'SwapInterestDeal', 'StructuredDeal',
                         'FXForwardDeal')
    #: Block fields an artifact is NOT a function of - the lifecycle switches, read when one is
    #: published or ridden rather than when it is fitted. `plan_key` shadows them out so a knob
    #: governing the ride cannot also hide the artifact it governs. `Quote_Sensitivity` joins them
    #: because it provably moves neither theta* nor J.
    lifecycle_fields = ('Quote_Sensitivity', 'Quote_Propagation', 'Drift_Tolerance')
    fields = [
        F('Currency', 'Text', default=REQUIRED, description='The currency of the curve to build'),
        F('Day_Count', 'Text', default='ACT_365',
          values=['ACT_365', 'ACT_360', 'ACT_365_ISDA', '_30_360', '_30E_360', 'ACT_ACT_ICMA'],
          description='Daycount the solved curve\'s tenors are expressed in'),
        F('Discount_Rate', 'Text', default='',
          description='The curve the quotes discount on; blank builds a self-discounting curve'),
        F('N_Iter', 'Integer', default=50,
          description='Newton iteration cap. Newton is quadratic near the root and a par-rate seed '
                      'is already within a few basis points, so a well-posed strip converges in '
                      'single digits; reaching the cap raises rather than returning a half-solved '
                      'curve'),
        F('Tol', 'Float', default=1e-14,
          description='Convergence tolerance on the Newton STEP, in rate space. A zero rate is '
                      'O(1e-2), so 1e-14 is about 1e-12 relative - inside the 1e-10 a round trip '
                      'asks for, and where the linear solve\'s own rounding stops the iteration '
                      'improving'),
        F('Damping_Halvings', 'Integer', default=6,
          description='How many times the line search may halve a Newton step before giving up. '
                      'Below that the step LENGTH is not what is wrong, so the solve says so '
                      'rather than creeping towards a root it will not reach'),
        F('Quote_Sensitivity', 'Text', default='No', values=['Yes', 'No'],
          description='Keep the solved curve connected to its quotes, so a calculation\'s backward '
                      'pass reports dV/dq beside dV/dtheta. Costs one extra compile of the '
                      'benchmark set and holds the residual graph for the life of the config; the '
                      'solved numbers are identical either way'),
        F('Quote_Propagation', 'Text', default='No', values=['No', 'Linear'],
          description='How a quote that moves between bootstraps reaches the curve. No re-solves, '
                      'which is what a job does today. Linear publishes a calibration artifact '
                      '(theta*, dtheta/dq, q0) at each bootstrap and RIDES it at every calculation '
                      'after - theta* + dtheta/dq (q_now - q0), a matvec instead of a solve - '
                      'refusing when the ridden curve no longer reprices the benchmarks inside '
                      'Drift_Tolerance, and refusing when no artifact answers to the plan. It is a '
                      'property of a COUPLED SET rather than of a block: blocks whose residuals '
                      'read each other\'s curves are solved as one system and ridden as one '
                      'operator, so every block of such a set must declare it or none may. Costs '
                      'one extra compile and one backward pass per block to measure the set, plus '
                      'the compile Quote_Sensitivity costs'),
        F('Drift_Tolerance', 'Float', default=1e-3,
          description='How far out of par a ridden curve may leave this block\'s own benchmarks '
                      'before Quote_Propagation refuses and asks for a refit, measured in PERCENT '
                      'OF QUOTE - so 1e-3 is a tenth of a basis point of mispricing. The ride is '
                      'second-order accurate, so this is a bound on the SQUARE of the tick: on the '
                      'round-trip worlds it admits about 11bp and refuses a 25bp move. Only read '
                      'when Quote_Propagation is Linear'),
        F('Points', 'Container', default={
            'Use': 'Yes', 'Deal': {}, 'Descriptor': '', 'DealType': 'DepositDeal',
            'Quote_Type': 'Par_Rate', 'Quoted_Market_Value': 0.0},
          sub_fields=[
            F('Use', 'Text', default='Yes', values=['Yes', 'No'],
              description='Whether this quote enters the solve'),
            F('Deal', 'Container', default={},
              description='The instrument itself, authored as a deal of type DealType'),
            F('Descriptor', 'Text', default='', description='Free text naming the quote'),
            F('DealType', 'Text', default='DepositDeal', values=list(quote_instruments),
              description='The instrument type the quote is a price for'),
            F('Quote_Type', 'Text', default='Par_Rate', values=['Par_Rate'],
              description='What Quoted_Market_Value is; the solve holds the instrument at par'),
            F('Quoted_Market_Value', 'Float',
              description='The quote the instrument is authored at, in the unit its own DealType '
                          'reads: a rate benchmark is quoted in percent, and an FXForwardDeal is '
                          'quoted as a forward OUTRIGHT - units of Buy_Currency per one unit of '
                          'Sell_Currency. The family scales nothing; each type\'s field semantics '
                          'do - see QUOTE_WRITERS. The one value key a patch cannot clear '
                          '(schema.MARKET_QUOTE_REQUIRED): a mid is moved, never removed'),
            F('Quoted_Bid', 'Float',
              description='The bid side of this quote, in the same unit as the mid. QUOTE-LAYER '
                          'data: nothing below reads it and the curve is solved from '
                          'Quoted_Market_Value alone - the mid is what the book runs on. Optional '
                          'because a benchmark the terminal quotes no two-way for stays mid-only '
                          'rather than borrowing a spread'),
            F('Quoted_Ask', 'Float',
              description='The offer side, the pair of Quoted_Bid. Optional on the same terms, and '
                          'read by nothing in the solve - it is the evidence a desk charges a '
                          'spread off, not an input to the strip'),
            F('Timestamp', 'Date', default='',
              description='When this quote was observed. Stored and reported, never read by the '
                          'solve - what counts as too old is the consumer\'s policy')],
          description='One market quote: an instrument, what kind of number is quoted, the number, '
                      'its two-way sides where the source printed them, and when it was seen')
    ]

    def __init__(self, param, device, dtype):
        self.device = device
        self.prec = dtype
        self.param = param
        #: What `Quote_Sensitivity` leaves behind: the solved nodes still connected to their quotes,
        #: per curve, plus the quote leaf per block. `Config.bootstrap` harvests both - tensors
        #: cannot live in `Price Factors`.
        self.calibrated = {}
        self.quote_leaves = {}

    @staticmethod
    def benchmark_curves(block):
        """Every `InterestRate` curve this block's used benchmark deals NAME, read off each deal
        type's own `factor_fields` and recursing into `Children`.

        `Discount_Rate` orders the ordinary multi-curve case but cannot order a CROSS-CURRENCY
        benchmark: an `FXForwardDeal` names the other leg's curve in `Sell_Discount_Rate`, inside
        the deal, so a block with a blank `Discount_Rate` can still read a curve nobody has built.

        Read off the deal CLASS, because this runs before anything is seeded. Being a declaration
        read it is strictly weaker than `BenchmarkInstruments.reads`, which measures the same
        coupling but needs every curve to exist first.
        """
        def walk(deal, object_type):
            declared = getattr(instruments, object_type, None)
            for field, candidates in getattr(declared, 'factor_fields', {}).items():
                if 'InterestRate' in candidates and deal.get(field):
                    yield '.'.join(utils.check_rate_name(deal[field]))
            for child in deal.get('Children', ()):
                yield from walk(child, child.get('Object', ''))

        return {curve for point in block['Points'] if point.get('Use', 'Yes') == 'Yes'
                for curve in walk(point['Deal'], point['DealType'])}

    def in_dependency_order(self, market_prices):
        """This family's blocks, one that READS a curve another block BUILDS coming after it.

        A block reads a curve two ways and both order it: `Discount_Rate`, and what its benchmark
        deals NAME (`benchmark_curves`). A block naming its own curve is the self-discounting
        configuration and orders nothing. A cycle is refused by name here rather than as the bare
        `RuntimeError` the sort would raise.
        """
        blocks = {}
        for name, implied_params in market_prices.items():
            rate = utils.check_rate_name(name)
            market_factor = utils.Factor(rate[0], rate[1:])
            if market_factor.type == self.market_factor_type:
                blocks[name] = implied_params
        # keyed by the curve name a `Discount_Rate` carries - the block's name without its type
        builds = {'.'.join(utils.check_rate_name(name)[1:]): name for name in blocks}
        graph = {}
        for name, implied_params in blocks.items():
            block = implied_params['instrument']
            reads = {block['Discount_Rate']} | self.benchmark_curves(block)
            graph[name] = sorted({builds[curve] for curve in reads
                                  if curve in builds and builds[curve] != name})
        # `topological_sort` deletes what it resolves, so what is left is exactly the cycle
        unresolved = dict(graph)
        try:
            order = utils.topological_sort(unresolved)
        except RuntimeError:
            raise Exception(
                'Curve bootstrap: {} cannot be put in a solve order - each reads a curve another '
                'builds, so whichever is solved first is solved against a curve that does not '
                'exist yet ({}). A mutually-referencing set has to be solved as ONE system; this '
                'family solves a block at a time.'.format(
                    ' + '.join(sorted(unresolved)),
                    '; '.join('{} reads {}'.format(name, ' + '.join(edges))
                              for name, edges in sorted(unresolved.items()))))
        return [(name, blocks[name]) for name in order]

    def bootstrap(self, sys_params, price_models, price_factors, factor_interp, market_prices,
                  calendars, debug=None):
        """Solve every block for the zero curve that reprices its used quotes to par, one COUPLED
        SET at a time.

        A set is the group of blocks whose residuals read each other's curves, measured rather than
        declared (`coupled_sets`). Forming one costs a compile and a backward pass per block and
        buys an operator whose Jacobian carries the coupling, so it is formed only where one was
        asked for; with no `Quote_Propagation` this is a dependency-ordered loop.
        """
        base_date = sys_params['Base_Date']
        blocks = self.in_dependency_order(market_prices)
        groups = self.coupled_sets(blocks, price_factors, factor_interp, base_date, calendars) \
            if any(entry['instrument'].get('Quote_Propagation', 'No') == 'Linear'
                   for _, entry in blocks) else [[block] for block in blocks]

        for group in groups:
            self.solve_set(group, price_factors, factor_interp, base_date, calendars)

    def seed(self, market_price, block, price_factors, base_date, calendars):
        """Write this block's par-rate seed curve into `Price Factors`, and give back what the solve
        reads off it: `(curve factor, used quotes, deal nodes, discount rate)`.

        Seeding comes first because the benchmark closure constructs the curve factor OUT of
        `Price Factors`; a par rate is within a few basis points of the zero rate at the same
        maturity, so it is also the seed. `Curve` sorts the pairs, so each knot keeps its quote.

        The `/100` is the SEED's and not the quote's - `author_quote` scales nothing. So an
        amount-valued quote seeds nonsense and converges anyway: an 18.32 outright seeds an 18.32%
        zero rate against a true 8.99% and damped Newton walks it to zero residual. Branching on the
        deal type here would put knowledge of a type somewhere other than `QUOTE_WRITERS`.
        """
        curve = utils.Factor('InterestRate', utils.check_rate_name(market_price)[1:])
        discount_rate = block['Discount_Rate'] or '.'.join(curve.name)
        points = self.used_quotes(block, market_price)
        nodes = quote_nodes(points, discount_rate)
        price_factors[utils.check_tuple_name(curve)] = {
            'Property_Aliases': None, 'Sub_Type': None, 'Currency': block['Currency'],
            'Day_Count': block['Day_Count'], 'Curve': utils.Curve([], list(zip(
                quote_knots(nodes, base_date, block['Day_Count'], calendars),
                [point['Quoted_Market_Value'] / 100.0 for point in points])))}
        return curve, points, nodes, discount_rate

    def coupled_sets(self, blocks, price_factors, factor_interp, base_date, calendars):
        """This family's blocks grouped into the SETS that have to solve as one system - measured.

        Two blocks are coupled when one's residual READS the curve the other builds, which is not
        the question `Discount_Rate` answers: what a benchmark projects off is authored inside its
        own deal block, so a strip declaring a blank `Discount_Rate` can still forecast off a
        neighbour's curve - on such a world a 10bp tick moved the "independent" curve 568bp.
        `BenchmarkInstruments.reads` answers by differentiation instead.

        The groups are the connected components of that relation, in dependency order, and a group
        is solved and ridden WHOLE - which is what puts `dtheta_2/dq_1` inside `J`, an ordering
        carrying a coupling through a bootstrap but nothing through a ride.

        Every block is seeded before anything is measured, a block forecasting off an unbuilt curve
        being uncompilable.
        """
        seeded, builds, ordered = {}, {}, dict(blocks)
        for market_price, entry in blocks:
            curve, points, nodes, _ = self.seed(
                market_price, entry['instrument'], price_factors, base_date, calendars)
            seeded[market_price] = (curve, nodes)
            builds[curve] = market_price

        components = {market_price: {market_price} for market_price in ordered}
        for market_price, entry in blocks:
            curve, nodes = seeded[market_price]
            benchmarks = BenchmarkInstruments(
                nodes, price_factors, factor_interp, base_date,
                entry['instrument']['Currency'], calendars, [curve], self.device)
            reads = benchmarks.reads({curve: torch.tensor(
                benchmarks.factors[curve].current_value(),
                dtype=BenchmarkInstruments.dtype, device=self.device)})
            for factor in reads & set(builds):
                merged = components[market_price] | components[builds[factor]]
                for name in merged:
                    components[name] = merged

        groups, seen = [], set()
        for market_price in ordered:
            if market_price not in seen:
                seen |= components[market_price]
                groups.append([(name, ordered[name]) for name in ordered
                               if name in components[market_price]])
        return groups

    def solve_set(self, group, price_factors, factor_interp, base_date, calendars):
        """Solve one coupled set: one Newton system over every curve in it, one Jacobian, one
        artifact.

        Flattening a multi-curve set is `damped_newton`'s own shape rather than a new solver -
        `solve_for` is a list, the residual takes a `{Factor: nodes}` over it, and the block
        Jacobian that falls out is what `calibration_jacobian` inverts in one go.

        The seed theta is read off the CONSTRUCTED factor, so it is aligned with the tenor grid the
        pricers gather against whatever `get_tenor` made of the block. The solve goes through the
        implicit-function wrapper either way; with no quotes on the tape no edge is recorded.

        Solver knobs are declared per block and a set takes the STRICTEST of them.
        """
        members = [(market_price, entry['instrument']) for market_price, entry in group]
        propagate = [block.get('Quote_Propagation', 'No') == 'Linear' for _, block in members]
        connect = [block.get('Quote_Sensitivity', 'No') == 'Yes' for _, block in members]
        if any(propagate) and not all(propagate):
            raise Exception(
                'Quote_Propagation is a property of a COUPLED SET, and {} solve as one system - '
                'measured, not declared. {} declares it while {} does not, and a partial ride is '
                'the one configuration this operator cannot express: the declining block would be '
                'priced off the curve the last bootstrap wrote while its partner rode the tick, '
                'and the drift metric would report the ridden half as perfectly fresh. Measured on '
                'the USD world at a 10bp OIS tick, that reads a PV of 9829.62 where the refit says '
                '9621.25, against a true move of -23.36 - wrong sign, 8.9x the size, drift 4.5e-4. '
                'Declare Quote_Propagation on every block of the set, or on none.'.format(
                    ' + '.join(name for name, _ in members),
                    ' + '.join(name for (name, _), asks in zip(members, propagate) if asks),
                    ' + '.join(name for (name, _), asks in zip(members, propagate) if not asks)))
        currencies = {block['Currency'] for _, block in members}
        if len(currencies) > 1:
            raise Exception(
                'Quote_Propagation: {} are coupled but priced in {} - a benchmark set has one '
                'reporting currency, so this set cannot be compiled as one system. Leave '
                'Quote_Propagation at No on a cross-currency curve set.'.format(
                    ' + '.join(name for name, _ in members), ' and '.join(sorted(currencies))))

        seeded = [self.seed(market_price, block, price_factors, base_date, calendars)
                  for market_price, block in members]
        # both switches want the quote side of the residual, one extra compile either way
        carry = any(connect) or any(propagate)
        points = [point for _, block_points, _, _ in seeded for point in block_points]
        curves = [curve for curve, _, _, _ in seeded]

        time_now = time.monotonic()
        benchmarks = BenchmarkInstruments(
            [node for _, _, nodes, _ in seeded for node in nodes], price_factors, factor_interp,
            base_date, members[0][1]['Currency'], calendars, curves, self.device,
            quotes=[point['Quoted_Market_Value'] for point in points] if carry else None,
            bumped_nodes=[node for _, block_points, _, discount_rate in seeded
                          for node in quote_nodes(block_points, discount_rate, 1.0)]
            if carry else None)
        # seed theta off the constructed factor - see the docstring on grid alignment
        theta = CalibrationSolve.apply(
            benchmarks,
            {curve: torch.tensor(benchmarks.factors[curve].current_value(),
                                 dtype=BenchmarkInstruments.dtype, device=self.device)
             for curve in curves},
            max(int(block.get('N_Iter', 50)) for _, block in members),
            min(float(block.get('Tol', 1e-14)) for _, block in members),
            max(int(block.get('Damping_Halvings', 6)) for _, block in members),
            benchmarks.quotes)

        solved = split_theta(benchmarks, theta)
        # a set-wide quote leaf reports dV/dq across the system, so its descriptors name the block
        # each quote came off; a set of one is the block's own list unchanged
        descriptors = [point['Descriptor'] if len(members) == 1 else
                       '{}: {}'.format(market_price, point['Descriptor'])
                       for (market_price, _), (_, block_points, _, _) in zip(members, seeded)
                       for point in block_points]
        for curve, (market_price, _), wants in zip(curves, members, connect):
            price_factors[utils.check_tuple_name(curve)]['Curve'] = utils.Curve(
                [], list(zip(benchmarks.tenors[curve], solved[curve].detach().cpu().numpy())))
            if wants:
                self.calibrated[curve] = solved[curve]
                self.quote_leaves[market_price] = (descriptors, benchmarks.quotes)
        if all(propagate):
            self.publish(members, factor_interp, base_date, benchmarks, theta.detach())

        residuals = benchmarks(split_theta(benchmarks, theta.detach())).detach()
        logging.info('{} bootstrapped from {} quotes in {:.2f} seconds, residual {:.3g}'.format(
            ' + '.join(utils.check_tuple_name(curve) for curve in curves), len(points),
            time.monotonic() - time_now, float(residuals.abs().max())))
        for point, residual in zip(points, residuals):
                logging.info('  {} at {:.4f} reprices to {:.3g}'.format(
                    point['Descriptor'], point['Quoted_Market_Value'], float(residual)))

    @classmethod
    def takes(cls, point, market_price):
        """Whether this family prices the quote. `Par_Rate` is the only convention built - every
        benchmark is held at PV zero. A futures price and a money-market rate on a different basis
        would have to be authored differently.
        """
        if point['Quote_Type'] == 'Par_Rate':
            return True
        logging.error('{} quote {} - Quote_Type {} not supported yet'.format(
            market_price, point['Descriptor'], point['Quote_Type']))
        return False

    @classmethod
    def used_quotes(cls, block, market_price):
        """The quotes that enter the solve, in the order theta, `J` and `q0` are all indexed by.

        A classmethod because the RIDE needs the same list off a block nobody is bootstrapping; a
        second filter beside this one is how a ridden theta ends up indexed differently from the
        artifact it rode.
        """
        return [point for point in block['Points']
                if point['Use'] == 'Yes' and cls.takes(point, market_price)]

    @classmethod
    def plan_key(cls, members, factor_interp, base_date):
        """The SLOT an artifact lives in: every member block of the coupled set, the base date, the
        interpolation scheme and the engine version - with the `lifecycle_fields` shadowed out and
        the quote VALUES projected away.

        Literally `schema.partition_market_price`'s structural half, the split `Config.plan_hash`
        takes over the same section, so the two cannot drift. Every tick of one strip lands on the
        same slot, which is what makes a ride possible, while a re-authored instrument, a flipped
        `Use`, a different `Day_Count`, a different solver knob or a new engine build lands
        elsewhere. A row that gains a `Quoted_Bid` keeps its slot: the solve reads neither side.

        The key names the SET rather than the block, so re-authoring a discount strip moves the slot
        of every curve solved against it.

        `base_date` and `Price Factor Interpolation` are in it because the SOLVE reads them and the
        block does not carry them. Without them two jobs 45 days apart share a slot, and a Linear
        job rides a Hermite solve 0.53bp away from its own.
        """
        # `config` imports from this module, so the package edge runs one way only
        from . import content_hash

        return content_hash({
            'engine_version': __version__, 'base_date': base_date, 'interpolation': factor_interp,
            'set': [{'market_price': market_price,
                     'block': dict(partition_market_price({'instrument': block})[0]['instrument'],
                                   **{field: None for field in cls.lifecycle_fields})}
                    for market_price, block in members]})

    @classmethod
    def slot(cls, names, market_prices, factor_interp, base_date):
        """The key those member blocks address in `market_prices` NOW, or `None` if one is gone.

        What turns `ArtifactStore.find`'s scan back into content addressing: an artifact answers for
        a curve only if the plan it was fitted against is still the plan standing.
        """
        members = [(name, market_prices.get(name, {}).get('instrument')) for name in names]
        if any(block is None for _, block in members):
            return None
        return cls.plan_key(members, factor_interp, base_date)

    @classmethod
    def publish(cls, members, factor_interp, base_date, benchmarks, theta):
        """Freeze this solve as an artifact, and measure what the last one would have been worth.

        With the previous artifact still in the slot, `theta_refit - theta_ridden` says how far the
        operator had drifted by the time it was replaced, and the ridden theta's benchmark residual
        says the same in the space the tolerance is declared in. Both are published ON the new
        artifact, so the record of how stale the last calibration got travels with its replacement.

        The refreshed artifact takes the old one's SLOT under a new `artifact_id`.
        """
        key = cls.plan_key(members, factor_interp, base_date)
        artifact = CalibrationArtifact(
            key, [market_price for market_price, _ in members], theta,
            calibration_jacobian(benchmarks, split_theta(benchmarks, theta)),
            benchmarks.quotes.detach(), benchmarks)
        name = ' + '.join(market_price for market_price, _ in members)

        previous = ARTIFACTS.get(key)
        if previous is not None:
            ridden = previous.ride(artifact.quotes)
            artifact.drift = {
                'tick': float((artifact.quotes - previous.quotes).abs().max()),
                'theta': float((theta - ridden).abs().max()),
                'quote': float(artifact.mispricing(ridden, artifact.quotes).abs().max()),
                'rode': previous.artifact_id, 'fitted': previous.timestamp}
            logging.info(
                '{} refit: artifact {} (fitted {}) rode a {:.4g}% tick to a drift of {:.3g} in '
                'theta and {:.3g}% in quote space (solver Tol {:.3g}), replaced by {}'.format(
                    name, previous.artifact_id[:12], previous.timestamp, artifact.drift['tick'],
                    artifact.drift['theta'], artifact.drift['quote'],
                    min(float(block.get('Tol', 1e-14)) for _, block in members),
                    artifact.artifact_id[:12]))
        else:
            logging.info('{} refit: artifact {} published, nothing in the slot to score'.format(
                name, artifact.artifact_id[:12]))
        ARTIFACTS.put(artifact)

    @classmethod
    def propagate(cls, factor, market_prices, factor_interp, base_date):
        """The curve `factor` RIDDEN to the quotes standing in `market_prices` now, or `None` where
        no block asks for one - the operator, evaluated per EXECUTE and storing nothing.

        Two ways to get `None`: a factor this family does not write, and a block that did not ask
        for `Quote_Propagation`.

        A block that DID ask and finds no artifact REFUSES - a miss is a 404 rather than a different
        number. That closes the replay hole: falling back to `theta*` reprices the book (13.4% on
        the eviction probe) while `plan_hash`, `values_hash`, the engine version and the seed all
        stay identical. A cold process rides nothing and says so, an artifact being unserialisable.

        A ride leaving the benchmarks further out of par than `Drift_Tolerance` refuses too. The
        tolerance is the SET's strictest, so a coupled set rides or refuses whole, and `slot`
        rechecks that the artifact's plan is the one still standing.
        """
        if factor.type != cls.price_factor_type:
            return None
        market_price = utils.check_tuple_name(utils.Factor(cls.market_factor_type, factor.name))
        block = market_prices.get(market_price, {}).get('instrument')
        if block is None or block.get('Quote_Propagation', 'No') != 'Linear':
            return None

        covering = ARTIFACTS.covering(factor)
        artifact = next((found for found in covering if found.key == cls.slot(
            found.members, market_prices, factor_interp, base_date)), None)
        if artifact is None:
            raise utils.CalibrationStale(
                '{}: Quote_Propagation is Linear and no calibration artifact answers to this plan '
                '- {}. Bootstrap the job and the same EXECUTE runs off the artifact that publishes; '
                'an artifact holds tensors and a compiled benchmark set, so it cannot be serialised '
                'and a fresh process has none. A plan the store cannot answer is a MISS, and a miss '
                'is not permission to price off the curve the last bootstrap wrote.'.format(
                    market_price, 'the store holds none for this curve' if not covering else
                    '{} cover it, each fitted against a different plan ({})'.format(
                        len(covering), '; '.join(' + '.join(found.members) for found in covering))))
        # a ride is a USE: a ridden slot must not age out under one merely published beside it
        ARTIFACTS.get(artifact.key)

        tolerance = min(float(market_prices[name]['instrument'].get('Drift_Tolerance', 1e-3))
                        for name in artifact.members)
        quotes = torch.tensor(
            [point['Quoted_Market_Value'] for name in artifact.members
             for point in cls.used_quotes(market_prices[name]['instrument'], name)],
            dtype=artifact.theta.dtype, device=artifact.theta.device)
        theta = artifact.ride(quotes)
        drift = float(artifact.mispricing(theta, quotes).abs().max())
        tick = float((quotes - artifact.quotes).abs().max())
        # the tolerance is in percent of quote; ||J||inf converts it to the curve units it is felt in
        curve_units = tolerance * artifact.jacobian_norm * 1e4
        if drift > tolerance:
            raise utils.CalibrationStale(
                '{}: Quote_Propagation refused - riding artifact {} (fitted {}) over a {:.4g}% '
                'tick leaves its benchmarks {:.3g}% of quote out of par, past the declared '
                'Drift_Tolerance {:.3g} (at most {:.3g}bp of zero rate on this set, ||J||inf '
                '{:.4g}). The linear operator is only second-order accurate and this move is too '
                'big for it, so re-bootstrap and the same job runs off the refit.'.format(
                    market_price, artifact.artifact_id[:12], artifact.timestamp, tick, drift,
                    tolerance, curve_units, artifact.jacobian_norm))
        logging.info(
            '{}: rode artifact {} (fitted {}) over a {:.4g}% tick, benchmarks {:.3g}% of quote out '
            'of par against a tolerance of {:.3g} ({:.3g}bp of zero rate, ||J||inf {:.4g})'.format(
                market_price, artifact.artifact_id[:12], artifact.timestamp, tick, drift,
                tolerance, curve_units, artifact.jacobian_norm))
        return artifact.nodes(theta, factor), artifact.artifact_id


class FXVolSurfaceParameters(object):
    """An `FXVol` surface bootstrapped from the ATM / risk-reversal / butterfly quotes it ticks in as.

    An FX smile is quoted in DELTA - one ATM vol per expiry and, per delta pillar, the risk reversal
    and butterfly that say how the two wings sit around it - while the surface the engine prices off
    is a log-moneyness one. The algebra between them is the strangle pair,
    `vol(call) = ATM + BF + RR/2` and `vol(put) = ATM + BF - RR/2`, followed by the
    delta-to-log-moneyness solve `Factor2D` carries for a `Malz` surface. What this family fixes is
    WHERE they run.

    **The x-grid is pinned.** The solve refines a log-moneyness grid until interpolating between its
    nodes resolves the smile, so the grid is a function of the quotes. Run at factor-construction
    time that would make every vol tick STRUCTURAL - a moved node is a moved tenor grid, a new plan
    and a recompile. So the refinement runs here, once, and the grid is part of the written factor;
    a re-bootstrap finding a surface already written for the same expiries at the same tolerance
    reuses it and moves only the vols, which is what makes a tick a `bind='value'` patch.
    `Grid_Tolerance` SIZES the grid, so it is structural and asking for a different one breaks the
    pin. The log says what the pinned grid resolves the CURRENT quotes to.

    **The conventions are declared because the solve implements exactly one of each.** The delta a
    pillar names is a premium-adjusted FORWARD delta ((K/F)N(d2) for a call), and the ATM quote is
    that convention's delta-neutral straddle, K = F exp(-sigma^2 T / 2).

    A quote `Timestamp` survives a save at the resolution it was authored.

    **A point may carry a two-way, and nothing here reads it.** `Quoted_Bid`/`Quoted_Ask` ride the
    row beside the mid for `derivus.structures` to charge a spread on. Every line below addresses
    `Quoted_Market_Value` by name, so what this writes is the mid surface either way.

    Like `InterestRatePrices` this writes a typed price factor rather than a `<ClassName>`
    parameter block, which is what `price_factor_type` declares.
    """
    market_factor_type = 'FXVolPrices'
    #: The `Price Factors` type this family writes - a `Malz` `FXVol`, minus the delta surface: it
    #: arrives SOLVED, so `Factor2D.solves_delta_surface` is false and the pinned grid survives.
    price_factor_type = 'FXVol'
    #: `Surface_Type` names the moneyness convention the engine reads the block at (log(F/K),
    #: interpolated in total variance); `Moneyness_Rule` is the factor's own declared default and no
    #: Malz code path reads it.
    surface_type, moneyness_rule = 'Malz', 'Sticky_Moneyness'
    #: The tolerances a grid can be BUILT at, enforced by `bootstrap`. Refinement halves an interval
    #: until the midpoint's vol error falls under the tolerance, so at 0.0 no midpoint qualifies
    #: (7.6M nodes on one expiry after 21 passes, still doubling) while 1e-8 is 4599 nodes for a
    #: four-expiry smile. At 1 the seed grid already passes.
    grid_tolerance_bounds = (1e-8, 1.0)
    #: The precision the TAPE runs in - the value path is numpy and has no dtype to pick. Float64 on
    #: the CPU whatever the job asked for, the twin dividing by the residual's slope at the root.
    dtype = torch.float64
    fields = [
        F('Currency', 'Text', default='',
          description='The currency stamped on the surface this builds'),
        F('Delta_Type', 'Text', default='Forward', values=['Forward'],
          description='The delta a Pillar names. The solve inverts a FORWARD delta - there is no '
                      'spot-delta discounting in it - so that is the one convention offered'),
        F('Premium_Adjusted', 'Text', default='Yes', values=['Yes'],
          description='Whether the pillar delta is premium adjusted. The solve inverts '
                      '(K/F)N(d2), which is the premium-adjusted (percentage-foreign) delta'),
        F('ATM_Convention', 'Text', default='Delta_Neutral_Straddle',
          values=['Delta_Neutral_Straddle'],
          description='What an ATM quote is the vol of. The solve places it at the strike whose '
                      'premium-adjusted straddle is delta neutral, K = F exp(-sigma^2 T/2); an '
                      'ATMF quote would sit at a different strike and is not built'),
        F('Grid_Tolerance', 'Float', default=1e-4, bounds=grid_tolerance_bounds,
          description='The vol error the log-moneyness grid is refined to when it is BUILT. Not '
                      'reached again per tick: the grid is pinned, so this sizes the plan rather '
                      'than the quote fit, and the log reports what the pinned grid still '
                      'resolves the quotes to. Changing it is STRUCTURAL - it breaks the pin and '
                      'refines a new grid. Bounded because refinement does not terminate below '
                      'the floor'),
        F('Quote_Sensitivity', 'Text', default='No', values=['Yes', 'No'],
          description='Keep the log-moneyness surface connected to the ATM / RR / BF quotes it was '
                      'built from, so a calculation\'s backward pass reports dV/dq beside '
                      'dV/dtheta. The written surface is identical either way'),
        F('Points', 'Table', default='null', row=Row([
            F('Use', 'Text', default='Yes', values=['Yes', 'No'],
              description='Whether this quote enters the surface'),
            F('Expiry', 'Float',
              description='Expiry in YEARS - the surface\'s own expiry axis, so no day count '
                          'stands between the quote and the coordinate it lands on'),
            F('Pillar', 'Float',
              description='The delta the wings are quoted at, as a magnitude (0.25 is the 25 '
                          'delta pair). Not read on an ATM row, which is quoted at no pillar'),
            F('Quote_Type', 'Text', default='ATM', values=['ATM', 'RR', 'BF'],
              description='The ATM vol, the risk reversal (call less put) or the butterfly (the '
                          'wing pair\'s average over ATM)'),
            F('Quoted_Market_Value', 'Float',
              description='The quote, in the surface\'s own units - 0.12 for 12 vols, and a risk '
                          'reversal of -0.35 vols is -0.0035. The one value key a patch cannot '
                          'clear (schema.MARKET_QUOTE_REQUIRED): a mid is moved, never removed, '
                          'where the sides and the stamp are absent whenever nothing printed'),
            F('Quoted_Bid', 'Float',
              description='The bid side of this quote, in the surface\'s own units. QUOTE-LAYER '
                          'data: nothing below reads it, and the surface, the pinned grid and '
                          'every mark are built from Quoted_Market_Value alone - the mid is what '
                          'the book runs on. Optional because a pillar the terminal quotes no '
                          'two-way for stays mid-only rather than borrowing a spread'),
            F('Quoted_Ask', 'Float',
              description='The offer side, the pair of Quoted_Bid and read only where that is - '
                          'derivus.structures, which shifts a leg\'s own copy of the written '
                          'surface by the ATM half-spread to quote a client two-sided. Absent '
                          'here, a structure quotes at mid, which is what it has always done'),
            F('Timestamp', 'Date', default='',
              description='When this quote was observed. Stored and reported - the surface '
                          'carries the latest of them - and never read by pricing')]),
          description='One quote: an expiry, a delta pillar, what kind of number is quoted, the '
                      'number, and when it was seen')
    ]

    def __init__(self, param, device, dtype):
        self.device = device
        self.prec = dtype
        self.param = param
        #: What `Quote_Sensitivity` leaves behind: the log-moneyness surface still connected to its
        #: quotes, keyed as `_build_factor_state` mints the `FXVol` leaf, plus the quote leaf per
        #: block. `Config.bootstrap` harvests both - tensors cannot live in `Price Factors`.
        self.calibrated = {}
        self.quote_leaves = {}

    @staticmethod
    def used(block):
        """The block's quotes that enter the surface - `Use` holds one out without deleting it."""
        return [point for point in block['Points'] if point['Use'] == 'Yes']

    @staticmethod
    def descriptor(point):
        """What a quote is CALLED where `dV/dq` is reported: its type, its pillar, its expiry."""
        return ('ATM {:g}'.format(point['Expiry']) if point['Quote_Type'] == 'ATM' else
                '{} {:g} {:g}'.format(point['Quote_Type'], point['Pillar'], point['Expiry']))

    @staticmethod
    def atm_quotes(quotes):
        """`{expiry: ATM vol}` - the ATM row per expiry, the number that expiry's wings sit around.

        The surface's ATM vol at an expiry IS this number: `Factor2D.malz_skew` places it at the
        delta-neutral straddle strike. So it is what `smile` builds the wings off, and what
        `GBMAssetPriceTSModelParameters` takes as the ATM column of a surface this family built.
        """
        return {point['Expiry']: point['Quoted_Market_Value']
                for point in quotes if point['Quote_Type'] == 'ATM'}

    @classmethod
    def smile(cls, quotes):
        """The quotes as a `(delta, expiry, vol)` surface - the strangle pair, per expiry pillar.

        `vol(call) = ATM + BF + RR/2` and `vol(put) = ATM + BF - RR/2`, with the ATM vol carried at
        the +-0.5 LABEL the delta solve reads it off (0.5 is not a delta there - the solve replaces
        the label with the delta-neutral straddle's own delta). A pillar quoted with only one of the
        two is read with the other at zero; an expiry with wings but no ATM quote raises `KeyError`.

        A `Pillar` of 0.5 is refused: a wing quoted at the ATM label would land a second vol on the
        ATM row's coordinate. A 50 delta pair is quoted as the ATM row.
        """
        atm = cls.atm_quotes(quotes)
        wings = {(point['Expiry'], point['Pillar'], point['Quote_Type']):
                 point['Quoted_Market_Value'] for point in quotes if point['Quote_Type'] != 'ATM'}
        pillars = sorted({key[:2] for key in wings})

        surface = [[0.5, expiry, vol] for expiry, vol in atm.items()]
        for expiry, pillar in pillars:
            if np.isclose(pillar, 0.5):
                raise ValueError(
                    'the {} quote at expiry {:g} is on Pillar {:g}, which collides with the ATM '
                    'label - a 50 delta pair is quoted as the ATM row'.format(
                        '/'.join(sorted(k[2] for k in wings if k[:2] == (expiry, pillar))),
                        expiry, pillar))
            rr, bf = wings.get((expiry, pillar, 'RR'), 0.0), wings.get((expiry, pillar, 'BF'), 0.0)
            surface.append([pillar, expiry, atm[expiry] + bf + 0.5 * rr])
            surface.append([-pillar, expiry, atm[expiry] + bf - 0.5 * rr])
        return np.array(sorted(surface))

    @staticmethod
    def carried_smile(quotes, values):
        """`smile`'s vol column on a tape - the strangle algebra, mirrored, and nothing else.

        Row for row and in `smile`'s own order, so the frozen structure the value path leaves
        addresses this vector. The algebra is `+`, `*` and a sort, on which torch and numpy agree to
        the last bit in float64, so the mirror is bit-identical and gated as such.
        """
        atm = {point['Expiry']: value for point, value in zip(quotes, values)
               if point['Quote_Type'] == 'ATM'}
        wings = {(point['Expiry'], point['Pillar'], point['Quote_Type']): value
                 for point, value in zip(quotes, values) if point['Quote_Type'] != 'ATM'}

        zero = values.new_zeros(())
        rows = [(0.5, expiry, vol) for expiry, vol in atm.items()]
        for expiry, pillar in sorted({key[:2] for key in wings}):
            rr = wings.get((expiry, pillar, 'RR'), zero)
            bf = wings.get((expiry, pillar, 'BF'), zero)
            rows.append((pillar, expiry, atm[expiry] + bf + 0.5 * rr))
            rows.append((-pillar, expiry, atm[expiry] + bf - 0.5 * rr))
        return torch.stack([vol for _, _, vol in sorted(rows, key=lambda row: row[:2])])

    @classmethod
    def carried_skews(cls, delta_surface, expiries, vols):
        """`Factor2D.malz_skews` on a tape - `vols` is that surface's vol column, still connected."""
        return {T: cls.carried_skew(delta_surface[delta_surface[:, 1] == T][:, 0],
                                    vols[delta_surface[:, 1] == T].clamp(min=1e-4), T)
                for T in expiries}

    @staticmethod
    def carried_skew(delta, vols, T):
        """`Factor2D.malz_skew` on a tape - the same wing pair, node for node, still connected.

        The wing vols are taped and so is `delta_atm`, the ATM quote saying where the delta-neutral
        straddle sits and so MOVING the two ATM nodes of the delta grid. What is read off the
        numbers rather than differentiated is the LAYOUT - the ordering, which node carries the
        +-0.5 label, which side had its ATM node mirrored in - a permutation having no derivative.
        """
        d = np.asarray(delta, dtype=float)
        order = np.argsort(d)
        d, v = d[order], list(vols[order])

        atm = np.isclose(np.abs(d), 0.5)
        sigma_atm = v[np.flatnonzero(atm)[-1]]  # ascending d, so the PREFERRED +0.5 label
        delta_atm = 0.5 * torch.exp(-0.5 * sigma_atm * sigma_atm * T)
        label = float(delta_atm.detach())
        nodes = [np.sign(di) * delta_atm if a else sigma_atm.new_tensor(di)
                 for di, a in zip(d, atm)]
        d = np.where(atm, np.sign(d) * label, d)

        # both wings need the ATM node - a smile quoted on one side only is mirrored onto the other
        for side in (-1.0, 1.0):
            if not np.any(np.isclose(d, side * label)):
                d, nodes, v = np.append(d, side * label), nodes + [side * delta_atm], v + [sigma_atm]

        order = np.argsort(d)
        d = d[order]
        deltas = torch.stack([nodes[i] for i in order])
        vols = torch.stack([v[i] for i in order])
        return {'d_put': deltas[d <= 0.0], 'v_put': vols[d <= 0.0],
                'd_call': deltas[d >= 0.0], 'v_call': vols[d >= 0.0],
                'sigma_atm': sigma_atm, 'delta_atm': delta_atm}

    @classmethod
    def carried_sigma(cls, skew, carried, T, x):
        """`Factor2D.malz_sigma` on a tape - and the bisection is NOT on it.

        A bisection's iterates are dyadic combinations of the bracket endpoints, so a tape through
        the 64 halvings differentiates where the BRACKET is rather than where the root is: on the
        call wing that derivative carries the ATM quote and no risk reversal or butterfly at all,
        while the true root moves with the wing vols.

        So the tape starts at the CONVERGED root. `delta*` is a constant here and the differentiable
        one is one Newton step off it, `delta - R(delta, q) / (dR/ddelta)` - the implicit function
        theorem as an expression, worth the solve's own residual forward and exactly `-R_q/R_delta`
        backward. What makes it the theorem is that `delta*` is the ROOT, not that the slope is
        detached.

        The CLAMPED nodes take the other branch, and it is not a repair: outside the wing's bracket
        there is no root to differentiate, the vol IS the endpoint knot's, and the derivative is
        that knot vol's own. The two branches meet where the root arrives at the endpoint, so the
        switch is a kink and autograd reports the branch's one-sided derivative.

        The wing span is guarded, and an ordinary config reaches it: an ATM-only smile has
        `malz_skew` mirror its one node onto both sides, so each wing is a single knot of zero span.
        Dividing before selecting would NaN the whole Jacobian while the value path writes a
        perfectly good flat surface.
        """
        delta_star, is_call, bracketed = riskfactors.Factor2D.malz_delta(skew, T, x)
        sigma = carried['sigma_atm'].new_zeros(np.shape(x))

        for side, wing in ((1.0, 'call'), (-1.0, 'put')):
            on_wing = is_call if side > 0 else ~is_call
            if not on_wing.any():
                continue
            knots, values, grid = carried['d_' + wing], carried['v_' + wing], skew['d_' + wing]
            xs, root, live = x[on_wing], delta_star[on_wing], bracketed[on_wing]
            # the segment the root sits in and, for a clamped node, the endpoint knot it sits on -
            # both frozen, an interval index not being a differentiable quantity
            seg = np.clip(np.searchsorted(grid, root, side='right') - 1,
                          0, max(grid.size - 2, 0))
            top = np.minimum(seg + 1, grid.size - 1)
            near = np.abs(root[:, None] - grid[None, :]).argmin(1)
            span = knots[top] - knots[seg]
            wide = values.new_tensor(grid[top] != grid[seg], dtype=torch.bool)
            k_over_f, log_mny = values.new_tensor(np.exp(-xs)), values.new_tensor(xs)

            def wing_vol(delta):
                # the double where over a ONE-KNOT wing's zero span - see the docstring
                low = values[seg]
                rise = torch.where(wide, (values[top] - low) / torch.where(
                    wide, span, torch.ones_like(span)), torch.zeros_like(span))
                return low + (delta - knots[seg]) * rise

            def residual(delta):
                vol = wing_vol(delta)
                d2 = (log_mny - 0.5 * vol * vol * T) / (vol * np.sqrt(T))
                return k_over_f * side * utils.norm_cdf(side * d2) - delta

            # `base` carries no graph, so its `.detach()` and the slope's missing `create_graph`
            # are no-ops - the theorem holds off the ROOT. |dR/ddelta| stays above 0.948 in a
            # 375-point sweep, so the `on_tape` guard is idiom
            base = values.new_tensor(root)
            probe = base.detach().requires_grad_(True)
            d_delta = torch.autograd.grad(residual(probe).sum(), probe)[0]
            on_tape = values.new_tensor(live, dtype=torch.bool)
            step = residual(base) / torch.where(on_tape, d_delta, torch.ones_like(d_delta))
            sigma[on_wing] = wing_vol(torch.where(on_tape, base - step, knots[near]))

        return sigma

    @classmethod
    def carried_surface(cls, skews, carried, grid):
        """`Factor2D.malz_surface`'s vol column on a tape, row for row and in its order."""
        return torch.cat([cls.carried_sigma(skews[T], carried[T], T, nodes)
                          for T, nodes in grid.items()])

    @classmethod
    def pinned_grid(cls, written, expiries, tolerance):
        """The log-moneyness grid a previously written surface already carries, or None.

        `written` is the PRICE FACTOR block this family wrote last, not a quote block. Four ways to
        get None, each a grid the quotes are not asking for: nothing to pin to; a different SUBTYPE,
        whose moneyness axis is S/K rather than log(F/K); a different set of EXPIRIES, which a
        rebuild answers and stretching does not; and a different TOLERANCE.
        """
        surface = written.get('Surface') if written else None
        if surface is None or not surface.array.any():
            return None
        if written.get('Surface_Type') != cls.surface_type:
            return None
        if not np.array_equal(np.unique(surface.array[:, 1]), expiries):
            return None
        if written.get('Grid_Tolerance') != tolerance:
            return None
        return {T: surface.array[surface.array[:, 1] == T][:, 0] for T in expiries}

    def bootstrap(self, sys_params, price_models, price_factors, factor_interp, market_prices,
                  calendars, debug=None):
        """Turn each block's quotes into the log-moneyness `FXVol` surface the pricers read.

        The x-grid is taken from the factor this wrote last if it still describes the same expiries
        at the same tolerance - see the class docstring on pinning - and refined otherwise.

        `Quote_Sensitivity` leaves that surface behind still connected to its ATM / RR / BF quotes,
        so `Calculation.factor_leaf` can offer the connected tensor rather than minting an `FXVol`
        leaf out of numpy. The tape is a SPLICE over the shipped conversion (`carried_sigma`), so
        every number written comes out of `Factor2D.malz_surface`.

        The grid is NOT differentiated. It is refined against the quotes when built and pinned from
        then on, which is what makes a tick a values patch; the twin moves the vols on frozen nodes.
        """
        for market_price, implied_params in market_prices.items():
            rate = utils.check_rate_name(market_price)
            market_factor = utils.Factor(rate[0], rate[1:])

            if market_factor.type == self.market_factor_type:
                block = implied_params['instrument']
                vol_name = utils.check_tuple_name(
                    utils.Factor(self.price_factor_type, market_factor.name))

                tolerance = float(block.get('Grid_Tolerance', riskfactors.Factor2D.malz_tol))
                # the one bounds= the engine reads: outside it there is no grid to refine to
                if not self.grid_tolerance_bounds[0] <= tolerance <= self.grid_tolerance_bounds[1]:
                    raise ValueError(
                        '{}: Grid_Tolerance {:g} is outside [{:g}, {:g}] - the refinement does not '
                        'terminate there'.format(market_price, tolerance,
                                                 *self.grid_tolerance_bounds))

                quotes = self.used(block)
                delta_surface = self.smile(quotes)
                expiries = np.unique(delta_surface[:, 1])
                skews = riskfactors.Factor2D.malz_skews(delta_surface, expiries)

                grid = self.pinned_grid(price_factors.get(vol_name), expiries, tolerance)
                pinned = grid is not None
                if not pinned:
                    grid = riskfactors.Factor2D.malz_grid(skews, tolerance)

                surface = riskfactors.Factor2D.malz_surface(skews, grid)
                stamps = [point['Timestamp'] for point in quotes if point['Timestamp']]
                price_factors[vol_name] = {
                    'Property_Aliases': None, 'Surface_Type': self.surface_type,
                    'Moneyness_Rule': self.moneyness_rule,
                    'Currency': block.get('Currency', ''),
                    'Grid_Tolerance': tolerance,
                    'Quote_Timestamp': max(stamps) if stamps else '',
                    'Surface': utils.Curve([], surface)}

                if block.get('Quote_Sensitivity', 'No') == 'Yes':
                    leaves = torch.tensor([point['Quoted_Market_Value'] for point in quotes],
                                          dtype=self.dtype, requires_grad=True)
                    carried = self.carried_surface(skews, self.carried_skews(
                        delta_surface, expiries, self.carried_smile(quotes, leaves)), grid)
                    # `Factor2D` sorts by (expiry, moneyness) and mints a leaf out of THAT column,
                    # so the twin is put in the same order rather than assumed to be in it
                    rows = np.array(surface)
                    order = np.lexsort((rows[:, 0], rows[:, 1]))
                    self.calibrated[utils.Factor(self.price_factor_type, market_factor.name)] = \
                        torch.tensor(rows[order, 2], dtype=self.dtype) + (
                            carried[order] - carried[order].detach())
                    self.quote_leaves[market_price] = (
                        [self.descriptor(point) for point in quotes], leaves)

                logging.info('{} built from {} quotes on a {} grid of {} nodes as at {}'.format(
                    vol_name, len(quotes), 'pinned' if pinned else 'refined',
                    sum(len(nodes) for nodes in grid.values()),
                    price_factors[vol_name]['Quote_Timestamp'] or 'no stated time'))
                for T, nodes in grid.items():
                    logging.info('  expiry {:.4f}: {} nodes resolving the smile to {:.3g} vol, '
                                 'built at {:.3g}'.format(
                                     T, len(nodes),
                                     float(riskfactors.Factor2D.malz_error(
                                         skews[T], T, nodes).max()), tolerance))


def family_class(btype):
    """The price family `Bootstrapper Configuration` names; an unknown name refuses by name."""
    cls = globals().get(btype)
    if not (isinstance(cls, type) and 'market_factor_type' in cls.__dict__):
        raise ValueError('Bootstrapper Configuration names {}, which is no price family; the families '
                         'are {}'.format(btype, ', '.join(sorted(FAMILIES))))
    return cls


def construct_bootstrapper(btype, param, dtype=torch.float32):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    return family_class(btype)(param, device, dtype)


#: class name -> the `Market Prices` type it reads, one row per family (the emitter's own rule)
FAMILIES = {name: cls.__dict__['market_factor_type'] for name, cls in list(globals().items())
            if isinstance(cls, type) and 'market_factor_type' in cls.__dict__}
