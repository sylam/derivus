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
from .schema import F, OPTION_QUOTE, QUOTE_TWO_WAY, REQUIRED, Row, partition_market_price
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


class HestonNandiModelParameters(object):
    documentation = (
        'Fx And Equity',
        ['For Risk Neutral simulation, the Heston-Nandi GARCH(1,1) model is calibrated to a set of European',
         'options $J$ on a spot underlying. The model is ASSET CLASS AGNOSTIC - the *Underlying* may be any',
         'spot (0D) price factor (**FxRate**, **EquityPrice**, **CommodityPrice**, **FuturesPrice**) and the',
         '*Volatility* any (moneyness, expiry) vol surface (**FXVol**, **EquityPriceVol**,',
         '**CommodityPriceVol**); the type of each is looked up from the price factors, or named explicitly',
         'with *Underlying_Type* / *Volatility_Type*. Under the locally risk neutral valuation relationship',
         '(LRNVR) $\\lambda^*=-\\frac{1}{2}$, so the model is parameterised directly in $\\gamma^*$:',
         '',
         '$$\\log\\frac{S_{t+1}}{S_t}=(r-q)-\\frac{h_{t+1}}{2}+\\sqrt{h_{t+1}}z_{t+1}$$',
         '',
         '$$h_{t+1}=\\omega+\\beta h_t+\\alpha\\Big(z_t-\\gamma^*\\sqrt{h_t}\\Big)^2$$',
         '',
         'with $z\\sim N(0,1)$ i.i.d. and $h_{t+1}$ predictable (known at $t$), hence the fitted initial',
         'variance is $h_1$ - the variance of the *first* step - and is stored as **H0**. Option values come',
         'from the recursive characteristic function of Heston and Nandi (2000) inverted by Gauss-Legendre',
         'quadrature (see the Heston-Nandi section of `derivus.utils`). The optional *Yield* (a dividend, repo, convenience or carry',
         'curve) enters as $q$ - the drift is $r-q$ and the value carries the extra $e^{-qt}$ factor - so',
         'equity, FX and commodity underlyings are all handled by the same objective.',
         '',
         'THE FORWARD IS THE PRICER\'S. $r$ is the *Discount_Rate* curve and is what the premium',
         'discounts on; the forward GROWS at the optional *Funding_Rate* curve instead where one is',
         'named, which is the curve `utils.calc_eq_forward` integrates - an equity\'s own repo curve',
         '(**EquityPrice.Interest_Rate**), not the curve its deals discount on. Left blank the two are',
         'one curve, which is the one-curve world and what an FX pair always was; named, an index',
         'carrying a repo/borrow spread calibrates at the forward it is priced at rather than one a',
         'spread away from it.',
         '',
         'Writing the persistence as $\\psi=\\beta+\\alpha\\gamma^{*2}$ and the stationary per-step variance as',
         '$m=\\frac{\\omega+\\alpha}{1-\\psi}$, the objective',
         '',
         '$$\\sum_{j\\in J}w_j\\Big(V_j-V_j(\\omega,\\alpha,\\beta,\\gamma^*,h_1)\\Big)^2$$',
         '',
         'is minimized with L-BFGS-B over',
         '$\\Big(\\log\\omega,\\psi,l,\\frac{|\\gamma^*|}{1000},\\log h_1\\Big)$ where',
         '$\\alpha=\\frac{|l|\\psi}{\\gamma^{*2}}$, $\\beta=\\psi(1-|l|)$ and',
         '$\\gamma^*=\\mathrm{sgn}(l)\\,|\\gamma^*|$ for a SIGNED leverage share $l\\in[-1,1]$.',
         'Stationarity is therefore a *box constraint on a fitted parameter* ($\\psi\\le1-10^{-6}$) and',
         'holds at every point the optimizer visits - there is no penalty term and no infeasible iterate.',
         'The share carries the sign because $\\gamma^*$ cannot: $\\alpha$ is singular at',
         '$\\gamma^*=0$, so the fitted magnitude is bounded away from zero and BOTH skew directions',
         'live in one box - the equity leverage shape (vol falling with strike in the underlying\'s',
         'own units, $\\gamma^*>0$) and the shape an FX pair read on its **FxRate** axis routinely',
         'wants ($\\gamma^*<0$). At $l=0$ there is no leverage channel and $\\gamma^*$ is',
         'unidentified, which is what a flat surface legitimately reports. Gradients are exact',
         '(torch autograd through the inversion).',
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

    # The Fourier inversion needs double precision - the framework default (float32) destroys the
    # cancellation in P1/P2 - so the dtype this is constructed with is deliberately ignored.
    prec = torch.float64
    # x = (log Omega, psi, SIGNED leverage share, |Gamma_Star|/1000, log H0) - see reparam
    bounds = [(np.log(1e-12), np.log(1e-3)), (0.0, 1.0 - 1e-6), (-1.0, 1.0),
              (1e-3, 5.0), (np.log(1e-10), np.log(1e-2))]
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

    market_factor_type = 'HestonNandiModelPrices'
    #: What a collapsed ladder costs, interpolated into the refusal: the component family inherits
    #: the emitter and its ladder identifies something else.
    identification_note = ('five parameters: the ATM term structure is what identifies H0, Beta '
                           'and Omega')
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
        F('Quadrature_Panels', 'Integer', default=64,
          description='Gauss-Legendre panels the characteristic function is inverted on'),
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

    @staticmethod
    def reparam(x):
        """Maps the fitted vector x to (Omega, Alpha, Beta, Gamma_Star, H0).

        Stationarity is enforced by construction: the optimizer fits the persistence
        psi = Beta + Alpha*Gamma_Star^2 under a box bound psi <= 1-1e-6 and splits it between the
        two channels with a leverage share l. Omega and H0 are fitted in logs so they stay positive
        and their ~1e-6 scale does not wreck the line search against Gamma_Star (~1e3, hence /1000).

        The share carries Gamma_Star's SIGN, so both skew directions live in one box - positive is
        the equity leverage shape, and an FX pair read on its `FxRate` axis wants the other. Widening
        Gamma_Star across zero instead is not available: Alpha = l*psi/Gamma_Star^2 is singular
        there. So x[3] is the magnitude, bounded away from zero, and x[2] a signed share in [-1, 1]:
        Alpha = |l|*psi/Gamma_Star^2, Beta = psi*(1-|l|). The price is continuous across l = 0, where
        Alpha is zero and Gamma_Star has no effect - which is what a flat surface reports.
        """
        psi, share, magnitude = x[1], x[2], x[3] * 1000.0
        gamma = torch.where(share < 0.0, -magnitude, magnitude)
        lev = torch.abs(share)
        return torch.exp(x[0]), lev * psi / gamma ** 2, psi * (1.0 - lev), gamma, torch.exp(x[4])

    @staticmethod
    def unreparam(omega, alpha, beta, gamma, h0):
        """Inverse of reparam (used to warm start off an existing price factor)."""
        psi = beta + alpha * gamma ** 2
        share = alpha * gamma ** 2 / psi
        return np.array([np.log(omega), psi, -share if gamma < 0.0 else share,
                         abs(gamma) / 1000.0, np.log(h0)])

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
    fx_wing_pillar = 0.25
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

        THE LADDER: ten vega-weighted implied vols read off the surface - ATM at 1M, 2M, 3M, 6M, 9M
        and 1Y, plus 25 delta wings at 3M and 6M - normalised by Black vega off the same surface.
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
        `K = F exp(-sigma^2 T/2)`, each wing the strike whose premium-adjusted forward delta is the
        pillar, found by inverting the delta the Malz solve inverted off the same vols.

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
                'family can fit, because an FxRate is priced in the domestic currency. Author the '
                'HestonNandiModelPrices block by hand, naming the Underlying and its '
                'Discount_Rate/Yield explicitly'.format(pair, domestic))
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

        for expiry in cls.fx_wing_expiries:
            found = pillar(expiry)
            if found is None:
                substituted.append('{:g}d {:g} DROPPED - no pillar at or under {:g}'.format(
                    cls.fx_wing_pillar, expiry, cap))
                continue
            T, moved, days, t, forward, rate, vol_at = found
            x_atm, _ = cls.fx_atm_coordinate(vol_at, T)
            for side in (1.0, -1.0):
                x = cls.fx_pillar_coordinate(vol_at, T, cls.fx_wing_pillar, side, x_atm)
                quote(days, forward, rate, t, x, vol_at(x))
            if moved:
                substituted.append('{:g}d {:g} -> {:g}'.format(cls.fx_wing_pillar, expiry, T))

        # a repeated contract is a weight rather than an observation, so what is counted is the
        # number of DISTINCT (expiry, strike) contracts
        contracts = {(point['Expiry_Date'], point['Strike']) for point in quotes}
        if len(contracts) < cls.fx_minimum_contracts:
            raise ValueError(
                '{} carries pillars {} - the ladder (ATM {}, {:g}d wings {}) collapses onto {} '
                'distinct contract{} on it, and {} do not identify {}, and a collapsed ladder has '
                'no term structure in it. Quote the pair at more expiries (at least {} distinct '
                'contracts, so at least three pillars at or under {:g}), or author the '
                '{} block by hand. What each rung did: {}'.format(
                    vol_name, '/'.join('{:g}'.format(x) for x in surface.expiry),
                    '/'.join('{:g}'.format(x) for x in cls.fx_atm_expiries), cls.fx_wing_pillar,
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

        source = '{} ATM {} + {:g}d wings {}, off {} as at {}'.format(
            len(quotes), '/'.join('{:g}'.format(x) for x in cls.fx_atm_expiries),
            cls.fx_wing_pillar, '/'.join('{:g}'.format(x) for x in cls.fx_wing_expiries), vol_name,
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
                # must be this number - so it is stated, and read off the field's own declaration
                'Steps_Per_Year': declared['Steps_Per_Year'],
                'Quadrature_Panels': declared['Quadrature_Panels'],
                'Quote_Timestamp': price_factors[vol_name].get('Quote_Timestamp') or '',
                'Quote_Source': source,
                'European_Options': quotes}}

    @staticmethod
    def price(spot, strike, is_call, units, omega, alpha, beta, gamma, r, n, h0, panels, yield_discount=1.0):
        """Heston-Nandi European option value - puts by put-call parity off the call.

        ``r`` is the per-step cost of carry r-q and ``yield_discount`` = exp(-q*t) converts the
        internal price exp(-(r-q)t)[F P1 - K P2] back to a value discounting at r. Parity survives
        the rescale, so puts are still call - S + K exp(-(r-q)n) times the same factor."""
        call = utils.hn_call(spot, strike, n, h0, omega, alpha, beta, gamma, r, panels=panels)
        return units * yield_discount * (call - (1.0 - is_call) * (spot - strike * torch.exp(-r * n)))

    def calc_error(self, x, groups, spot, panels, scale):
        """Weighted squared premium error and its exact gradient (autograd).

        ``scale`` is the mean squared quoted premium: L-BFGS-B's gradient tolerance is ABSOLUTE, so
        without it the fit would stop early on a low priced underlying (an fx rate) and late on a
        high priced one. Dividing by a constant leaves the relative Weights untouched."""
        x_t = torch.tensor(x, device=self.device, dtype=self.prec, requires_grad=True)
        omega, alpha, beta, gamma, h0 = self.reparam(x_t)
        error = 0.0
        for n, b, q, strike, is_call, units, weight, premium in groups:
            fitted = self.price(spot, strike, is_call, units,
                                omega, alpha, beta, gamma, b, n, h0, panels, q)
            error = error + (weight * (premium - fitted) ** 2).sum() / scale
        error.backward()
        return float(error.detach()), x_t.grad.cpu().numpy()

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

    def bootstrap(self, sys_params, price_models, price_factors, factor_interp, market_prices, calendars, debug=None):
        '''
        Calibrates the risk neutral Heston-Nandi GARCH(1,1) parameters to a set of European options
        on any spot underlying and writes a HestonNandiModelParameters price factor.

        `resolve_references` resolves every reference this quote type reads before an option is
        looked at, and a missing one refuses by name.
        '''

        def tensor(x):
            return torch.tensor(x, device=self.device, dtype=self.prec)

        for market_price, implied_params in market_prices.items():
            rate = utils.check_rate_name(market_price)
            market_factor = utils.Factor(rate[0], rate[1:])

            if market_factor.type == self.market_factor_type:
                instrument = implied_params['instrument']

                # the spot and discount curve, plus whatever else this quote type reads
                factors = self.resolve_references(
                    market_price, instrument, price_factors, factor_interp)
                underlying, discount = factors['Underlying'], factors['Discount_Rate']
                carry, funding = factors.get('Yield'), factors.get('Funding_Rate')
                vol_surface = factors.get('Volatility')
                if vol_surface is not None:
                    vol_surface.delta = sys_params.get('Volatility_Delta', 0.0)

                spot = float(underlying.current_value()[0])
                quote_type = instrument['Quote_Type']
                steps_per_year = instrument.get('Steps_Per_Year', 252.0)
                panels = instrument.get('Quadrature_Panels', 64)
                use_forward = instrument.get('Use_Forward') == 'Yes'
                invert_moneyness = instrument.get('Invert_Moneyness') == 'Yes'

                # a mis-looked-up vol converges to the wrong answer, so refuse rather than guess
                if quote_type == 'Implied_Volatility':
                    subtype = vol_surface.get_subtype()
                    if subtype[0] not in self.tabular_surfaces:
                        raise ValueError(
                            '{0}: volatility {1} has Surface_Type {2} (Moneyness_Rule {3}); only '
                            '{4} surfaces carry a vol AT A STRIKE, which is what Quote_Type '
                            'Implied_Volatility prices its target premium at. Quote the premiums '
                            'directly (Quote_Type Premium) instead'.format(
                                market_price, instrument['Volatility'], subtype[0], subtype[1],
                                '/'.join(self.tabular_surfaces)))

                # grouped by expiry, so one expiry's strikes share a characteristic function
                expiries = {}
                for option in instrument['European_Options']:
                    t = discount.get_day_count_accrual(
                        sys_params['Base_Date'], (option['Expiry_Date'] - sys_params['Base_Date']).days)
                    r = float(discount.current_value(t))
                    q = self.effective_yield(r, funding, carry, t)
                    forward = spot * np.exp((r - q) * t)
                    sign = 1.0 if option['Option_Type'] == 'Call' else -1.0
                    option['Strike'] = forward if not option['Strike'] else option['Strike']
                    option['r'] = r
                    option['q'] = q
                    option['T'] = t
                    # GARCH steps to expiry; the carry is spread so exp(-b_step*n) is exp(-(r-q)*t)
                    option['n'] = max(int(round(t * steps_per_year)), 1)
                    if quote_type == 'Implied_Volatility':
                        moneyness = self.moneyness(
                            option['Strike'], spot, forward, vol_surface, use_forward, invert_moneyness)
                        sigma = vol_surface.current_value([[moneyness, t]])[0] if not option[
                            'Quoted_Market_Value'] else option['Quoted_Market_Value']
                        sigma += vol_surface.delta
                        option['Moneyness'] = moneyness
                        option['Premium'] = utils.black_european_option_price(
                            forward, option['Strike'], r, sigma, t, option['Units'], sign)
                    else:
                        option['Premium'] = option['Units'] * option['Quoted_Market_Value']
                        # back out the Black vol of the quote (seeds the fit and the diagnostics)
                        call = option['Quoted_Market_Value'] + (0.0 if sign > 0 else
                                                                forward - option['Strike']) * np.exp(-r * t)
                        sigma = np.sqrt(utils.bs_implied_total_var(
                            call, spot * np.exp(-q * t), option['Strike'], r * t, 1) / t)
                    option['sigma'] = sigma
                    expiries.setdefault(option['n'], []).append(option)

                groups = [(n, tensor((opts[0]['r'] - opts[0]['q']) * opts[0]['T'] / n),
                            tensor(np.exp(-opts[0]['q'] * opts[0]['T'])),
                            tensor([x['Strike'] for x in opts]),
                            tensor([1.0 if x['Option_Type'] == 'Call' else 0.0 for x in opts]),
                            tensor([x['Units'] for x in opts]),
                            tensor([x['Weight'] for x in opts]),
                            tensor([x['Premium'] for x in opts])) for n, opts in expiries.items()]

                price_param = utils.Factor(self.__class__.__name__, market_factor.name)
                param_name = utils.check_tuple_name(price_param)
                if param_name in price_factors:
                    # warm start off the previous fit
                    old = price_factors[param_name]
                    x0 = np.clip(self.unreparam(*(old[k] for k in utils.HN_PARAM_NAMES)),
                                 *np.array(self.bounds).T)
                else:
                    var = np.mean([x['sigma'] for opts in expiries.values()
                                   for x in opts]) ** 2 / steps_per_year
                    # the sign is seeded off the quotes: the objective kinks at zero leverage, and a
                    # smile rising with strike in the underlying's units is a negative Gamma_Star
                    rise = sum(max(opts, key=lambda o: o['Strike'])['sigma'] -
                               min(opts, key=lambda o: o['Strike'])['sigma']
                               for opts in expiries.values() if len(opts) > 1)
                    x0 = np.array([np.log(0.1 * var), 0.9, -0.5 if rise > 0.0 else 0.5, 0.1,
                                   np.log(var)])

                scale = np.mean([x['Premium'] ** 2 for opts in expiries.values() for x in opts])
                result = scipy.optimize.minimize(
                    self.calc_error, x0, args=(groups, spot, panels, scale), jac=True,
                    method='L-BFGS-B', bounds=self.bounds,
                    # the defaults suit an O(1e2) objective; the normalised one starts at O(1) and
                    # a good fit is O(1e-12)
                    options={'ftol': 1e-15, 'gtol': 1e-12})

                omega, alpha, beta, gamma, h0 = [
                    float(x) for x in self.reparam(tensor(result.x))]

                # log the results
                with torch.no_grad():
                    for n, b, q, strike, is_call, units, weight, premium in groups:
                        pt = [tensor(x) for x in (omega, alpha, beta, gamma)]     # the four recursion params
                        fitted = self.price(spot, strike, is_call, units, *pt, b, n, tensor(h0), panels, q)
                        for option, fitted_premium in zip(expiries[n], fitted.cpu().numpy()):
                            vol = utils.hn_implied_vol(
                                spot, option['Strike'], n, tensor(h0), *pt, b, steps_per_year, panels=panels)
                            logging.info(
                                'Underlying {} strike {}, expiry {}, steps {}, vol {}, c_vol {}, premium {}, '
                                'c_premium {}, err {}'.format(
                                    instrument['Underlying'], option['Strike'], option['Expiry_Date'], n,
                                    option['sigma'], vol, option['Premium'], fitted_premium,
                                    (fitted_premium - option['Premium']) ** 2))

                logging.info(
                    'Underlying {} Heston-Nandi Omega {}, Alpha {}, Beta {}, Gamma_Star {}, H0 {}, '
                    'persistence {}, long run vol {}, sse {} ({})'.format(
                        instrument['Underlying'], omega, alpha, beta, gamma, h0,
                        utils.hn_persistence(alpha, beta, gamma),
                        utils.hn_ann_vol(omega, alpha, beta, gamma, steps_per_year),
                        result.fun, result.message))
                # where the quotes came from, in the record beside the parameters they produced
                if instrument.get('Quote_Source') or instrument.get('Quote_Timestamp'):
                    logging.info('  quotes: {} (as at {})'.format(
                        instrument.get('Quote_Source') or 'authored by hand',
                        instrument.get('Quote_Timestamp') or 'no stated time'))

                # canonical HN_PARAM_NAMES order, paired with reparam's output tuple
                price_factors[param_name] = {
                    'Property_Aliases': None,
                    **dict(zip(utils.HN_PARAM_NAMES, (omega, alpha, beta, gamma, h0)))}


class HestonNandiComponentModelParameters(HestonNandiModelParameters):
    documentation = (
        'Fx And Equity',
        ['The COMPONENT Heston-Nandi model of Christoffersen, Jacobs, Ornthanalai and Wang splits the',
         'variance into a long-run component $q_t$ and a short-run deviation. Under the LRNVR measure',
         '',
         '$$h_{t+1}=q_{t+1}+\\beta(h_t-q_t)+\\alpha\\Big[(z_t-\\gamma_1\\sqrt{h_t})^2-'
         '(1+\\gamma_1^2h_t)\\Big]$$',
         '',
         '$$q_{t+1}=\\omega_t+\\rho q_t+\\phi\\Big[(z_t-\\gamma_2\\sqrt{h_t})^2-(1+\\gamma_2^2h_t)'
         '\\Big]$$',
         '',
         'Both bracketed terms are EXACTLY centered, so $h_t-q_t$ is a pure AR(1) at $\\beta$ and',
         '$E_t[q_{t+k}]$ is driven by $\\omega$ alone. Setting $\\phi=0$ and holding $q$ flat recovers',
         'plain Heston-Nandi exactly, with $\\beta$ its persistence $\\psi$ and the flat level its',
         'stationary variance - so this family is a strict extension of *HestonNandiModelPrices*.',
         '',
         'THE L CURVE. The intercept is parametrised by a curve rather than a constant:',
         '$\\omega_t=L_{t+1}-\\rho L_t$. Then $q_t-L_t$ is a homogeneous AR(1), so ANCHORING',
         '$q_0=L(0)$ gives $E_0[q_t]=L_t$ exactly - the fitted $L$ IS the model\'s expected long-run',
         'variance path and is directly comparable to the market\'s forward variance strip. $L$ is',
         'piecewise-linear in $t$ between pillar knots and flat outside them, so $\\omega_t$ is',
         'AFFINE within a pillar - it drifts by $(B-A)(1-\\rho)/n$ per step over a segment of $n$',
         'steps from $A$ to $B$ - and KINKS only at one. The curve carries a knot at tenor 0 whose',
         'value is $h_0$: at the base date the two states are held equal, because no option is quoted',
         'at zero maturity to separate them, and that knot is what makes $q_0=L(0)$ a property of the',
         'written factor rather than a convention.',
         '',
         'THE FIT IS TWO NESTED SOLVES, because the two halves of the surface identify two different',
         'things.',
         '',
         '1. THE INNER TRIANGULAR BOOTSTRAP. Given candidate globals, the $L$ pillars are solved',
         'SEQUENTIALLY, each against its own ATM expiry\'s premium, by *brentq* on the pillar level.',
         'An option to $T$ never reads $L$ beyond $T$, so the system is exactly triangular; and the',
         'price is monotone in the pillar\'s level (raising it raises $\\omega_t$ over that segment,',
         'hence $E[h_t]$, hence the premium), so a bracketed root is unique. A pillar with no',
         'admissible level REFUSES BY NAME with the bracket it searched.',
         '',
         '2. THE OUTER FIT concentrates $L$ out: the skew globals are fitted to the WING quotes with',
         'the whole $L$ strip re-bootstrapped at every iterate, so every candidate reprices the ATM',
         'term structure exactly and is judged only on the smile. It inherits the plain family\'s',
         'SIGN-FREE LEVERAGE REPARAMETRISATION - $\\alpha=|l|\\beta/\\gamma_1^2$, $\\gamma_1=\\mathrm{sgn}',
         '(l)|\\gamma_1|$ for a signed share $l\\in[-1,1]$ - so both skew directions live in one box and',
         '$\\beta(1-|l|)\\ge0$ keeps the variance recursion positive. $\\phi$ is fitted as a SHARE of',
         '$\\alpha$ (same units, and the share is scale-free), and the search is derivative-free',
         '(Nelder-Mead): the inner solve is a root find, so no gradient passes through it.',
         '',
         'TWO PINS, both declared rather than hidden. *Rho* is PINNED (default 0.99 per step): the',
         'L-parametrisation has evicted $\\rho$ from the ATM fit - $L$ hits the term structure',
         'whatever $\\rho$ is - into the smile\'s term structure alone, and sub-year wings do not',
         'identify it. *Tie_Gamma_2* holds $\\gamma_2=\\gamma_1$ by default; set it to **No** to fit the',
         'long-run leverage separately, which needs a wing ladder deep enough to tell the two apart.',
         '',
         'THE NEGATIVE-OMEGA GUARD. A pillar demanding $L$ to fall FASTER than $\\rho$ decays it makes',
         '$\\omega_t<0$, which drives $q$ - and then $h$ - negative. *Declining_Variance* decides:',
         '**Refuse** (default) names the pillar, the level it wanted and the least admissible one;',
         '**Floor** takes that least admissible level and says so. There is no silent third option.',
         '',
         'THE LADDER is the plain family\'s, WIDENED AT THE WINGS: the same six ATM rungs, plus 25',
         'delta wings at 1M, 3M, 6M and 1Y rather than 3M and 6M alone. Six globals reduce to five',
         'free ones under the two pins, and five free globals judged on the smile alone want more',
         'than four wing quotes - the ATM rungs are spent on the $L$ pillars and identify nothing',
         'else. Everything else - the vega weights, the surface\'s own strikes, nothing past 1Y, the',
         'substitution note - is inherited unchanged from *HestonNandiModelPrices*.',
         '',
         '*Quote_Sensitivity* is REFUSED on this family. The quote derivative would have to pass',
         'through the inner root find by the implicit function theorem, which is real work and is',
         'not built; a family that answered zeros would be worse than one that says so.'
         ]
    )

    market_factor_type = 'HestonNandiComponentModelPrices'

    #: The fit runs on the CPU whatever device the job was constructed with. The A/B/C recursion is
    #: `n` sequential steps of about ten elementwise operations over a 512-element complex vector,
    #: so it is kernel-launch bound. On an RTX 3090, one 126-step price: 47 ms CPU against 186 ms
    #: CUDA, and the adaptive phi_max scan 172 ms against 775 ms.
    device = torch.device('cpu')

    def __init__(self, param, device, dtype):
        # the constructed device is ignored, as the constructed dtype is - see the `device` note
        self.param = param

    #: Four wing expiries rather than the plain family's two: five free globals are judged on the
    #: smile alone, the ATM rungs being spent on the L pillars. The ATM rungs, the cap and the
    #: snapping rule are inherited unchanged.
    fx_wing_expiries = (1.0 / 12.0, 0.25, 0.5, 1.0)
    #: The ATM ladder is a CONSTRAINT, not a term: concentrating L out presumes the term structure
    #: is hit exactly, so no smile improvement may pay for a pillar on the declining-variance floor.
    #: The floor's relative miss enters at this weight, on the same scale as the normalised wing
    #: residual - at 1e4 a one basis point ATM miss costs a good fit's whole smile residual. At
    #: weight 1 the simplex settled 0.63% inside the infeasible region on the USDZAR fixture.
    atm_constraint_weight = 1.0e4

    #: More contracts than the plain family's six, the ATM rungs being consumed by the bootstrap: a
    #: ladder whose wings collapse onto one expiry has no smile term structure in it.
    fx_minimum_contracts = 8
    identification_note = ('five free globals off the smile: the ATM term structure is spent on '
                           'the L pillars, which are bootstrapped rather than fitted')

    #: x = (beta, signed leverage share, ARCH share of the level's own room, phi share of alpha,
    #: log H0) - see `reparam`. beta is the short-run persistence, bounded below 1 because it is the
    #: plain family's psi under the exact nesting map.
    bounds = [(1e-4, 1.0 - 1e-6), (-1.0, 1.0), (1e-3, 1.0 - 1e-6), (0.0, 1.0),
              (np.log(1e-10), np.log(1e-2))]

    fields = HestonNandiModelParameters.fields[:-1] + [
        F('Rho', 'Float', default=0.99,
          description='PINNED long-run persistence per step, 0 <= Rho < 1 and REFUSED outside it '
                      '(q is an AR(1) at rho: at rho >= 1 the long-run component is '
                      'non-stationary and the negative-omega floor turns negative, which disables '
                      'the guard). The L parametrisation evicts rho from '
                      'the ATM fit - L reprices the term structure whatever rho is - into the '
                      'smile\'s own term structure, and sub-year wings under-identify it. Declared '
                      'so a desk that has a view can state it, not so the fit can wander'),
        F('Tie_Gamma_2', 'Text', default='Yes', values=['Yes', 'No'],
          description='Hold the long-run leverage equal to the short-run one. No fits Gamma_2 '
                      'separately, which needs wings deep enough to tell the two apart'),
        F('Declining_Variance', 'Text', default='Refuse', values=['Refuse', 'Floor'],
          description='What a pillar demanding L to fall faster than rho decays it does. Refuse '
                      'names the pillar; Floor takes the least admissible level and says so. '
                      'Never a silent negative variance'),
        F('Max_Iterations', 'Integer', default=300,
          description='Outer (Nelder-Mead) function evaluations. The objective re-bootstraps the '
                      'whole L strip per iterate and every price derives its own quadrature bound, '
                      'so this is THE wall-clock knob: measured at 4.79 s an evaluation on the '
                      'four-pillar USDZAR ladder, which puts 300 at 24 minutes and 400 at 32. The '
                      'default is the largest that fits the half hour; a fit that stops here '
                      'reports itself CAPPED with the residual it actually reached rather than the '
                      'tolerance it did not'),
        F('Tolerance', 'Float', default=1e-8,
          description='Outer convergence tolerance on the weighted premium residual'),
        F('Pillar_Tolerance', 'Float', default=1e-14,
          description='brentq tolerance on a pillar\'s L level, relative to its bracket'),
        F('Quote_Sensitivity', 'Text', default='No', values=['Yes', 'No'],
          description='REFUSED on this family: the quote derivative would have to pass through the '
                      'inner root find by the implicit function theorem, which is not built')
    ] + HestonNandiModelParameters.fields[-1:]

    # ----------------------------------------------------------------------------------
    # the parametrisation
    # ----------------------------------------------------------------------------------

    @staticmethod
    def reparam(x):
        """Maps the fitted vector ``x = (beta, signed leverage share, ARCH share, phi share,
        log H0)`` to (Alpha, Beta, Gamma_1, Phi, H0).

        Two shares, each making a different positivity constraint automatic, so no iterate needs a
        penalty. The box buys feasible ALGEBRA and not a finite PRICE: away from the nested face the
        moment generating function can still diverge, and there the phi_max scan caps and the
        objective reads the candidate as infeasible (+inf).

        * The leverage share l in [-1, 1] holds Alpha*Gamma_1^2 = |l|*Beta, so the plain-equivalent
          GARCH coefficient Beta(1-|l|) is non-negative. It carries Gamma_1's SIGN, so both skew
          directions live in one box - an FX pair read on its `FxRate` axis routinely wants the
          negative one, and a one-signed box answers such a surface with a flat converged smile.

        * The ARCH share a in (0, 1) holds Alpha = a*H0*(1-Beta), so the nested-face intercept
          H0(1-Beta)(1-a) is positive. The plain family gets this by fitting omega in logs; here the
          intercept is derived from the L curve, so an Alpha larger than the level's own room makes
          the variance recursion - and the MGF the pricer inverts - diverge.

        Gamma_1 is therefore DERIVED: sgn(l) sqrt(|l| Beta / Alpha). Its scale is set by the shares
        alone, and over the box it runs from 0 at l = 0 to 31,623 at the (Beta 1-1e-6, |l| 1,
        a 1e-3) corner, which still prices finite. Landed fits read 0.56 to 7.4.

        Phi is a share of Alpha - the two multiply the same squared normal - and phi_share = 0 is
        exactly the nested face where this model is the plain one.

        No box gives positivity of the FULL recursion away from the nested face: h_{t+1} >= omega_t
        + (rho-beta) q_t + [beta(1-|l|) - phi gamma_2^2] h_t - alpha - phi has no sign for free once
        rho != beta and phi > 0. It fails loudly - a divergent MGF caps the phi_max scan and reads
        as infeasible, and a negative h in the simulator is a NaN out of sqrt.
        """
        beta, share, arch = x[0], x[1], x[2]
        h0 = torch.exp(x[4])
        alpha = arch * h0 * (1.0 - beta)
        lev = torch.abs(share)
        magnitude = torch.sqrt(lev * beta / alpha)
        gamma1 = torch.where(share < 0.0, -magnitude, magnitude)
        return alpha, beta, gamma1, x[3] * alpha, h0

    @staticmethod
    def unreparam(alpha, beta, gamma1, phi, h0):
        """Inverse of reparam (used to warm start off an existing price factor)."""
        share = alpha * gamma1 ** 2 / beta
        return np.array([beta, -share if gamma1 < 0.0 else share,
                         alpha / (h0 * (1.0 - beta)),
                         phi / alpha if alpha else 0.0, np.log(h0)])

    @staticmethod
    def worst_case_variance_drift(alpha, beta, gamma1, rho, phi, gamma2, omega_min):
        """The deterministic lower bound on one step's SHORT-run variance, as
        `(intercept, q_loading, h_slope)`:

            h_{t+1} >= intercept + q_loading * q_t + h_slope * h_t

        Both quadratics are dropped. Substituting the q step into the h step leaves
        `alpha(z - gamma_1 sqrt h)^2 + phi(z - gamma_2 sqrt h)^2` on top of an affine part, and no
        single innovation zeroes both unless gamma_1 = gamma_2 - so this is the looser of the two
        bounds, still a bound. The centering terms cancel the quadratics' own h coefficients.
        `omega_min` is the smallest intercept over the whole horizon, strip and tail (`omega_floor`).

        Three non-negative numbers certifies h GIVEN q >= 0, and nothing more. q has no certificate
        of its own whenever phi*gamma_2^2 > 0, so a fit with a live long-run ARCH channel cannot be
        certified positive at all - a CJOW property, not this parametrisation's. The simulator's
        `utils.HN_COMPONENT_VARIANCE_FLOOR` is what keeps a tail path finite.
        """
        return (float(omega_min) - float(alpha) - float(phi),
                float(rho) - float(beta),
                float(beta) - float(alpha) * float(gamma1) ** 2
                - float(phi) * float(gamma2) ** 2)

    @staticmethod
    def admissible_level(previous, days, rho):
        """The LEAST pillar level whose segment keeps omega_t >= 0 - the declining-variance floor.

        On a segment of `days` steps running linearly from `previous` to a level B,
        L_i = A + (B-A)i/n and omega_i = A(1-rho) + (B-A)(1 + i(1-rho))/n, which is increasing in B
        and (for a FALLING segment) smallest at the last step. Setting that to zero gives

            B_min = A * (1 - (1-rho)*n / (1 + (n-1)(1-rho)))

        a closed form rather than a search, so the refusal can name the number it wanted. A rising
        segment has its minimum at i = 0 and is admissible whenever A(1-rho) >= 0, which it is.

        The margin is one part in 1e9 above the exact crossing: the level is written to a Curve and
        the omega strip is rebuilt from it in a different order, so the exact crossing comes back an
        ulp either side of zero (-2.7e-20 on the humped fixture), which still reads as negative.
        """
        gap = 1.0 - rho
        exact = float(previous) * (1.0 - gap * days / (1.0 + (days - 1) * gap))
        return exact * (1.0 + 1e-9) if exact > 0.0 else exact

    # ----------------------------------------------------------------------------------
    # pricing + the inner triangular bootstrap
    # ----------------------------------------------------------------------------------

    @classmethod
    def price(cls, spot, strike, is_call, units, omegas, h0, q0, params, r, panels,
              yield_discount=1.0):
        """Component European option value - puts by put-call parity off the call, exactly as the
        plain family's `price` does, and with the same `yield_discount` rescale (the internal price
        discounts at the carry r-q; the value discounts at r, and parity survives the rescale)."""
        # no phi_max knob: every price derives its own quadrature bound, a reused one not being
        # conservative for this model - see `bootstrap_l` and `utils.hn_component_auto_phi_max`
        call = utils.hn_component_call(spot, strike, omegas, h0, q0, *params, r, panels=panels)
        n = len(omegas)
        return units * yield_discount * (
            call - (1.0 - is_call) * (spot - strike * torch.exp(-r * n)))

    @classmethod
    def l_strip(cls, knots, levels, steps, rho, spy):
        """The omega strip over `steps` daily steps from the (knots, levels) L curve."""
        l_path = utils.hn_component_l_path(knots, levels, steps, spy)
        return list(utils.hn_component_omega_path(l_path, rho))

    @classmethod
    def omega_floor(cls, knots, levels, rho, spy):
        """The smallest omega_t over the whole horizon - the strip between the knots and the flat
        tail past the last one, where omega = L_last(1-rho). The certificate's `omega_min`.

        The tail is routinely the minimum: on a rising segment of n steps the least intercept is
        below the tail's only while n > 1/(1-rho), 100 steps at the pinned rho. On
        L = [9e-5, 9.9e-5, 1e-4] at knots 0/0.25/0.5y a strip-only read gives 1.006e-6 against
        1.000e-6 true.
        """
        strip = cls.l_strip(knots, levels, max(int(round(float(knots[-1]) * spy)), 1), rho, spy)
        return min([float(x) for x in strip] + [float(levels[-1]) * (1.0 - float(rho))])

    def bootstrap_l(self, atm, params, h0, rho, spy, panels, knots, declining, tolerance,
                    refuse=True):
        """The inner triangular bootstrap: the L pillars, solved one at a time against their own ATM
        premium. Returns `(levels, notes, shortfall)` - the level at each knot (levels[0] is
        L(0) = h0 by the anchoring), any floors applied by name, and the summed squared relative
        premium miss on the floored pillars (zero when every pillar solved).

        Triangular because the model is: an option to T reads L only on [0, T], so pillar k's
        premium is a function of pillars 0..k and nothing later, and each is a one-dimensional root
        find. Monotone in the pillar's level, so brentq is the tool; the bracket starts around the
        previous pillar and doubles out, and one that runs out refuses by name with what it searched.

        `refuse` separates the search from the answer. Inside the outer optimizer the floor is taken
        and the miss returned as a shortfall the objective adds - a simplex walks into a box corner
        routinely, and a shortfall is a slope out of the infeasible region where `inf` is a wall. On
        the FINAL strip `Declining_Variance` decides for real.

        Every price derives its own phi_max, at about 4x the cost of one reused scan, because a
        reused bound is not conservative: past a parameter-dependent point the A/B/C recursion
        diverges and an over-large bound integrates garbage. At one converged optimum the 21-step
        contract's bound is 512 and the 126-step contract's 256, and that 126-step price reads
        0.7353321384 at phi_max 512, 0.7323069671 at 1024 and 9.4e+55 at 2048.

        The floor binds even on a rising term structure. A piecewise-linear L matched to segment
        integrals is the recurrence L_k = 2*A_k - L_(k-1), whose multiplier is -1: marginally stable,
        so an error in L(0) alternates in sign and never decays. H0 sets the PHASE of the strip,
        which is what identifies it here - the outer fit's smile residual being the other half.
        """
        levels, notes, shortfall = [h0], [], 0.0
        for k, (n, quote) in enumerate(atm):
            days = int(n - (0 if not k else atm[k - 1][0]))
            floor = self.admissible_level(levels[-1], days, float(rho))
            spot, strike, is_call, units, target, b, q = quote
            if not target:
                # every reading of this pillar is relative to its own premium, so a zero divides
                raise ValueError(
                    'HestonNandiComponent: the {:g}y ATM pillar quotes a premium of {:g}, and a '
                    'pillar with no premium cannot anchor an L level - the bootstrap solves each '
                    'level against its own quote and reports every miss relative to it. Drop that '
                    'rung, or quote it at a positive vol (Quote_Type Implied_Volatility) or a '
                    'positive premium'.format(n / spy, target))

            def premium(level):
                omegas = self.l_strip(knots[:k + 2], torch.stack(levels + [level]),
                                      int(n), rho, spy)
                return float(self.price(spot, strike, is_call, units, omegas, h0, levels[0],
                                        params, b, panels, q))

            low = max(floor, 1e-12)
            high = max(float(levels[-1]) * 2.0, low * 4.0)
            error = lambda x: premium(self.tensor(x)) - target
            lo_err = error(low)
            if lo_err > 0.0:
                # the pillar wants less variance than the floor admits - named on every path out
                message = (
                    'HestonNandiComponent: the {:g}y ATM pillar demands a long-run variance BELOW '
                    '{:.6g}, the least level whose segment keeps omega_t = L_(t+1) - rho*L_t '
                    'non-negative from {:.6g} over {} steps at rho={:g}. Its premium there is '
                    '{:.6g} against a target of {:.6g} ({:+.2%}). A negative omega drives the '
                    'long-run component - and then the variance - negative, so this refuses rather '
                    'than floors: set Declining_Variance to Floor to take {:.6g} and have the fit '
                    'say so, or lower Rho so the level is allowed to decay faster'.format(
                        n / spy, low, float(levels[-1]), days, float(rho),
                        lo_err + target, target, lo_err / target, low))
                if refuse and declining == 'Refuse':
                    raise ValueError(message)
                notes.append('pillar {:g}y FLOORED at {:.6g} - its own level would need L to fall '
                             'faster than rho={:g} decays it; the pillar reprices {:+.2%}'.format(
                                 n / spy, low, float(rho), lo_err / target))
                shortfall += (lo_err / target) ** 2
                levels.append(self.tensor(low))
                continue
            expansions = 0
            while error(high) < 0.0:
                high *= 2.0
                expansions += 1
                if expansions > 40:
                    raise ValueError(
                        'HestonNandiComponent: no long-run variance level reprices the {:g}y ATM '
                        'pillar - its premium is still {:.6g} below the target {:.6g} at a level '
                        'of {:.6g} (annualised vol {:.1%}), searched up from {:.6g}. The quote is '
                        'not reachable under these globals; check the ATM vol on that rung'.format(
                            n / spy, -error(high), target, high, float(np.sqrt(high * spy)), low))
            levels.append(self.tensor(scipy.optimize.brentq(
                error, low, high, xtol=tolerance * max(high, 1e-12), rtol=8.9e-16)))
        return levels, notes, shortfall

    def tensor(self, x):
        return torch.tensor(float(x), device=self.device, dtype=self.prec)

    # ----------------------------------------------------------------------------------
    # the outer fit
    # ----------------------------------------------------------------------------------

    def bootstrap(self, sys_params, price_models, price_factors, factor_interp, market_prices,
                  calendars, debug=None):
        """Calibrates the component Heston-Nandi parameters and writes them out as a
        `HestonNandiComponentModelParameters` price factor.

        The quote preparation is the plain family's, quote for quote. The quotes then SPLIT into an
        ATM ladder the L bootstrap consumes exactly and a wing ladder the outer fit is judged on.

        The ATM quote at an expiry is the one nearest its own forward, so a hand-authored block
        reads the same way as an emitted one.

        One ATM equation per expiry, and repeated wings stay as WEIGHT. A thin surface snaps several
        rungs onto one pillar: on the ATM side that would solve the same pillar against itself, so
        the group answers one equation; on the wing side a repeat is the heavier weight the
        emitter's normalisation intends.
        """
        for market_price, implied_params in market_prices.items():
            rate = utils.check_rate_name(market_price)
            market_factor = utils.Factor(rate[0], rate[1:])
            if market_factor.type != self.market_factor_type:
                continue
            instrument = implied_params['instrument']

            if instrument.get('Quote_Sensitivity', 'No') == 'Yes':
                raise Exception(
                    'Quote_Sensitivity: {} concentrates the L curve out through a bracketed root '
                    'find (brentq), which carries no derivative. Propagating a quote tick would '
                    'need the implicit function theorem across the inner solve AND the outer '
                    'Nelder-Mead, which is not built - the IFT half is a roadmap row. Set '
                    'Quote_Sensitivity to No. The plain HestonNandiModelPrices family is NOT the '
                    'remedy - it declares no Quote_Sensitivity field at all; the differentiable '
                    'quote chains are the surface and curve families (FXVolPrices, '
                    'InterestRatePrices, GBMAssetPriceTSModelPrices, HullWhite2FactorModelPrices), '
                    'which solve through torch rather than through brentq'.format(market_price))

            # resolved before an option is looked at; a missing one refuses by name
            factors = self.resolve_references(
                market_price, instrument, price_factors, factor_interp)
            underlying, discount = factors['Underlying'], factors['Discount_Rate']
            carry, funding = factors.get('Yield'), factors.get('Funding_Rate')
            vol_surface = factors.get('Volatility')
            if vol_surface is not None:
                vol_surface.delta = sys_params.get('Volatility_Delta', 0.0)

            spot = float(underlying.current_value()[0])
            quote_type = instrument['Quote_Type']
            # every default read off the field's own declaration, so the two cannot disagree
            declared = {field.name: field.default for field in self.fields}
            spy = float(instrument.get('Steps_Per_Year', declared['Steps_Per_Year']))
            panels = instrument.get('Quadrature_Panels', declared['Quadrature_Panels'])
            use_forward = instrument.get('Use_Forward') == 'Yes'
            invert_moneyness = instrument.get('Invert_Moneyness') == 'Yes'
            rho = float(instrument.get('Rho', declared['Rho']))
            if not 0.0 <= rho < 1.0:
                raise ValueError(
                    'Rho: {} declares Rho={:g}, which is outside [0, 1). q_t is an AR(1) at Rho, '
                    'so at Rho >= 1 the long-run component is NON-STATIONARY - E_0[q_t] = L_t no '
                    'longer holds and the L curve stops meaning the expected variance path it is '
                    'fitted as - and the least admissible level '
                    'A(1 - (1-Rho)n/(1 + (n-1)(1-Rho))) goes NEGATIVE, so max(floor, 1e-12) '
                    'silently disables the negative-omega guard rather than tripping it. Pin Rho '
                    'in [0, 1); the declared default is {:g}'.format(
                        market_price, rho, declared['Rho']))
            rho = self.tensor(rho)
            tie = instrument.get('Tie_Gamma_2', declared['Tie_Gamma_2']) == 'Yes'
            declining = instrument.get('Declining_Variance', declared['Declining_Variance'])
            max_iter = int(instrument.get('Max_Iterations', declared['Max_Iterations']))
            tolerance = float(instrument.get('Tolerance', declared['Tolerance']))
            pillar_tol = float(instrument.get('Pillar_Tolerance', declared['Pillar_Tolerance']))

            if quote_type == 'Implied_Volatility':
                subtype = vol_surface.get_subtype()
                if subtype[0] not in self.tabular_surfaces:
                    raise ValueError(
                        '{0}: volatility {1} has Surface_Type {2} (Moneyness_Rule {3}); only {4} '
                        'surfaces carry a vol AT A STRIKE, which is what Quote_Type '
                        'Implied_Volatility prices its target premium at. Quote the premiums '
                        'directly (Quote_Type Premium) instead'.format(
                            market_price, instrument['Volatility'], subtype[0], subtype[1],
                            '/'.join(self.tabular_surfaces)))

            rows = []
            for option in instrument['European_Options']:
                t = discount.get_day_count_accrual(
                    sys_params['Base_Date'], (option['Expiry_Date'] - sys_params['Base_Date']).days)
                r = float(discount.current_value(t))
                q = self.effective_yield(r, funding, carry, t)
                forward = spot * np.exp((r - q) * t)
                sign = 1.0 if option['Option_Type'] == 'Call' else -1.0
                option['Strike'] = forward if not option['Strike'] else option['Strike']
                option['T'] = t
                option['n'] = max(int(round(t * spy)), 1)
                if quote_type == 'Implied_Volatility':
                    moneyness = self.moneyness(
                        option['Strike'], spot, forward, vol_surface, use_forward, invert_moneyness)
                    sigma = vol_surface.current_value([[moneyness, t]])[0] if not option[
                        'Quoted_Market_Value'] else option['Quoted_Market_Value']
                    sigma += vol_surface.delta
                    option['Premium'] = utils.black_european_option_price(
                        forward, option['Strike'], r, sigma, t, option['Units'], sign)
                else:
                    option['Premium'] = option['Units'] * option['Quoted_Market_Value']
                    call = option['Quoted_Market_Value'] + (0.0 if sign > 0 else
                                                            forward - option['Strike']) * np.exp(-r * t)
                    sigma = np.sqrt(utils.bs_implied_total_var(
                        call, spot * np.exp(-q * t), option['Strike'], r * t, 1) / t)
                option['sigma'] = sigma
                # the per-step carry and the yield rescale, as the plain family builds them
                rows.append((option, (r - q) * t / option['n'], np.exp(-q * t), forward))

            # the split: one ATM per distinct expiry (nearest its own forward), the rest wings
            by_expiry = {}
            for option, b, yq, forward in rows:
                by_expiry.setdefault(option['n'], []).append((option, b, yq, forward))
            atm_rows, wing_rows = [], []
            for n in sorted(by_expiry):
                group = by_expiry[n]
                pick = min(group, key=lambda row: abs(row[0]['Strike'] / row[3] - 1.0))
                atm_rows.append(pick)
                wing_rows += [row for row in group
                              if row[0]['Strike'] != pick[0]['Strike']]
            if not atm_rows:
                logging.error('{} carries no quotes - nothing to bootstrap'.format(market_price))
                continue

            # knots land on the step count each pillar is priced at, so the L path's day index and
            # the option's step count are one clock (see `utils.hn_component_l_path`)
            knots = np.array([0.0] + [row[0]['n'] / spy for row in atm_rows])
            atm = [(row[0]['n'],
                    (spot, self.tensor(row[0]['Strike']),
                     1.0 if row[0]['Option_Type'] == 'Call' else 0.0,
                     self.tensor(row[0]['Units']), row[0]['Premium'],
                     self.tensor(row[1]), self.tensor(row[2])))
                   for row in atm_rows]
            wings = [(row[0]['n'], self.tensor(row[0]['Strike']),
                      1.0 if row[0]['Option_Type'] == 'Call' else 0.0,
                      self.tensor(row[0]['Units']), row[0]['Premium'], row[0]['Weight'],
                      self.tensor(row[1]), self.tensor(row[2])) for row in wing_rows]
            if not wings:
                logging.warning(
                    '{} carries no wing quotes - the L bootstrap will reprice the ATM ladder '
                    'exactly and the skew globals are unidentified by it'.format(market_price))

            price_param = utils.Factor(self.__class__.__name__, market_factor.name)
            param_name = utils.check_tuple_name(price_param)
            x0 = self.seed(price_factors.get(param_name), atm_rows, by_expiry, spy, tie)

            scale = np.mean([w[4] ** 2 for w in wings]) if wings else 1.0
            calls = {'n': 0}
            state = {}

            def objective(x):
                """One outer iterate: re-bootstrap L, then score the wings.

                L is concentrated out, so the ATM ladder is repriced exactly at every candidate the
                bootstrap could solve and this is a pure smile residual. A candidate whose pillars
                hit the declining-variance floor carries that miss at `atm_constraint_weight`, a
                slope the simplex can walk down; one whose MGF diverges scores +inf.
                """
                calls['n'] += 1
                params = self.unpack(x, tie, rho)
                h0 = float(params[-1])
                try:
                    # notes are dropped here: what is reported is the FINAL strip's, bootstrapped
                    # at the parameters actually written
                    levels, _, shortfall = self.bootstrap_l(
                        atm, params[:-1], self.tensor(h0), rho, spy, panels, knots, declining,
                        pillar_tol, refuse=False)
                    error = self.atm_constraint_weight * shortfall
                    for n, strike, is_call, units, premium, weight, b, yq in wings:
                        omegas = self.l_strip(knots, torch.stack(levels), int(n), rho, spy)
                        fitted = float(self.price(spot, strike, is_call, units, omegas,
                                                  self.tensor(h0), levels[0], params[:-1], b,
                                                  panels, yq))
                        error += weight * (premium - fitted) ** 2 / scale
                except (ValueError, RuntimeError) as refusal:
                    state['last_refusal'] = str(refusal)
                    return np.inf
                if not np.isfinite(error):
                    state['last_refusal'] = 'the characteristic function diverged at this candidate'
                    return np.inf
                return error

            started = time.time()
            result = scipy.optimize.minimize(
                objective, x0, method='Nelder-Mead', bounds=self.box(tie),
                options={'maxfev': max_iter, 'xatol': 1e-10, 'fatol': tolerance})
            elapsed = time.time() - started

            params = self.unpack(result.x, tie, rho)
            alpha, beta, gamma1, phi, gamma2, h0 = [float(x) for x in (
                params[0], params[1], params[2], params[4], params[5], params[6])]
            # the final strip is bootstrapped at the reported parameters, not the last iterate the
            # simplex tried, and this is the call `Declining_Variance` decides for real
            levels, notes, _ = self.bootstrap_l(
                atm, params[:-1], self.tensor(h0), rho, spy, panels, knots, declining, pillar_tol)
            curve = utils.Curve([], [[float(k), float(v)] for k, v in zip(knots, levels)])

            self.report(instrument, market_price, spot, atm, wings, knots, levels, params,
                        rho, spy, panels, result, elapsed, calls['n'], notes, state, max_iter)

            price_factors[param_name] = {
                'Property_Aliases': None,
                **dict(zip(utils.HN_COMPONENT_PARAM_NAMES,
                           (alpha, beta, gamma1, float(rho), phi, gamma2, h0))),
                utils.HN_COMPONENT_CURVE_NAME: curve}

    def unpack(self, x, tie, rho):
        """The fitted vector as `(alpha, beta, gamma1, rho, phi, gamma2, h0)` - the six-parameter
        positional block every `utils.hn_component_*` function takes, plus h0.

        The pin and the tie live here rather than at each call site, so nothing downstream has to
        remember which two of the seven the fit did not move.

        Untied, the vector grows a sixth coordinate: Gamma_2 as a RATIO to Gamma_1, in [0, 5]. The
        magnitude is free - the long-run smile's own width - but the direction is not, a smile
        rising at one horizon and falling at another being a kink no sub-year ladder identifies.
        Ratio 1 is the tie, and where the untied fit starts."""
        x = torch.tensor(np.asarray(x, dtype=float), device=self.device, dtype=self.prec)
        alpha, beta, gamma1, phi, h0 = self.reparam(x)
        return alpha, beta, gamma1, rho, phi, gamma1 if tie else gamma1 * x[5], h0

    @classmethod
    def box(cls, tie):
        """The fitted box - `bounds`, plus the untied Gamma_2 magnitude's own."""
        return cls.bounds if tie else cls.bounds + [(0.0, 5.0)]

    def seed(self, previous, atm_rows, by_expiry, spy, tie):
        """The cold/warm start. The sign is seeded off the quotes, as the plain family seeds it: the
        objective kinks at zero leverage, and a smile rising with strike in the underlying's own
        units is a negative Gamma_1.
        """
        box = np.array(self.box(tie)).T
        if previous:
            start = list(self.unreparam(
                *(previous[k] for k in ('Alpha', 'Beta', 'Gamma_1', 'Phi', 'H0'))))
            if not tie:
                start.append(abs(previous['Gamma_2'] / previous['Gamma_1']))
            return np.clip(np.array(start), *box)
        # H0 seeds off the FRONT pillar and not the ladder's mean: it sets L(0), whose phase the
        # strip alternates about (`bootstrap_l`), so a mean starts half a cycle out
        var = atm_rows[0][0]['sigma'] ** 2 / spy
        # the sign off the quotes, read over the SMILE (`by_expiry`) rather than over the ATM
        # ladder, which carries one strike per expiry
        rise = sum(max(group, key=lambda r: r[0]['Strike'])[0]['sigma'] -
                   min(group, key=lambda r: r[0]['Strike'])[0]['sigma']
                   for group in by_expiry.values() if len(group) > 1)
        start = [0.9, -0.5 if rise > 0.0 else 0.5, 0.05, 0.05, np.log(var)]
        return np.clip(np.array(start + ([1.0] if not tie else [])), *box)

    @staticmethod
    def quote_vol(fitted, premium, spot, strike, is_call, units, b, n, yield_discount):
        """The fitted-minus-quoted difference in Black vol, per step-count `n`.

        Both premia go back through `bs_implied_total_var` off the forward, with the units and the
        yield rescale stripped so the two sides are the same contract. The objective minimises a
        premium residual; a desk reads a vol one. Returned absolute - the caller scales to points."""
        forward = float(spot) * np.exp(float(b) * int(n))
        scale = float(units) * float(yield_discount)
        both = []
        for value in (fitted, premium):
            call = value / scale + (0.0 if is_call else (float(spot) - float(strike)))
            both.append(np.sqrt(max(utils.bs_implied_total_var(
                call, float(spot), float(strike), float(b), int(n)), 0.0) / (int(n) / 252.0)))
        return both[0] - both[1]

    def report(self, instrument, market_price, spot, atm, wings, knots, levels, params,
               rho, spy, panels, result, elapsed, calls, notes, state, max_iter):
        """What the fit MEASURED, logged beside what it wrote - the ATM residual (which is the
        bootstrap's own convergence, not a fit quality), the worst wing, the L curve as annualised
        vol, and the wall clock against the iteration cap it was given."""
        atm_resid = 0.0
        for n, (s, strike, is_call, units, target, b, q) in atm:
            omegas = self.l_strip(knots, torch.stack(levels), int(n), rho, spy)
            fitted = float(self.price(s, strike, is_call, units, omegas, levels[0], levels[0],
                                      params[:-1], b, panels, q))
            atm_resid = max(atm_resid, abs(fitted / target - 1.0) if target else abs(fitted))
        worst, worst_vol, total = 0.0, 0.0, 0.0
        for n, strike, is_call, units, premium, weight, b, yq in wings:
            omegas = self.l_strip(knots, torch.stack(levels), int(n), rho, spy)
            fitted = float(self.price(spot, strike, is_call, units, omegas, params[-1], levels[0],
                                      params[:-1], b, panels, yq))
            worst = max(worst, abs(fitted / premium - 1.0))
            # x100 for vol points: a 5% miss on a 25 delta wing premium is a few tenths of a vol
            worst_vol = max(worst_vol, 100.0 * abs(self.quote_vol(
                fitted, premium, spot, strike, is_call, units, b, n, yq)))
            total += weight * (premium - fitted) ** 2
        logging.info(
            '{} component Heston-Nandi: Alpha {:.6g}, Beta {:.6g}, Gamma_1 {:.6g}, Rho {:g} '
            '(pinned), Phi {:.6g}, Gamma_2 {:.6g}, H0 {:.6g}'.format(
                market_price, float(params[0]), float(params[1]), float(params[2]), float(rho),
                float(params[4]), float(params[5]), float(params[6])))
        logging.info('  L curve (annualised vol): {}'.format(', '.join(
            '{:g}y {:.2%}'.format(k, float(np.sqrt(float(v) * spy)))
            for k, v in zip(knots, levels))))
        # the positivity certificate, reported rather than enforced - no box guarantees it
        certificate = self.worst_case_variance_drift(
            *[float(x) for x in (params[0], params[1], params[2], rho, params[4], params[5])],
            self.omega_floor(knots, torch.stack(levels), rho, spy))
        certified = all(x >= 0.0 for x in certificate) and not float(params[4])
        logging.info(
            '  worst-case variance step: h_(t+1) >= {:.3e} + {:.4f}*q_t + {:.4f}*h_t - {}'.format(
                *certificate + ('POSITIVE for every reachable state' if certified else
                                'NOT a certificate (Phi > 0 leaves q itself uncertified); the '
                                'simulator floors at {:g} per step and the closed form does not, '
                                'so the two part company in the tail'.format(
                                    utils.HN_COMPONENT_VARIANCE_FLOOR),)))
        logging.info(
            '  {} outer evaluations in {:.1f}s ({}), ATM residual {:.3e} (bootstrapped), worst '
            'wing {:.2%} of premium / {:.3f} vol points, weighted wing residual {:.3e}'.format(
                calls, elapsed,
                'converged: ' + str(result.message) if calls < max_iter else
                'CAPPED at Max_Iterations={} - the tolerance actually reached is the residual '
                'above, not the declared one'.format(max_iter),
                atm_resid, worst, worst_vol, total))
        for note in notes:
            logging.warning('  declining variance: {}'.format(note))
        if state.get('last_refusal'):
            logging.info('  the search visited infeasible candidates; the last said: {}'.format(
                state['last_refusal']))
        if instrument.get('Quote_Source') or instrument.get('Quote_Timestamp'):
            logging.info('  quotes: {} (as at {})'.format(
                instrument.get('Quote_Source') or 'authored by hand',
                instrument.get('Quote_Timestamp') or 'no stated time'))


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

    @property
    def quotes(self):
        """The quote leaf per benchmark, or `()` where the block asked for no `Quote_Sensitivity` -
        which is what makes the wrapper a pass-through with no edge recorded."""
        return tuple(swap.quote for swap in self.market_swaps.values() if swap.quote is not None)

    @property
    def descriptors(self):
        """The benchmark names of `quotes`, in its order - what `quote_leaves` pairs them with."""
        return [name for name, swap in self.market_swaps.items() if swap.quote is not None]

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
        `objective.reduce` rather than a `sum` spelled three times.
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

        theta = np.concatenate([soln[1][key] for key in self.keys])
        return torch.tensor(theta, dtype=self.implied_var[self.keys[0]].dtype,
                            device=self.implied_var[self.keys[0]].device)


class LeastSquaresSolve(torch.autograd.Function):
    """The swaption calibration as one differentiable node: quotes in, calibrated parameters out.

    FORWARD IS THE ORDINARY SOLVE - `SwaptionCalibration.solve` and nothing else - so enabling quote
    gradients cannot move theta*. Autograd runs `forward` with grad mode off and both optimizers
    need it on, so it is re-enabled here and each evaluation's graph is discarded with it.

    BACKWARD IS THE IMPLICIT FUNCTION THEOREM at the stationarity fixed point. This is a
    least-squares minimum, not a root: `r(theta*, q)` is never zero, so what is held fixed is
    `g = J^T r = 0`. Differentiating that and dropping the term in `d(J^T)/dtheta . r` - the
    Gauss-Newton approximation - gives

        (J^T J) dtheta/dq = -J^T dr/dq

    so a cotangent `v = dL/dtheta*` contracts as `w = (J^T J)^+ v` then `dL/dq = -(dr/dq)^T (J w)`.

    One contraction, two residuals, exact to leading order for two different reasons. The Monte
    Carlo residual is already a square, so both dropped terms are half what they correct and cancel.
    The analytic residual is separable, so the cross term is absent rather than cancelled and the
    theta-side term is the textbook `O(||r||)`: 1.50e-4 of `J^T J` in Frobenius norm beside a
    `||r||` of 1.48e-3 on the identified block, against the Monte Carlo path's 0.500064.

    Both derivatives come from autograd on ONE fresh evaluation at `(theta*, q)`, through
    `autograd.grad` rather than off `.grad`: the quote leaves accumulate across the optimizer's
    evaluations, so a harvested `.grad` is a path sum rather than the derivative at the answer.

    `J^T J` is rank deficient - J has one row per benchmark and 23 columns - so the inverse is a
    PSEUDO-inverse at a declared relative cutoff and `dtheta/dq` in a null direction is the
    minimum-norm representative. No ridge: a Tikhonov term answers a different problem.

    Stationarity is CHECKED. `solve` accepts whatever the chain returned, possibly the seed, and the
    contraction is worthless off the fixed point - so `||J^T r||` above tolerance raises.

    Every `grad` retains the graph, for the reason `CalibrationSolve` gives.
    """

    @staticmethod
    def forward(ctx, calibration, rcond, stationarity, *quotes):
        with torch.enable_grad():
            theta = calibration.solve()
        ctx.calibration, ctx.theta = calibration, theta
        ctx.rcond, ctx.stationarity = rcond, stationarity
        return theta

    @staticmethod
    def backward(ctx, cotangent):
        # grad mode here means `create_graph` - a second differentiation Gauss-Newton cannot give
        if torch.is_grad_enabled():
            raise Exception('Swaption calibration: create_graph is not supported - the backward is '
                            'a Gauss-Newton contraction and carries no second derivative')
        calibration = ctx.calibration
        with torch.enable_grad():
            x = ctx.theta.detach().requires_grad_(True)
            residual = calibration(x)
            jacobian = torch.stack([torch.autograd.grad(residual[i], x, retain_graph=True)[0]
                                    for i in range(residual.numel())]).double()
            gradient = jacobian.t() @ residual.detach().double()
            if float(gradient.norm()) > ctx.stationarity:
                raise Exception(
                    'Swaption calibration: theta* is not stationary - ||J^T r|| is {:.6g} against a '
                    'Stationarity_Tol of {:.6g}, so the implicit function theorem does not hold '
                    'there'.format(float(gradient.norm()), ctx.stationarity))
            w = torch.linalg.pinv(jacobian.t() @ jacobian, hermitian=True,
                                  rtol=ctx.rcond) @ cotangent.double()
            grads = torch.autograd.grad(residual, calibration.quotes, retain_graph=True,
                                        grad_outputs=-(jacobian @ w).to(residual.dtype))
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
                    base_currency, price_factors, price_models, ir_curve, rate)

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
                    float(implied_params['instrument'].get('Jacobian_Rcond', 1e-8)),
                    float(implied_params['instrument'].get('Stationarity_Tol', 1e-3)),
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
        F('Jacobian_Rcond', 'Float', default=1e-8,
          description='Relative cutoff on the eigenvalues of the Gauss-Newton matrix J\'J when the '
                      'backward pass inverts it. J has one row per benchmark and 23 columns, so '
                      'that matrix is rank deficient on every block quoting fewer swaptions than '
                      'that and the inverse is a pseudo-inverse: below the cutoff a direction is '
                      'one the quotes do not identify and its dtheta/dq is the minimum-norm '
                      'representative. Only used when Quote_Sensitivity is Yes'),
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

    def implied_process(self, base_currency, price_factors, price_models, ir_curve, rate):
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


def construct_bootstrapper(btype, param, dtype=torch.float32):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    return globals().get(btype)(param, device, dtype)
